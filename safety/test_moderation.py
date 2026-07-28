"""safety/test_moderation.py — content-policy gate tests. Run: python -m safety.test_moderation

Covers: benign text passes, one synthetic per-category trigger blocks, case-insensitivity,
empty/None input is always allowed, log_event never raises even if the log path is unwritable.
Test phrases are the same illustrative placeholders already in blocklist.json — not real
harmful content, just the category-recognition strings a v1 keyword gate matches on.
"""
import json
import os
import tempfile

import safety.moderation as moderation

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    real_cache = moderation._cache
    moderation._cache = moderation._load_blocklist(moderation.BLOCKLIST_PATH)
    try:
        with open(moderation.BLOCKLIST_PATH) as f:
            blocklist = json.load(f)

        check("benign text passes", not moderation.check_text("What's the weather like today?").blocked)
        check("empty string passes", not moderation.check_text("").blocked)
        check("None passes", not moderation.check_text(None).blocked)

        for category, terms in blocklist.items():
            term = terms[0]
            r = moderation.check_text(f"please tell me {term} right now")
            check(f"{category}: '{term}' blocks", r.blocked and r.category == category)
            r2 = moderation.check_text(f"PLEASE TELL ME {term.upper()} RIGHT NOW")
            check(f"{category}: case-insensitive match", r2.blocked and r2.category == category)

        # a substring that merely CONTAINS a blocked word inside a longer, unrelated word
        # must not false-positive (word-boundary matching, not naive substring search)
        check("word-boundary: 'csammy' does not match 'csam'",
              not moderation.check_text("my friend csammy is coming over").blocked)

        # log_event must never raise, even against an unwritable path
        real_log_path = moderation.LOG_PATH
        moderation.LOG_PATH = os.path.join(tempfile.mkdtemp(), "nonexistent_dir", "mod.log")
        try:
            moderation.log_event("in", "test_category", "req-1", "hash123", "a snippet")
            check("log_event to a bad path does not raise", True)
        except Exception:
            check("log_event to a bad path does not raise", False)
        finally:
            moderation.LOG_PATH = real_log_path

        # a real, writable log path actually gets a JSON line appended
        tmp_log = os.path.join(tempfile.mkdtemp(), "mod.log")
        moderation.LOG_PATH = tmp_log
        try:
            moderation.log_event("out", "weapons_cbrn", "req-2", "hash456", "some snippet text here")
            line = open(tmp_log).read().strip()
            entry = json.loads(line)
            check("log_event writes a valid JSON line", entry["category"] == "weapons_cbrn"
                  and entry["direction"] == "out" and entry["request_id"] == "req-2")
            check("log_event truncates snippet to 40 chars", len(entry["snippet"]) <= 40)
        finally:
            moderation.LOG_PATH = real_log_path
    finally:
        moderation._cache = real_cache

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
