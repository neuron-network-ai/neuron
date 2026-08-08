"""agent/test_probation_is_visible.py — run: python -m agent.test_probation_is_visible

A stranger's node registered, downloaded its slice, bound its port, opened its relay tunnel,
and then logged `heartbeat ok — active` for three days. It was PROBATIONARY the whole time:
excluded from routing, earning nothing, serving nothing, because the operator's verifier had
died. PROBLEMS.md [P24].

Nothing was broken on that machine. The failure was that the agent had no way to say so — it
learned its standing once, in the reply to its registration, and never asked again. So the one
line that mattered scrolled away on day one and every line after it said the node was fine.

These tests are about that property: **a node that is online and useless says so, repeatedly,
and notices when it stops being useless.**
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class Captured(logging.Handler):
    """Collect (level, rendered message) for everything the agent logs."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def at(self, level):
        return [m for lvl, m in self.records if lvl == level]

    def clear(self):
        self.records.clear()


def _agent(tmp):
    """An Agent built against a throwaway config, with its log captured."""
    from agent import agent as mod
    cfg = os.path.join(tmp, "config.json")
    mod.ensure_config(cfg)
    a = mod.Agent(config_path=cfg)
    cap = Captured()
    parent = logging.getLogger("neuron")
    for h in list(parent.handlers):
        parent.removeHandler(h)
    parent.addHandler(cap)
    parent.setLevel(logging.DEBUG)
    return mod, a, cap


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-probation-")
    mod, a, cap = _agent(tmp)
    every = mod.PROBATION_WARN_BEATS

    print("\n-- a node stuck probationary eventually says so")
    for _ in range(every - 1):
        a.note_standing("probationary")
    check("it does not cry wolf on the first beat", not cap.at(logging.WARNING),
          str(cap.at(logging.WARNING)))
    a.note_standing("probationary")
    warned = cap.at(logging.WARNING)
    check("after PROBATION_WARN_BEATS it warns", len(warned) == 1, str(warned))
    check("...saying it earns nothing", "earns no NRN" in warned[0], str(warned))
    check("...and that this machine is not the problem",
          "Nothing here needs fixing" in warned[0], str(warned))
    check("...pointing at the operator's verifier", "verifier" in warned[0], str(warned))

    print("\n-- and keeps saying it, because a log scrolls")
    cap.clear()
    for _ in range(every):
        a.note_standing("probationary")
    check("it warns again a cycle later", len(cap.at(logging.WARNING)) == 1,
          str(cap.at(logging.WARNING)))

    print("\n-- promotion is announced, not silent")
    cap.clear()
    a.note_standing("verified")
    info = cap.at(logging.INFO)
    check("becoming verified is logged", any("VERIFIED" in m for m in info), str(info))
    check("standing is tracked", a.standing == "verified", a.standing)
    check("the probation counter resets", a._probation_beats == 0, a._probation_beats)

    print("\n-- a verified node is never nagged")
    cap.clear()
    for _ in range(every * 2):
        a.note_standing("verified")
    check("no warnings for a healthy node", not cap.at(logging.WARNING),
          str(cap.at(logging.WARNING)))
    check("promotion is announced once, not every beat",
          len([m for m in cap.at(logging.INFO) if "VERIFIED" in m]) == 0,
          str(cap.at(logging.INFO)))

    print("\n-- an older coordinator that reports no standing changes nothing")
    # The ping only started carrying `standing` in this change. A node talking to a
    # coordinator that predates it must not conclude anything from the absence.
    cap.clear()
    before = a.standing
    for _ in range(every * 2):
        a.note_standing(None)
    check("no warning is invented from missing data", not cap.at(logging.WARNING),
          str(cap.at(logging.WARNING)))
    check("standing is left alone", a.standing == before, a.standing)

    print("\n-- the coordinator actually sends it")
    # The agent's half is useless if the ping does not carry the field.
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "coordinator", "main.py"), encoding="utf-8").read()
    ping = src.split("def ping(", 1)[1].split("\n@app.", 1)[0]
    check("GET /node/{id}/ping returns standing", '"standing"' in ping, ping[:400])

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
