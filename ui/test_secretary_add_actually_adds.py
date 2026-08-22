"""The assistant's task list actually stores a task — run:
    python -m ui.test_secretary_add_actually_adds

Reported as "assistant is not adding task", and it had never worked. The workspace UI sends
`{kind, title}` and renders `item.title`; this Python reimplementation of its Express API
invented `{text, done}` instead. So every task anybody typed was answered with 400 "text is
required" — and the client does not read the response, so the box cleared, no row appeared, and
nothing anywhere said why. A feature that is 100% broken and 0% noisy.

The shape mismatch went further than the one field:

  * LIST returned items whose `title` was undefined, so even a stored task rendered as a blank
    line;
  * UPDATE copied only `text`/`done`, so ticking one off did nothing either.

What is pinned here is the CONTRACT, in the direction that matters: the client's declared
`SecretaryItem` is the shape, because that is what ships in the bundle and what the panel
renders. The server reads the old field names too, so a list already on somebody's disk from
the previous shape is not thrown away — this is their list of things to remember, and a rename
is not a reason to lose it.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio  # noqa: E402

from ui import workspace_api as wa  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeRequest:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def call(fn, payload):
    return asyncio.run(fn(FakeRequest(payload)))


def main():
    tmp = tempfile.mkdtemp(prefix="neuron_sec_")
    real_state = wa._state_dir
    wa._state_dir = lambda: __import__("pathlib").Path(tmp)
    try:
        # -- WHAT THE CLIENT ACTUALLY SENDS ------------------------------------------- #
        res = call(wa.secretary_add, {"kind": "task", "title": "buy milk"})
        added = not hasattr(res, "status_code")
        check("a task sent as the client sends it is accepted", added)
        if not added:
            print(f"        server said: {getattr(res, 'body', b'')!r}")
        check("...and comes back with the title the client renders",
              added and res["item"]["title"] == "buy milk")
        def stored_titles():
            # Guarded: with the bug present nothing is written at all, and a test that raises
            # here reports a traceback instead of naming which promise broke.
            try:
                with open(os.path.join(tmp, "secretary.json"), encoding="utf-8") as f:
                    return [i.get("title") or i.get("text") for i in json.load(f)["items"]]
            except (OSError, ValueError, KeyError):
                return []

        check("...and is stored, not just echoed", stored_titles() == ["buy milk"])

        # -- and the list the panel reads shows it --------------------------------------- #
        listed = wa.secretary_list()
        rows = listed["items"]
        check("the list returns it", len(rows) == 1)
        check("...with a title, so the row is not blank",
              bool(rows) and rows[0].get("title") == "buy milk")
        check("...and counts it as open", listed["stats"]["open"] == 1)

        # -- ticking it off, which arrives as a status ---------------------------------- #
        # Guarded like the rest: with add broken there is nothing to tick, and the run has to
        # keep going so every broken promise is named rather than the first one aborting it.
        sid = rows[0]["id"] if rows else "no-such-id"
        upd = call(wa.secretary_update, {"id": sid, "status": "done"})
        check("marking it done is accepted", not hasattr(upd, "status_code"))
        check("...and both fields agree afterwards, never one open and one done",
              upd["item"]["status"] == "done" and upd["item"]["done"] is True)
        check("...and the open count drops", wa.secretary_list()["stats"]["open"] == 0)
        check("open=1 filters it out", len(wa.secretary_list(open="1")["items"]) == 0)

        # -- an empty title is still refused, and says which field ---------------------- #
        bad = call(wa.secretary_add, {"kind": "task", "title": "   "})
        check("an empty title is refused", getattr(bad, "status_code", 200) == 400)

        # -- A LIST WRITTEN IN THE OLD SHAPE IS NOT LOST -------------------------------- #
        # Somebody's existing secretary.json holds {text, done}. It is their list; a rename on
        # our side must not empty it.
        with open(os.path.join(tmp, "secretary.json"), "w", encoding="utf-8") as f:
            json.dump({"items": [{"id": "old1", "text": "written before the fix",
                                  "kind": "task", "done": False, "createdAt": 1}]}, f)
        old = wa.secretary_list()
        check("an item stored in the OLD shape still lists", len(old["items"]) == 1)
        check("...and is shown with a title rather than blank",
              old["items"][0]["title"] == "written before the fix")
        check("...and is counted as open", old["stats"]["open"] == 1)
        upd2 = call(wa.secretary_update, {"id": "old1", "status": "done"})
        check("...and can still be ticked off", upd2["item"]["done"] is True)

        # -- removing works on either shape ---------------------------------------------- #
        rem = call(wa.secretary_remove, {"id": "old1"})
        check("removing it leaves an empty list", rem["items"] == [])
    finally:
        wa._state_dir = real_state
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
