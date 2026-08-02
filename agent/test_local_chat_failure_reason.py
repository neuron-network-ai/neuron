"""agent/test_local_chat_failure_reason.py — run:
python -m agent.test_local_chat_failure_reason

When the personal Chat UI fails, the agent logged:

    local Chat UI failed to start on port 8080 — ... check whether that port is already in use

regardless of what actually went wrong, because start() returns None and the caller had no
reason to report. On a packaged build the real failure was `No module named '_sqlite3'`, and
that message sent everyone — including the person debugging it — to inspect a port that was
free. A confident wrong diagnosis costs more than no diagnosis.

So: start() records the reason, the agent reports THAT, and the port guess is gone.
"""
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import local_chat                     # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class Recorder(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage() % () if not record.args
                          else record.getMessage())

    def text(self):
        return "\n".join(self.lines)


def main():
    from agent import agent as agent_mod

    print("\n-- start() records why it failed")
    local_chat.LAST_ERROR = None
    # ensure_driver_slice is the first thing start() touches; make it blow up the way the
    # packaged build did.
    real_ensure = local_chat.ensure_driver_slice
    local_chat.ensure_driver_slice = lambda *a, **k: (_ for _ in ()).throw(
        ModuleNotFoundError("No module named '_sqlite3'"))
    try:
        server = local_chat.start("http://c", "some/model", tempfile.mkdtemp(), port=8080)
    finally:
        local_chat.ensure_driver_slice = real_ensure
    check("a failed start returns None", server is None)
    check("and the reason is recorded", local_chat.LAST_ERROR is not None)
    check("naming the real exception",
          "_sqlite3" in (local_chat.LAST_ERROR or ""), str(local_chat.LAST_ERROR))
    check("with its type", "ModuleNotFoundError" in (local_chat.LAST_ERROR or ""),
          str(local_chat.LAST_ERROR))

    print("\n-- the agent reports that reason, not a guess about the port")
    rec = Recorder()
    parent = logging.getLogger("neuron")
    parent.addHandler(rec)
    prev_level = parent.level
    parent.setLevel(logging.INFO)
    try:
        a = agent_mod.Agent.__new__(agent_mod.Agent)      # no __init__: no network, no disk
        a.cfg = {"local_chat": True, "model_id": "some/model", "local_chat_port": 8080}
        a.base = "http://c"
        a.local_chat_state = "pending"
        a.local_chat_error = None
        real_start = local_chat.start
        local_chat.start = lambda *args, **kw: None       # simulate the failure
        local_chat.LAST_ERROR = "ModuleNotFoundError: No module named '_sqlite3'"
        try:
            a.start_local_chat()
        finally:
            local_chat.start = real_start
    finally:
        parent.removeHandler(rec)
        parent.setLevel(prev_level)

    body = rec.text()
    check("state is marked failed", a.local_chat_state == "failed")
    check("the reason is kept on the agent for the tray",
          "_sqlite3" in (a.local_chat_error or ""), str(a.local_chat_error))
    check("the log line carries the real cause", "_sqlite3" in body, body)
    check("and no longer asserts the port is in use",
          "already in use" not in body, body)
    check("the port is still mentioned, as context not as a verdict",
          "8080" in body, body)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
