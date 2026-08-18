"""agent/test_agent_death_is_loud.py — the agent thread may never die quietly ([P51]).

    python -m agent.test_agent_death_is_loud

**Live 2026-08-18, 81 minutes.** `neuron-agent.exe` was alive, holding port 50999, answering
proof-of-compute probes correctly with `holds: [0, 9]`, serving its Chat UI — and the
coordinator had it down as `offline` while `agent.log` carried not one line from that process.
Auto-repair reacted to what it could see and collapsed the whole network onto the other machine.

The hole: `Tray.run` was

    threading.Thread(target=self.agent.run, daemon=True).start()

with no wrapper. An exception there goes to `threading.excepthook`, which writes to
`sys.stderr` — and the tray is a FROZEN WINDOWED app whose console `_hide_console()` has
already hidden, so stderr goes nowhere. The agent loop could stop dead while the tray icon, the
poll thread, the Chat UI and `node_server`'s listener all carried on. No log line, no `[CRASH]`
marker; on the real machine none has ever been written.

Whether that is exactly what happened on 18 August cannot be proved after the fact — the restart
destroyed the evidence. What IS proved here is that it is a path to precisely that symptom, and
that the path is now closed. Two halves, because the loop can fail in two ways:

  * it RAISES, or returns — `_supervise_agent`;
  * it keeps running and stops reaching the coordinator — `_watch_heartbeat`.

The second is the one that actually matches the incident, and no amount of exception handling
would have caught it.
"""
import io
import logging
import os
import sys
import time
import types

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


class _FakeAgent:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.state = {"status": "starting"}
        self.cfg = {}
        self.base = "http://c"
        self.user_paused = types.SimpleNamespace(is_set=lambda: False)

    def run(self):
        if self.behaviour == "raise":
            raise RuntimeError("boom in the agent loop")
        if self.behaviour == "return":
            return
        while True:                      # pragma: no cover - the healthy case
            time.sleep(3600)


def main():
    from agent import tray as T

    # Capture what reaches the log AND what reaches crash_log, since the whole point is that
    # the failure must survive a broken logging config.
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logging.getLogger("neuron").addHandler(handler)
    logging.getLogger("neuron").setLevel(logging.DEBUG)
    crashes = []
    real_crash = T.crash_log
    T.crash_log = lambda what: crashes.append(what)

    try:
        print("\n-- a loop that RAISES is recorded, not swallowed")
        t = T.Tray.__new__(T.Tray)
        t.agent = _FakeAgent("raise")
        T.Tray._supervise_agent(t)
        out = stream.getvalue()
        check("the exception reaches the log", "STOPPED" in out and "boom" in out, out[-200:])
        check("...and crash_log, which does not depend on logging",
              any("boom" in c for c in crashes), str(crashes))
        check("...and the tray goes to an error state a person can see",
              t.agent.state.get("status") == "error", str(t.agent.state))
        check("...naming the remedy rather than the traceback",
              "restart" in (t.agent.state.get("detail") or "").lower(),
              str(t.agent.state))
        check("supervising does not re-raise into a dead stderr", True)

        print("\n-- a loop that RETURNS is also a stop")
        # An endless loop reaching its end is indistinguishable from a healthy agent unless
        # somebody says so. This is the quieter half and the easier one to miss.
        stream.truncate(0), stream.seek(0)
        crashes.clear()
        t2 = T.Tray.__new__(T.Tray)
        t2.agent = _FakeAgent("return")
        T.Tray._supervise_agent(t2)
        out = stream.getvalue()
        check("a silent return is reported", "RETURNED" in out, out[-200:])
        check("...and says the node has stopped earning", "stopped earning" in out)
        check("...and shows in the tray", t2.agent.state.get("status") == "error")

        print("\n-- a loop that is ALIVE but not getting through — the actual incident")
        stream.truncate(0), stream.seek(0)
        crashes.clear()
        t3 = T.Tray.__new__(T.Tray)
        t3.agent = _FakeAgent("hang")

        t3.agent.state["last_beat_at"] = time.time()
        T.Tray._watch_heartbeat(t3)
        check("a fresh beat is quiet", stream.getvalue() == "", stream.getvalue())

        t3.agent.state["last_beat_at"] = time.time() - (T.Tray.BEAT_STALE_S + 60)
        T.Tray._watch_heartbeat(t3)
        out = stream.getvalue()
        check("a stale beat is reported LOUDLY", "no successful heartbeat" in out, out[-200:])
        check("...saying the node is earning nothing while the process runs",
              "earning nothing" in out and "still running" in out)
        check("...through crash_log too", bool(crashes), str(crashes))

        # An alarm that repeats every 30s is one people silence, and then it cannot report
        # the next thing.
        stream.truncate(0), stream.seek(0)
        T.Tray._watch_heartbeat(t3)
        check("it is not repeated every poll", "no successful heartbeat" not in stream.getvalue())

        stream.truncate(0), stream.seek(0)
        t3.agent.state["last_beat_at"] = time.time()
        T.Tray._watch_heartbeat(t3)
        check("recovery is announced, so the all-clear is visible too",
              "getting through again" in stream.getvalue(), stream.getvalue())

        print("\n-- a node that has never beaten is starting, not stalled")
        stream.truncate(0), stream.seek(0)
        t4 = T.Tray.__new__(T.Tray)
        t4.agent = _FakeAgent("hang")
        T.Tray._watch_heartbeat(t4)
        check("no alarm before the first beat", stream.getvalue() == "", stream.getvalue())

        print("\n-- and the thread is actually wired to the supervisor")
        # A supervisor nothing calls is the same bug with more code in it.
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tray.py"),
                   encoding="utf-8").read()
        # Scoped to run()'s BODY, not the whole file: `_supervise_agent`'s docstring quotes
        # the old line verbatim to explain the bug, and a file-wide ban matched that and failed
        # on a file that was already correct. Banning a string that also appears in the prose
        # explaining why it is banned is its own small trap.
        body = src[src.index("    def run(self):"):]
        body = body[:body.index(chr(10) + chr(10))]
        check("Tray.run starts _supervise_agent, not agent.run directly",
              "target=self._supervise_agent" in body and "target=self.agent.run" not in body,
              body)
        check("the poll loop calls the watchdog", "self._watch_heartbeat()" in src)
    finally:
        logging.getLogger("neuron").removeHandler(handler)
        T.crash_log = real_crash

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
