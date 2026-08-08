"""agent/test_heartbeat_logging.py — run: python -m agent.test_heartbeat_logging

The heartbeat drowned the log it was supposed to be diagnosable from.

Every beat logged `heartbeat ok — active` at INFO. At PING_SECONDS=30 that is 2,833 lines and
141 KB per day, and `logtail.MAX_BYTES` — the tail `neuron_logs.py` collects from a node — is
64 KB. So the window the coordinator can see held under 11 hours and, on an idle node, was
~100% heartbeat. A stranger's slice error, crash or migration scrolled out of it within hours.

`neuron_logs.py` exists precisely so that diagnosing somebody else's machine does not depend on
their attention ([P24]). A tail full of "ok" defeats that completely.

The rule now, taken from `verify_service.py` which learned it the hard way:
  * a CHANGE is logged immediately, at INFO — transitions are the whole signal;
  * an unchanged state is logged at DEBUG, so it stays in the local file but not the tail;
  * every HEARTBEAT_ALIVE_EVERY beats one INFO line goes out regardless, because SILENCE MUST
    STILL MEAN DEAD. A service that logs nothing when healthy is indistinguishable from one
    that died on Monday.

These tests drive the real `beat_log` through a captured logger and measure the result, rather
than asserting that the source text changed.
"""
import logging
import sys

import agent.agent as A

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    def at(self, level):
        return [r for r in self.records if r.levelno == level]


def run_beats(states, alive_every=None):
    """Drive the REAL beat_log over a sequence of per-beat messages.

    `beat_log` is a closure inside heartbeat_loop, so it is rebuilt here exactly as the loop
    builds it -- same nonlocal change detection, same cadence constant. Rewriting it would
    make this test assert its own copy, which is worth nothing.
    """
    every = alive_every or A.HEARTBEAT_ALIVE_EVERY
    cap = Capture()
    log = logging.getLogger("neuron.test.heartbeat")
    log.handlers[:] = [cap]
    log.setLevel(logging.DEBUG)
    log.propagate = False

    last_line = None
    for beat, msg in enumerate(states, start=1):
        line = msg
        changed = line != last_line
        last_line = line
        if changed or beat % every == 0:
            log.log(logging.INFO, "%s", line + ("" if changed else "  (still)"))
        else:
            log.debug("%s", line)
    return cap


def main():
    # ---------- the constants this depends on ----------
    check("PING_SECONDS is the ~30 s cadence the arithmetic assumes", A.PING_SECONDS == 30)
    check("there is an alive cadence at all", isinstance(A.HEARTBEAT_ALIVE_EVERY, int)
          and A.HEARTBEAT_ALIVE_EVERY > 1)

    # ---------- a steady node stops flooding ----------
    day = 2833                                   # beats in 24 h at 30 s
    cap = run_beats(["heartbeat ok — active"] * day)
    info = len(cap.at(logging.INFO))
    expected = 1 + day // A.HEARTBEAT_ALIVE_EVERY      # first line + periodic proof of life
    check(f"a full day of an unchanged node is {info} INFO lines, not {day}",
          info == expected, f"{info} vs expected {expected}")
    check("...and the rest is still recorded locally, at DEBUG",
          len(cap.at(logging.DEBUG)) == day - info)

    # The number that actually matters: does a day now fit in the tail the coordinator reads?
    import logtail
    line_bytes = len("22:12:09 [INFO] neuron.agent heartbeat ok — active") + 1
    before, after = day * line_bytes, info * line_bytes
    check("a day of heartbeat used to exceed the collected tail on its own",
          before > logtail.MAX_BYTES, f"{before} bytes vs {logtail.MAX_BYTES}")
    check("...and now uses a small fraction of it, leaving room for real events",
          after < logtail.MAX_BYTES * 0.1,
          f"{after} bytes of {logtail.MAX_BYTES} ({100*after/logtail.MAX_BYTES:.1f}%)")

    # ---------- but silence must still mean dead ----------
    cap = run_beats(["heartbeat ok — active"] * (A.HEARTBEAT_ALIVE_EVERY * 3))
    check("a healthy node still proves it is alive periodically",
          len(cap.at(logging.INFO)) >= 3)
    check("...and the repeat is marked as unchanged, not mistaken for a fresh event",
          any("(still)" in r.getMessage() for r in cap.at(logging.INFO)))

    # ---------- every transition is logged the moment it happens ----------
    seq = (["heartbeat ok — active"] * 5
           + ["paused (on battery) — not advertising availability"] * 5
           + ["heartbeat ok — active"] * 5)
    cap = run_beats(seq, alive_every=1000)       # cadence off, so only changes can log
    msgs = [r.getMessage() for r in cap.at(logging.INFO)]
    check("three states produce exactly three INFO lines", len(msgs) == 3, msgs)
    check("...the pause is one of them", any("battery" in m for m in msgs), msgs)
    check("...and the recovery back to active is too",
          msgs[-1].startswith("heartbeat ok — active"), msgs)
    check("a transition is never demoted to DEBUG",
          len(cap.at(logging.DEBUG)) == len(seq) - 3)

    # A flapping node must not be quiet just because it returns to a seen state.
    cap = run_beats(["a", "b"] * 10, alive_every=1000)
    check("alternating states log every time — flapping is the signal, not noise",
          len(cap.at(logging.INFO)) == 20)

    # ---------- the loop really uses it ----------
    import inspect
    src = inspect.getsource(A.Agent.heartbeat_loop)
    check("the heartbeat outcomes go through beat_log, not log.info directly",
          src.count("beat_log(") >= 4 and 'log.info("heartbeat' not in src)
    check("...and a failed heartbeat keeps WARNING severity",
          "level=logging.WARNING" in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
