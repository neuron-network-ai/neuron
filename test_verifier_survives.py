"""test_verifier_survives.py — run: python test_verifier_survives.py

The verifier is the service that promotes probationary nodes. When it is down, every node that
joins the network sits at zero NRN forever and nothing anywhere says why — which is exactly
what happened for two days in PROBLEMS.md [P24].

Two failures made that possible, and these tests cover both:

  1. **A transient coordinator error cost a whole cycle.** The last three lines the service
     ever wrote were one 502 and two DNS failures — the normal weather on a home connection.
     One bad read must not skip a sweep's promotions.
  2. **A healthy verifier logged nothing at all**, so its log was identical whether it was
     running or had been dead since Monday. Silence has to mean dead.
"""
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests                                                        # noqa: E402

import verify_service                                                  # noqa: E402

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
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def at(self, level):
        return [m for lvl, m in self.records if lvl == level]

    def clear(self):
        self.records.clear()


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise requests.HTTPError(f"{self.status} Server Error")

    def json(self):
        return self.payload


def main():
    cap = Capture()
    verify_service.log.handlers = [cap]
    verify_service.log.setLevel(logging.DEBUG)
    verify_service.ROSTER_BACKOFF_S = 0            # do not actually sleep in a test

    v = verify_service.Verifier("http://coordinator.test", "secret", interval=1)
    roster = {"nodes": [{"node_id": "agent-x", "standing": "verified", "status": "online",
                         "tailscale_ip": "10.0.0.1", "port": 50999,
                         "layer_start": 0, "layer_end": 9}]}

    print("\n-- a transient 502 does not cost the cycle")
    calls = {"n": 0}

    def flaky(url, **kw):
        calls["n"] += 1
        return FakeResponse(status=502) if calls["n"] == 1 else FakeResponse(roster)

    real_get = requests.get
    try:
        requests.get = flaky
        nodes = v.nodes()
    finally:
        requests.get = real_get
    check("the roster is read on the retry", len(nodes) == 1, str(nodes))
    check("it took a second attempt to get there", calls["n"] == 2, str(calls))

    print("\n-- a real outage is reported, and escalates")
    def dead(url, **kw):
        raise requests.ConnectionError("Failed to resolve 'neuronnet.duckdns.org'")

    try:
        requests.get = dead
        cap.clear()
        v.sweep()
        first = cap.at(logging.WARNING)
        check("one bad cycle is a warning", len(first) == 1, str(first))
        check("...naming the consequence", "promoted" in first[0], str(first))
        for _ in range(verify_service.UNREACHABLE_ESCALATE - 1):
            v.sweep()
        check("a sustained outage escalates to ERROR", len(cap.at(logging.ERROR)) == 1,
              str(cap.at(logging.ERROR)))
        check("every attempt was retried",
              v.unreachable == verify_service.UNREACHABLE_ESCALATE, v.unreachable)

        print("\n-- recovery is stated, so the log is not left on a cliffhanger")
        cap.clear()
        requests.get = lambda url, **kw: FakeResponse(roster)
        v.sweep()
        info = cap.at(logging.INFO)
        check("coming back is logged", any("reachable again" in m for m in info), str(info))
        check("the counter resets", v.unreachable == 0, v.unreachable)
    finally:
        requests.get = real_get

    print("\n-- a healthy verifier proves it is alive")
    # The [P24] detection gap: `nothing to verify` is debug-level, so a running verifier and a
    # dead one wrote the same thing -- nothing.
    v.last_roster = (3, 0)
    line = v._alive_line()
    check("the alive line says what it saw", "3 node(s), 0 awaiting verification" == line, line)
    v.unreachable = 2
    check("an unreachable verifier says that instead", "unreachable" in v._alive_line(),
          v._alive_line())

    print("\n-- run() emits it on schedule, and a bad sweep never kills the loop")
    v.unreachable, v.last_roster = 0, (1, 0)
    cap.clear()
    sweeps = {"n": 0}

    def exploding_sweep():
        sweeps["n"] += 1
        if sweeps["n"] == 1:
            raise RuntimeError("one bad cycle")
        if sweeps["n"] > verify_service.ALIVE_EVERY:
            raise KeyboardInterrupt          # stop the loop from the inside
        return 0

    v.sweep = exploding_sweep
    real_sleep, time.sleep = time.sleep, lambda s: None
    try:
        v.run()
    except KeyboardInterrupt:
        pass
    finally:
        time.sleep = real_sleep
    alive = [m for m in cap.at(logging.INFO) if m.startswith("alive |")]
    check("a crashing sweep is logged, not fatal",
          any("sweep failed" in m for m in cap.at(logging.ERROR)), str(cap.at(logging.ERROR)))
    check("the loop kept going after it", sweeps["n"] > verify_service.ALIVE_EVERY,
          str(sweeps))
    check("an alive line lands every ALIVE_EVERY sweeps", len(alive) == 1, str(alive))

    print("\n-- the keepalive does not mistake 'a process exists' for 'it is working'")
    # The first version of agent/verifier_keepalive.py counted any process whose command line
    # mentioned verify_service.py as healthy. A verifier that HUNG would then be reported fine
    # forever and never restarted -- the same "online means nothing" mistake as [P21] (an agent
    # logging heartbeats while its listener had never bound) and [P22] (a relay accepting
    # connections and carrying nothing), made by the very file written to prevent [P24].
    from agent import verifier_keepalive as ka
    import tempfile as _tf
    tmp = _tf.mkdtemp(prefix="neuron-ka-")
    ka.VERIFIER_LOG = os.path.join(tmp, "verify_service.log")

    real_procs, real_age = ka.verifier_processes, ka.oldest_process_age_s
    try:
        def scenario(pids, log_age_min, proc_age_min):
            ka.verifier_processes = lambda: pids
            ka.oldest_process_age_s = lambda p: (proc_age_min * 60
                                                 if proc_age_min is not None else None)
            if log_age_min is None:
                if os.path.exists(ka.VERIFIER_LOG):
                    os.remove(ka.VERIFIER_LOG)
            else:
                open(ka.VERIFIER_LOG, "w").close()
                t = time.time() - log_age_min * 60
                os.utime(ka.VERIFIER_LOG, (t, t))
            return ka.verifier_state()

        st, _ = scenario([], 5, None)
        check("no process at all is dead", st == "dead", st)

        st, _ = scenario([101], 5, 120)
        check("a process writing to its log is running", st == "running", st)

        st, why = scenario([101, 102], 200, 300)
        check("a long-silent process is HUNG, not running", st == "hung", f"{st}: {why}")
        check("...and the reason names the silence", "has not been written" in why, why)

        # The trap: right after a real outage the log is days stale and the process is seconds
        # old. That is the verifier that was just restarted, still loading torch.
        st, _ = scenario([101], 2880, 1)
        check("a just-started process with an old log is NOT killed", st == "running", st)

        st, _ = scenario([101], None, 1)
        check("a process that has not logged yet is given the benefit of the doubt",
              st == "running", st)

        ka.verifier_processes = lambda: None
        st, _ = ka.verifier_state()
        check("no psutil means 'unknown', never 'dead'", st == "unknown", st)
        # Assert on the ACTION, not the exit code: "did it launch anything?" is the question.
        launched = []
        real_start, ka.start_verifier = ka.start_verifier, lambda: launched.append(1)
        try:
            ka.main([])
            check("...and unknown starts nothing", not launched,
                  "a second verifier would double every proof-of-compute on the network")
            ka.verifier_processes = lambda: []          # genuinely dead
            ka.main([])
            check("a genuinely dead verifier IS started", len(launched) == 1, str(launched))
        finally:
            ka.start_verifier = real_start

        print("\n-- one logical verifier can be two processes")
        # Found live: on Windows a venv's pythonw.exe is a redirector that spawns the base
        # interpreter as a child, so `verify_service.py` matched twice. A restart that killed
        # only the first would leave the half doing the actual work running.
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "agent", "verifier_keepalive.py"), encoding="utf-8").read()
        check("every matching process is returned, not the first",
              "pids.append" in src and "return pids" in src)
        check("...and all of them are stopped together",
              "def stop_verifier(pids)" in src and "wait_procs" in src)
        check("terminate is tried before kill", src.index("p.terminate()") < src.index("p.kill()"))
    finally:
        ka.verifier_processes, ka.oldest_process_age_s = real_procs, real_age

    print("\n-- the doctor calls a stale verifier log dead")
    import neuron_doctor
    rep = neuron_doctor.Report()
    stale = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_test_stale_verifier.log")
    with open(stale, "w") as f:
        f.write("2026-08-03 17:27:47 coordinator unreachable\n")
    old = time.time() - neuron_doctor.VERIFIER_STALE_S - 60
    os.utime(stale, (old, old))
    real_path, neuron_doctor.VERIFIER_LOG = neuron_doctor.VERIFIER_LOG, stale
    try:
        neuron_doctor.check_verifier(rep)
    finally:
        neuron_doctor.VERIFIER_LOG = real_path
        os.remove(stale)
    check("a log that stopped growing fails the check", len(rep.failed) == 1,
          str(rep.rows))
    check("...and says what it costs", "probationary" in rep.rows[0]["fix"], str(rep.rows))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
