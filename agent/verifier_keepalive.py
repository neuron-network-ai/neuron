"""agent/verifier_keepalive.py — start the operator's verifier if it is not running.

[P24]. `verify_service.py` promotes probationary nodes; while it is down, every node that
joins the network sits at zero NRN forever and nothing says why. On Windows its auto-start is
an HKCU Run key, which fires **once, at login, and never again** — so a crash, an OOM kill or
a closed session removes it until a human notices. Nobody noticed for two days, because a dead
verifier and an idle one looked identical from the outside.

This is the Windows counterpart of the cron keepalive `install.py` already installs on Linux
for the agent (`agent/neuron-keepalive.sh`): run every few minutes by a scheduled task, it
does nothing at all while the verifier is alive and restarts it when it is not, bounding an
outage at one interval instead of "until somebody looks".

    python -m agent.verifier_keepalive            # one check
    python -m agent.verifier_keepalive --verbose  # say what it found

Note it cannot match itself the way an inline shell grep would (the reason
`neuron-keepalive.sh` lives in its own file): this process's command line contains
`verifier_keepalive`, never `verify_service.py`, so the scan below cannot mistake the guard
for the thing it guards.
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VERIFIER = os.path.join(REPO, "verify_service.py")
VERIFIER_LOG = os.path.join(REPO, "verify_service.log")
IS_WINDOWS = sys.platform == "win32"

# A healthy verifier writes an "alive" line every ALIVE_EVERY (30) sweeps of its interval (60s)
# -- once every 30 minutes. Three missed heartbeats is not slow, it is stopped. Same number
# neuron_doctor.py uses, deliberately: one cadence, one threshold.
STALE_AFTER_S = 90 * 60


def verifier_processes():
    """PIDs running verify_service.py, or None if that cannot be determined.

    None means "could not ask" (no psutil) and must never be read as "not running": launching a
    second verifier would double every proof-of-compute on the network, and two of them
    attesting the same node is worse than a delayed restart.

    Returns every match, not the first. On Windows a venv's `pythonw.exe` is a redirector that
    spawns the base interpreter as a CHILD, so one logical verifier is two processes -- and a
    restart that killed only one of them would leave the working half behind.
    """
    try:
        import psutil
    except ImportError:
        return None
    pids, me = [], os.getpid()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = p.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if p.info["pid"] == me:
            continue
        if any("verify_service.py" in str(part) for part in cmd):
            pids.append(p.info["pid"])
    return pids


def log_age_s():
    """Seconds since the verifier last wrote to its log, or None if it never has."""
    try:
        return time.time() - os.path.getmtime(VERIFIER_LOG)
    except OSError:
        return None


def oldest_process_age_s(pids):
    """Age of the longest-running verifier process, or None if unknown.

    Load-bearing. Right after a real outage the log is hours or days stale while the process is
    seconds old -- that is a verifier that was JUST restarted and is still loading torch, not a
    hung one. Without this guard the keepalive would kill the verifier it had just started, on
    every run, forever.
    """
    try:
        import psutil
    except ImportError:
        return None
    ages = []
    for pid in pids:
        try:
            ages.append(time.time() - psutil.Process(pid).create_time())
        except Exception:
            continue
    return max(ages) if ages else None


def verifier_state():
    """'running' | 'hung' | 'dead' | 'unknown', and a human-readable reason.

    Process presence is NOT liveness. That mistake is catalogued twice in PROBLEMS.md -- [P21]
    (an agent logging "heartbeat ok" while its listener had never bound) and [P22] (a relay
    tunnel accepting connections and carrying nothing) -- and the first version of this file
    made it a third time: any process whose command line mentioned verify_service.py counted as
    healthy, so a verifier that HUNG rather than died would be reported fine forever and never
    restarted. That is exactly the [P24] outcome this file exists to prevent.
    """
    pids = verifier_processes()
    if pids is None:
        return "unknown", "psutil is not installed, so processes cannot be inspected"
    if not pids:
        return "dead", "no verify_service.py process"
    age = log_age_s()
    if age is None:
        return "running", f"{len(pids)} process(es), no log written yet"
    proc_age = oldest_process_age_s(pids)
    if age > STALE_AFTER_S and (proc_age is None or proc_age > STALE_AFTER_S):
        return "hung", (f"{len(pids)} process(es) alive but the log has not been written for "
                        f"{int(age // 60)} minutes")
    return "running", f"{len(pids)} process(es), log written {int(age // 60)}m ago"


def stop_verifier(pids):
    """Terminate a hung verifier so it can be replaced. SIGTERM first, then kill what is left.

    Killing a process is the one thing here with consequences, so it is gated on the 90-minute
    silence above -- a working verifier speaks every 30 minutes, so this cannot fire against
    one that is doing its job.
    """
    try:
        import psutil
    except ImportError:
        return
    procs = []
    for pid in pids:
        try:
            procs.append(psutil.Process(pid))
        except Exception:
            continue
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    gone, alive = psutil.wait_procs(procs, timeout=10)
    for p in alive:
        try:
            p.kill()
        except Exception:
            pass


def start_verifier():
    """Launch the verifier detached, with the SAME interpreter that runs this keepalive.

    Never a PATH lookup: this repo's working interpreter is a venv with PyTorch, while `python`
    on PATH here is a bare install without it — install.py already learned that the hard way
    when a Run key written against PATH died on import at every boot.
    """
    py = sys.executable
    if IS_WINDOWS:
        cand = os.path.join(os.path.dirname(py), "pythonw.exe")
        if os.path.exists(cand):
            py = cand
        subprocess.Popen([py, VERIFIER], cwd=REPO,
                         creationflags=0x00000008)          # DETACHED_PROCESS
    else:
        subprocess.Popen([py, VERIFIER], cwd=REPO, start_new_session=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Restart verify_service.py if it is not working.")
    ap.add_argument("--verbose", action="store_true", help="report what was found")
    ap.add_argument("--check-only", action="store_true",
                    help="report the state and change nothing (exit 1 if not running)")
    args = ap.parse_args(argv)

    if not os.path.exists(VERIFIER):
        if args.verbose:
            # ASCII: this runs from a scheduled task whose console codepage is not UTF-8.
            print(f"{VERIFIER} not found - nothing to keep alive")
        return 0

    state, why = verifier_state()
    if args.check_only:
        print(f"{state}: {why}")
        return 0 if state in ("running", "unknown") else 1
    if state == "unknown":
        if args.verbose:
            print(f"doing nothing - {why}")
        return 0
    if state == "running":
        if args.verbose:
            print(f"verifier is running - {why}")
        return 0
    if state == "hung":
        # Silent for 90 minutes with a process still up. Whatever it is doing, it is not
        # verifying anything, and a stranger who joined is sitting at zero NRN because of it.
        print(f"verifier appears hung ({why}) - stopping it and starting a fresh one")
        stop_verifier(verifier_processes() or [])
    start_verifier()
    print(f"verifier was not working ({state}) - started a fresh one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
