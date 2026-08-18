"""
neuron_doctor.py — is this network actually able to serve right now?

Run it after an install, after an update, after changing code, or any time the answer to
"is NEURON working?" should not be "type hi and see".

    python neuron_doctor.py                          # check the default coordinator
    python neuron_doctor.py --coordinator http://... # check another one
    python neuron_doctor.py --quiet                  # only print problems
    python neuron_doctor.py --json                   # machine-readable, for CI/installers

Exit code is 0 when everything a user depends on is working, 1 when it is not, so an
installer or a cron job can act on it without parsing the output.

WHY THIS EXISTS
---------------
2026-08-04: the network sat at 21 of 28 layers covered for an unknown length of time. Every
chat silently failed. The coordinator's own /status endpoint was returning
`"network_healthy": false` the entire time, and the dashboard was showing a red DEGRADED
banner -- but nothing consumed either, so the failure surfaced as a person typing "hi" eight
times and getting nothing back. See RESILIENCE.md [R6].

The data was never missing. The check was. That is the whole point of this file.

WHAT IT CHECKS, IN THE ORDER A REQUEST DEPENDS ON THEM
  1. coordinator reachable            -- nothing works without it
  2. every model layer has an owner   -- a hole here means ZERO requests can complete
  3. nodes online and eligible        -- registered but offline is not capacity
  4. spare capacity for failover      -- RESILIENCE.md [R2]: recovery needs somewhere to go
  5. a model is actually serving      -- covered layers but no serving tier is still dead
  6. the operator's verifier is alive -- only where it runs; a dead one means every node
                                         that joins stays probationary forever ([P24])

Deliberately NOT a correctness test. `selftest_shard.py` proves the maths; this proves the
network is up. Different questions, different runtimes -- this one takes a second and needs
no model in RAM, so it can run anywhere, including on a stranger's machine after install.
"""

import argparse
import json
import os
import sys
import time

try:
    import requests
except ImportError:
    print("neuron_doctor needs `requests` (pip install requests)")
    sys.exit(1)

DEFAULT_COORDINATOR = "https://neuronnet.duckdns.org"

OK, WARN, BAD = "ok", "warn", "bad"
MARK = {OK: "  OK  ", WARN: " WARN ", BAD: " FAIL "}

# The operator's verifier writes an "alive" line every ~30 minutes (verify_service.ALIVE_EVERY).
# Three missed heartbeats is dead, not slow.
VERIFIER_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_service.log")

# The AGENT's log, on the machine that runs a node. Same idea as VERIFIER_LOG one line below and
# the same incident shape: a process that is up, holding its port, and writing nothing looks
# exactly like a healthy one. Live 2026-08-18 — the agent restarted after a wake, went on
# answering proof-of-compute challenges on port 50999, served its Chat UI, and did not write a
# single log line or heartbeat for 81 minutes. The coordinator read it as offline and
# auto-repair collapsed the network onto the other machine. Nothing on the box said so.
#
# A node beats every PING_INTERVAL_S (~30s) and logs "heartbeat ok" periodically, so a log
# untouched for half an hour is not a quiet agent, it is a stopped one.
_AGENT_BASE = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"),
                                                             ".local", "share")
AGENT_LOG = os.path.join(_AGENT_BASE, "NEURON", "agent.log")
AGENT_STALE_S = 30 * 60
VERIFIER_STALE_S = 90 * 60


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail="", fix=""):
        self.rows.append({"status": status, "check": name, "detail": detail, "fix": fix})

    @property
    def failed(self):
        return [r for r in self.rows if r["status"] == BAD]

    @property
    def warned(self):
        return [r for r in self.rows if r["status"] == WARN]


def fetch(base, path, timeout=10):
    r = requests.get(f"{base.rstrip('/')}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def check_network(base, rep):
    """Everything a user's request depends on, in dependency order."""
    try:
        st = fetch(base, "/status")
    except Exception as e:
        rep.add(BAD, "coordinator reachable", f"{base}: {e.__class__.__name__}: {e}",
                "is the coordinator running, and is this the right URL?")
        return None
    rep.add(OK, "coordinator reachable", base)

    net = st.get("network", {}) or {}
    covered = net.get("total_layers_covered")
    total = net.get("total_layers")
    online = net.get("online_nodes", 0)
    eligible = net.get("eligible_nodes", 0)
    registered = net.get("total_nodes", 0)

    # 2. THE ONE THAT MATTERS MOST. A single uncovered layer means no request can finish.
    if covered is None or total is None:
        rep.add(WARN, "model layers covered", "coordinator did not report layer coverage")
    elif covered >= total:
        rep.add(OK, "model layers covered", f"{covered}/{total}")
    else:
        # Name the gap when the coordinator reports it. "7 layers have no node" tells you the
        # chain is broken; "layers 21-27" tells you where to put one. Older coordinators do
        # not send `uncovered_layers`, so its absence is not an error.
        gaps = net.get("uncovered_layers") or []
        where = ", ".join(f"{lo}-{hi}" if lo != hi else f"{lo}" for lo, hi in gaps)
        rep.add(BAD, "model layers covered", f"{covered}/{total} -- "
                f"{total - covered} layer(s) have no node"
                f"{f' (missing: {where})' if where else ''}. NO request can complete.",
                "POST /network/rebalance (admin) to re-split across the nodes that are "
                "online, or bring another node up. See RESILIENCE.md [R6].")

    # 3. registered != serving.
    if online == 0:
        rep.add(BAD, "nodes online", f"0 of {registered} registered",
                "start the agent on at least one machine")
    elif online < registered:
        rep.add(WARN, "nodes online", f"{online} of {registered} registered are online")
    else:
        rep.add(OK, "nodes online", f"{online} of {registered}")

    if eligible < online:
        rep.add(WARN, "nodes eligible to serve",
                f"{eligible} of {online} online are eligible "
                f"(probationary {net.get('probationary_nodes', 0)}, "
                f"flagged {net.get('flagged_nodes', 0)})",
                "probationary nodes need proof-of-compute; verify_service.py does this")
    else:
        rep.add(OK, "nodes eligible to serve", str(eligible))

    # 3b. PROBATIONARY NODES. [P24]: a stranger registered, downloaded a slice, served
    # heartbeats for three days and was never promoted, because the operator's verifier had
    # been dead for two of them. The node looked healthy from every angle except this one.
    prob = net.get("probationary_nodes", 0)
    if prob:
        rep.add(BAD, "nodes awaiting verification",
                f"{prob} node(s) probationary -- they serve nothing and earn nothing "
                f"until promoted",
                "is verify_service.py running? A probationary node that never gets promoted "
                "is a stranger who installed NEURON and got nothing. See PROBLEMS.md [P24].")
    else:
        rep.add(OK, "nodes awaiting verification", "none stuck probationary")

    # 4. Failover needs somewhere to go -- RESILIENCE.md [R2].
    if covered and total and covered >= total and eligible <= 2:
        rep.add(WARN, "spare capacity for failover",
                f"{eligible} eligible node(s): no replica, so a node dying mid-answer has "
                f"nowhere to reroute to",
                "add a node. Recovery is built and tested (test_node_death.py) but it "
                "needs a replacement to exist. RESILIENCE.md [R2]")
    elif eligible > 2:
        rep.add(OK, "spare capacity for failover", f"{eligible} eligible nodes")

    # 5. Coverage without a serving tier is still not serving.
    # Routability, reported separately from coverage because they fail independently and only
    # one of them is visible anywhere else. A roster can cover every layer and still walk to a
    # 1-stage chain, which no driver accepts -- and until 2026-08-09 that state reported
    # `network_healthy: true` while every chat failed. PROBLEMS.md [P32].
    if net.get("routable") is False:
        stages = net.get("stages")
        if net.get("uncovered_layers"):
            rep.add(BAD, "chain is routable", f"chain stops at a gap after {stages} stage(s)",
                    "fill the uncovered layers above; the chain cannot be walked past them")
        else:
            rep.add(BAD, "chain is routable",
                    f"every layer is covered, but the chain walks to {stages} stage(s)",
                    "a node holding a range that spans another node's wins the chain walk and "
                    "swallows it. Re-split: ./coordinator/pin_layers.sh --driver <this machine>")
    elif net.get("routable") is True:
        rep.add(OK, "chain is routable",
                f"{net.get('stages')} stage(s) {net.get('chain_ranges')}")

    healthy = net.get("network_healthy")
    if healthy is False:
        rep.add(BAD, "coordinator reports healthy", "network_healthy = false",
                "see the failing check above; the coordinator already knows")
    elif healthy is True:
        rep.add(OK, "coordinator reports healthy", "network_healthy = true")

    return st


def check_serving_model(base, rep):
    try:
        models = fetch(base, "/models")
    except Exception:
        return  # optional endpoint; absence is not a failure
    entries = models if isinstance(models, list) else models.get("models", [])
    # `ready` is the field /models actually returns, and it is a BOOLEAN. This read only
    # `state`/`status`, which that payload has never carried, so `serving` was always empty and
    # the check reported "no model tier is in a serving state" on a perfectly healthy network —
    # observed 2026-08-18 with routable true, stage1_ok true and both nodes online.
    #
    # A check that cries wolf is worse than no check: it is the one people learn to scroll past,
    # and then it cannot report the day a tier really has stopped. The string forms are kept
    # because a future payload may use them, but `ready` is what is asked first.
    def _is_serving(m):
        if isinstance(m.get("ready"), bool):
            return m["ready"]
        return str(m.get("state", m.get("status", ""))).lower() in ("serving", "ready")

    serving = [m for m in entries if _is_serving(m)]
    if not entries:
        return
    if serving:
        names = ", ".join(str(m.get("id") or m.get("model") or m.get("model_id")
                                    or m.get("tier")) for m in serving)
        rep.add(OK, "a model is serving", names)
    else:
        rep.add(BAD, "a model is serving", "no model tier is in a serving state",
                "not enough nodes or RAM for any tier; see the dashboard")


def check_agent(rep):
    """Is the LOCAL agent still writing anything? Skipped where there is no agent.log.

    Deliberately a check on the log's MTIME rather than on the process list: a running process
    proves nothing here, and that is the whole point — the failure being guarded is a process
    that is alive, listening, and mute. It also costs nothing on a machine that only runs the
    Chat UI, where the file simply does not exist.
    """
    if not os.path.exists(AGENT_LOG):
        return
    age = time.time() - os.path.getmtime(AGENT_LOG)
    mins = int(age // 60)
    if age > AGENT_STALE_S:
        rep.add(BAD, "agent alive",
                f"agent.log has not been written for {mins} minutes — this node is almost "
                f"certainly not heartbeating, even if the process is still running",
                "check the coordinator's node list: if it reads `offline` while the process is "
                "up, the agent is mute rather than stopped. Restart it. A node that stops "
                "beating is dropped from routing and earns nothing, and on a small network "
                "auto-repair will re-place its layers onto whoever is left. See PROBLEMS.md "
                "[P51].")
    else:
        rep.add(OK, "agent alive", f"last wrote {mins} minute(s) ago")


def check_verifier(rep):
    """Is the OPERATOR's verifier still alive? Local check, skipped where it does not apply.

    Only the machine running verify_service.py has this log, so a stranger running the doctor
    sees nothing about it. On the operator's machine it is the check that would have caught
    [P24] two days early: the service had been dead since 17:27 on 2026-08-03, and the only
    evidence was a log file that had stopped growing — which nobody was looking at, and which
    looked the same as a quiet, healthy verifier until it started writing an alive line.
    """
    if not os.path.exists(VERIFIER_LOG):
        return
    age = time.time() - os.path.getmtime(VERIFIER_LOG)
    mins = int(age // 60)
    if age > VERIFIER_STALE_S:
        rep.add(BAD, "verifier alive", f"verify_service.log has not been written for {mins} "
                f"minutes -- the verifier is almost certainly dead",
                "restart it (python verify_service.py) and make sure something supervises it. "
                "While it is down, every node that joins stays probationary forever: it earns "
                "nothing and serves nothing. See PROBLEMS.md [P24].")
    else:
        rep.add(OK, "verifier alive", f"last wrote {mins} minute(s) ago")


def render(rep, quiet=False):
    width = max(len(r["check"]) for r in rep.rows) if rep.rows else 20
    for r in rep.rows:
        if quiet and r["status"] == OK:
            continue
        print(f"[{MARK[r['status']]}] {r['check']:<{width}}  {r['detail']}")
        if r["fix"] and r["status"] != OK:
            print(f"{'':>10}  {'':<{width}}  -> {r['fix']}")

    print()
    if rep.failed:
        print(f"NOT HEALTHY -- {len(rep.failed)} failing check(s)"
              + (f", {len(rep.warned)} warning(s)" if rep.warned else ""))
        print("Users cannot be served reliably right now.")
    elif rep.warned:
        print(f"SERVING, with {len(rep.warned)} warning(s) -- requests work, "
              f"but something is not resilient.")
    else:
        print("HEALTHY -- everything a request depends on is working.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coordinator", default=DEFAULT_COORDINATOR)
    ap.add_argument("--quiet", action="store_true", help="only show problems")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    rep = Report()
    if check_network(args.coordinator, rep) is not None:
        check_serving_model(args.coordinator, rep)
    check_verifier(rep)
    check_agent(rep)

    if args.json:
        print(json.dumps({"healthy": not rep.failed, "checks": rep.rows}, indent=1))
    else:
        if not args.quiet:
            print(f"\nNEURON doctor — {args.coordinator}\n")
        render(rep, args.quiet)

    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
