"""
neuron_logs.py — read the logs of machines you are not sitting at.

    python neuron_logs.py                      # every node
    python neuron_logs.py --node agent-bhpc012104-5452f9
    python neuron_logs.py --coordinator        # the VM's own journal, over SSH
    python neuron_logs.py --list               # who has answered, who is silent
    python neuron_logs.py --cached             # show what is already held, ask for nothing

Until this existed, a NEURON node's log was readable only by whoever was sitting at that machine.
That is not a small inconvenience -- it is why [P24] ran for three days: a stranger's node logged
`heartbeat ok — active` while doing nothing at all, and the only way to find out was to ask a
human to open a file. Every diagnosis was hostage to somebody else's attention.

**How it works, and why it looks indirect.** Nodes are behind NAT -- that is the entire reason
the relay exists -- so nothing here can connect to them. Instead this raises a flag on the
coordinator; each node notices it on its next heartbeat (<=30 s), uploads a redacted tail of
`agent.log`, and the flag clears. So a fetch is: ask, wait one heartbeat, read. A node that is
offline uploads when it returns, which means "no log yet" is itself a finding.

Reading needs the operator secret (`NEURON_REGISTER_SECRET`) -- these are other people's
machines. Secrets and home-directory names are stripped at both ends (`logtail.py`).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "https://neuronnet.duckdns.org")
SECRET = os.environ.get("NEURON_REGISTER_SECRET", "")
SSH_HOST = os.environ.get("NEURON_DEPLOY_HOST", "ubuntu@150.230.22.250")
SSH_KEY = os.environ.get("NEURON_DEPLOY_KEY", os.path.expanduser("~/.ssh/oracle_coordinator"))
SERVICE = "neuron-coordinator"
HEARTBEAT_S = 30


def _call(path, method="GET", body=None, timeout=30):
    req = urllib.request.Request(f"{COORDINATOR.rstrip('/')}{path}", method=method)  # noqa: S310
    req.add_header("X-Register-Secret", SECRET)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:           # noqa: S310
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise SystemExit(
                "the coordinator refused: these are other people's machines, so reading their "
                "logs needs the operator secret.\n"
                "  set NEURON_REGISTER_SECRET, or pass --secret") from None
        raise SystemExit(f"coordinator returned {e.code} for {path}") from None
    except urllib.error.URLError as e:
        raise SystemExit(f"could not reach {COORDINATOR}: {e.reason}") from None


def _ago(ts):
    if not ts:
        return "never"
    d = max(0, int(time.time() - ts))
    if d < 60:
        return f"{d}s ago"
    if d < 3600:
        return f"{d // 60}m ago"
    return f"{d // 3600}h ago"


def show_coordinator(lines):
    """The coordinator's journal. A plain SSH pull -- it is a server we own with a fixed
    address, so none of the NAT dance below applies to it."""
    cmd = ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=accept-new", SSH_HOST,
           f"journalctl -u {SERVICE} -n {lines} --no-pager"]
    print(f"===== coordinator ({SSH_HOST}) — last {lines} lines =====")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  could not reach the coordinator over SSH: {e}")
        return 1
    print(r.stdout or "  (no output)")
    if r.returncode != 0 and r.stderr:
        print(f"  ssh: {r.stderr.strip()}")
    return r.returncode


def show_nodes(targets, wait_s, cached_only):
    roster = _call("/admin/logs")          # [{node_id, uploaded_at, lines, bytes}]
    known = {r["node_id"] for r in roster}
    nodes = _call("/node/list")
    all_ids = [n["node_id"] for n in (nodes.get("nodes") if isinstance(nodes, dict) else nodes)]
    wanted = targets or all_ids
    unknown = [t for t in wanted if t not in all_ids]
    for t in unknown:
        print(f"  no such node: {t}")
    wanted = [t for t in wanted if t in all_ids]
    if not wanted:
        return 1

    if not cached_only:
        _call("/admin/logs/request", method="POST", body={"nodes": wanted})
        print(f"asked {len(wanted)} node(s) for a log tail; they upload on their next "
              f"heartbeat (<={HEARTBEAT_S}s)")
        deadline = time.time() + wait_s
        before = {r["node_id"]: r["uploaded_at"] for r in roster}
        while time.time() < deadline:
            time.sleep(3)
            roster = _call("/admin/logs")
            fresh = {r["node_id"] for r in roster
                     if r["uploaded_at"] > before.get(r["node_id"], 0)}
            if set(wanted) <= fresh:
                break
            print(f"  {len(fresh & set(wanted))}/{len(wanted)} uploaded ...", end="\r")
        print(" " * 60, end="\r")

    rc = 0
    for nid in wanted:
        entry = _call(f"/admin/logs/{nid}")
        print(f"\n===== {nid} =====")
        if not entry or not entry.get("body"):
            # Not an error. A node that never answers is usually the node with the problem.
            print(f"  no log uploaded. Last seen {_ago(entry.get('uploaded_at') if entry else 0)}"
                  f" — the node is offline, or its agent is not running.")
            rc = 1
            continue
        print(f"  ({entry['lines']} lines, uploaded {_ago(entry['uploaded_at'])})\n")
        print(entry["body"])
    if known and not targets:
        pass
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", action="append", default=[],
                   help="a node id; repeatable. Default: every node.")
    p.add_argument("--coordinator", action="store_true",
                   help="the coordinator VM's journal over SSH, instead of the nodes")
    p.add_argument("--list", action="store_true",
                   help="who has a log on file and how old it is; fetch nothing")
    p.add_argument("--cached", action="store_true",
                   help="print what the coordinator already holds; ask for nothing new")
    p.add_argument("--wait", type=int, default=45,
                   help="seconds to wait for uploads (default %(default)s)")
    p.add_argument("--lines", type=int, default=200,
                   help="journal lines for --coordinator (default %(default)s)")
    p.add_argument("--secret", default=None, help="operator secret (else NEURON_REGISTER_SECRET)")
    args = p.parse_args(argv)

    global SECRET
    if args.secret:
        SECRET = args.secret

    if args.coordinator:
        return show_coordinator(args.lines)

    if args.list:
        rows = _call("/admin/logs")
        if not rows:
            print("no node has uploaded a log yet — run without --list to ask them to.")
            return 0
        print(f"{'node':<34} {'uploaded':<12} {'lines':>6} {'bytes':>8}")
        for r in sorted(rows, key=lambda r: r["node_id"]):
            print(f"{r['node_id']:<34} {_ago(r['uploaded_at']):<12} "
                  f"{r['lines']:>6} {r['bytes']:>8}")
        return 0

    return show_nodes(args.node, args.wait, args.cached)


if __name__ == "__main__":
    sys.exit(main())
