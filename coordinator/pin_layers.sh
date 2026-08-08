#!/usr/bin/env bash
# coordinator/pin_layers.sh — give every ONLINE node a stage the chat driver can actually route.
#
#   ./coordinator/pin_layers.sh              # work out the split and apply it
#   ./coordinator/pin_layers.sh --dry-run    # print the split, change nothing
#
# The balancer optimises for stage TIME and will produce a split no client can use. The driver
# holds a fixed shard -- layers 0..S1-1, S1 from neuron_driver.py (env NEURON_S1, default 10) --
# and node_a.py REFUSES any chain whose first stage is not exactly that. So a speed-optimal
# 0-12 / 13-27 split is unroutable by a driver built for 0-9, and every request dies client-side
# after the coordinator has already taken a wallet hold.
#
# This pins stage 1 to exactly the driver's shard and splits the rest evenly across whoever else
# is online, then POSTs it to /network/layers, which validates the whole split before writing any
# of it.
#
# EVERY online node must get a stage. A node left spanning the whole model wins the chain walk
# (build_chain advances to the farthest layer_end from each cursor) and collapses the pinned
# stages back into a 1-stage chain -- which is exactly the live 2026-08-07 state, two machines
# both holding 0-27.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVFILE="${NEURON_ENV_FILE:-$here/.env.coordinator}"
COORD="${NEURON_COORDINATOR:-https://neuronnet.duckdns.org}"
S1="${NEURON_S1:-10}"                     # must match neuron_driver.S1
# Which node is the DRIVER — the machine you chat from. It must be stage 1, because that is the
# machine holding layers 0..S1-1 and node_a.py rejects any chain whose first stage is not exactly
# its own shard. Nothing in the roster identifies it: head_ms would, but it is None on every node
# on this network, so it cannot be inferred and is not guessed.
DRIVER="${NEURON_DRIVER_NODE:-}"
DRY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --driver)  DRIVER="$2"; shift ;;
    *) echo "usage: $0 [--driver <node_id>] [--dry-run]"; exit 2 ;;
  esac
  shift
done

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

SECRET="${NEURON_REGISTER_SECRET:-}"
if [ -z "$SECRET" ] && [ -f "$ENVFILE" ]; then
  SECRET="$(grep '^NEURON_REGISTER_SECRET=' "$ENVFILE" | cut -d= -f2- | tr -d '"'"'"' \r')"
fi
[ -z "$SECRET" ] && { echo "no operator secret (looked in \$NEURON_REGISTER_SECRET and $ENVFILE)"; exit 2; }

PY="$here/.venv/Scripts/python.exe"
[ -x "$PY" ] || PY="$here/.venv/bin/python"
[ -x "$PY" ] || PY="python"

say "working out the split"
PLAN="$("$PY" - "$COORD" "$S1" "$SECRET" "$DRIVER" <<'PYEOF'
import json, sys, urllib.request
coord, s1, secret = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3]
driver = sys.argv[4] if len(sys.argv) > 4 else ""
st = json.load(urllib.request.urlopen(f"{coord}/status", timeout=20))
total = st["network"]["total_layers"]
# /node/list with the operator secret, NOT the dashboard HTML. Scraping the rendered page missed
# nodes whenever the markup shifted, and a layer plan built from a partial roster silently leaves
# machines out of the chain.
req = urllib.request.Request(f"{coord}/node/list")
req.add_header("X-Register-Secret", secret)
roster = json.load(urllib.request.urlopen(req, timeout=20))
nodes = roster["nodes"] if isinstance(roster, dict) else roster
# Only ELIGIBLE online nodes: routing skips everything else, so giving a stage to a node the
# router will not use is the same as leaving a hole in the chain.
online = [n["node_id"] for n in nodes
          if n.get("status") == "online" and n.get("eligible")]
if not online:
    print(json.dumps({"error": "no online eligible nodes"})); raise SystemExit
if total <= s1:
    print(json.dumps({"error": f"model has {total} layers, driver wants 0..{s1-1}"})); raise SystemExit
if driver and driver not in online:
    print(json.dumps({"error": f"driver {driver!r} is not online+eligible. online: {online}"}))
    raise SystemExit
if not driver:
    if len(online) == 1:
        driver = online[0]              # only one machine; it is necessarily stage 1
    else:
        print(json.dumps({"error":
            "which machine do you chat from? It must be stage 1, and nothing in the roster says "
            "which it is (head_ms is None on every node). Re-run with --driver <node_id>. "
            f"online: {online}"}))
        raise SystemExit
if len(online) < 2:
    print(json.dumps({"error":
        f"only {len(online)} node online ({online[0]}). The driver holds 0..{s1-1} and needs at "
        f"least one more machine for the rest of the model -- a 1-stage chain is not routable."}))
    raise SystemExit

# Stage 1 is the driver's shard, fixed. The rest is split as evenly as possible.
rest_nodes = [n for n in online if n != driver]
layers = {driver: [0, s1 - 1]}
# rest_nodes is never empty: the <2 guard above already exited.
remaining, cur = total - s1, s1
base, extra = divmod(remaining, len(rest_nodes))
last = None
for i, nid in enumerate(rest_nodes):
    cnt = base + (1 if i < extra else 0)
    if cnt == 0:
        # More machines than layers left. An extra machine becomes a REPLICA of the last stage
        # rather than a stage of its own -- that is what turns added machines into throughput
        # instead of a deeper pipeline, and a zero-width stage would break the chain walk.
        layers[nid] = list(last)
        continue
    last = [cur, cur + cnt - 1]
    layers[nid] = list(last)
    cur += cnt
print(json.dumps({"layers": layers, "total": total, "online": online}))
PYEOF
)"

if echo "$PLAN" | grep -q '"error"'; then
  echo "   $(echo "$PLAN" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["error"])')"
  exit 1
fi
echo "$PLAN" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); [print(f"   {k:<32} {v[0]:>2}-{v[1]}") for k,v in d["layers"].items()]'

if [ -n "$DRY" ]; then
  say "dry run — nothing applied"
  exit 0
fi

say "applying"
BODY="$(echo "$PLAN" | "$PY" -c 'import json,sys; print(json.dumps({"layers": json.load(sys.stdin)["layers"]}))')"
if ! curl -fsS -m 30 -X POST -H "X-Register-Secret: $SECRET" -H "Content-Type: application/json" \
     -d "$BODY" "$COORD/network/layers"; then
  echo
  echo "FAILED. A 401 means the secret in $ENVFILE does not match the one the coordinator runs"
  echo "with. A 404 means /network/layers is not deployed yet — run ./coordinator/deploy.sh."
  exit 1
fi
echo

say "after"
curl -fsS -m 20 "$COORD/status"; echo
echo
echo "   Now restart the NEURON agent on each machine, and the chat UI."
echo "   A node reuses a slice that already CONTAINS its new range, so a machine that has been"
echo "   through a re-split before downloads nothing."
