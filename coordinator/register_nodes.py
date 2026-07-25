"""register_nodes.py — register the 3 NEURON nodes and keep them alive.

Standalone (only `requests` + stdlib). Registers each node with the coordinator,
saves the returned tokens to node_tokens.json, then runs a heartbeat prober:
every PING_INTERVAL it checks whether each server node's inference port is really
listening and pings the coordinator on its behalf. node_a is the local driver (no
server port) so it is pinged while this script runs. Killing node_b/node_c's
server stops its pings -> the coordinator marks it offline after the timeout.

(In a fully decentralised build each node would self-ping; probing the port here
keeps node_b.py/node_c.py unmodified while still making health-checking real.)

Usage:
  python coordinator/register_nodes.py                  # register + keep pinging
  python coordinator/register_nodes.py --register-only  # just register, then exit
  python coordinator/register_nodes.py --coordinator http://localhost:8000
"""
import argparse
import json
import os
import socket
import time
from pathlib import Path

import requests

# Must match the coordinator's NEURON_REGISTER_SECRET. Defaults to the dev secret
# for the local setup; set the env var to use a real (e.g. cloud) coordinator's secret.
REGISTER_SECRET = os.environ.get("NEURON_REGISTER_SECRET", "neuron-dev-secret")
PING_INTERVAL = 30

# The dev nodes are loaded from coordinator/nodes.local.json (GITIGNORED — holds your
# real Tailscale IPs) so the public repo carries no private infra. Falls back to
# placeholders; copy coordinator/nodes.example.json to nodes.local.json and set your IPs.
# `probe` (liveness (ip,port), or None for the driver) is derived from the "driver" flag;
# ms_per_layer/head_ms come from benchmark.py (Session 14) and feed the auto-balancer.
_NODES_PATH = Path(__file__).resolve().parent / "nodes.local.json"
_PLACEHOLDER_NODES = [
    {"node_id": "node_a", "tailscale_ip": "100.100.100.1", "port": 50999,
     "layer_start": 0, "layer_end": 9, "cores": 16, "ram_gb": 63, "driver": True,
     "ms_per_layer": 9.0, "head_ms": 38.0},
    {"node_id": "node_c", "tailscale_ip": "100.100.100.2", "port": 50999,
     "layer_start": 10, "layer_end": 18, "cores": 4, "ram_gb": 11, "ms_per_layer": 12.4},
    {"node_id": "node_b", "tailscale_ip": "100.100.100.3", "port": 50999,
     "layer_start": 19, "layer_end": 27, "cores": 6, "ram_gb": 15, "ms_per_layer": 12.2},
]


def _load_nodes():
    if _NODES_PATH.exists():
        nodes = json.loads(_NODES_PATH.read_text())
    else:
        print(f"[register] {_NODES_PATH.name} not found — using placeholder IPs. Copy "
              f"nodes.example.json -> nodes.local.json and set your real node IPs.")
        nodes = [dict(n) for n in _PLACEHOLDER_NODES]
    for n in nodes:
        n["probe"] = None if n.get("driver") else (n["tailscale_ip"], n["port"])
    return nodes


NODES = _load_nodes()
TOKENS_PATH = Path(__file__).resolve().parent / "node_tokens.json"


def port_alive(hostport, timeout=2.0):
    try:
        with socket.create_connection(hostport, timeout=timeout):
            return True
    except OSError:
        return False


def register_all(base):
    tokens = {}
    for n in NODES:
        body = {k: n[k] for k in ("node_id", "tailscale_ip", "port", "layer_start",
                                  "layer_end", "cores", "ram_gb")}
        for opt in ("ms_per_layer", "head_ms"):    # Session 14: speeds for auto-balance
            if opt in n:
                body[opt] = n[opt]
        r = requests.post(f"{base}/node/register", json=body,
                          headers={"X-Register-Secret": REGISTER_SECRET}, timeout=10)
        r.raise_for_status()
        data = r.json()
        tokens[n["node_id"]] = data["node_token"]
        print(f"registered {n['node_id']:7s} layers {data['assigned_layers']} "
              f"token {data['node_token'][:8]}...")
    TOKENS_PATH.write_text(json.dumps(tokens, indent=2))
    print(f"tokens saved -> {TOKENS_PATH}")
    return tokens


def heartbeat_loop(base, tokens):
    print(f"heartbeat: pinging live nodes every {PING_INTERVAL}s (Ctrl-C to stop)")
    while True:
        line = []
        for n in NODES:
            nid = n["node_id"]
            alive = True if n["probe"] is None else port_alive(n["probe"])
            if not alive:
                line.append(f"{nid}=DOWN")
                continue
            try:
                requests.get(f"{base}/node/{nid}/ping",
                             headers={"X-Node-Token": tokens[nid]}, timeout=5)
                line.append(f"{nid}=ok")
            except requests.RequestException as e:
                line.append(f"{nid}=ERR({e.__class__.__name__})")
        print(f"[{time.strftime('%H:%M:%S')}] " + "  ".join(line))
        time.sleep(PING_INTERVAL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", default="http://150.230.22.250:8001",
                    help="coordinator URL (default: cloud host on :8001)")
    ap.add_argument("--register-only", action="store_true")
    ap.add_argument("--node-c-host", default=None,
                    help="override node_c address (e.g. run node_c locally as 127.0.0.1)")
    ap.add_argument("--node-c-port", type=int, default=None)
    args = ap.parse_args()
    base = args.coordinator.rstrip("/")

    if args.node_c_host or args.node_c_port:
        for nd in NODES:
            if nd["node_id"] == "node_c":
                nd["tailscale_ip"] = args.node_c_host or nd["tailscale_ip"]
                nd["port"] = args.node_c_port or nd["port"]
                nd["probe"] = (nd["tailscale_ip"], nd["port"])
                print(f"node_c overridden -> {nd['tailscale_ip']}:{nd['port']}")

    tokens = register_all(base)
    if args.register_only:
        return
    try:
        heartbeat_loop(base, tokens)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
