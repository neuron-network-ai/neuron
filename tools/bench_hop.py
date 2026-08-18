"""tools/bench_hop.py — what does ONE real network hop cost per token? ([P30] phase 1)

    python -m tools.bench_hop --node node-c-pavilion

**The number this exists for.** Decode is sequential: every token crosses every hop and waits
for the answer before the next one starts. So the product's latency is
`compute + hops x round_trip`, and which of those two terms dominates decides the whole
architecture:

  * if COMPUTE dominates, a faster engine ([P30]) is the win and pipelining across machines is
    the right shape;
  * if the ROUND TRIP dominates, a faster engine buys almost nothing across a real network,
    and the answer is REPLICATION -- several machines each holding the whole model, serving
    different users -- rather than stages.

The loopback spike measured ggml-rpc's protocol overhead at ~8% with zero network latency. That
is a floor, not a prediction. This measures the real path: through NEURON's relay on a public
VM, to a node behind a home NAT, using the actual wire protocol.

**It measures the round trip a node is ASKED for, not a synthetic ping.** A `config` probe is
one request and one reply over the same socket, framed exactly as an `act` message is, and
answered before the node touches a model -- so what comes back is transport, not inference.
That is the term being isolated.

Read-only: it opens a socket, sends a config probe, and closes. It does not route a request,
spend NRN, or change any placement.
"""
from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common                                                # noqa: E402


def _probe_once(host, port, s1, s2, timeout=30.0):
    """One connect + config round trip. Returns (connect_s, roundtrip_s)."""
    t0 = time.perf_counter()
    sock = socket.create_connection((host, port), timeout=timeout)
    t1 = time.perf_counter()
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        common.send_msg(sock, {"type": "config", "s1": s1, "s2": s2})
        common.recv_msg(sock)
        t2 = time.perf_counter()
    finally:
        try:
            sock.close()
        except OSError:
            pass
    return t1 - t0, t2 - t1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--node", default=None, help="node_id to probe (default: the first online)")
    ap.add_argument("--n", type=int, default=12, help="round trips to time")
    ap.add_argument("--s1", type=int, default=0)
    ap.add_argument("--s2", type=int, default=10)
    args = ap.parse_args(argv)

    import verify_service as vs
    import requests
    secret = vs.load_secret()
    d = requests.get(vs.DEFAULT_COORDINATOR + "/node/list",
                     headers={"X-Register-Secret": secret}, timeout=30).json()
    nodes = d["nodes"] if isinstance(d, dict) else d
    online = [n for n in nodes if n.get("status") == "online"]
    if args.node:
        online = [n for n in online if n["node_id"] == args.node]
    if not online:
        print("no matching online node")
        return 2
    n = online[0]
    host, port = n["tailscale_ip"], n["port"]
    print(f"\nnode    : {n['node_id']}  layers {n['layer_start']}-{n['layer_end']}")
    print(f"address : {host}:{port}"
          + ("   (via the public relay)" if str(host).startswith("150.230") else "   (direct)"))
    print(f"probe   : config s1={args.s1} s2={args.s2}, answered before any model work\n")

    connects, trips = [], []
    for i in range(args.n):
        try:
            c, r = _probe_once(host, port, args.s1, args.s2)
        except Exception as e:                               # noqa: BLE001
            print(f"  {i+1:>2}. FAILED {e.__class__.__name__}: {e}")
            continue
        connects.append(c)
        trips.append(r)
        print(f"  {i+1:>2}. connect {c*1000:7.1f} ms   round trip {r*1000:7.1f} ms")

    if not trips:
        print("\nno successful probes")
        return 1
    med = statistics.median(trips)
    print(f"\n  round trip : median {med*1000:.1f} ms   "
          f"min {min(trips)*1000:.1f}   max {max(trips)*1000:.1f}")
    print(f"  connect    : median {statistics.median(connects)*1000:.1f} ms")

    # What it means for the engine decision, in the only terms that matter.
    print("\n  per-token budget at this round trip, ONE hop:")
    for label, compute_ms in (("PyTorch fp32 (today)", 1000 / 3.09),
                              ("llama.cpp q4_k_m", 1000 / 26.93)):
        total = compute_ms + med * 1000
        share = (med * 1000) / total * 100
        print(f"    {label:<22} compute {compute_ms:6.1f} ms + hop {med*1000:6.1f} ms "
              f"= {total:6.1f} ms/token  ({share:.0f}% network)  -> {1000/total:5.2f} tok/s")
    print("\n  If the network share is dominant, a faster ENGINE cannot fix the product and")
    print("  replication beats pipelining. That is [P30] phase 1's gate.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
