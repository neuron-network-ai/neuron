"""
security/proof_of_compute.py — verify a node actually did the work  [Session 16]

A lazy or malicious node could return garbage to farm NRN without computing. Proof-of-
compute catches that: a verifier sends a node a challenge (a known input for its layer
range), the node runs its slice, and the verifier checks the output against the locally
computed expected result. Wrong output ⇒ the node failed ⇒ the coordinator drops its
reputation and withholds NRN (see coordinator reputation, Session 16).

The verifier needs torch (it computes the expected output); the coordinator only
aggregates pass/fail into reputation, so it stays torch-free. This challenges a
LAST-stage node (layers[s2:n] + final norm — node_b's wire protocol). Challenging a
middle node needs a no-relay mode on node_c (extension).

    python -m security.proof_of_compute --host <ip> --port 50999 --s2 19 --n 28
"""
import argparse
import json
import socket
import time

import requests
import torch

import common


def make_challenge(s2, n, seed=0):
    """Deterministic (input, expected) for a last-stage node holding layers[s2:n] + norm."""
    _tok, model, _N = common.load_model_shard(s2, n, norm=True)
    H = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, H, generator=g, dtype=common.DTYPE)
    expected = common.last_stage(model, s2, inp, common.new_cache(), 0)
    return inp, expected


def make_middle_challenge(s1, s2, seed=0):
    """Deterministic (input, expected) for a MIDDLE node holding layers[s1:s2] only (no norm,
    no head) -- the isolated no-relay probe (node_c.py / agent/node_server.py's "probe" role)."""
    _tok, model, _N = common.load_model_shard(s1, s2)
    H = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, H, generator=g, dtype=common.DTYPE)
    expected = common.mid_stage(model, s1, s2, inp, common.new_cache(), 0)
    return inp, expected


def verify(output, expected, atol=0.05):
    """(passed, max_abs_err). The tolerance absorbs any cross-hardware fp jitter while
    still rejecting garbage, which is off by many orders of magnitude."""
    if not torch.is_tensor(output) or output.shape != expected.shape:
        return False, float("inf")
    err = (output.float() - expected.float()).abs().max().item()
    return err <= atol, err


def challenge_node(host, port, s2, n, inp, timeout=30):
    """Speak the last-stage wire protocol: config -> act(challenge) -> read output."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        common.send_msg(s, {"type": "config", "s2": s2, "n": n})
        ack = common.recv_msg(s)
        if not ack.get("ok"):
            raise RuntimeError(f"node refused config: {ack}")
        common.send_msg(s, {"type": "act", "hidden": inp})
        resp = common.recv_msg(s)
        common.send_msg(s, {"type": "bye"})
        return resp["hidden"]
    finally:
        s.close()


def challenge_middle_node(host, port, s1, s2, inp, timeout=30):
    """Speak the PROBE wire protocol: a config with NO host_b tells a middle-shard node
    (node_c.py / agent/node_server.py) to run its own layers in isolation and return the raw
    result, instead of relaying to a next hop. The node answers with ITS OWN actual s1/s2
    (self.lo/self.hi) in the ack -- checked here so a node lying about its own range fails
    loudly instead of silently passing a challenge for a range it doesn't really hold."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        common.send_msg(s, {"type": "config", "s1": s1, "s2": s2})
        ack = common.recv_msg(s)
        if not ack.get("ok"):
            raise RuntimeError(f"node refused config: {ack}")
        if (ack.get("s1"), ack.get("s2")) != (s1, s2):
            raise RuntimeError(f"node's actual range {ack.get('s1'), ack.get('s2')} does not "
                               f"match the expected {s1, s2} -- registration/slice mismatch")
        common.send_msg(s, {"type": "act", "hidden": inp})
        resp = common.recv_msg(s)
        common.send_msg(s, {"type": "bye"})
        return resp["hidden"]
    finally:
        s.close()


def attest(host, port, s2, n, seed=0, atol=0.05):
    """Full challenge: make -> send -> verify. Returns a result dict."""
    inp, expected = make_challenge(s2, n, seed)
    t0 = time.time()
    output = challenge_node(host, port, s2, n, inp)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6),
            "ms": int((time.time() - t0) * 1000), "layers": [s2, n - 1]}


def attest_middle(host, port, s1, s2, seed=0, atol=0.05):
    """Full MIDDLE-node challenge: make -> send -> verify. Returns a result dict."""
    inp, expected = make_middle_challenge(s1, s2, seed)
    t0 = time.time()
    output = challenge_middle_node(host, port, s1, s2, inp)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6),
            "ms": int((time.time() - t0) * 1000), "layers": [s1, s2 - 1]}


def attest_via_coordinator(coordinator, node_id, register_secret, n=None,
                           seed=0, atol=0.05):
    """Verify a node the coordinator knows about and record the result (Session 12 — open
    join). Looks the node up in /node/list, challenges it (last-stage nodes get the full
    challenge incl. norm; any other node gets the middle no-relay probe), then POSTs the
    pass/fail to /node/{id}/attest so a passing probationary node becomes eligible to serve
    and earn. Returns {'challenge', 'attestation'}."""
    coordinator = coordinator.rstrip("/")
    # node addresses are operator-private -> authenticate to see them (S25 privacy)
    nodes = requests.get(f"{coordinator}/node/list", timeout=10,
                         headers={"X-Register-Secret": register_secret}).json()["nodes"]
    node = next((x for x in nodes if x["node_id"] == node_id), None)
    if node is None:
        raise SystemExit(f"node '{node_id}' not found at {coordinator}")
    if "tailscale_ip" not in node:
        raise SystemExit("coordinator did not return node addresses — wrong register secret?")
    total = n or node.get("total_layers") or 28
    is_last = node["layer_end"] == total - 1
    if is_last:
        res = attest(node["tailscale_ip"], node["port"], node["layer_start"], total, seed, atol)
    else:
        res = attest_middle(node["tailscale_ip"], node["port"],
                            node["layer_start"], node["layer_end"] + 1, seed, atol)
    r = requests.post(f"{coordinator}/node/{node_id}/attest",
                      json={"passed": res["passed"], "max_err": res["max_err"]},
                      headers={"X-Register-Secret": register_secret}, timeout=10)
    r.raise_for_status()
    return {"challenge": res, "attestation": r.json()}


def verify_loop(coordinator, register_secret, interval=60, seed=0, atol=0.05, n=None):
    """Continuously find probationary nodes and verify them -- no more running the CLI by
    hand for every new arrival. Skips already-flagged nodes (the coordinator already excludes
    them from routing after repeated failures, per Session 16 reputation; re-challenging a
    hopeless node forever would just waste cycles). A single node's failure (network hiccup,
    genuinely bad node, wrong secret) is logged and the loop moves on -- one bad node must
    never stop the whole verifier."""
    coordinator = coordinator.rstrip("/")
    print(f"auto-verify: checking {coordinator} for probationary nodes every {interval}s "
         f"(Ctrl-C to stop)")
    while True:
        try:
            nodes = requests.get(f"{coordinator}/node/list", timeout=10,
                                 headers={"X-Register-Secret": register_secret}).json()["nodes"]
            pending = [nd for nd in nodes
                      if nd.get("standing") == "probationary" and not nd.get("flagged")]
            if not pending:
                print(f"[{time.strftime('%H:%M:%S')}] nothing pending")
            for nd in pending:
                try:
                    out = attest_via_coordinator(coordinator, nd["node_id"], register_secret,
                                                 n=n, seed=seed, atol=atol)
                    ch = out["challenge"]
                    print(f"[{time.strftime('%H:%M:%S')}] {nd['node_id']} "
                         f"({'PASSED' if ch['passed'] else 'FAILED'}, "
                         f"max_err={ch['max_err']}, layers={ch['layers']})")
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] could not verify "
                         f"{nd['node_id']}: {e}")
        except requests.RequestException as e:
            print(f"auto-verify: coordinator unreachable: {e}")
        time.sleep(interval)


def main():
    import os

    ap = argparse.ArgumentParser(
        description="Proof-of-compute verifier. Direct mode (--host + --s2, or --host + "
                    "--s1/--s2 for a middle node) challenges one node and prints the result; "
                    "coordinator mode (--coordinator/--node-id) also records the attestation "
                    "so a probationary node gets promoted; --auto runs continuously and "
                    "verifies every probationary node it finds, with no per-node command.")
    # coordinator mode (Session 12)
    ap.add_argument("--coordinator", help="coordinator URL; with --node-id, verify + attest")
    ap.add_argument("--node-id", help="node to verify (coordinator mode)")
    ap.add_argument("--register-secret", default=os.environ.get("NEURON_REGISTER_SECRET"),
                    help="secret to post the attestation (defaults to $NEURON_REGISTER_SECRET)")
    ap.add_argument("--auto", action="store_true",
                    help="with --coordinator (no --node-id): loop forever, auto-verifying "
                         "every probationary node found, instead of a one-shot check")
    ap.add_argument("--interval", type=int, default=60, help="--auto poll interval, seconds")
    # direct mode
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=50999)
    ap.add_argument("--s1", type=int, help="middle node: holds layers[s1:s2) (no norm/head)")
    ap.add_argument("--s2", type=int, help="last node: holds layers[s2:n]+norm; "
                                          "middle node: end of its range (with --s1)")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=0.05)
    args = ap.parse_args()

    if args.coordinator and args.auto:
        if not args.register_secret:
            ap.error("--register-secret (or $NEURON_REGISTER_SECRET) is required for --auto")
        verify_loop(args.coordinator, args.register_secret, interval=args.interval,
                   seed=args.seed, atol=args.atol)
    elif args.coordinator:
        if not args.node_id:
            ap.error("--coordinator requires --node-id (or --auto to verify all pending)")
        if not args.register_secret:
            ap.error("--register-secret (or $NEURON_REGISTER_SECRET) is required to attest")
        out = attest_via_coordinator(args.coordinator, args.node_id, args.register_secret,
                                     n=args.n, seed=args.seed, atol=args.atol)
        print(json.dumps(out, indent=2))
    elif args.host is not None and args.s1 is not None and args.s2 is not None:
        print(json.dumps(attest_middle(args.host, args.port, args.s1, args.s2,
                                       args.seed, args.atol)))
    else:
        if args.host is None or args.s2 is None:
            ap.error("direct mode requires --host and --s2 (+ --s1 for a middle node), "
                     "or --coordinator/--node-id, or --coordinator/--auto")
        print(json.dumps(attest(args.host, args.port, args.s2, args.n, args.seed, args.atol)))


if __name__ == "__main__":
    main()
