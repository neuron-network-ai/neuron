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


def attest(host, port, s2, n, seed=0, atol=0.05):
    """Full challenge: make -> send -> verify. Returns a result dict."""
    inp, expected = make_challenge(s2, n, seed)
    t0 = time.time()
    output = challenge_node(host, port, s2, n, inp)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6),
            "ms": int((time.time() - t0) * 1000), "layers": [s2, n - 1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=50999)
    ap.add_argument("--s2", type=int, required=True, help="node holds layers[s2:n]")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=0.05)
    args = ap.parse_args()
    print(json.dumps(attest(args.host, args.port, args.s2, args.n, args.seed, args.atol)))


if __name__ == "__main__":
    main()
