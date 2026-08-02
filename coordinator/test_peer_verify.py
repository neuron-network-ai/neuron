"""coordinator/test_peer_verify.py — run: python -m coordinator.test_peer_verify

Peer verification exists to remove the last human from the join path. Before it, a newcomer
stayed probationary — reachable, earning nothing — until the operator personally ran
proof_of_compute against them, so the network could only grow while one laptop was on and one
person held a secret nobody else could be given.

The interesting cases are not the happy path but the ways a decentralised vote goes wrong:
one node promoting its own sybils, a node promoting itself, an unverified node voting at all.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-peer-"), "t.db")

from coordinator import config, models          # noqa: E402
from coordinator import main as coord           # noqa: E402
from fastapi import HTTPException                # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def mk(node_id, lo, hi, trusted=False):
    tok = f"tok-{node_id}"
    models.register_node(node_id, "1.2.3.4", 50999, lo, hi, 4, 8, tok, trusted=trusted)
    return tok


def main():
    models.init_db()
    config.PEER_VERIFY_QUORUM = 2

    a = mk("verifier-a", 0, 9, trusted=True)
    b = mk("verifier-b", 10, 18, trusted=True)
    c = mk("newcomer", 19, 27)                     # open-join, probationary
    mk("outsider", 19, 27)                          # probationary, so may NOT vote

    check("a newcomer starts probationary",
          models.get_node("newcomer")["standing"] == "probationary")

    # --- only a verified node may vote ------------------------------------------- #
    body = coord.AttestBody(passed=True, max_err=0.0)
    try:
        coord.peer_attest("newcomer", body, x_node_token="tok-outsider")
        check("an unverified node cannot attest", False)
    except HTTPException as e:
        check("an unverified node cannot attest", e.status_code == 403)
    try:
        coord.peer_attest("newcomer", body, x_node_token="not-a-real-token")
        check("a bogus token cannot attest", False)
    except HTTPException as e:
        check("a bogus token cannot attest", e.status_code == 401)

    # --- a node cannot verify itself ---------------------------------------------- #
    try:
        coord.peer_attest("verifier-a", body, x_node_token=a)
        check("a node cannot verify itself", False)
    except HTTPException as e:
        check("a node cannot verify itself", e.status_code == 400)

    # --- one verifier is not a quorum, and cannot become one by repeating --------- #
    coord.peer_attest("newcomer", body, x_node_token=a)
    check("one vote does not promote", models.get_node("newcomer")["standing"] == "probationary")
    for _ in range(5):
        coord.peer_attest("newcomer", body, x_node_token=a)
    passes, _f = models.peer_verdicts("newcomer")
    check("the same verifier voting 6 times still counts once (sybil guard)", passes == 1)
    check("...so it still is not promoted",
          models.get_node("newcomer")["standing"] == "probationary")

    # --- a second, distinct verifier reaches quorum ------------------------------- #
    out = coord.peer_attest("newcomer", body, x_node_token=b)
    check("two distinct verifiers promote it", out["standing"] == "verified")
    check("...and it becomes eligible to serve and earn",
          models.get_node("newcomer")["eligible"] is True)
    check("the response reports the quorum honestly",
          out["distinct_passes"] == 2 and out["quorum"] == 2)

    # --- assignments: only to verified nodes, never yourself, never twice --------- #
    mk("newcomer2", 19, 27)
    models.touch_node("newcomer2")
    models.touch_node("verifier-a")
    job = coord.verify_assignment(x_node_token=a)
    # which probationary node it picks is not specified — only that it is one of them, and
    # never an already-verified node or the caller itself
    check("a verified node is handed a probationary target",
          job.get("node_id") in {"outsider", "newcomer2"})
    check("the assignment carries the address a challenge needs",
          job.get("host") and job.get("port"))
    coord.peer_attest(job["node_id"], body, x_node_token=a)
    job2 = coord.verify_assignment(x_node_token=a)
    check("a verifier is not handed the same target twice",
          job2.get("node_id") != job["node_id"])
    try:
        coord.verify_assignment(x_node_token="tok-outsider")
        check("a probationary node gets no assignments", False)
    except HTTPException as e:
        check("a probationary node gets no assignments", e.status_code == 403)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
