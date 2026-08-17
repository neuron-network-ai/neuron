"""test_stage1_challenge.py — the stage-1 node answers the middle probe  [P47]

    python test_stage1_challenge.py

**This is the measurement that retired a skip, so it lives in the repo rather than in a
scratch file.** `verify_service` refused to challenge any node with `layer_start == 0` from
2026-08-11 to 2026-08-17, on the stated grounds that `make_middle_challenge` computes
`layers[s1:s2]` on a raw hidden state while "a first-stage node normally embeds token ids
first, so the two are not computing the same function and the node answers wrong every time."
That reasoning was labelled a hypothesis in the code, and it is false: `node_server`'s probe
role does not embed either. It runs `common.mid_stage(model, self.lo, self.hi + 1, hidden)`,
which is precisely what `make_middle_challenge` computes.

The cost of the skip was not a gap in reputation, it was money. Emission pays only on a passing
proof-of-compute challenge (`models.mark_slot_poc`), so a node that is never challenged can
never earn an availability hour — and the skipped node is the DRIVER, which [P43] shows carries
the largest fixed cost on the network. [P40]'s reconciliation measured the damage on the live
ledger: 81 unearnable hours on `agent-optinovate-6ff49d`, ~243 NRN, against 333 distributed.

Two properties, both end to end against a real `NodeServer` on a real slice — no mocks, because
a mock of the thing in question would have agreed with the hypothesis:

  1. a stage-1 node ANSWERS the middle probe correctly;
  2. a node challenged on a range it does not hold raises `RangeMismatch` — refused, and
     recorded in neither direction — rather than returning a wrong-looking answer.

(2) is what makes removing the skip safe. It is the difference between "the driver is not
checked" and "the driver can be flagged for the coordinator's own bookkeeping", which is the
[P37] mistake and the thing the skip was really guarding against.

Requires torch and a downloaded slice. Skips cleanly without them: a checkout with no weights
is a legitimate state, and a test that cannot run must say so rather than fail.
"""
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SLICE_0_9 = os.path.join(HERE, "agent", "model_slice_0_9")
PORT_OK, PORT_DRIFT = 51997, 51996

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def _serve(slice_dir, lo, hi, n, port):
    from agent.node_server import NodeServer
    srv = NodeServer(slice_dir, lo, hi, n)
    threading.Thread(target=srv.run, kwargs={"host": "127.0.0.1", "port": port},
                     daemon=True).start()
    if not srv.listening.wait(timeout=180):
        raise RuntimeError(f"node server never bound: {srv.bind_error}")
    return srv


def main():
    if not os.path.isdir(SLICE_0_9):
        print(f"SKIP: no slice at {SLICE_0_9} — this check needs real weights.")
        return True
    try:
        import torch                                                   # noqa: F401
    except ImportError:
        print("SKIP: torch is not installed in this environment.")
        return True

    from security import proof_of_compute as poc

    LO, HI, N = 0, 9, 28

    print("\n-- a stage-1 node answers the middle probe")
    _serve(SLICE_0_9, LO, HI, N, PORT_OK)
    inp, expected = poc.make_middle_challenge(LO, HI + 1)
    out = poc.challenge_middle_node("127.0.0.1", PORT_OK, LO, HI + 1, inp)
    passed, err = poc.verify(out, expected)
    check("it passes", passed, f"max_err {err}")
    # Not merely inside atol. The node and the verifier run the same function over the same
    # weights, so anything above float noise would mean they are not the same function after
    # all -- which is the hypothesis this test exists to settle.
    check("...exactly, not just within tolerance", err == 0.0, f"max_err {err}")

    print("\n-- a node challenged for a range it does not hold is REFUSED, not scored")
    _serve(SLICE_0_9, LO, HI, N, PORT_DRIFT)          # really holds 0-9
    inp2, _ = poc.make_middle_challenge(10, 19)       # coordinator thinks it holds 10-18
    try:
        poc.challenge_middle_node("127.0.0.1", PORT_DRIFT, 10, 19, inp2)
        check("RangeMismatch is raised", False, "the challenge ran and returned an answer")
    except poc.RangeMismatch as e:
        check("RangeMismatch is raised", True)
        check("...naming both ranges", "(0, 10)" in str(e) and "(10, 19)" in str(e), str(e))
    except poc.ChallengeRefused as e:
        check("RangeMismatch is raised", False, f"got plain ChallengeRefused: {e}")

    print("\n-- the verifier no longer skips stage 1")
    import verify_service
    src = open(os.path.join(HERE, "verify_service.py"), encoding="utf-8").read()
    check("the skip is gone", "is NOT BEING VERIFIED" not in src)
    check("stage-1 failures stay unscored until the live driver is seen to pass",
          verify_service.STAGE1_FAILURES_ARE_SCORED is False)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
