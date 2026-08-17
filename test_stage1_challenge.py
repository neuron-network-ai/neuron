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
PORT_OK, PORT_DRIFT, PORT_FULL = 51997, 51996, 51995

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
    import common

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
        # Wording changed when `holds` started being checked FIRST ([P47], 2026-08-17): the node
        # reports what it really has, so the message can name that instead of two abstract
        # tuples. Strictly more informative — it says which side is wrong.
        check("...naming both ranges", "0-9" in str(e) and "10-18" in str(e), str(e))
    except poc.ChallengeRefused as e:
        check("RangeMismatch is raised", False, f"got plain ChallengeRefused: {e}")

    print("\n-- a node holding the WHOLE model is refused, not answered wrongly")
    # The case that was actually costing the driver every hour it ever worked, found live on
    # 2026-08-17 and reproduced here. `agent-optinovate-6ff49d` is assigned 0-9 and its
    # NodeServer serves 0-27 -- a 64 GB machine that loaded the whole model -- so
    # `is_true_last` (self.hi == n-1) was TRUE and a probe fell into the LAST-stage branch.
    # It ran layers[10:] + the final norm for a challenge about layers 0-9 and answered
    # max_err 28.5958: deterministic, confident, and about a different question entirely.
    # That is the unexplained figure of 2026-08-11, and it was never a bad machine.
    #
    # Built on the 0-9 slice and then moved to hi == N-1, because [P42]'s reload guard rightly
    # refuses to CONSTRUCT a 0-27 server over a 0-9 slice and this machine has no full slice to
    # hand. What is under test is role SELECTION, which reads self.hi and self.n and nothing
    # else, so this reproduces the live driver's state exactly where it matters. The forward
    # pass is never reached: the range is refused first, which is the whole point.
    full = _serve(SLICE_0_9, LO, HI, N, PORT_FULL)
    full.hi = N - 1                    # now `is_true_last` is True, as on the live driver
    inp3, _ = poc.make_middle_challenge(LO, HI + 1)
    try:
        poc.challenge_middle_node("127.0.0.1", PORT_FULL, LO, HI + 1, inp3)
        check("a full-model node refuses a stage-1 probe instead of answering it", False,
              "it returned an answer — this is the 28.6")
    except poc.RangeMismatch as e:
        check("a full-model node refuses a stage-1 probe instead of answering it", True)
        check("...naming what it really holds, so the operator is told the placement is stale",
              "0-27" in str(e), str(e))

    # WHICH BRANCH ANSWERED, pinned separately. The two fixes are independent and either one
    # alone makes the check above pass: `challenge_middle_node` now reads `holds`, so it would
    # raise RangeMismatch even against an unpatched node still using the last-stage branch.
    # That is deliberate (it works against today's agents with no release) and it is exactly why
    # it cannot stand as evidence for the node-side fix. The ack tells them apart: only the
    # PROBE branch sends `s1`, and only the probe branch computes this node's own layers.
    import socket as _socket
    _s = _socket.create_connection(("127.0.0.1", PORT_FULL), timeout=30)
    try:
        common.send_msg(_s, {"type": "config", "s1": LO, "s2": HI + 1})
        ack = common.recv_msg(_s)
    finally:
        _s.close()
    check("a probe config is answered by the PROBE branch, not the last-stage one",
          ack.get("s1") is not None, f"ack={ack}")
    check("...and it reports its own real range rather than echoing the caller's",
          (ack.get("s1"), ack.get("s2")) == (0, N), f"ack={ack}")

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
