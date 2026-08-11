"""test_drift_is_not_evidence.py — a node is never penalised for the coordinator's bookkeeping.

[P37], 2026-08-10: three honest machines were flagged for a PLACEMENT bug. Each was on a range
the coordinator had moved under it, and `placement_drift` was true on exactly those three and
false on the two unflagged ones — a perfect predictor, sitting in `/node/list`, read by nothing.

Re-verification ([P35]) then turned that bookkeeping error into a reputation. It was the right
mechanism given the wrong input: the verifier asks about the range the COORDINATOR believes, and
a node that holds something else fails deterministically, forever, through no fault of its own.

`RangeMismatch` covers the case where the node says so out loud. The two paths here are the ones
that flagged the live machines instead, because drift usually does NOT arrive as a typed answer:

  - **it hangs up.** Asked for layers it never downloaded, the forward pass raises on
    uninitialized meta tensors, `_handle` catches only (ConnectionError, TimeoutError, EOFError),
    the thread dies, `finally: conn.close()` slams the socket. That reaches the verifier as
    `socket closed mid-message` — the generic branch — and after UNREACHABLE_STRIKES cycles it
    was attested as a real failure. Live 2026-08-11: `agent-bhpc012101-18f1da` went 1/18 to 1/22
    in twenty minutes while its ms/layer sat at a figure routing had already discarded.
  - **it answers the wrong question correctly.** `node-c-pavilion`'s `max_err 33.79` was
    deterministic across attempts and verifier restarts: it computed 10-18 with no final norm
    while the verifier compared against 10-27 + norm.

Both are now suppressed while `placement_drift` is set, and the suppression is ASYMMETRIC — a
PASS under drift is still recorded, because it is real evidence (it proves the node does hold the
assigned range, so `reported_*` is the stale field) and it is how a wrongly flagged node climbs
back out of cumulative counters that only ever grow.

Run:  python test_drift_is_not_evidence.py     (from repo root)
"""
import sys
import types

import verify_service
from security import proof_of_compute


class _Recorder:
    """A verifier with the network cut off: challenges are scripted, attestations are collected."""

    def __init__(self, outcome, drifted):
        self.outcome = outcome            # "hangup" | "wrong" | "pass"
        self.drifted = drifted
        self.attested = []                # (node_id, passed)
        v = verify_service.Verifier.__new__(verify_service.Verifier)
        v.base = "http://x"
        v.strikes = {}
        v.unreachable_strikes = {}
        v.unchallengeable = set()
        v.last_checked = {}
        v.last_roster = None
        v.unreachable = 0
        v.challenge = self._challenge
        v.attest = self._attest
        v.nodes = self._nodes
        self.v = v

    def _nodes(self):
        return [{"node_id": "n1", "layer_start": 19, "layer_end": 27, "status": "online",
                 "standing": "verified", "placement_drift": self.drifted,
                 "tailscale_ip": "1.1.1.1", "port": 50999}]

    def _challenge(self, n, total):
        if self.outcome == "hangup":
            raise ConnectionError("socket closed mid-message")
        if self.outcome == "wrong":
            return {"passed": False, "max_err": 33.79, "ms": 12}
        return {"passed": True, "max_err": 1e-6, "ms": 12}

    def _attest(self, nid, passed, max_err):
        self.attested.append((nid, passed))
        return {"standing": "verified", "reputation": 1.0}

    def sweep_n(self, times):
        for _ in range(times):
            self.v.sweep()
        return self.attested


def _total():
    return 28


# The sweep needs a layer total and a roster; everything else is stubbed above.
verify_service.Verifier.total_layers = lambda self: 28


# --------------------------------------------------------------------------- #
# the hang-up path — how [P37]'s two socket-closing nodes were flagged
# --------------------------------------------------------------------------- #
def test_a_drifted_node_that_hangs_up_accumulates_nothing():
    """The live 18f1da case. Ten sweeps is twice UNREACHABLE_STRIKES, so an unguarded verifier
    would have attested at least one failure."""
    r = _Recorder("hangup", drifted=True)
    assert r.sweep_n(10) == [], "a failure was recorded against a node with drifted placement"


def test_an_undrifted_node_that_hangs_up_is_still_penalised():
    """The guard must not become a blanket amnesty: persistent inability to answer IS a fact
    about a node whose placement everyone agrees on ([P35])."""
    r = _Recorder("hangup", drifted=False)
    out = r.sweep_n(verify_service.UNREACHABLE_STRIKES + 1)
    assert out and all(p is False for _, p in out), out


# --------------------------------------------------------------------------- #
# the wrong-answer path — pavilion's deterministic max_err 33.79
# --------------------------------------------------------------------------- #
def test_a_drifted_node_answering_the_wrong_question_accumulates_nothing():
    r = _Recorder("wrong", drifted=True)
    assert r.sweep_n(verify_service.FAIL_STRIKES + 3) == []


def test_an_undrifted_node_giving_wrong_answers_is_still_penalised():
    r = _Recorder("wrong", drifted=False)
    out = r.sweep_n(verify_service.FAIL_STRIKES + 1)
    assert out and all(p is False for _, p in out), out


# --------------------------------------------------------------------------- #
# asymmetry — the half that lets a wrongly flagged node recover
# --------------------------------------------------------------------------- #
def test_a_pass_is_recorded_even_under_drift():
    """Drift suppresses failures, NOT passes. A pass proves the node holds the assigned range,
    so `reported_*` is the stale field -- and cumulative counters only ever grow, so a suppressed
    pass would leave a wrongly flagged node with no way back."""
    r = _Recorder("pass", drifted=True)
    out = r.sweep_n(3)
    assert out and all(p is True for _, p in out), out


def test_a_stage_one_node_is_never_scored_by_the_middle_probe():
    """`make_middle_challenge` computes layers[s1:s2] on a raw hidden state, with no embedding —
    a first-stage node applies one, so the two compute different functions and it answers "wrong"
    every time. Live 2026-08-11 that reached `strike 1 of 3` against the DRIVER; three would have
    flagged the only machine holding stage 1. Not challenged, and said out loud."""
    r = _Recorder("wrong", drifted=False)
    r._nodes = lambda: [{"node_id": "driver", "layer_start": 0, "layer_end": 9,
                         "status": "online", "standing": "verified",
                         "placement_drift": False, "tailscale_ip": "1.1.1.1", "port": 50999}]
    r.v.nodes = r._nodes
    out = r.sweep_n(verify_service.FAIL_STRIKES + 3)
    assert out == [], f"the driver was scored by a challenge it cannot pass: {out}"
    assert r.v.last_checked.get("driver"), "and it must still be stamped, or it starves the rest"


def test_a_stage_one_node_that_is_also_the_last_stage_is_still_challenged():
    """A single-node network holds 0-27: that IS the last stage, so `make_challenge` applies and
    the skip must not swallow it."""
    r = _Recorder("wrong", drifted=False)
    r._nodes = lambda: [{"node_id": "solo", "layer_start": 0, "layer_end": 27,
                         "status": "online", "standing": "verified",
                         "placement_drift": False, "tailscale_ip": "1.1.1.1", "port": 50999}]
    r.v.nodes = r._nodes
    out = r.sweep_n(verify_service.FAIL_STRIKES + 1)
    assert out and all(p is False for _, p in out), out


def test_a_node_that_never_answers_does_not_monopolise_the_rotation():
    """The re-check rotation is `last_checked`-ordered with a default of 0.0 and takes ONE node
    per cycle. An unstamped failure therefore parks that node at the front forever and starves
    every other node -- which is why `82cbee` sat at 2/4, one pass from clearing its flag, while
    `18f1da` was challenged four times in twenty minutes. The attempt is what the rotation
    measures; `unreachable_strikes` counts the outcome."""
    r = _Recorder("hangup", drifted=False)
    r.v.sweep()
    assert r.v.last_checked.get("n1"), (
        "a node that failed to answer was never stamped, so it sorts first on every future "
        "cycle and no other node is ever re-checked")


def test_the_guard_reads_the_field_that_predicted_the_incident():
    """`placement_drift` is derived, not stored, and it must survive the trip through
    /node/list -- the endpoint hides addresses and fingerprints, and hiding this one would
    silently restore the old behaviour."""
    import inspect

    from coordinator import main as coord_main
    src = inspect.getsource(coord_main.node_list)
    # The endpoint strips fields by name into `hidden`. If `placement_drift` ever joins that
    # set, the verifier silently stops seeing drift and every guard above quietly reverts to
    # the 2026-08-10 behaviour -- with all these tests still green, because they stub the
    # roster. This is the one assertion that would catch it.
    assert "placement_drift" not in src, (
        "/node/list now mentions placement_drift; if it is being hidden, the verifier is blind "
        "to drift again and the guards in verify_service.py are dead code")

    # And it is genuinely derived for the verifier to read, not a field only the dashboard sees.
    from coordinator import models
    assert "placement_drift" in inspect.getsource(models._node_dict)


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
