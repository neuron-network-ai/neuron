"""test_flag_recovery.py — run: python test_flag_recovery.py

A node challenged on layers it does not hold must not be recorded as a bad node, and a node
that HAS been flagged must be able to earn its way back.

Both halves failed live on 2026-08-10 and together they took the network down to a single
point of failure. Three nodes were on ranges the coordinator had moved ([P32] drift):

    node-c-pavilion          coordinator 10-27  |  config claims 0-27  |  disk holds 10-18
    agent-bhpc012104-82cbee  coordinator 10-27  |  config claims 14-27
    agent-bhpc012101-18f1da  coordinator 19-27  |  config claims 10-13

Challenged on the coordinator's range, each failed in one of two ways, and the verifier could
tell neither from cheating:

  * ANSWERED FOR THE WRONG RANGE. Pavilion is not a last-stage node, so it replied from the
    probe branch with its real range and computed layers 10-18 with no final norm. Compared
    against layers 10-27 + norm that is max_err 33.79, deterministic across restarts, and
    identical in shape to a node running corrupted weights. `challenge_node` never read the
    ack that said so -- the information was on the wire the whole time.
  * DIED RUNNING LAYERS IT NEVER DOWNLOADED. The last-stage role took `s2` from the caller, so
    layers below its own start are uninitialized meta tensors; the forward pass raised
    something that is not a ConnectionError, the thread died, and `finally: conn.close()` made
    it look like a hangup ("socket closed mid-message").

Then the flag was permanent: `flagged` is derived from cumulative counters, and the verifier
skipped flagged nodes -- so a flagged node was never challenged again, its counters never
moved, and no endpoint reset them. Three honest machines were excluded from routing forever
for a coordinator bookkeeping error, leaving one eligible node holding all 28 layers:
28/28 covered, 1 stage, unroutable.

Needs no model weights: the protocol decisions are all made on the ack, so a loopback socket
and a zero tensor are enough to exercise the real challenge path.
"""
import sys


ok = fail = 0


def check(label, cond, extra=""):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{('  -- ' + str(extra)) if extra and not cond else ''}")


import security.proof_of_compute as poc


# --------------------------------------------------------------------------- #
# 1. the ack now says what the node actually holds, and disagreement is its own type
# --------------------------------------------------------------------------- #
def test_range_check():
    print("\n-- a placement disagreement is not a wrong answer")

    # pavilion, exactly: challenged as last-stage 10-27, answers from the probe branch with its
    # real range 10-18. This is the ack that was already being sent and never read.
    try:
        poc._check_range({"ok": True, "layers": 28, "s1": 10, "s2": 19}, 10, 27, "s2=10, n=28")
        check("the live pavilion ack is rejected", False, "no exception raised")
    except poc.RangeMismatch as e:
        check("the live pavilion ack is rejected as a RangeMismatch", True)
        check("...and the message names both ranges, so it needs no other evidence",
              "10-27" in str(e) and "10-18" in str(e), str(e))

    # RangeMismatch must be catchable as a refusal but NOT as a generic protocol error, or the
    # verifier's blanket `except Exception` swallows it back into the strike path it escaped.
    check("RangeMismatch is a ChallengeRefused", issubclass(poc.RangeMismatch,
                                                            poc.ChallengeRefused))
    check("ChallengeRefused is not itself a RangeMismatch",
          not issubclass(poc.ChallengeRefused, poc.RangeMismatch))

    # a node that agrees passes straight through
    poc._check_range({"ok": True, "holds": [19, 27]}, 19, 27, "")
    poc._check_range({"ok": True, "s1": 19, "s2": 28}, 19, 27, "")
    check("a node reporting the range it was challenged on is accepted", True)

    # COVERAGE, not equality. A node holding a superset serves the challenged range correctly
    # -- `layers[lo:]` on a full skeleton -- and `ensure_slice` reuses such a slice on purpose
    # rather than re-downloading it. Requiring equality would refuse to verify exactly those
    # nodes, so they could never be promoted: the same harm by the opposite route.
    poc._check_range({"ok": True, "holds": [0, 27]}, 10, 27, "")
    check("a node holding MORE than it was challenged on is accepted, not failed", True)

    # `holds` wins when both are present: it is the node's own range on every role, whereas a
    # last-stage ack echoes the caller's s2 back and so can never disagree with itself.
    try:
        poc._check_range({"ok": True, "holds": [14, 27], "s2": 10}, 10, 27, "")
        check("`holds` is authoritative over the echoed s2", False, "no exception")
    except poc.RangeMismatch:
        check("`holds` is authoritative over the echoed s2", True)

    # An agent too old to send either must not be failed for it. Silence is not disagreement,
    # and treating it as such would flag every node on the network the day this ships.
    poc._check_range({"ok": True, "layers": 28, "s2": 19}, 19, 27, "")
    check("a pre-0.20 agent that reports no range is not failed for the omission", True)


def test_challenge_node_calls_it():
    """The check has to run on the REAL path, not merely exist.

    Testing `_check_range` alone would pass just as happily with the call site deleted -- and
    a call site that was never there is the entire bug: the node has always sent its range on
    the probe path and `challenge_node` has always thrown it away. So this speaks the protocol
    over a loopback socket, which is the cheapest thing that can tell the two apart.
    """
    print("\n-- ...and challenge_node actually performs the check")
    import socket
    import threading

    import common

    def serve_one(ack):
        """A one-shot node that answers the config with `ack` and nothing else."""
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)

        def run():
            conn, _ = srv.accept()
            try:
                common.recv_msg(conn)                       # the config
                common.send_msg(conn, ack)
                common.recv_msg(conn)                       # act, if the caller gets that far
            except Exception:                               # noqa: BLE001,S110
                pass                                        # the caller hanging up is the pass
            finally:
                conn.close()
                srv.close()

        threading.Thread(target=run, daemon=True).start()
        return srv.getsockname()[1]

    import torch
    inp = torch.zeros(1, 1, 4)

    # pavilion's live ack, over a real socket, through the real function.
    port = serve_one({"ok": True, "layers": 28, "s1": 10, "s2": 19})
    try:
        poc.challenge_node("127.0.0.1", port, 10, 28, inp, timeout=10)
        check("challenge_node rejects the live pavilion ack", False, "no exception raised")
    except poc.RangeMismatch:
        check("challenge_node rejects the live pavilion ack", True)
    except Exception as e:                                                   # noqa: BLE001
        check("challenge_node rejects the live pavilion ack", False,
              f"{e.__class__.__name__}: {e}")

    # the new typed refusal, which replaces a thread death that read as a hangup
    port = serve_one({"ok": False, "error": "range_mismatch",
                      "detail": "asked to serve layers 10-27, but this node holds 14-27",
                      "holds": [14, 27]})
    try:
        poc.challenge_node("127.0.0.1", port, 10, 28, inp, timeout=10)
        check("a range_mismatch refusal raises RangeMismatch", False, "no exception raised")
    except poc.RangeMismatch as e:
        check("a range_mismatch refusal raises RangeMismatch", True)
        check("...carrying the node's own explanation, not a generic protocol error",
              "14-27" in str(e), str(e))
    except Exception as e:                                                   # noqa: BLE001
        check("a range_mismatch refusal raises RangeMismatch", False,
              f"{e.__class__.__name__}: {e}")

    # a paused node is refused, but it is NOT a range problem and must not be confused for one
    port = serve_one({"ok": False, "error": "paused", "detail": "paused by its owner"})
    try:
        poc.challenge_node("127.0.0.1", port, 10, 28, inp, timeout=10)
        check("a paused node raises ChallengeRefused", False, "no exception raised")
    except poc.RangeMismatch:
        check("a paused node raises ChallengeRefused, not RangeMismatch", False)
    except poc.ChallengeRefused as e:
        check("a paused node raises ChallengeRefused, not RangeMismatch", "paused" in str(e))


# --------------------------------------------------------------------------- #
# 2. the verifier records nothing for a disagreement, and keeps checking flagged nodes
# --------------------------------------------------------------------------- #
def test_verifier():
    print("\n-- the verifier attributes the fault to placement, not to the node")
    import verify_service

    class V(verify_service.Verifier):
        def __init__(self, outcome):
            super().__init__("http://c", "s", interval=0)
            self.outcome = outcome
            self.attested = []
            self.roster = []

        def nodes(self):
            return self.roster

        def total_layers(self):
            return 28

        def heartbeat(self):
            """Stubbed with the rest of the network ([P47] cause 2).

            This harness talks to a coordinator at `http://c` that does not exist. `nodes` and
            `attest` are already overridden for that reason; the heartbeat was not, so every
            sweep spent ~2.7s failing to resolve a hostname and this file took a minute. Not a
            timeout — DNS gives up long before one — which is why lowering it did nothing.
            """

        def attest(self, node_id, passed, max_err):
            self.attested.append((node_id, passed))
            return {"node_id": node_id, "passed": passed, "reputation": 1.0,
                    "standing": "verified", "flagged": False}

        def challenge(self, node, total):
            out = self.outcome(node)
            if isinstance(out, Exception):
                raise out
            return out

    def node(nid, standing="verified", flagged=False, status="online", lo=10, hi=27):
        # tailscale_ip/port are present because sweep() refuses to run without them -- an
        # address-less roster means the operator secret was rejected, and verifying nothing
        # while reporting success is the failure that guard exists to prevent.
        return {"node_id": nid, "standing": standing, "flagged": flagged, "status": status,
                "layer_start": lo, "layer_end": hi, "tailscale_ip": "1.2.3.4", "port": 9}

    # --- a range mismatch is attested NEITHER way and takes no strike --------- #
    v = V(lambda n: poc.RangeMismatch("challenged on layers 10-27 but the node holds 10-18"))
    v.roster = [node("pavilion")]
    v.sweep()
    check("a range mismatch is not attested as a failure",
          ("pavilion", False) not in v.attested, v.attested)
    check("...nor smuggled through as a pass", ("pavilion", True) not in v.attested, v.attested)
    check("...and records nothing at all", v.attested == [], v.attested)
    check("...and takes no unreachable strike", not v.unreachable_strikes.get("pavilion"),
          v.unreachable_strikes)
    check("...but IS stamped as checked, or the rotation never reaches another node",
          "pavilion" in v.last_checked)

    # Five consecutive mismatches is what the live incident looked like. Under the old code
    # that was UNREACHABLE_STRIKES and an attested failure; it must now still record nothing.
    for _ in range(5):
        v.sweep()
    check("five mismatches in a row still record nothing", v.attested == [], v.attested)

    # --- a genuine hangup still counts, or [P35] regresses ------------------- #
    v2 = V(lambda n: ConnectionError("socket closed mid-message"))
    v2.roster = [node("really-broken")]
    for _ in range(verify_service.UNREACHABLE_STRIKES):
        v2.sweep()
    check("a node that genuinely cannot answer is still failed after UNREACHABLE_STRIKES",
          ("really-broken", False) in v2.attested, v2.attested)

    # --- a wrong answer still counts, or proof-of-compute means nothing ------ #
    v3 = V(lambda n: {"passed": False, "max_err": 33.79, "ms": 500})
    v3.roster = [node("liar")]
    for _ in range(verify_service.FAIL_STRIKES):
        v3.sweep()
    check("a genuinely wrong answer is still recorded as a failure",
          ("liar", False) in v3.attested, v3.attested)

    print("\n-- a flagged node can earn its way back")

    # --- flagged nodes are on the recheck rotation --------------------------- #
    v4 = V(lambda n: {"passed": True, "max_err": 1e-5, "ms": 300})
    v4.roster = [node("pavilion", standing="flagged", flagged=True)]
    v4.sweep()
    check("a FLAGGED node is challenged again", v4.attested != [], v4.attested)
    check("...and its pass is RECORDED, which is the only thing that can lift the flag",
          ("pavilion", True) in v4.attested, v4.attested)

    # --- and a re-check's pass is recorded too -------------------------------- #
    # Failures were written and passes were not, so re-verification could only ever move a
    # node's ratio DOWN -- one bad afternoon was permanent, however long it behaved after.
    v5 = V(lambda n: {"passed": True, "max_err": 1e-5, "ms": 300})
    v5.roster = [node("healthy", standing="verified")]
    v5.sweep()
    check("a healthy node's re-check pass is recorded, not discarded",
          ("healthy", True) in v5.attested, v5.attested)

    # --- an offline node is still left alone --------------------------------- #
    v6 = V(lambda n: {"passed": True, "max_err": 0.0, "ms": 1})
    v6.roster = [node("gone", standing="flagged", flagged=True, status="offline")]
    v6.sweep()
    check("an offline flagged node is not challenged", v6.attested == [], v6.attested)

    # --- the rotation still moves, one node per cycle ------------------------ #
    v7 = V(lambda n: {"passed": True, "max_err": 0.0, "ms": 1})
    v7.roster = [node("a"), node("b", standing="flagged", flagged=True), node("c")]
    v7.sweep()
    check("only one node is re-challenged per cycle, whatever its standing",
          len(v7.attested) == 1, v7.attested)
    v7.sweep()
    v7.sweep()
    check("...and three cycles reach all three, oldest-checked first",
          {n for n, _ in v7.attested} == {"a", "b", "c"}, v7.attested)

    # --- a named refusal is not evidence of anything ------------------------- #
    v8 = V(lambda n: poc.ChallengeRefused("node refused config (paused): paused by its owner"))
    v8.roster = [node("paused-node")]
    for _ in range(verify_service.UNREACHABLE_STRIKES + 1):
        v8.sweep()
    check("a node paused by its owner is never recorded as a failure", v8.attested == [],
          v8.attested)


# --------------------------------------------------------------------------- #
# 3. the operator can retire evidence the network manufactured
# --------------------------------------------------------------------------- #
def test_reset():
    print("\n-- a verdict produced by a coordinator bug can be retired")
    import os
    import tempfile

    os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-flag-"), "t.db")
    for m in [k for k in sys.modules if k.startswith("coordinator")]:
        del sys.modules[m]
    from coordinator import models

    models.init_db()
    models.register_node("n1", "1.2.3.4", 9, 10, 27, 4, 12.0, "tok1")
    models.record_attestation("n1", True)
    models.record_attestation("n1", False)
    models.record_attestation("n1", False)
    n = models.get_node("n1")
    check("1 pass / 2 fails is flagged, as it was live", n["flagged"] and n["standing"] == "flagged",
          n["standing"])
    check("...and therefore ineligible to serve or earn", not n["eligible"])

    check("resetting an unknown node reports it rather than pretending",
          models.reset_attestations("nope") is False)

    models.reset_attestations("n1")
    n = models.get_node("n1")
    check("after a reset the flag is gone", not n["flagged"], n["standing"])
    check("...the counters read as never-challenged, not as a forged pass",
          n["challenges_passed"] == 0 and n["challenges_failed"] == 0)
    check("...reputation is None (no samples), not a fabricated 1.0", n["reputation"] is None)
    check("...and the node is back to probationary, NOT silently promoted",
          n["standing"] == "probationary" and not n["eligible"], n["standing"])

    # Two more passes from where pavilion actually was: 1/3 -> 3/5 = 0.6, which is not BELOW
    # the 0.6 threshold. So a flagged node genuinely can clear itself once it is challenged
    # again -- the arithmetic was never the obstacle, being skipped by the verifier was.
    models.record_attestation("n2", True)                     # unknown node: no row to update
    models.register_node("n2", "1.2.3.4", 8, 10, 27, 4, 12.0, "tok2")
    for passed in (True, False, False):
        models.record_attestation("n2", passed)
    check("n2 starts flagged at 1/3", models.get_node("n2")["flagged"])
    models.record_attestation("n2", True)
    models.record_attestation("n2", True)
    n2 = models.get_node("n2")
    check("two further passes lift the flag with no operator involved (3/5 >= 0.6)",
          not n2["flagged"] and n2["eligible"], f"{n2['reputation']} {n2['standing']}")


def main():
    test_range_check()
    test_challenge_node_calls_it()
    test_verifier()
    test_reset()
    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
