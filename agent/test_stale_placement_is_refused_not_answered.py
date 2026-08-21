"""agent/test_stale_placement_is_refused_not_answered.py — run:
       python -m agent.test_stale_placement_is_refused_not_answered

**[P60]. A node answered real pipeline traffic in a role it could not serve, and the answer
looked like an answer.**

`node_server.serve` chose its role in three arms: middle if the config carries `host_b`, LAST if
this node holds the model's final layer and the message is not a probe, and PROBE for everything
else. That last arm was a catch-all, and its comment justified itself with an inference that is
simply not true:

    a config with no host_b that ... reaches a node whose own range does NOT include the
    model's final layer can only be a verifier challenging this node in isolation

It can also be a node the COORDINATOR believes is last while the node knows it is not.
Placement moves in the coordinator's database and the node finds out when it next registers
([P37]) — so between those two moments, the driver asks a middle node for a last stage.

Live 2026-08-21: the coordinator moved a node to 10-27 while it held 10-18. The driver asked for
a last stage. The node answered as a PROBE — `mid_stage(10, 19)`, correct arithmetic over the
nine layers it really had, and **no final norm**. The driver ran `lm_head` on an un-normed
hidden state, and a question about the sky came back as
`Sovereberg Sovere ABCDEFGHITestCategory`.

Every indicator was green: `routable`, `stage1_ok`, `network_healthy`, 28/28 covered — and
decode hit **3.00 tok/s, the fastest this network has ever produced**, because a node running
nine layers instead of eighteen is genuinely quicker. **Speed rose while correctness went to
zero.** Same ending as [P55], down a different road.

The rule this pins, stated positively: **a node answers real pipeline traffic only for a role it
can actually serve.** Anything else is a named refusal, never a plausible-looking tensor.
Refusing costs nothing — `range_mismatch` is already a reroute to the driver, and
proof-of-compute already declines to score it as a failed challenge, because a node telling the
truth about stale placement is not a node computing badly.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import common
from agent.node_server import _is_range_probe

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def role_of(msg, lo, hi, n):
    """`node_server.serve`'s role decision, in its own order — including the arm [P60] added.
    Returns "middle" | "last" | "refuse" | "probe"."""
    is_true_last = (hi == n - 1)
    if "host_b" in msg:
        return "middle"
    if is_true_last and not _is_range_probe(msg):
        return "last"
    if not _is_range_probe(msg):
        return "refuse"
    return "probe"


def main():
    N = 28

    print("-- the live shape: coordinator says 10-27, the node holds 10-18")
    # Exactly what neuron_driver._connect sends a last hop, via the shared constructor.
    traffic = common.stage_config(10, stage="last", hidden_size=1536, lan_hint=["192.168.1"])
    check("real last-stage traffic to a node that is NOT last is REFUSED",
          role_of(traffic, 10, 18, N) == "refuse",
          f"got {role_of(traffic, 10, 18, N)!r} — 'probe' is the bug: it answers "
          f"mid_stage() with no final norm and the driver runs lm_head on it")

    print("\n-- and a driver that predates `stage` gets the same protection")
    # A 2-stage chain from an older driver: s1 == s2 is a junction, i.e. traffic, not a probe.
    old_driver = {"type": "config", "s1": 10, "s2": 10, "wire": ["f32"]}
    check("s1 == s2 is traffic, so it is refused too",
          role_of(old_driver, 10, 18, N) == "refuse", role_of(old_driver, 10, 18, N))
    # A middle relay forwarding on carries no s1 at all.
    relayed = {"type": "config", "s2": 19, "n": N, "wire": ["f32"], "stage": "last"}
    check("a middle relay's onward hop is refused by a node that is not last",
          role_of(relayed, 19, 25, N) == "refuse", role_of(relayed, 19, 25, N))

    print("\n-- nothing that WORKED before is refused now")
    check("a true last stage still serves real traffic",
          role_of(traffic, 10, 27, N) == "last")
    check("a middle relay still relays", role_of(
        common.stage_config(19, stage="middle", s1=10, host_b="h", port_b=1), 10, 18, N)
        == "middle")
    check("a verifier's probe to a mid-range node is still a probe",
          role_of({"type": "config", "s1": 10, "s2": 19, "probe": True}, 10, 18, N) == "probe")
    check("a verifier's probe to a FULL-model node is still a probe",
          role_of({"type": "config", "s1": 0, "s2": 10, "probe": True}, 0, 27, N) == "probe",
          "the 2026-08-17 case: is_true_last is true even for a question about layers 0-9")

    print("\n-- the refusal is the one the rest of the system already understands")
    src = open(os.path.join(HERE, "node_server.py"), encoding="utf-8").read()
    i = src.find("elif not _is_range_probe(msg):")
    arm = src[i:src.index("else:", i)]
    check("it refuses with `range_mismatch`", '"error": "range_mismatch"' in arm)
    check("...and reports what it ACTUALLY holds, so the caller can diagnose it",
          '"holds": [self.lo, self.hi]' in arm,
          "a refusal that does not say what is true leaves the operator where the outage did")
    check("...and returns rather than falling through to a compute branch",
          arm.rstrip().endswith("return"))

    print("\n-- and the far end treats it as a reroute, not as a bad node")
    drv = open(os.path.join(os.path.dirname(HERE), "neuron_driver.py"), encoding="utf-8").read()
    check("the driver turns a refused config into PeerUnavailable",
          "raise PeerUnavailable(f\"next hop refused config: " in drv)
    check("...which is a ConnectionError, so the existing reroute path takes it",
          "class PeerUnavailable(ConnectionError):" in drv)
    poc = open(os.path.join(os.path.dirname(HERE), "security", "proof_of_compute.py"),
               encoding="utf-8").read()
    check("proof-of-compute reads it as placement, never as a failed challenge",
          'if err == "range_mismatch":' in poc and "class RangeMismatch(ChallengeRefused)" in poc,
          "flagging an honest node for the coordinator's stale bookkeeping is [P28]")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
