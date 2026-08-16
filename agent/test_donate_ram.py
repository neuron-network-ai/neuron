"""agent/test_donate_ram.py - run: python -m agent.test_donate_ram

A volunteer can cap how much of their machine they lend.

`donation_mode` has always governed WHEN a node serves. Nothing governed HOW MUCH of the
machine it commits while it does, so someone with a 64 GB workstation happy to lend 8 GB had to
choose between donating the whole thing and not donating at all. That is the worst trade to put
in front of the person most able to help, and it is also what stood between this project and
its own capacity case: with a 64 GiB machine in a two-node network, no model is "too big for
any one machine" until it is nearly 50 GB.

**It is a declaration the COORDINATOR enforces, not a runtime limiter.** The agent reports the
capped figure as `ram_gb`; `balancer.max_layers_for` sizes the slice from it; the node is never
handed layers that do not fit inside the cap. There is nothing to police while it runs, because
the work is never given out. That is why this is four lines of agent and no new coordinator
code — the sizing path already asked exactly the right question, it was just always being
answered with the hardware's number.

What must hold:
  1. no cap set -> exactly the old behaviour, to the byte;
  2. a cap below the machine is honoured;
  3. a cap ABOVE the machine is ignored -- registration takes no credential under open join, and
     this is a field that turns into a memory budget (`balancer.sane_vram_gb`'s reasoning, a
     layer earlier);
  4. nonsense is "no cap", never "donate nothing" -- a typo must not take a node off the network;
  5. the coordinator sizes from it, which is the entire point.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import agent
from coordinator import balancer

ok = fail = 0
GB = 10**9


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def donated(cap, total_gib=64):
    return agent.donated_ram_gb({"donate_ram_gb": cap}, total_bytes=total_gib * 2**30)


def main():
    # A 64 GiB machine reports 68, because agent.py converts binary GiB to decimal GB ([P43]).
    check("no cap set -> the machine's own figure, unchanged", donated(None) == 68)
    check("...and that is the same arithmetic as before this existed",
          donated(None) == int(64 * 2**30 // GB))
    check("the DEFAULT_CONFIG ships with no cap, so nothing changes for anyone",
          agent.DEFAULT_CONFIG["donate_ram_gb"] is None)

    check("a cap below the machine is honoured", donated(8) == 8)
    check("...including one that is most of it", donated(60) == 60)
    check("a float cap is floored rather than refused", donated(8.9) == 8)

    check("a cap ABOVE the machine is ignored, not honoured", donated(512) == 68,
          "otherwise it is a way to be assigned a slice this node cannot hold, on an "
          "endpoint that takes no credential under open join")
    check("...and one exactly at the machine is a no-op", donated(68) == 68)

    for junk in (0, -4, "eight", "", [], {}, float("nan")):
        check(f"{junk!r} means NO CAP, not 'donate nothing'", donated(junk) == 68,
              "a typo in a config file must not silently take a node off the network")
    # A sub-1 GB cap is an INTENTION ("almost nothing"), not a typo, so it floors to 1 rather
    # than being ignored. Ignoring it would donate the whole machine — the exact opposite of
    # what was asked for, which is the worst way to be wrong about a consent setting.
    check("a cap under 1 GB floors to 1, it does not fall back to donating everything",
          donated(0.4) == 1 and donated(1) == 1)

    # ---------- the point of it: the coordinator sizes from the donated figure ----------
    Q4_GPL, Q4_HEAD, Q4_LAYERS = 0.2019, 0.7779, 36
    full = {"node_id": "big", "ram_gb": donated(None), "weight_dtype": "fp16"}
    capped = {"node_id": "big", "ram_gb": donated(8), "weight_dtype": "fp16"}
    check("uncapped, the 64 GiB machine holds the whole 4B model by itself",
          balancer.capacity_shortfall([full], Q4_LAYERS, Q4_GPL, Q4_HEAD) == 0)
    check("capped at 8 GB it does NOT, which is what makes a capacity case possible at all",
          balancer.capacity_shortfall([capped], Q4_LAYERS, Q4_GPL, Q4_HEAD) > 0)

    pav = {"node_id": "pavilion", "ram_gb": 12.0, "weight_dtype": "fp16"}
    check("...and together with the Pavilion it fits, neither machine alone",
          balancer.capacity_shortfall([capped, pav], Q4_LAYERS, Q4_GPL, Q4_HEAD) == 0
          and balancer.capacity_shortfall([pav], Q4_LAYERS, Q4_GPL, Q4_HEAD) > 0)

    # The cap must also move the DRIVER: stage 1 carries the head, and the router picks the
    # machine with the most RAM. A cap that sized slices but not that choice would put the
    # embedding on the machine that just said it had least to give.
    check("a capped machine stops being chosen as the driver",
          balancer.head_node_index([capped, pav]) == 1
          and balancer.head_node_index([full, pav]) == 0)

    # ---------- and it is actually wired into what gets sent ----------
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py"),
               encoding="utf-8").read()
    check("the registration body reports the donated figure",
          '"ram_gb": donated_ram_gb(self.cfg),' in src)
    check("...and the log says 'donated of' so a small assignment is not read as a misjudgement",
          "GB donated of" in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
