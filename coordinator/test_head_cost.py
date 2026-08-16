"""coordinator/test_head_cost.py - run: python -m coordinator.test_head_cost

The driver holds weights nobody charged it for.

Every node in a plan is sized by `gb_per_layer * layers`. The first one additionally holds the
embedding, and the `lm_head` when the model does not tie them — `slice_downloader` fetches both
for `is_first_node`, and `common.apply_lm_head` runs the projection on the driver. Nothing in
the coordinator's arithmetic knew that. `model_tiers` said so in a comment and left it:

    NOT modelled: the driver additionally holds the embedding and lm_head (~2.2 GB for the
    7B), so the first node in a plan is under-charged by that much. Worth fixing with a
    `head_gb` column once a real measurement exists; recorded here rather than guessed.

The measurement now exists (`tools/measure_model.py`, from the model's published safetensors
header), so this is that column and the arithmetic behind it.

**Why it is worth a file of its own rather than a line in the sizing tests.** The head is a
FIXED cost, so its share of a machine grows as the network shrinks. Spread over ten nodes it is
noise; on two it is the whole question. Qwen3-4B's embedding is 389M params — 1.56 GB at the
fp32 the runtime actually defaults to, against the 3.75 GB an 8 GB machine gets after the OS
reserve and headroom. That is 41% of a node's budget, and until now it was spent invisibly.

What must hold:

  1. The head is charged to exactly ONE node. Charging every node refuses networks that fit.
  2. It is charged to the node that will actually HOLD it — which means this module and
     `router.canonical_assignment` must name the same machine, and a test has to say so, because
     they are separate implementations of one rule.
  3. It is corrected for storage dtype on the SAME basis as `gb_per_layer`. Two figures from one
     table drifting onto different bases is the fault `effective_gb` exists to prevent.
  4. Absent, it changes nothing. Every tier shipped before this one has no `head_gb`, and must
     be sized exactly as it was.
  5. It can flip a verdict. A charge that never changes an answer is not being applied.
"""
import sys

from coordinator import balancer, migration, model_tiers, router

ok = fail = 0

# Qwen3-4B-Instruct-2507, measured from its published safetensors header on 2026-08-16 by
# tools/measure_model.py. Both figures are on the tier table's fp16 basis (2 bytes/param);
# `balancer.effective_gb` scales them to whatever a node really stores at.
#   36 layers x 100,930,816 params      -> 0.2019 GB/layer
#   embed_tokens 388,956,160 params     -> 0.7779 GB, and tie_word_embeddings=true, so
#                                          lm_head IS the embedding and is not counted twice
Q4_LAYERS = 36
Q4_GB_PER_LAYER = 0.2019
Q4_HEAD_GB = 0.7779
HEADROOM = 0.75


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def node(node_id, ram_gb, **kw):
    """A roster entry shaped like `models._node_dict` gives one."""
    n = {"node_id": node_id, "ram_gb": ram_gb, "ms_per_layer": 12.0,
         "status": "online", "eligible": True, "layer_start": 0, "layer_end": 0}
    n.update(kw)
    return n


def budget(ram_gb):
    """What `max_layers_for` has to spend on weights, before any head charge."""
    return max(ram_gb - balancer.RAM_OS_RESERVE_GB, 0.0) * HEADROOM


def main():
    # ---------- 1. one node pays, and it is the right one ----------------------
    pav, small = node("pavilion", 12.0), node("node-b", 8.0)
    roster = [small, pav]                      # deliberately NOT in RAM order

    check("head_node_index picks the machine with the most RAM, not the first in the list",
          balancer.head_node_index(roster) == 1)
    check("...deterministically, by node_id, when RAM ties",
          balancer.head_node_index([node("zeta", 8.0), node("alpha", 8.0)]) == 1)
    check("...and answers None for an empty roster rather than raising",
          balancer.head_node_index([]) is None)

    caps = balancer.layer_caps(roster, Q4_GB_PER_LAYER, Q4_LAYERS, Q4_HEAD_GB)
    bare = balancer.layer_caps(roster, Q4_GB_PER_LAYER, Q4_LAYERS)
    check("exactly one node's cap moves when a head is charged",
          sum(1 for a, b in zip(caps, bare) if a != b) == 1,
          f"charged {caps}, uncharged {bare}")
    check("...and it is the max-RAM node that pays", caps[0] == bare[0] and caps[1] < bare[1])

    # ---------- 2. the pin: this module and the router must agree --------------
    # Two separate implementations of "who is the driver". If they drift, feasibility charges
    # the head to a machine the placer does not put it on -- which either clears a plan that
    # OOMs the driver, or refuses one that would have fitted.
    for ram_a, ram_b in ((12.0, 8.0), (8.0, 12.0), (8.0, 8.0), (68.0, 12.0)):
        # ranges that are NOT the incumbent stage 1 (0..DRIVER_STAGE1_LAYERS-1), so the
        # router falls through to the capacity rule this module mirrors
        rs = [node("aaa", ram_a, layer_start=0, layer_end=4),
              node("bbb", ram_b, layer_start=5, layer_end=27)]
        assigned = router.canonical_assignment(rs, 28)
        mine = rs[balancer.head_node_index(rs)]["node_id"]
        theirs = assigned[0]["node_id"] if assigned else None
        check(f"router and balancer name the same driver at {ram_a:g}/{ram_b:g} GB "
              f"({mine})", mine == theirs, f"balancer says {mine}, router says {theirs}")
    check("...and the router really did put that node on layer 0 (the head's home)",
          router.canonical_assignment(
              [node("aaa", 8.0, layer_start=0, layer_end=4),
               node("bbb", 12.0, layer_start=5, layer_end=27)], 28)[0]["layer_start"] == 0)

    # ---------- 3. the arithmetic, against a hand-computed figure ---------------
    # fp32 is what a node really stores at, so both table figures double.
    gpl32, head32 = Q4_GB_PER_LAYER * 2, Q4_HEAD_GB * 2
    expect = int((budget(12.0) - head32) / gpl32)
    check(f"driver cap is (budget - head) / layer, exactly ({expect} layers on 12 GB)",
          balancer.max_layers_for(pav, Q4_GB_PER_LAYER, head_gb=Q4_HEAD_GB) == expect,
          f"got {balancer.max_layers_for(pav, Q4_GB_PER_LAYER, head_gb=Q4_HEAD_GB)}")
    check(f"the head costs this driver {int(budget(12.0)/gpl32) - expect} layers",
          balancer.max_layers_for(pav, Q4_GB_PER_LAYER) > expect)
    check("a non-driver is unaffected",
          balancer.max_layers_for(small, Q4_GB_PER_LAYER)
          == balancer.max_layers_for(small, Q4_GB_PER_LAYER, head_gb=None)
          == int(budget(8.0) / gpl32))

    # ---------- 4. same dtype basis as gb_per_layer -----------------------------
    # The head is weights. If it were left on the table's fp16 basis while layers were
    # corrected to fp32, the driver would be under-charged by exactly the amount this fixes.
    fp16_pav = node("pavilion", 12.0, weight_dtype="fp16")
    check("a node storing fp16 is charged half the head, like it is charged half a layer",
          balancer.effective_gb(Q4_HEAD_GB, fp16_pav) == Q4_HEAD_GB
          and balancer.effective_gb(Q4_HEAD_GB, pav) == head32)
    check("...so it fits more layers even while paying the head",
          balancer.max_layers_for(fp16_pav, Q4_GB_PER_LAYER, head_gb=Q4_HEAD_GB)
          == int((budget(12.0) - Q4_HEAD_GB) / Q4_GB_PER_LAYER))
    check("effective_gb_per_layer still answers as before (the sizing tests call it)",
          balancer.effective_gb_per_layer(Q4_GB_PER_LAYER, pav) == gpl32)
    check("unknown figure stays unknown rather than becoming 0",
          balancer.effective_gb(None, pav) is None)

    # ---------- 5. absent head changes nothing ---------------------------------
    for tier in model_tiers.TIERS:
        live = [node("a", 8.0), node("b", 12.0), node("c", 68.0)]
        check(f"tier {tier['name']} is sized exactly as before this change",
              balancer.capacity_shortfall(live, tier["layers"], tier.get("gb_per_layer"),
                                          tier.get("head_gb"))
              == balancer.capacity_shortfall(live, tier["layers"], tier.get("gb_per_layer")))
    check("no tier declares a head without the per-layer figure it is measured beside",
          all(t.get("gb_per_layer") for t in model_tiers.TIERS if t.get("head_gb")))

    # ---------- 6. it can flip a verdict ---------------------------------------
    # A charge that never changes an answer is not being applied. Sized so the layers alone
    # just fit and the head alone is what breaks it.
    tight = [node("a", 8.0), node("b", 8.0)]
    n_fits = sum(balancer.layer_caps(tight, Q4_GB_PER_LAYER, 99))
    check(f"without the head these two machines hold {n_fits} layers of Qwen3-4B",
          balancer.capacity_shortfall(tight, n_fits, Q4_GB_PER_LAYER) == 0)
    check("...and with it they do not - the head is the difference",
          balancer.capacity_shortfall(tight, n_fits, Q4_GB_PER_LAYER, Q4_HEAD_GB) > 0)

    # ---------- 7. solve charges the driver, not the biggest machine -----------
    # solve()'s contract is PIPELINE ORDER, driver first, and `head_ms` is already keyed to
    # index 0. The memory cost of the head and its time cost are the same weights; they must
    # land on the same machine or the plan is internally inconsistent.
    ordered = [dict(small, ms_per_layer=10.0), dict(pav, ms_per_layer=10.0)]
    a_head = balancer.solve(ordered, 20, Q4_GB_PER_LAYER, Q4_HEAD_GB)
    a_bare = balancer.solve(ordered, 20, Q4_GB_PER_LAYER)
    check("solve charges nodes[0] - the driver - even though it is the smaller machine",
          a_head[0]["max_layers"] < a_bare[0]["max_layers"]
          and a_head[1]["max_layers"] == a_bare[1]["max_layers"])
    check("solve still covers every layer with contiguous ranges",
          a_head[0]["layer_start"] == 0 and a_head[-1]["layer_end"] == 19
          and all(a_head[i + 1]["layer_start"] == a_head[i]["layer_end"] + 1
                  for i in range(len(a_head) - 1)))

    # ---------- 8. the placement path, not just the feasibility path -----------
    # plan_migration is what turns a tier into real layer ranges. Its own sort puts the
    # head-carrying node first (head_ms > 0), so it charges index 0 rather than asking
    # head_node_index -- here that is the SMALL machine, which is the case worth catching.
    #
    # 20 layers, not 36: plan_migration always covers every layer by design, and leaves an
    # over-cap node over-cap when there is nowhere to shift to (`partition_shortfall` is the
    # caller's refusal signal, not this function's). At 36 both nodes are over cap and the
    # split cannot move, so a count that FITS is what shows the charge doing anything.
    def _plan(head):
        p = migration.plan_migration(
            [node("fast-small", 8.0, ms_per_layer=5.0, head_ms=3.0),
             node("slow-big", 12.0, ms_per_layer=20.0)],
            20, gb_per_layer=Q4_GB_PER_LAYER, head_gb=head)
        return {a["node_id"]: a["layer_end"] - a["layer_start"] + 1 for a in p}

    by_id, bare_by_id = _plan(Q4_HEAD_GB), _plan(None)
    check("plan_migration moves layers OFF the driver once the head is charged",
          by_id.get("fast-small", 0) < bare_by_id.get("fast-small", 0),
          f"charged {by_id}, uncharged {bare_by_id}")
    check("...onto the machine with room, still covering every layer",
          sum(by_id.values()) == 20)

    # ---------- 9. the known edge, asserted so it is visible --------------------
    # A driver charged a head bigger than its whole budget reports 1 rather than 0, which
    # understates the shortfall by a layer. Deliberate: dropping the floor would let solve()
    # emit a zero-width stage 1. Pinned here so the limit is a decision, not a surprise.
    starved = node("tiny", 4.0)
    check("a node that cannot hold even the head still reports 1, not 0 (documented floor)",
          balancer.max_layers_for(starved, Q4_GB_PER_LAYER, head_gb=Q4_HEAD_GB) == 1
          and budget(4.0) < head32)

    # ---------- 10. the capacity case itself ------------------------------------
    # The thing this was built for: a model neither machine can hold, across both. The 68 GB
    # box is deliberately absent -- with it there is no capacity case, only a big node.
    two = [node("pavilion", 12.0), node("node-b", 8.0)]
    short32 = balancer.capacity_shortfall(two, Q4_LAYERS, Q4_GB_PER_LAYER, Q4_HEAD_GB)
    two16 = [dict(n, weight_dtype="fp16") for n in two]
    short16 = balancer.capacity_shortfall(two16, Q4_LAYERS, Q4_GB_PER_LAYER, Q4_HEAD_GB)
    check(f"12 GB + 8 GB CANNOT hold Qwen3-4B at fp32 ({short32} layers short)", short32 > 0,
          "16.1 GB of weights against 20 GB of total RAM is not a headroom problem, it is "
          "arithmetic - see PROBLEMS.md [P43]")
    check("12 GB + 8 GB CAN hold it at fp16 storage (fp32 compute, per test_weight_dtype.py)",
          short16 == 0)
    check("...and neither machine could hold it alone, which is the whole claim",
          balancer.capacity_shortfall([two16[0]], Q4_LAYERS, Q4_GB_PER_LAYER, Q4_HEAD_GB) > 0
          and balancer.capacity_shortfall([two16[1]], Q4_LAYERS, Q4_GB_PER_LAYER,
                                          Q4_HEAD_GB) > 0)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
