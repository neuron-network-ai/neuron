"""coordinator/test_weight_dtype_sizing.py — run: python -m coordinator.test_weight_dtype_sizing

The tier table and the runtime disagreed about how big a layer is, by exactly 2x, and the
disagreement was documented as agreement.

`model_tiers.gb_per_layer` is computed at the fp16 storage dtype (2 bytes/param). Its comment
justified that with "common.WEIGHT_DTYPE=fp16". `common.py:62` reads `NEURON_WEIGHT_DTYPE` and
defaults to **fp32** — 4 bytes/param — nothing in the agent or installer sets it, and
`cast_linears` is a no-op at fp32. So every node on every tier was cleared to hold twice what
it can.

Latent rather than firing only because the network serves the 1.5B, where 28 layers is small
enough that the cap never binds. At the 7B tier an 8 GB machine was cleared for 8 layers =
7.46 GB of weights against a 3.75 GB budget.

What must hold:

  1. A node is sized against its REAL footprint, not the table's fp16 basis.
  2. The correction is applied exactly ONCE, however the caller reaches it (`max_layers_for`,
     `layer_caps`, `capacity_shortfall`, `solve`). Applying it twice would be the same class of
     bug in the opposite direction.
  3. The default is pessimistic. Over-estimating a slice costs throughput; under-estimating it
     costs a volunteer's machine.
  4. A node that genuinely runs fp16 is sized on ITS terms once it says so — the reason the fix
     lives in the balancer (dtype is a property of the node) and not in the tier table (a layer
     count is a property of the model).
  5. An unknown footprint is still no constraint at all.
"""
import sys

from coordinator import balancer, model_tiers

ok = fail = 0

GB16 = 0.466            # Qwen2.5-7B in the tier table, fp16 basis
GB32 = GB16 * 2         # what a default node actually stores
HEADROOM = 0.75


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def budget(ram_gb):
    return max(ram_gb - balancer.RAM_OS_RESERVE_GB, 0.0) * HEADROOM


def main():
    # ---------- 1. the runtime's real dtype, asserted rather than assumed ----------
    import common
    check("common.WEIGHT_DTYPE really is fp32 by default (the premise of this whole file)",
          str(common.WEIGHT_DTYPE) == "torch.float32", str(common.WEIGHT_DTYPE))
    check("...so a default node stores 4 bytes/param",
          balancer.weight_bytes_for(None) == 4.0)
    check("the tier column is on a 2 bytes/param basis", balancer.TIER_BASIS_BYTES == 2.0)

    # ---------- 2. every node fits its own budget now ----------
    for label, ram in [("volunteer", 8.0), ("pavilion", 11.0), ("optiplex", 15.0),
                       ("driver", 63.0)]:
        cap = balancer.max_layers_for({"node_id": label, "ram_gb": ram}, gb_per_layer=GB16)
        check(f"{label} ({ram:.0f} GB): {cap} layers actually fit in its budget",
              cap * GB32 <= budget(ram),
              f"{cap} layers = {cap * GB32:.2f} GB against a {budget(ram):.2f} GB budget")

    # The specific machine the 7B tier would have OOM-killed.
    v = {"node_id": "v", "ram_gb": 8.0}
    check("the 8 GB machine is cleared for 4 layers, not 8",
          balancer.max_layers_for(v, gb_per_layer=GB16) == 4,
          balancer.max_layers_for(v, gb_per_layer=GB16))

    # ---------- 3. applied exactly once, by every route ----------
    expected = balancer.max_layers_for(v, gb_per_layer=GB16)
    check("layer_caps agrees with max_layers_for (no second application)",
          balancer.layer_caps([v], GB16, 28) == [expected])
    # solve() reports the same cap it sized with.
    a = balancer.solve([dict(v, ms_per_layer=10.0)], 4, gb_per_layer=GB16)
    check("solve reports the same cap it sized with", a[0]["max_layers"] == expected, a)
    # And the correction is real: sizing with the already-doubled figure must give the same
    # answer as sizing with the table figure. If it did not, it is being applied twice.
    raw = max(int((budget(8.0)) / GB32), 1)
    check("the corrected cap equals sizing directly at the true footprint",
          expected == raw, f"{expected} vs {raw}")

    # ---------- 4. capacity_shortfall inherits it, and the fleet still holds the 7B ----------
    fleet = [{"node_id": n, "ram_gb": r}
             for n, r in [("v", 8.0), ("p", 11.0), ("o", 15.0), ("d", 63.0)]]
    check("the live fleet can still hold the 7B at the honest figure (no forced demotion)",
          balancer.capacity_shortfall(fleet, 28, GB16) == 0,
          balancer.capacity_shortfall(fleet, 28, GB16))
    check("...and the 1.5B, comfortably",
          balancer.capacity_shortfall(fleet, 28, 0.094) == 0)
    # A network that genuinely cannot hold a model must still say so.
    check("a fleet that cannot hold a model still reports a shortfall",
          balancer.capacity_shortfall([{"node_id": "tiny", "ram_gb": 4.0}], 80, 1.755) > 0)

    # ---------- 5. pessimistic by default, honest when told ----------
    fp16 = dict(v, weight_dtype="fp16")
    check("a node reporting fp16 is sized on its own terms",
          balancer.max_layers_for(fp16, gb_per_layer=GB16) == 8,
          balancer.max_layers_for(fp16, gb_per_layer=GB16))
    check("...and what it is cleared for genuinely fits at fp16",
          balancer.max_layers_for(fp16, gb_per_layer=GB16) * GB16 <= budget(8.0))
    check("an fp16 node is cleared for MORE than an fp32 one, not less",
          balancer.max_layers_for(fp16, gb_per_layer=GB16)
          > balancer.max_layers_for(v, gb_per_layer=GB16))
    for junk in ("", None, "float64", "nonsense", 7):
        check(f"an unrecognised weight_dtype ({junk!r}) falls back to the pessimistic default",
              balancer.weight_bytes_for({"weight_dtype": junk})
              == balancer.ASSUMED_WEIGHT_BYTES)

    # ---------- 6. unknown footprint is still no constraint ----------
    check("no gb_per_layer -> no constraint, as before",
          balancer.max_layers_for(v, gb_per_layer=None) is None)
    check("effective_gb_per_layer passes None through",
          balancer.effective_gb_per_layer(None) is None)
    check("...and 0 through (a tier with no measurement behind it)",
          not balancer.effective_gb_per_layer(0))

    # ---------- 7. the tier table itself ----------
    check("gb_per_layer_for still returns the raw table figure, uncorrected",
          model_tiers.gb_per_layer_for("Qwen/Qwen2.5-7B-Instruct") == GB16,
          "the correction belongs in the balancer: a layer's size is a property of the MODEL, "
          "the dtype is a property of the NODE. Doubling the table would conflate them and "
          "make a genuine fp16 node impossible to size correctly.")
    # Checked as the presence of the CORRECTION, not the absence of the old claim: the file now
    # quotes that claim in order to refute it, so an absence test matches its own refutation
    # and fails. (It did, on the first run of this file.)
    tiers_src = open(model_tiers.__file__, encoding="utf-8").read()
    check("model_tiers states the runtime default is fp32, not fp16",
          "defaults" in tiers_src and "**fp32**" in tiers_src)
    check("...and points at where the correction is applied",
          "effective_gb_per_layer" in tiers_src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
