"""
tools/capacity_dryrun.py — what would the coordinator decide, before deploying it?

`neuron_doctor.py` answers "is the live network working". This answers the question that comes
BEFORE that one: given these machines and this model, what will the coordinator do when it
comes up? Which node drives, which layers each gets, whether anyone is over its budget, and how
much each has to download.

It touches no database, starts no server and needs no VM. It calls the coordinator's OWN
functions on an in-memory roster -- `model_tiers.stage1_for`, `router.canonical_assignment`,
`router.chain_shape`, `router.assignment_overflow`, `balancer.capacity_shortfall` -- so the
answer is the one the real coordinator will give, not a second implementation that can drift
from it. That property is the whole point: a dry run that agrees with itself and disagrees with
production is worse than no dry run.

    python tools/capacity_dryrun.py
    python tools/capacity_dryrun.py --model Qwen/Qwen3-4B-Instruct-2507 --nodes pavilion:12,node-b:8
    python tools/capacity_dryrun.py --dtype fp32          # see it refused, and why
    python tools/capacity_dryrun.py --no-sizes            # skip the HuggingFace header fetch

Exit code is 0 when the model would be placed with every node inside its budget, 1 when it
would not -- so it can gate a deploy.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Before importing the coordinator: point it at a throwaway DB. Nothing here reads or writes
# one, but `config` resolves a path at import and a dry run must not go near a real ledger.
os.environ.setdefault("NEURON_DB", os.path.join(tempfile.mkdtemp(prefix="neuron_dryrun_"),
                                                "dryrun.db"))

from coordinator import balancer, config, model_tiers, router  # noqa: E402

DEFAULT_NODES = "pavilion:12,node-b:8"


def roster(spec, dtype):
    """A node list shaped like `models.list_nodes()` returns, without a database.

    `layer_start`/`layer_end` are the ranges these machines hold TODAY (the 28-layer floor);
    they matter because the router prefers to leave an incumbent driver where it is.
    """
    out, cur = [], 0
    parts = [p for p in spec.split(",") if p.strip()]
    for i, part in enumerate(parts):
        name, _, ram = part.partition(":")
        try:
            ram_gb = float(ram)
        except ValueError:
            raise SystemExit(f"--nodes: '{part}' is not name:GB")
        span = config.TOTAL_LAYERS // max(len(parts), 1)
        out.append({"node_id": name.strip(), "ram_gb": ram_gb,
                    "status": "online", "eligible": True, "ms_per_layer": 12.0,
                    "layer_start": cur,
                    "layer_end": (cur + span - 1) if i < len(parts) - 1
                    else config.TOTAL_LAYERS - 1,
                    **({"weight_dtype": dtype} if dtype else {})})
        cur += span
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--nodes", default=DEFAULT_NODES, help=f"name:GB,... (default {DEFAULT_NODES})")
    ap.add_argument("--dtype", default="fp16", choices=sorted(balancer._DTYPE_BYTES) + ["unset"],
                    help="what the nodes report storing at; 'unset' is an agent too old to say")
    ap.add_argument("--no-sizes", action="store_true", help="skip the HuggingFace header fetch")
    args = ap.parse_args()

    dtype = None if args.dtype == "unset" else args.dtype
    nodes = roster(args.nodes, dtype)

    tier = model_tiers.tier_for(args.model)
    if tier is None:
        raise SystemExit(f"'{args.model}' is not in the tier table, so the coordinator has no "
                         f"layer count or footprint for it and would ignore a pin naming it. "
                         f"Known: {[t['model_id'] for t in model_tiers.TIERS]}")

    layers = int(tier["layers"])
    s1 = model_tiers.stage1_for(args.model)
    gpl, head = tier.get("gb_per_layer"), tier.get("head_gb")

    print(f"\n  model    {args.model}")
    print(f"           {layers} layers, {gpl} GB/layer + {head} GB head (fp16 basis)")
    print(f"           tier '{tier['name']}'"
          + ("  MANUAL ONLY - the capacity ladder will never select it; pin it"
             if tier.get("manual_only") else ""))
    print(f"  stage 1  {s1} layers"
          + ("  (per-model; the global default is "
             f"{config.DRIVER_STAGE1_LAYERS})" if s1 != config.DRIVER_STAGE1_LAYERS
             else "  (the global default)"))
    print(f"  nodes    {len(nodes)}, storing "
          + (dtype or f"nothing reported -> sized at {balancer.ASSUMED_WEIGHT_BYTES:g} B/param"))

    print("\n  per-node budget")
    for n in nodes:
        budget = max(n["ram_gb"] - balancer.RAM_OS_RESERVE_GB, 0.0) * 0.75
        cap = balancer.max_layers_for(n, gpl)
        alone = balancer.capacity_shortfall([n], layers, gpl, head)
        print(f"    {n['node_id']:<14} {n['ram_gb']:>5.1f} GB -> {budget:5.2f} GB budget"
              f" -> {cap:>3} layers"
              f"   alone: {'HOLDS THE WHOLE MODEL' if not alone else f'{alone} short'}")

    short = balancer.capacity_shortfall(nodes, layers, gpl, head)
    print(f"\n  together: {sum(balancer.layer_caps(nodes, gpl, layers, head))} layer-slots "
          f"for {layers} needed"
          + ("" if not short else f"  -> {short} SHORT"))

    plan = router.canonical_assignment(nodes, layers, serving_model_id=args.model)
    if not plan:
        print("\n  VERDICT: REFUSED - no node can hold stage 1 of this model.")
        print("  That is the coordinator declining to place a plan that would OOM a machine.")
        print("  The remedy is a smaller model (the TierController demotes), more RAM, or "
              "fp16 storage\n  (NEURON_WEIGHT_DTYPE=fp16) if these nodes are not already "
              "reporting it.")
        return 1

    for a in plan:
        a["_n"] = next(n for n in nodes if n["node_id"] == a["node_id"])
    print("\n  placement")
    for a in plan:
        got = a["layer_end"] - a["layer_start"] + 1
        wt = balancer.effective_gb(gpl, a["_n"]) * got
        if a["layer_start"] == 0:
            wt += balancer.effective_gb(head, a["_n"])
        print(f"    {a['node_id']:<14} layers {a['layer_start']:>3}-{a['layer_end']:<3}"
              f" ({got:>2})   {wt:5.2f} GB resident"
              + ("   <- driver: embedding + lm_head" if a["layer_start"] == 0 else ""))

    # Applying the plan means these ARE the ranges, so validation is asked about the result.
    applied = [dict(n, layer_start=a["layer_start"], layer_end=a["layer_end"])
               for a in plan for n in nodes if n["node_id"] == a["node_id"]]
    shape = router.chain_shape(applied, layers, serving_model_id=args.model)
    over = router.assignment_overflow(nodes, plan, args.model)

    print(f"\n  routable   {shape['routable']}   stages {shape['stages']}   "
          f"stage1_ok {shape['stage1_ok']} (expects {shape['expected_stage1']})")
    if over:
        for o in over:
            print(f"  OVERFLOW   {o['node_id']} would be given {o['layers']} layers but can "
                  f"hold {o['max_layers']} - {o['over']} over")
        print("             The tail is covered anyway (a gap means not one request "
              "completes),\n             but that node is expected to be OOM-killed.")
    else:
        print("  overflow   none - every node holds what it was given")

    if not args.no_sizes:
        try:
            from coordinator import sliceinfo
            print("\n  downloads (per-tensor byte ranges off the published checkpoint)")
            total = 0.0
            for a in plan:
                i = sliceinfo.slice_info(args.model, a["layer_start"], a["layer_end"], layers)
                total += i["estimated_download_gb"]
                print(f"    {a['node_id']:<14} {i['estimated_download_gb']:>5.2f} GB")
            print(f"    {'':<14} {total:>5.2f} GB total")
        except Exception as e:
            print(f"\n  (could not size the downloads: {e} - pass --no-sizes to skip)")

    ok = shape["routable"] and not over
    print(f"\n  VERDICT: {'YES' if ok else 'PLACED, BUT A NODE IS OVER ITS BUDGET'}"
          + (" - this is what the coordinator will do once deployed." if ok else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
