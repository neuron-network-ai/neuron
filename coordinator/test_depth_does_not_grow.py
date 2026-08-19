"""coordinator/test_depth_does_not_grow.py — run: python -m coordinator.test_depth_does_not_grow

**The property this file exists to protect is the one that decides whether NEURON works at
scale**, and it is not a speed property — it is a shape property.

Decode is sequential: every token must pass through every stage of the chain in order. So a
chain N machines deep pays N network hops PER TOKEN. Measured on the two live machines
2026-08-19, same model, same engine:

    one machine                32.11 tok/s
    split across two            3.97 tok/s      <- 8x slower

That is not a bug to optimise away, it is what pipelining costs. Which means there are exactly
two things a new machine can be, and they are opposites:

    another STAGE   -> every user's answer gets slower, forever, with every machine that joins
    another REPLICA -> another independent chain, so more users served at once, latency flat

Only the second survives a million nodes. `canonical_assignment` caps depth at
`PIPELINE_STAGES` and spills the surplus into replicas, and that was already right. What was
wrong is that it took the most stages the roster allowed rather than the fewest the MODEL
needed: `n_stages = min(max_stages - 1, len(rest))`. On a roster where one node could hold
every remaining layer, a second node still became a second stage — the worst possible use of a
machine, buying an extra hop per token and adding no throughput at all.

The tests below pin both halves, because the fix has a failure mode of its own: minimising
depth must never RE-SPLIT a chain that already works. Moving a node's range under it is [P37],
the bookkeeping that flagged three honest machines. Growth is refused; existing shape is kept.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-depth-"), "t.db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinator import config, models, router          # noqa: E402

ok = fail = 0
N = config.TOTAL_LAYERS
S1 = config.DRIVER_STAGE1_LAYERS


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def reg(node_id, ls, le, ram=68.0):
    models.register_node(node_id, "1.1.1.1", 50999, ls, le, 8, ram, f"tok-{node_id}",
                         ms_per_layer=10, trusted=True)


def clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def stages(plan):
    """Distinct layer ranges in a plan = the DEPTH of the chain. Repeated ranges are replicas."""
    return len({(a["layer_start"], a["layer_end"]) for a in plan})


def main():
    models.init_db()

    print("\n-- a machine joining an already-sufficient roster becomes a REPLICA, not a stage")
    clear()
    reg("driver", 0, S1 - 1)
    reg("tail", S1, N - 1)                       # one node already covers everything else
    plan = router.canonical_assignment(models.list_nodes(), N)
    check("two machines that cover the model form a 2-deep chain", stages(plan) == 2,
          str(plan))

    reg("joiner", 0, 0)                          # a third machine arrives, range not yet set
    plan = router.canonical_assignment(models.list_nodes(), N)
    check("a third machine does NOT become a third stage", stages(plan) == 2,
          f"depth {stages(plan)}: every token would pay an extra hop, forever — {plan}")
    check("...it is placed as a replica instead", len(plan) == 3, str(plan))
    dupes = [a for a in plan if (a["layer_start"], a["layer_end"]) ==
             (plan[1]["layer_start"], plan[1]["layer_end"])]
    check("...sharing an existing stage's exact range, which is what a replica IS",
          len(dupes) >= 2, str(plan))

    print("\n-- and it keeps holding at roster sizes where the old code deepened every time")
    for extra in range(4, 9):
        reg(f"n{extra}", 0, 0)
        plan = router.canonical_assignment(models.list_nodes(), N)
        check(f"{extra} machines still form a {stages(plan)}-deep chain (cap "
              f"{config.PIPELINE_STAGES})",
              stages(plan) <= config.PIPELINE_STAGES and stages(plan) == 2,
              f"depth grew to {stages(plan)} at {extra} machines")
    check("every machine is still given work", len(models.list_nodes()) == len(plan), str(plan))

    print("\n-- a chain that already routes is NOT re-split to save a hop ([P37])")
    clear()
    reg("driver", 0, S1 - 1)
    reg("mid", S1, 18)
    reg("tail", 19, N - 1)                       # healthy 3-deep chain, already routable
    before = {n["node_id"]: (n["layer_start"], n["layer_end"]) for n in models.list_nodes()}
    plan = router.canonical_assignment(models.list_nodes(), N)
    after = {a["node_id"]: (a["layer_start"], a["layer_end"]) for a in plan}
    check("a working 3-deep chain is left exactly as it is", before == after,
          f"before {before} after {after} — moving a node's range under it is [P37]")

    print("\n-- the cap still holds, because a 4th STAGE is unroutable by construction")
    check("depth never exceeds PIPELINE_STAGES",
          stages(plan) <= config.PIPELINE_STAGES, str(plan))

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
