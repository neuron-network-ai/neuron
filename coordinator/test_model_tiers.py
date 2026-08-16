"""coordinator/test_model_tiers.py — capacity-driven tiering logic.

Run:  python -m coordinator.test_model_tiers      (from the repo root)
      pytest coordinator/test_model_tiers.py

Deterministic: every test supplies its own `now`, and the hysteresis knobs are pinned
below so assertions don't depend on the shipped defaults.
"""
from coordinator import model_tiers as mt

# Pin the knobs so the tests are independent of env / default drift.
mt.PROMOTE_MARGIN = 0.15
mt.PROMOTE_DWELL_S = 300.0
mt.DEMOTE_GRACE_S = 300.0


def T(name):
    """A tier BY NAME. These were addressed by list position until the `4b` capacity tier was
    inserted between `1.5b` and `7b` and silently redefined TIERS[1] and TIERS[2] under every
    assertion in this file. A tier's identity is its name; its index is an ordering detail."""
    for t in mt.TIERS:
        if t["name"] == name:
            return t
    raise AssertionError(f"no tier named {name!r} -- TIERS is "
                         f"{[x['name'] for x in mt.TIERS]}")


def snap_tier(snap, name):
    """The snapshot row for a named tier, for the same reason."""
    return next(t for t in snap["tiers"] if t["name"] == name)


def nodes(n, ram=8.0, status="online", eligible=True):
    """A fake population of n identical nodes."""
    return [{"node_id": f"n{i}", "status": status, "eligible": eligible, "ram_gb": ram}
            for i in range(n)]


# --------------------------------------------------------------------------- #
# capacity
# --------------------------------------------------------------------------- #
def test_capacity_counts_only_online_eligible():
    pop = (nodes(3, ram=8) +
           nodes(2, ram=8, status="offline") +      # offline: excluded
           nodes(2, ram=8, eligible=False))         # probationary/flagged: excluded
    cap = mt.network_capacity(pop)
    assert cap["nodes"] == 3, cap
    assert cap["total_ram_gb"] == 24.0, cap


def test_capacity_empty():
    cap = mt.network_capacity([])
    assert cap == {"nodes": 0, "total_ram_gb": 0.0}, cap


# --------------------------------------------------------------------------- #
# pure feasibility  (default tiers: 1.5b=2/6, 7b=3/20, 70b=20/180)
# --------------------------------------------------------------------------- #
def test_floor_when_tiny():
    cap = mt.network_capacity(nodes(2, ram=4))      # 2 nodes, 8 GB
    assert mt.best_feasible(cap)["name"] == "1.5b"


def test_nothing_feasible_when_below_floor():
    cap = mt.network_capacity(nodes(1, ram=4))      # 1 node < floor's 2
    assert mt.best_feasible(cap) is None
    assert mt.feasible_tier_index(cap) == -1


def test_best_feasible_picks_biggest():
    cap = mt.network_capacity(nodes(7, ram=8))      # 7 nodes, 56 GB -> 7b feasible
    assert mt.best_feasible(cap)["name"] == "7b"
    cap2 = mt.network_capacity(nodes(24, ram=8))    # 24 nodes, 192 GB -> 70b feasible
    assert mt.best_feasible(cap2)["name"] == "70b"


def test_next_tier_gap():
    # Derived from the tier table, not hardcoded. These were `6 - 3` and `40 - 24` until
    # b5b4f22 (half-precision weights) lowered 7b's thresholds to 3 nodes / 20 GB, which left
    # the assertions stale and this case failing -- unnoticed, because the suite raises on the
    # first bad assert and prints no summary line to notice it in.
    nxt = T("7b")
    have_nodes, have_ram = 2, 16.0
    cap = mt.network_capacity(nodes(have_nodes, ram=8))
    gap = mt.next_tier_gap(cap, 0)
    assert gap["name"] == nxt["name"]
    assert gap["need_nodes"] == max(0, nxt["min_nodes"] - have_nodes)
    assert gap["need_ram_gb"] == round(max(0.0, nxt["min_ram_gb"] - have_ram), 1)
    # The fixture has to sit BELOW the next tier or this asserts nothing at all.
    assert gap["need_nodes"] > 0 or gap["need_ram_gb"] > 0
    assert mt.next_tier_gap(cap, len(mt.TIERS) - 1) is None   # top tier -> no gap


# --------------------------------------------------------------------------- #
# hysteresis: promotion needs margin + dwell
# --------------------------------------------------------------------------- #
def test_no_promote_without_margin():
    c = mt.TierController(start_index=0)
    # Exactly at the next tier's minimum but NOT +15% -> never promotes. Sized from the tier
    # table: this said "6 nodes, 42 GB" against thresholds that b5b4f22 later lowered, which
    # left the fixture comfortably ABOVE the margin and the case asserting the opposite of
    # what it is named for.
    nxt = T("7b")
    n = nxt["min_nodes"]
    pop = nodes(n, ram=nxt["min_ram_gb"] / n)        # exactly min_nodes and min_ram_gb
    for t in range(0, 2000, 60):
        c.update(pop, now=t)
    assert c.active()["name"] == T("1.5b")["name"]


def test_promote_requires_sustained_dwell():
    c = mt.TierController(start_index=0)
    pop = nodes(8, ram=8)                            # 8 nodes, 64 GB -> 7b with margin
    assert c.update(pop, now=0)["name"] == "1.5b"    # candidate registered, not yet promoted
    assert c.update(pop, now=299)["name"] == "1.5b"  # still within dwell
    assert c.update(pop, now=300)["name"] == "7b"    # dwell elapsed -> promoted


def test_promote_dwell_resets_if_capacity_drops():
    c = mt.TierController(start_index=0)
    big = nodes(8, ram=8)                            # 7b-with-margin
    small = nodes(2, ram=4)                          # floor only
    c.update(big, now=0)                             # start dwell for 7b
    c.update(small, now=100)                         # capacity gone -> candidate cleared
    c.update(big, now=200)                           # dwell restarts here
    assert c.update(big, now=499)["name"] == "1.5b"  # only 299s sustained -> no promote
    assert c.update(big, now=500)["name"] == "7b"    # 300s sustained -> promote


# --------------------------------------------------------------------------- #
# hysteresis: demotion needs a sustained grace period (no flapping)
# --------------------------------------------------------------------------- #
def _below(tier):
    """A population genuinely INFEASIBLE for `tier`, asserted rather than assumed.

    The hysteresis tests below need a population that has actually fallen out of the current
    tier. They hardcoded `nodes(3, ram=8)` against a 7b that wanted 6 nodes / 40 GB; b5b4f22
    lowered it to 3 / 20, so that fixture became feasible and the demotion tests started
    asserting the opposite of their names. The assert is what stops that happening silently
    the next time a threshold moves.
    """
    pop = nodes(max(1, tier["min_nodes"] - 1), ram=8)
    assert not mt._meets(mt.network_capacity(pop), tier), (
        f"fixture is not below {tier['name']} -- this test would assert nothing")
    return pop


def _at_7b():
    c = mt.TierController(start_index=0)
    big = nodes(8, ram=8)
    c.update(big, now=0)
    c.update(big, now=300)
    assert c.active()["name"] == "7b"
    return c


def test_brief_dip_does_not_demote():
    c = _at_7b()
    small = _below(T("7b"))
    c.update(small, now=400)                         # infeasible starts at 400
    c.update(small, now=600)                         # 200s < grace -> hold
    assert c.active()["name"] == "7b"
    c.update(nodes(8, ram=8), now=650)               # recovered before grace elapsed
    assert c.active()["name"] == "7b"                # never demoted — no flap


def test_sustained_loss_demotes():
    c = _at_7b()
    small = _below(T("7b"))
    c.update(small, now=400)                         # infeasible starts
    c.update(small, now=699)                         # 299s < grace
    assert c.active()["name"] == "7b"
    c.update(small, now=700)                         # 300s >= grace -> demote
    assert c.active()["name"] == T("1.5b")["name"]


def test_demote_targets_biggest_still_servable():
    # start at 70b, collapse to a mid-size network -> should land on 8b, not the floor
    c = mt.TierController(start_index=2)             # 70b
    mid = nodes(8, ram=8)                            # 8b-feasible, 70b-not
    c.update(mid, now=0)                             # infeasible(70b) starts
    c.update(mid, now=300)                           # grace elapsed -> demote
    assert c.active()["name"] == "7b"


# --------------------------------------------------------------------------- #
# snapshot (endpoint payload)
# --------------------------------------------------------------------------- #
def test_snapshot_shape():
    c = mt.TierController(start_index=0)
    # Sized BELOW the 7b tier so "not feasible" is actually true: this used 3 nodes / 24 GB,
    # which cleared 7b outright once b5b4f22 lowered it to 3 nodes / 20 GB.
    snap = mt.snapshot(nodes(2, ram=8), c, now=0)     # 2 nodes, 16 GB
    assert snap["active_tier"] == T("1.5b")["name"]
    assert snap["active_model"] == T("1.5b")["model_id"]
    assert snap["capacity"]["nodes"] == 2
    assert snap["next_tier"]["name"] == T("7b")["name"]
    assert [t["name"] for t in snap["tiers"]] == [t["name"] for t in mt.TIERS]
    assert snap_tier(snap, "1.5b")["feasible"] is True
    assert snap_tier(snap, "7b")["feasible"] is False


# --------------------------------------------------------------------------- #
# per-node placement: aggregate RAM is not the same question as "does a slice fit"
#
# 2026-08-07, live: the network promoted itself to Qwen2.5-7B on "3 nodes · 20 GB" and the even
# split handed 9-10 layers to machines with 8 GB of total RAM. Every number in this section is
# derived from the tier table so a threshold change cannot leave a case asserting nothing --
# the same trap `_below` and `test_next_tier_gap` above were written for.
# --------------------------------------------------------------------------- #
def mixed(rams, status="online", eligible=True):
    """A population of nodes with the given TOTAL RAM.

    RAM figures here are what a real node needs, i.e. against the fp32 footprint
    `balancer.effective_gb_per_layer` charges -- twice the tier table's fp16 column. Several
    fixtures below grew when that correction landed; the numbers changed, the gating logic
    under test did not.
    """
    return [{"node_id": f"m{i}", "status": status, "eligible": eligible, "ram_gb": r}
            for i, r in enumerate(rams)]


def test_gb_per_layer_for_known_and_unknown_models():
    assert mt.gb_per_layer_for(T("7b")["model_id"]) == T("7b")["gb_per_layer"]
    assert mt.gb_per_layer_for("someone/not-a-tier") is None      # -> no memory constraint


def test_partition_shortfall_is_a_different_question_from_aggregate_ram():
    """Six 5 GB machines clear the 7b tier's aggregate bar twice over and cannot host it."""
    tier = T("7b")
    pop = mixed([5.0] * 6)                                        # 6 nodes, 30 GB
    assert mt._meets(mt.network_capacity(pop), tier), "fixture must PASS the aggregate gate"
    assert mt.partition_shortfall(pop, tier) > 0                  # ...and fail the real one
    assert mt.placeable(pop, tier) is False
    assert mt.placeable(pop, T("1.5b")) is True                 # the floor still fits


def test_partition_shortfall_ignores_offline_and_ineligible_nodes():
    tier = T("7b")
    big = mixed([64.0], status="offline") + mixed([64.0], eligible=False)
    assert mt.partition_shortfall(mixed([4.0] * 2) + big, tier) > 0
    assert mt.placeable(mixed([4.0] * 2) + mixed([64.0]), tier) is True


def test_a_tier_with_no_measured_footprint_is_always_placeable():
    """NEURON_MODEL_TIERS can inject a tier with no gb_per_layer. An unknown footprint is not
    grounds to refuse to serve -- it is the same "no constraint" the code had before it knew
    anything about memory."""
    unknown = dict(T("7b")); unknown.pop("gb_per_layer")
    assert mt.placeable(mixed([2.0] * 2), unknown) is True


def test_no_promote_when_no_node_can_hold_a_slice():
    """The gate this section exists for. Sustained far past the dwell, so the ONLY thing
    holding the promotion back is that the model cannot be laid out."""
    c = mt.TierController(start_index=0)
    tier = T("7b")
    pop = mixed([5.0] * 6)
    cap = mt.network_capacity(pop)
    assert mt._meets(cap, tier, mt.PROMOTE_MARGIN), "fixture must clear the margin, or this " \
                                                    "test passes for the wrong reason"
    for t in range(0, 2000, 60):
        c.update(pop, now=t)
    assert c.active()["name"] == T("1.5b")["name"]
    # and it is reported as such rather than silently withheld
    snap = mt.snapshot(pop, c)
    assert snap_tier(snap, "7b")["feasible"] is True
    assert snap_tier(snap, "7b")["placeable"] is False


def test_promote_still_happens_when_the_partition_fits():
    """Control for the case above: same tier, machines that can actually hold a slice.

    Was [8, 8, 12, 16] GB, which holds 4+4+7+10 = 25 of the 28 layers a 7B needs at the
    real fp32 footprint -- so the "partition fits" control did not fit. Sized to machines
    that genuinely do."""
    c = mt.TierController(start_index=0)
    pop = mixed([13.0, 13.0, 21.0, 27.0])
    assert mt.placeable(pop, T("7b")) is True
    c.update(pop, now=0)
    assert c.update(pop, now=mt.PROMOTE_DWELL_S)["name"] == T("7b")["name"]


def test_demotes_when_the_shape_stops_fitting_even_though_the_ram_is_there():
    """A network can keep its aggregate RAM and lose the ability to place a slice -- the big
    machine leaves, several small ones join. Demotion is what fixes that, by pointing everyone
    at a model they can hold."""
    c = _at_7b()
    reshaped = mixed([5.0] * 6)
    assert mt._meets(mt.network_capacity(reshaped), T("7b")), \
        "fixture must still PASS the aggregate gate, or this tests the old path"
    c.update(reshaped, now=400)                       # unplaceable from here
    c.update(reshaped, now=699)                       # 299s < grace -> hold, no flapping
    assert c.active()["name"] == T("7b")["name"]
    c.update(reshaped, now=700)                       # 300s >= grace -> demote
    assert c.active()["name"] == T("1.5b")["name"]


def test_demotion_lands_on_a_tier_that_can_actually_be_placed():
    """Thirty 8 GB machines can AFFORD the 72B outright -- 240 GB against a 180 GB bar -- and
    cannot hold 80 layers of it between them (2 layers each, 60 of 80). Demote to the biggest
    tier that fits on real machines, not the biggest the aggregate allows."""
    c = mt.TierController(start_index=2)              # 70b
    pop = mixed([8.0] * 30)
    assert mt._meets(mt.network_capacity(pop), T("70b")), "fixture must afford the top tier"
    assert mt.placeable(pop, T("70b")) is False
    c.update(pop, now=0)                              # unplaceable -> grace clock starts
    c.update(pop, now=mt.DEMOTE_GRACE_S)
    assert c.active()["name"] == T("7b")["name"]
    assert mt.placeable(pop, c.active()) is True


def test_falls_back_to_the_floor_when_nothing_is_placeable():
    """Documented edge: there is nowhere below the floor to go, so the floor is where a network
    that can hold nothing lands. Serving badly beats declaring the network non-existent -- and
    the migration controller refuses the move separately (test_migration.py)."""
    c = _at_7b()
    tiny = mixed([3.5] * 4)                           # can barely hold anything at all
    assert not any(mt.placeable(tiny, t) for t in mt.TIERS)
    c.update(tiny, now=400)
    c.update(tiny, now=400 + mt.DEMOTE_GRACE_S)
    assert c.active()["name"] == T("1.5b")["name"]


def test_snapshot_separates_feasible_from_placeable():
    c = mt.TierController(start_index=0)
    pop = mixed([13.0, 13.0, 21.0, 27.0])
    snap = mt.snapshot(pop, c, now=0)
    assert snap_tier(snap, "1.5b")["placeable"] is True
    assert snap_tier(snap, "7b")["placeable"] is True
    assert snap_tier(snap, "70b")["placeable"] is False   # 80 layers of a 72B, four machines
    assert all(("feasible" in t and "placeable" in t) for t in snap["tiers"])


# --------------------------------------------------------------------------- #
# manual_only: a tier the ladder can see and must never climb to
#
# The 4b capacity tier is an EXPERIMENT, and the live 3-node network would take it: min_nodes 2
# clears the 15% promote margin at 3 nodes, and the 68 GB machine makes it placeable even at
# fp32. Without this flag, adding the row migrates the production network onto a 4B model on
# the next health sweep -- the 2026-08-07 auto-promotion, again, from a different direction.
# --------------------------------------------------------------------------- #
def test_the_capacity_tier_exists_and_is_manual_only():
    t = T("4b")
    assert t["manual_only"] is True
    assert t["layers"] == 36 and t["gb_per_layer"] and t["head_gb"], \
        "a manual tier still has to be sized, or pinning it plans against nothing"


def test_a_manual_tier_is_never_promoted_to_however_big_the_network_gets():
    for count, ram in ((3, 30.0), (10, 60.0), (40, 200.0)):
        pop = nodes(count, ram=ram)
        c = mt.TierController(start_index=0)
        for t in range(0, 4000, 60):
            c.update(pop, now=t)
        assert c.active()["name"] != "4b", f"{count} nodes x {ram} GB promoted onto a manual tier"
        assert mt.best_feasible(mt.network_capacity(pop), pop)["name"] != "4b"


def test_a_manual_tier_is_never_the_answer_a_demotion_falls_back_to():
    """Demotion picks the biggest tier still servable. A manual tier must be invisible there
    too, or a network that loses a machine LANDS on the experiment."""
    c = _at_7b()
    reshaped = mixed([5.0] * 6)          # aggregate holds, no single node can place a 7B slice
    for t in range(400, 1400, 60):
        c.update(reshaped, now=t)
    assert c.active()["name"] == "1.5b"


def test_a_manual_tier_is_not_advertised_as_the_next_rung():
    """The growth prompt says "you are N nodes away from X". X must be reachable by growing."""
    c = mt.TierController(start_index=0)
    snap = mt.snapshot(nodes(2, ram=8), c, now=0)
    assert snap["next_tier"]["name"] == "7b", \
        "next_tier must skip past the manual tier, not promise a model growth cannot unlock"


def test_a_manual_tier_is_still_visible_and_still_pinnable():
    """Invisible to the LADDER, not to the operator: `tier_for` is what an operator pin
    resolves through (main.py), and a pin that resolves to None is silently ignored."""
    t = mt.tier_for("Qwen/Qwen3-4B-Instruct-2507")
    assert t is not None and t["name"] == "4b"
    assert mt.gb_per_layer_for(t["model_id"]) == t["gb_per_layer"]
    assert mt.head_gb_for(t["model_id"]) == t["head_gb"]
    snap = mt.snapshot(nodes(3, ram=30.0), mt.TierController(start_index=0), now=0)
    row = snap_tier(snap, "4b")
    assert row["manual_only"] is True, \
        "a feasible, placeable, unadopted tier needs to say why on the dashboard"


def test_the_capacity_case_is_the_one_that_was_measured():
    """The claim the tier exists to make: neither of these two machines can hold Qwen3-4B, and
    together they can -- at fp16 storage, and NOT at the fp32 the network runs today."""
    from coordinator import balancer
    t = T("4b")
    pav = {"node_id": "pavilion", "ram_gb": 12.0, "status": "online", "eligible": True}
    small = {"node_id": "node-b", "ram_gb": 8.0, "status": "online", "eligible": True}

    assert mt.placeable([pav, small], t) is False, "fp32 does not fit, and must not claim to"
    fp16 = [dict(pav, weight_dtype="fp16"), dict(small, weight_dtype="fp16")]
    assert mt.placeable(fp16, t) is True
    for one in fp16:
        assert mt.placeable([one], t) is False, f"{one['node_id']} alone must not hold it"
    # and the 68 GB machine is exactly what makes it NOT a capacity case
    assert mt.placeable([dict(pav, node_id="optiplex", ram_gb=68.0)], t) is True
    assert balancer.capacity_shortfall(fp16, t["layers"], t["gb_per_layer"], t["head_gb"]) == 0


# --------------------------------------------------------------------------- #
def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
