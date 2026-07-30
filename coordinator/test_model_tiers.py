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
    cap = mt.network_capacity(nodes(3, ram=8))      # 3 nodes, 24 GB, at floor (idx 0)
    gap = mt.next_tier_gap(cap, 0)
    assert gap["name"] == "7b"
    assert gap["need_nodes"] == 3        # 6 - 3
    assert gap["need_ram_gb"] == 16.0    # 40 - 24
    assert mt.next_tier_gap(cap, len(mt.TIERS) - 1) is None   # top tier -> no gap


# --------------------------------------------------------------------------- #
# hysteresis: promotion needs margin + dwell
# --------------------------------------------------------------------------- #
def test_no_promote_without_margin():
    c = mt.TierController(start_index=0)
    # exactly at the 7b minimum (6 nodes, 42 GB) but NOT +15% -> never promotes
    pop = nodes(6, ram=7)                            # 6 nodes, 42 GB
    for t in range(0, 2000, 60):
        c.update(pop, now=t)
    assert c.active()["name"] == "1.5b"


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
def _at_7b():
    c = mt.TierController(start_index=0)
    big = nodes(8, ram=8)
    c.update(big, now=0)
    c.update(big, now=300)
    assert c.active()["name"] == "7b"
    return c


def test_brief_dip_does_not_demote():
    c = _at_7b()
    small = nodes(3, ram=8)                          # below 8b (needs 6 nodes)
    c.update(small, now=400)                         # infeasible starts at 400
    c.update(small, now=600)                         # 200s < grace -> hold
    assert c.active()["name"] == "7b"
    c.update(nodes(8, ram=8), now=650)               # recovered before grace elapsed
    assert c.active()["name"] == "7b"                # never demoted — no flap


def test_sustained_loss_demotes():
    c = _at_7b()
    small = nodes(3, ram=8)                          # floor-feasible, 8b-infeasible
    c.update(small, now=400)                         # infeasible starts
    c.update(small, now=699)                         # 299s < grace
    assert c.active()["name"] == "7b"
    c.update(small, now=700)                         # 300s >= grace -> demote
    assert c.active()["name"] == "1.5b"


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
    snap = mt.snapshot(nodes(3, ram=8), c, now=0)
    assert snap["active_tier"] == "1.5b"
    assert snap["active_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert snap["capacity"]["nodes"] == 3
    assert snap["next_tier"]["name"] == "7b"
    assert [t["name"] for t in snap["tiers"]] == ["1.5b", "7b", "70b"]
    assert snap["tiers"][0]["feasible"] is True
    assert snap["tiers"][1]["feasible"] is False


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
