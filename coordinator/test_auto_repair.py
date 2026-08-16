"""coordinator/test_auto_repair.py — the chain repairs itself, without a human.

`pin_layers.sh` being a MANUAL step was the bug. Every join and every leave could push the chain
into a shape no driver accepts — four stages, or a stage 1 of the wrong width — and the only
repair was somebody noticing and running a command. On 2026-08-10 that state sat unrepaired
overnight: machines flapped, `self_heal` re-split on each flap, and by morning the chain was
[[0,9],[10,13],[14,20],[21,27]] with the driver stranded in the middle on 14-20.

`router.canonical_assignment` is that script expressed server-side. These tests pin the two
invariants it exists to hold — stage 1 is exactly the driver's shard, and extra machines
REPLICATE rather than deepening the pipeline — plus the property that makes it safe to run every
60 seconds: it is stable, so a healthy chain is never disturbed.

Run:  python -m coordinator.test_auto_repair     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import balancer, config, main, model_tiers, models, router

models.init_db()

N = config.TOTAL_LAYERS
S1 = config.DRIVER_STAGE1_LAYERS


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def _reg(node_id, ls, le, ram=16.0):
    models.register_node(node_id, "1.1.1.1", 50999, ls, le, 8, ram, f"tok-{node_id}",
                         ms_per_layer=10, trusted=True)


def _apply(plan):
    for a in plan:
        models.update_layers(a["node_id"], a["layer_start"], a["layer_end"])


def _shape():
    return router.chain_shape(models.list_nodes(), N)


# --------------------------------------------------------------------------- #
# The two invariants
# --------------------------------------------------------------------------- #
def test_repairs_the_live_2026_08_10_four_stage_chain():
    _clear()
    _reg("bhpc-a", 0, 9, ram=8.0)
    _reg("bhpc-b", 10, 13, ram=8.0)
    _reg("driver", 14, 20, ram=68.0)
    _reg("pavilion", 21, 27, ram=12.0)
    assert _shape()["routable"] is False, "premise: four stages is unroutable"
    _apply(router.canonical_assignment(models.list_nodes(), N))
    s = _shape()
    assert s["routable"] is True, s
    assert s["ranges"][0] == [0, S1 - 1], "stage 1 must be the driver's shard"
    assert s["stages"] <= config.PIPELINE_STAGES


def test_repairs_a_stage1_of_the_wrong_width():
    """The shape a halted migration left: three stages, 28/28 covered, still refused."""
    _clear()
    _reg("driver", 0, 16, ram=68.0); _reg("mid", 17, 23); _reg("tail", 24, 27)
    assert _shape()["routable"] is False
    _apply(router.canonical_assignment(models.list_nodes(), N))
    assert _shape()["ranges"][0] == [0, S1 - 1]
    assert _shape()["routable"] is True


def test_extra_machines_replicate_instead_of_adding_a_stage():
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0)
    for i in range(6):
        _reg(f"extra{i}", 0, 27)
    plan = router.canonical_assignment(models.list_nodes(), N)
    _apply(plan)
    s = _shape()
    assert s["stages"] <= config.PIPELINE_STAGES, s
    assert s["routable"] is True
    assert len(plan) == 7, "every node is assigned, none left idle"
    distinct = {(a["layer_start"], a["layer_end"]) for a in plan}
    assert len(distinct) <= config.PIPELINE_STAGES


def test_replicas_are_spread_not_piled_on_one_stage():
    """A pipeline is only as parallel as its least-replicated stage."""
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0)
    for i in range(4):
        _reg(f"extra{i}", 0, 27)
    plan = router.canonical_assignment(models.list_nodes(), N)
    counts = {}
    for a in plan:
        if a["node_id"] == "driver":
            continue
        key = (a["layer_start"], a["layer_end"])
        counts[key] = counts.get(key, 0) + 1
    assert max(counts.values()) - min(counts.values()) <= 1, counts


# --------------------------------------------------------------------------- #
# Stability — why this is safe to run every 60 seconds
# --------------------------------------------------------------------------- #
def test_the_incumbent_driver_keeps_stage_one():
    """Moving the driver breaks chat even when the chain looks legal, so a node already holding
    stage 1 must never be displaced by a bigger machine joining."""
    _clear()
    _reg("driver", 0, S1 - 1, ram=8.0)          # small, but it IS the driver
    _reg("beefy", S1, 27, ram=256.0)
    plan = router.canonical_assignment(models.list_nodes(), N)
    stage1 = [a for a in plan if a["layer_start"] == 0][0]
    assert stage1["node_id"] == "driver"


def test_a_healthy_chain_is_left_alone():
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0); _reg("mid", S1, 18); _reg("tail", 19, 27)
    before = {n["node_id"]: n["assigned_layers"] for n in models.list_nodes()}
    assert _shape()["routable"] is True
    plan = router.canonical_assignment(models.list_nodes(), N)
    _apply(plan)
    after = {n["node_id"]: n["assigned_layers"] for n in models.list_nodes()}
    assert before == after, "a routable chain must not be reshuffled"


def test_repeated_repairs_converge():
    """Run every sweep: the second pass must be a no-op, or the network churns forever."""
    _clear()
    _reg("a", 0, 27, ram=8.0); _reg("b", 0, 27, ram=12.0); _reg("c", 0, 27, ram=68.0)
    _apply(router.canonical_assignment(models.list_nodes(), N))
    first = {n["node_id"]: n["assigned_layers"] for n in models.list_nodes()}
    _apply(router.canonical_assignment(models.list_nodes(), N))
    second = {n["node_id"]: n["assigned_layers"] for n in models.list_nodes()}
    assert first == second
    assert _shape()["routable"] is True


def test_too_few_nodes_returns_no_plan_rather_than_inventing_one():
    _clear()
    _reg("solo", 0, 27)
    assert router.canonical_assignment(models.list_nodes(), N) == []


def test_offline_and_probationary_nodes_are_not_assigned():
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0); _reg("mid", S1, 27)
    models.register_node("newcomer", "1.1.1.1", 50999, 0, 27, 8, 8, "tok-n", trusted=False)
    plan = router.canonical_assignment(models.list_nodes(), N)
    assert "newcomer" not in {a["node_id"] for a in plan}


# --------------------------------------------------------------------------- #
# Memory awareness — [P26]'s hazard, which this function reproduced
# --------------------------------------------------------------------------- #
def test_the_split_respects_what_a_node_can_actually_hold():
    """An even split is a fine default and a bad promise. On the 7B tier an even 3-way split of
    the tail hands an 8 GB office PC ~8.4 GB of weights at fp32 -- exactly [P26]."""
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0)
    _reg("small", S1, 18, ram=8.0)
    _reg("big", 19, 27, ram=64.0)
    plan = router.canonical_assignment(models.list_nodes(), N,
                                       serving_model_id="Qwen/Qwen2.5-7B-Instruct")
    by_id = {a["node_id"]: a for a in plan}
    small = by_id["small"]
    held = small["layer_end"] - small["layer_start"] + 1
    cap = balancer.max_layers_for(models.get_node("small"),
                                  model_tiers.gb_per_layer_for("Qwen/Qwen2.5-7B-Instruct"))
    assert held <= cap, f"assigned {held} layers to a node that can hold {cap}"
    # ...and the model is still fully covered: what the cap sheds spills onto the next stage
    covered = set()
    for a in plan:
        covered.update(range(a["layer_start"], a["layer_end"] + 1))
    assert covered == set(range(N))


def test_an_unknown_model_imposes_no_memory_constraint():
    """gb_per_layer of None means 'unknown footprint', which must behave as it did before tiers
    carried the figure -- not as 'zero layers fit'."""
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0); _reg("a", S1, 18, ram=8.0); _reg("b", 19, 27, ram=8.0)
    plan = router.canonical_assignment(models.list_nodes(), N, serving_model_id="who/knows")
    covered = set()
    for a in plan:
        covered.update(range(a["layer_start"], a["layer_end"] + 1))
    assert covered == set(range(N))


# --------------------------------------------------------------------------- #
# A catastrophically slow replica must not be routed to
# --------------------------------------------------------------------------- #
def test_a_wildly_slow_replica_is_never_picked():
    """Live 2026-08-10: one replica reported 4150 ms/layer against 22 ms for its peer, and
    answers came back at 0.24 tok/s. Weighted-random still hands it the occasional request, and
    a chain runs at the speed of its slowest stage."""
    fast = {"node_id": "fast", "ms_per_layer": 22.0, "layer_start": 19, "layer_end": 27}
    slow = {"node_id": "slow", "ms_per_layer": 4150.0, "layer_start": 19, "layer_end": 27}
    pick = router.fastest_pick()
    chosen = {pick([fast, slow])["node_id"] for _ in range(400)}
    assert chosen == {"fast"}, chosen


def test_a_merely_slower_replica_still_gets_traffic():
    """The guard is for orders of magnitude, not for 'somewhat slower' -- a node at half speed
    should still carry half the traffic ([P16]: added machines are for throughput)."""
    a = {"node_id": "a", "ms_per_layer": 10.0, "layer_start": 0, "layer_end": 9}
    b = {"node_id": "b", "ms_per_layer": 20.0, "layer_start": 0, "layer_end": 9}
    pick = router.fastest_pick()
    chosen = {pick([a, b])["node_id"] for _ in range(400)}
    assert chosen == {"a", "b"}, chosen


def test_an_old_measurement_is_still_used_for_routing():
    """This test asserted the opposite until 2026-08-11, when expiry reached the live coordinator
    and cost the product an order of magnitude. `node-c-pavilion` -- 4 cores, 99% reliable,
    measured at 11.0 -- was scored at the 40.0 prior because its reading was 12.9 h old, while a
    slower peer kept its fresher 20.4. That inverts the preference between them, and chat fell to
    0.05 tok/s.

    The TTL's purpose ([P34]: an outlier is excluded, so it never serves, so nothing revises it)
    assumed `agent.remeasure_loop`, which ships in **0.20**. On a 0.19 fleet there is no second
    measurement to fall back to, so expiry trades evidence for a guess. REPLICA_SLOWDOWN_LIMIT
    still drops a genuine outlier before weighting, and an agent re-measures on restart.

    Revisit when 0.20 is on every node. The display half of the TTL is unchanged and still hides
    an old figure from strangers -- see coordinator/test_stale_speed_display.py."""
    import time as _t
    fresh = {"node_id": "f", "ms_per_layer": 4150.0, "layer_start": 0, "layer_end": 9,
             "ms_per_layer_at": _t.time()}
    stale = dict(fresh, node_id="s",
                 ms_per_layer_at=_t.time() - config.MS_PER_LAYER_TTL_S - 60)
    assert router.stage_ms(fresh) == router.stage_ms(stale), (
        "age must not change what routing believes about a measured node")
    good_but_old = {"node_id": "p", "ms_per_layer": 11.0, "layer_start": 10, "layer_end": 27,
                    "ms_per_layer_at": _t.time() - config.MS_PER_LAYER_TTL_S - 60}
    fresh_but_slower = {"node_id": "q", "ms_per_layer": 20.4, "layer_start": 10, "layer_end": 27,
                        "ms_per_layer_at": _t.time()}
    assert router.stage_ms(good_but_old) < router.stage_ms(fresh_but_slower), (
        "the live inversion: a reliable node's old figure must still beat a slower fresh one")
    # The prior is for a node that has NEVER been measured — the one case with no evidence at all.
    assert router.stage_ms({"node_id": "n", "layer_start": 0, "layer_end": 9,
                            "ms_per_layer": None}) == 10 * router.DEFAULT_MS_PER_LAYER


def test_a_fresh_good_measurement_is_not_discarded():
    import time as _t
    good = {"node_id": "g", "ms_per_layer": 8.3, "layer_start": 0, "layer_end": 9,
            "ms_per_layer_at": _t.time()}
    assert abs(router.stage_ms(good) - 83.0) < 1e-6


def test_a_node_that_never_reported_an_age_is_still_believed():
    """Rows written before the column existed have no timestamp. Treating NULL as expired would
    discard every good figure on the network the moment this shipped."""
    n = {"node_id": "old", "ms_per_layer": 8.3, "layer_start": 0, "layer_end": 9,
         "ms_per_layer_at": None}
    assert abs(router.stage_ms(n) - 83.0) < 1e-6


def test_all_replicas_slow_keeps_the_segment_served():
    """If every replica is an outlier of the best, emptying the segment would break the chain
    entirely -- worse than serving it slowly."""
    a = {"node_id": "a", "ms_per_layer": 4000.0, "layer_start": 0, "layer_end": 9}
    b = {"node_id": "b", "ms_per_layer": 4100.0, "layer_start": 0, "layer_end": 9}
    pick = router.fastest_pick()
    assert pick([a, b])["node_id"] in {"a", "b"}


# --------------------------------------------------------------------------- #
# Model mismatch is detectable at all — [P33]
# --------------------------------------------------------------------------- #
def test_a_node_on_the_wrong_model_is_detected_and_is_not_healthy():
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0); _reg("mid", S1, 18); _reg("tail", 19, 27)
    models.set_reported_model("driver", "Qwen/Qwen2.5-1.5B-Instruct")
    models.set_reported_model("tail", "Qwen/Qwen2.5-7B-Instruct")     # the 2026-08-10 state
    net, _ = main._network_summary()
    assert net["routable"] is True, "premise: ranges and coverage all look fine"
    assert net["model_mismatch"] == ["tail"]
    assert net["network_healthy"] is False


def test_a_node_too_old_to_report_its_model_is_not_a_mismatch():
    _clear()
    _reg("driver", 0, S1 - 1, ram=68.0); _reg("tail", S1, 27)
    net, _ = main._network_summary()
    assert net["model_mismatch"] == []
    assert net["network_healthy"] is True


# --------------------------------------------------------------------------- #
# [P44]: the two stages this function never checked it could fill
#
# `caps` covered `rest` only, so the DRIVER -- the node that also holds the embedding and
# lm_head, the largest fixed cost in the model -- was the one machine whose capacity was never
# asked about. And the LAST stage takes the whole tail regardless of its cap, which is the right
# call (a gap means not one request completes) made silently, which is not: main.py applied the
# plan and printed `routable=True`.
# --------------------------------------------------------------------------- #
Q4 = "Qwen/Qwen3-4B-Instruct-2507"
Q4_LAYERS = 36


def _capacity_roster(dtype=None):
    """The founder's two machines, as the coordinator holds them: a 12 GB Pavilion and an
    8 GB node, with the 64 GiB OptiPlex deliberately absent."""
    _clear()
    for nid, ram, ls, le in (("pavilion", 12.0, 0, 9), ("node-b", 8.0, 10, 27)):
        models.register_node(nid, "1.1.1.1", 50999, ls, le, 8, ram, f"tok-{nid}",
                             ms_per_layer=10, trusted=True, weight_dtype=dtype)
    return models.list_nodes()


def test_the_driver_is_checked_for_stage_1_like_every_other_stage():
    """fp32 Qwen3-4B is 0.4 GB/layer plus a 1.56 GB head. An 18-layer stage 1 is 8.8 GB, which
    no machine here has. Before this, that plan was emitted anyway."""
    roster = _capacity_roster()
    assert router.canonical_assignment(roster, Q4_LAYERS, s1=18, serving_model_id=Q4) == [], \
        "no node can hold 18 layers of a 4B at fp32; there is no routable shape to return"
    # and the remedy is a smaller model, which the roster CAN hold
    assert router.canonical_assignment(roster, N, serving_model_id=config.MODEL_ID)


def test_the_same_roster_places_it_once_the_nodes_say_they_store_fp16():
    """The capacity case. 8.04 GB across two machines that hold 12 and 8 -- neither alone."""
    roster = _capacity_roster(dtype="fp16")
    plan = router.canonical_assignment(roster, Q4_LAYERS, s1=18, serving_model_id=Q4)
    got = {a["node_id"]: (a["layer_start"], a["layer_end"]) for a in plan}
    assert got == {"pavilion": (0, 17), "node-b": (18, 35)}, got
    assert router.assignment_overflow(roster, plan, Q4) == [], \
        "every node must hold what it was given, or this is not a capacity case"


def test_an_overfilled_tail_is_still_assigned_but_no_longer_silent():
    """The tail must stay covered -- a gap means not one request completes. What must stop is
    covering it without saying so. s1=10 on a 36-layer model leaves 26 layers for one node."""
    roster = _capacity_roster(dtype="fp16")
    plan = router.canonical_assignment(roster, Q4_LAYERS, s1=10, serving_model_id=Q4)
    got = {a["node_id"]: (a["layer_start"], a["layer_end"]) for a in plan}
    assert got["node-b"] == (10, 35), got            # still covered, deliberately
    over = router.assignment_overflow(roster, plan, Q4)
    assert [o["node_id"] for o in over] == ["node-b"]
    assert over[0]["layers"] == 26 and over[0]["max_layers"] == 18 and over[0]["over"] == 8


def test_an_incumbent_driver_that_cannot_hold_its_seat_loses_it():
    """The incumbent rule exists for STABILITY, which is worth a lot -- but keeping a driver
    that is OOM-killed on the first token is not stability, it is a chain that breaks every
    time it is repaired."""
    _clear()
    # the small machine is the incumbent on exactly stage 1; the big one is not
    models.register_node("small", "1.1.1.1", 50999, 0, S1 - 1, 8, 8.0, "tok-small",
                         ms_per_layer=10, trusted=True)
    models.register_node("big", "1.1.1.1", 50999, S1, 27, 8, 64.0, "tok-big",
                         ms_per_layer=10, trusted=True)
    roster = models.list_nodes()
    plan = router.canonical_assignment(roster, Q4_LAYERS, s1=14, serving_model_id=Q4)
    assert plan and plan[0]["node_id"] == "big", (
        f"stage 1 went to {plan[0]['node_id'] if plan else None}; 14 fp32 layers + head is "
        f"7.2 GB and `small` has a 3.75 GB budget")
    # ...and on the model it CAN hold, the incumbent is left exactly where it is
    keep = router.canonical_assignment(roster, N, serving_model_id=config.MODEL_ID)
    assert keep[0]["node_id"] == "small", "a driver that fits must not be moved"


def test_overflow_reports_nothing_when_the_model_footprint_is_unknown():
    """An unknown model means unknown footprint, which means no constraint -- the behaviour
    from before tiers carried a figure, and the answer for an env-injected tier."""
    roster = _capacity_roster()
    plan = router.canonical_assignment(roster, N, serving_model_id=config.MODEL_ID)
    assert router.assignment_overflow(roster, plan, "some/model-nobody-measured") == []
    assert router.assignment_overflow(roster, plan, None) == []


# --------------------------------------------------------------------------- #
# [P44] part 3: the coordinator is now the SOLE owner of stage-1 width
#
# It used to be a value two processes on two machines had to agree on, both reading NEURON_S1 at
# import. `neuron_driver` now derives it from the shard it actually downloaded, and the
# coordinator publishes it on slice-info. That removes the cross-machine coupling and leaves a
# closer one: PLACEMENT (canonical_assignment) and VALIDATION (chain_shape) are still two
# separate readers of the same constant, and if they ever disagree auto-repair re-places a chain
# on every sweep that validation then calls unroutable -- a loop that never converges, on the
# path that runs every 60 seconds.
# --------------------------------------------------------------------------- #
def test_placement_and_validation_agree_on_how_wide_stage_one_is():
    _clear()
    _reg("driver", 0, 27, ram=68.0)          # one node holding everything: unroutable
    _reg("mid", 0, 27)
    _reg("tail", 0, 27)
    plan = router.canonical_assignment(models.list_nodes(), N)
    _apply(plan)
    shape = _shape()
    assert shape["stage1_ok"], (
        f"placement produced stage 1 {shape['ranges'][0]} and validation expects "
        f"{shape['expected_stage1']} — the repair loop cannot converge")
    assert shape["routable"]
    # ...and the value both of them used is the one published to drivers, so a driver that
    # believes slice-info believes what the chain will actually say.
    assert shape["expected_stage1"] == [0, config.DRIVER_STAGE1_LAYERS - 1]


def test_a_repair_at_a_different_width_still_validates_at_that_width():
    """The capacity case runs at s1=18. Placement and validation have to move together, so
    this asserts the pair rather than the constant: chain_shape reads the config, so a plan
    built at any OTHER width must read as unroutable -- which is what stops a hand-placed
    split from being silently accepted."""
    _clear()
    _reg("a", 0, 27, ram=68.0)
    _reg("b", 0, 27)
    _apply(router.canonical_assignment(models.list_nodes(), N, s1=S1 + 4))
    shape = _shape()
    assert not shape["stage1_ok"], (
        "a split placed at a width the coordinator does not validate at must NOT read as "
        "routable, or the driver refuses every request while the dashboard says green")
    assert shape["expected_stage1"] == [0, S1 - 1]


def test_stage1_width_is_per_model_now_that_the_driver_follows_it():
    """A global constant was the wrong shape: 10 is right for 28 layers over three machines
    and wrong for 36 over two, where the second node would be handed 26 layers. It could not
    vary while the driver read NEURON_S1 at import; it can now."""
    assert model_tiers.stage1_for(config.MODEL_ID) == config.DRIVER_STAGE1_LAYERS, \
        "the serving model must be placed exactly as before this change"
    assert model_tiers.stage1_for(Q4) == 18
    assert model_tiers.stage1_for("some/model-with-no-tier") == config.DRIVER_STAGE1_LAYERS, \
        "an unknown model keeps the global default rather than becoming unplaceable"


def test_the_capacity_case_places_with_no_environment_variables_at_all():
    """The end of the coordinated restart. Two machines, a 36-layer model, and the split that
    fits — chosen because the TIER says stage 1 is 18, not because someone exported NEURON_S1
    on the coordinator and on every driver."""
    roster = _capacity_roster(dtype="fp16")
    plan = router.canonical_assignment(roster, Q4_LAYERS, serving_model_id=Q4)   # no s1= !
    _apply(plan)
    got = {a["node_id"]: (a["layer_start"], a["layer_end"]) for a in plan}
    assert got == {"pavilion": (0, 17), "node-b": (18, 35)}, got
    assert router.assignment_overflow(roster, plan, Q4) == []
    shape = router.chain_shape(models.list_nodes(), Q4_LAYERS, serving_model_id=Q4)
    assert shape["routable"] and shape["expected_stage1"] == [0, 17], shape
    # ...and a driver asking slice-info is told the same 18, so what it builds is what the
    # chain will assert. That is the whole loop closed without an environment variable.
    assert model_tiers.stage1_for(Q4) == shape["expected_stage1"][1] + 1


def test_the_capacity_case_is_still_refused_at_fp32():
    """Placing it is not the same as it fitting. 16.09 GB does not go into two machines
    holding 20 GB between them, and no per-model stage-1 width changes that."""
    roster = _capacity_roster()                       # no weight_dtype -> pessimistic fp32
    assert router.canonical_assignment(roster, Q4_LAYERS, serving_model_id=Q4) == []


def test_slice_info_publishes_the_width_a_driver_must_build_at():
    """The field that replaces the shared env var. A driver reads it, downloads a shard of
    that width, and asserts that width back at the coordinator -- so it has to be the SAME
    number placement uses, not a second opinion."""
    _clear()
    _reg("solo", 0, 9, ram=68.0)
    from coordinator import sliceinfo
    real = sliceinfo.slice_info
    sliceinfo.slice_info = lambda *a, **k: {"model_id": a[0], "layer_start": a[1],
                                            "layer_end": a[2], "total_layers": a[3]}
    try:
        info = main.slice_info("solo")
    finally:
        sliceinfo.slice_info = real
    assert info["driver_stage1_layers"] == config.DRIVER_STAGE1_LAYERS
    assert info["driver_stage1_layers"] == router.chain_shape(
        models.list_nodes(), N)["expected_stage1"][1] + 1


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
