"""coordinator/test_migration.py — rolling migration state machine (Build 3).

Run:  python -m coordinator.test_migration      (from repo root)
"""
from coordinator import migration as mig
from coordinator import router


def N(node_id, ls=0, head=False, status="online", eligible=True):
    return {"node_id": node_id, "layer_start": ls, "status": status,
            "eligible": eligible, "head_ms": 38 if head else 0}


def _trio():
    return [N("a", 0, head=True), N("c", 10), N("b", 19)]


def NL(node_id, ls, le, head=False, status="online", eligible=True):
    """Like N(), but with layer_end too -- self-heal's gap detection (router.py) needs it,
    unlike plan_migration()'s own callers above which never touch layer_end."""
    return {"node_id": node_id, "layer_start": ls, "layer_end": le, "status": status,
           "eligible": eligible, "head_ms": 38 if head else 0}


def _serving():
    s = {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 28}

    def apply(mid, layers):
        s["model_id"] = mid
        s["layers"] = int(layers)
    return s, apply


# --------------------------------------------------------------------------- #
# partition
# --------------------------------------------------------------------------- #
def test_plan_even_split():
    plan = mig.plan_migration(_trio(), 30)          # 30/3 = 10 each, driver first
    assert plan == [
        {"node_id": "a", "layer_start": 0, "layer_end": 9},
        {"node_id": "c", "layer_start": 10, "layer_end": 19},
        {"node_id": "b", "layer_start": 20, "layer_end": 29},
    ]


def test_plan_remainder_goes_to_first_nodes():
    plan = mig.plan_migration(_trio(), 32)          # 10,10,10 + 2 -> 11,11,10
    assert [p["layer_end"] for p in plan] == [10, 21, 31]


def test_plan_never_makes_more_stages_than_the_driver_can_route():
    """Live 2026-08-07: a migration split across however many nodes were eligible, cut over onto
    a ONE-stage plan (one node holding every layer), and the dashboard read 28/28 healthy while
    every chat died on `expected a 3-node chain, got 1`. The pipeline is three programs with
    three roles; the planner has to produce that shape."""
    nodes = [NL("optinovate", 0, 27, head=True), NL("b1", 0, 13), NL("b2", 0, 13),
             NL("p", 14, 20)]
    plan = mig.plan_migration(nodes, 28)
    stages = {(a["layer_start"], a["layer_end"]) for a in plan}
    assert len(stages) == 3                          # what the driver can actually route
    covered = set()
    for lo, hi in stages:
        covered |= set(range(lo, hi + 1))
    assert covered == set(range(28))                 # and still the whole model
    assert len(plan) == 4                            # nobody is left idle...
    assert len({a["node_id"] for a in plan}) == 4    # ...and nobody is assigned twice


def test_plan_extra_nodes_become_replicas_of_the_thinnest_stage():
    """Extra machines are what turns into throughput ([P16]); they must land on the stage with
    the fewest copies, not all pile onto one."""
    nodes = [NL("a", 0, 9, head=True)] + [NL(f"n{i}", 10, 18) for i in range(1, 6)]
    plan = mig.plan_migration(nodes, 30)             # 6 nodes, 3 stages -> 3 extras
    from collections import Counter
    depth = Counter((a["layer_start"], a["layer_end"]) for a in plan)
    assert len(depth) == 3
    assert sorted(depth.values()) == [2, 2, 2]       # evenly replicated, none starved
    assert len(plan) == 6                            # every machine has work


def test_plan_stage_cap_is_overridable_for_callers_that_know_better():
    nodes = [NL(f"n{i}", 0, 9) for i in range(5)]
    assert len({(a["layer_start"], a["layer_end"])
                for a in mig.plan_migration(nodes, 20, max_stages=5)}) == 5


def test_plan_excludes_offline_and_ineligible():
    ns = [N("a", 0, head=True), N("x", 10, status="offline"),
          N("y", 10, eligible=False), N("c", 10)]
    plan = mig.plan_migration(ns, 20)               # only a,c usable
    assert [p["node_id"] for p in plan] == ["a", "c"]


# --------------------------------------------------------------------------- #
# state machine
# --------------------------------------------------------------------------- #
def test_steady_when_target_equals_serving():
    s, apply = _serving()
    c = mig.MigrationController()
    st = c.update(_trio(), {"model_id": s["model_id"], "layers": 28}, s, 0, apply)
    assert st["phase"] == "steady"


def test_preparing_starts_without_flipping_serving():
    s, apply = _serving()
    c = mig.MigrationController()
    st = c.update(_trio(), {"model_id": "meta/8b", "layers": 32}, s, 0, apply)
    assert st["phase"] == "preparing" and st["plan_size"] == 3 and st["ready_count"] == 0
    assert s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"   # serving NOT flipped yet


def test_partial_ready_holds():
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "meta/8b", "layers": 32}
    c.update(_trio(), tgt, s, 0, apply)
    c.mark_ready("a"); c.mark_ready("c")            # b missing
    st = c.update(_trio(), tgt, s, 1, apply)
    assert st["phase"] == "preparing" and s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_cutover_when_all_ready():
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "meta/8b", "layers": 32}
    c.update(_trio(), tgt, s, 0, apply)
    for nid in ("a", "c", "b"):
        assert c.mark_ready(nid)
    assert s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"    # not until update runs
    st = c.update(_trio(), tgt, s, 1, apply)
    assert s["model_id"] == "meta/8b" and s["layers"] == 32 and st["phase"] == "steady"


def test_abort_when_target_reverts():
    s, apply = _serving()
    c = mig.MigrationController()
    c.update(_trio(), {"model_id": "meta/8b", "layers": 32}, s, 0, apply)
    c.mark_ready("a")
    # capacity dropped: target reverts to the serving model -> abort, serving unchanged
    st = c.update(_trio(), {"model_id": s["model_id"], "layers": 28}, s, 1, apply)
    assert st["phase"] == "steady" and s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_replan_when_target_changes():
    s, apply = _serving()
    c = mig.MigrationController()
    c.update(_trio(), {"model_id": "meta/8b", "layers": 32}, s, 0, apply)
    c.mark_ready("a")
    st = c.update(_trio(), {"model_id": "meta/70b", "layers": 80}, s, 1, apply)
    assert st["phase"] == "preparing" and st["target"]["model_id"] == "meta/70b"
    assert st["ready_count"] == 0                    # ready reset on retarget


def test_mark_ready_guards():
    s, apply = _serving()
    c = mig.MigrationController()
    assert c.mark_ready("a") is False                # steady
    c.update(_trio(), {"model_id": "meta/8b", "layers": 32}, s, 0, apply)
    assert c.mark_ready("a") is True                 # preparing + in plan
    assert c.mark_ready("zzz") is False              # not in plan


def test_replan_when_a_planned_node_drops_offline():
    """Post-audit fix: without this, a planned node going offline mid-preparing (a real
    stranger's laptop pausing under the idle donation mode is a plausible trigger) would wedge
    cutover forever — `planned <= ready` can never become true again for a node that will never
    report ready. The controller must replan against whoever is CURRENTLY eligible instead."""
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "meta/8b", "layers": 30}
    c.update(_trio(), tgt, s, 0, apply)              # plan: a 0-9, c 10-19, b 20-29
    c.mark_ready("a"); c.mark_ready("c")
    assert c.status()["plan_size"] == 3 and c.status()["ready_count"] == 2

    # b drops offline before ever reporting ready
    nodes_b_offline = [N("a", 0, head=True), N("c", 10), N("b", 19, status="offline")]
    st = c.update(nodes_b_offline, tgt, s, 1, apply)
    assert st["phase"] == "preparing"                 # still migrating, not stuck/aborted
    assert st["plan_size"] == 2                        # replanned over a,c only
    assert {p["node_id"] for p in st["plan"]} == {"a", "c"}
    assert st["ready_count"] == 0                       # a/c must re-report against new ranges
    assert s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"  # never flipped mid-wedge

    # a,c re-report ready against their NEW assignment -> cutover completes without b
    c.mark_ready("a"); c.mark_ready("c")
    st2 = c.update(nodes_b_offline, tgt, s, 2, apply)
    assert st2["phase"] == "steady"
    assert s["model_id"] == "meta/8b" and s["layers"] == 30


def test_no_replan_when_nothing_changed():
    """A plain tick (nobody dropped, target unchanged) must NOT reset ready progress —
    only an actual node loss or target change should trigger a replan."""
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "meta/8b", "layers": 30}
    c.update(_trio(), tgt, s, 0, apply)
    c.mark_ready("a")
    st = c.update(_trio(), tgt, s, 1, apply)           # same nodes, same target
    assert st["ready_count"] == 1                       # progress preserved


def test_assignment_for():
    s, apply = _serving()
    c = mig.MigrationController()
    assert c.assignment_for("a") is None             # steady
    c.update(_trio(), {"model_id": "meta/8b", "layers": 30}, s, 0, apply)
    asg = c.assignment_for("a")
    assert asg["migrating"] and asg["model_id"] == "meta/8b"
    assert asg["layer_start"] == 0 and asg["layer_end"] == 9
    assert asg["total_layers"] == 30                # a node needs this to know is_last_node


# --------------------------------------------------------------------------- #
# self-heal: closing a coverage gap with no existing replica (surplus reassignment)
# --------------------------------------------------------------------------- #
def _trio28():
    return [NL("a", 0, 9, head=True), NL("c", 10, 18), NL("b", 19, 27)]


def _serving28():
    return {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 28}


def _heal_apply():
    calls = []
    return calls, (lambda assignments: calls.append(assignments))


def test_plan_migration_supports_start_offset():
    surplus = [NL("d", 99, 99)]
    plan = mig.plan_migration(surplus, 9, start=10)
    assert plan == [{"node_id": "d", "layer_start": 10, "layer_end": 18}]
    assert mig.plan_migration(surplus, 9)[0]["layer_start"] == 0   # default start=0 unaffected


def test_covering_and_missing_detects_gap_and_covering_nodes():
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"), NL("b", 19, 27)]
    missing, covering = router.covering_and_missing(nodes, 28)
    assert missing == [(10, 18)]
    assert covering == {"a", "b"}


def test_self_heal_noop_when_no_gap():
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    st = c.self_heal(_trio28(), _serving28(), 0, apply)
    assert st["healing"] is False and calls == []


def test_self_heal_assigns_true_surplus_node_to_gap():
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]        # d is genuinely idle -- not covering anything
    st = c.self_heal(nodes, _serving28(), 0, apply)
    assert st["healing"] is True
    assert st["plan"] == [{"node_id": "d", "layers": [10, 18], "ready": False}]
    assert calls == []                                # not cut over yet -- d hasn't reported ready


def test_self_heal_cutover_persists_layers_when_ready():
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    c.self_heal(nodes, _serving28(), 0, apply)
    assert c.mark_ready("d") is True
    st = c.self_heal(nodes, _serving28(), 1, apply)
    assert st["healing"] is False                     # cleared after cutover
    assert calls == [[{"node_id": "d", "layer_start": 10, "layer_end": 18}]]


def test_self_heal_resplits_when_no_surplus_available():
    """[R6]: the survivors of a shrunken network hold their old ranges, so the departed node's
    layers belong to nobody and there is no idle node to hand them to. Re-split across whoever
    is left rather than sitting on a chain that cannot serve a single request."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"), NL("b", 19, 27)]
    st = c.self_heal(nodes, _serving28(), 0, apply)
    assert st["healing"] is True and st["mode"] == "resplit"
    # a and b split all 28 layers -- contiguous, complete, driver first
    assert [p["layers"] for p in st["plan"]] == [[0, 13], [14, 27]]
    assert calls == []                                # not until both report ready

    st2 = c.self_heal(nodes, _serving28(), 1, apply)  # identical tick -> no replan, no churn
    assert st2["plan"] == st["plan"] and calls == []


def test_self_heal_resplit_cuts_over_when_every_node_is_ready():
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"), NL("b", 19, 27)]
    c.self_heal(nodes, _serving28(), 0, apply)
    assert c.mark_ready("a") is True and c.mark_ready("b") is True
    st = c.self_heal(nodes, _serving28(), 1, apply)
    assert st["healing"] is False and st["mode"] is None
    assert calls == [[{"node_id": "a", "layer_start": 0, "layer_end": 13},
                      {"node_id": "b", "layer_start": 14, "layer_end": 27}]]


def test_self_heal_noop_when_nothing_is_eligible():
    """Nobody left to re-split across: report no heal rather than raise or invent a plan."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True, status="offline"), NL("b", 19, 27, eligible=False)]
    st1 = c.self_heal(nodes, _serving28(), 0, apply)
    st2 = c.self_heal(nodes, _serving28(), 1, apply)  # idempotent -- no crash, no assignment
    assert st1["healing"] is False and st2["healing"] is False and calls == []


def test_self_heal_surplus_path_never_moves_a_sole_cover():
    """The surplus path may take idle nodes and redundant replicas, and nothing else.

    b and e are tied replicas of 19-27, so exactly one of them is movable. `a` is the only node
    holding 0-9 -- taking it would close the 10-18 gap by opening a 0-9 one, which is not a heal.
    """
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("e", 19, 27)]
    st = c.self_heal(nodes, _serving28(), 0, apply)
    moved = {p["node_id"] for p in st["plan"]}
    assert st["mode"] == "surplus"
    assert len(moved & {"b", "e"}) == 1                # one of the pair, never both
    assert "a" not in moved                            # never the sole cover of 0-9
    assert calls == []                                 # nothing cut over yet


def test_self_heal_moves_a_redundant_replica_instead_of_resplitting():
    """The live 2026-08-07 layout: three machines on 0-13, one on 14-20, 21-27 empty.

    No node is idle, so this used to fall straight through to a full re-split that moved all
    four -- including the two serving correctly -- and a four-node plan only cuts over when the
    slowest of four is ready. Two of the three on 0-13 are pure duplicates: moving them closes
    the gap and leaves the working stages untouched.
    """
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 13, head=True), NL("b1", 0, 13), NL("b2", 0, 13), NL("p", 14, 20)]
    st = c.self_heal(nodes, _serving28(), 0, apply)
    assert st["mode"] == "surplus"
    moved = {p["node_id"] for p in st["plan"]}
    assert moved == {"b1", "b2"}                       # the duplicates, never a sole cover
    assert "a" not in moved and "p" not in moved       # working stages left alone
    covered = set()
    for p in st["plan"]:
        covered |= set(range(p["layers"][0], p["layers"][1] + 1))
    assert covered == set(range(21, 28))               # and between them they close the gap


def test_self_heal_keeps_one_copy_of_a_replicated_segment():
    """A replicated segment may give up its spares, never its last node."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    # b1/b2 both hold 19-27; one is spare, one must stay or 19-27 goes dark closing 10-18.
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b1", 19, 27), NL("b2", 19, 27)]
    st = c.self_heal(nodes, _serving28(), 0, apply)
    assert st["mode"] == "surplus"
    assert [p["node_id"] for p in st["plan"]] == ["b2"]     # exactly one of the pair
    assert st["plan"][0]["layers"] == [10, 18]


def test_self_heal_replica_choice_is_stable_across_ticks():
    """Which replica is kept must not flip between ticks -- a plan that churns discards every
    node's partial download and never converges."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 13, head=True), NL("b1", 0, 13), NL("b2", 0, 13), NL("p", 14, 20)]
    def assignment(plan):
        return [(p["node_id"], p["layers"]) for p in plan]   # `ready` legitimately changes

    first = c.self_heal(nodes, _serving28(), 0, apply)["plan"]
    c.mark_ready(first[0]["node_id"])
    again = c.self_heal(list(reversed(nodes)), _serving28(), 1, apply)   # same set, new order
    assert assignment(again["plan"]) == assignment(first)   # same node -> same range
    assert again["ready_count"] == 1                        # progress preserved, not reset
    assert calls == []                                      # and no premature cutover


def test_self_heal_prefers_surplus_over_resplit():
    """An idle node exists, so the working segments must be left exactly where they are."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    st = c.self_heal(nodes, _serving28(), 0, apply)
    assert st["mode"] == "surplus"
    assert st["plan"] == [{"node_id": "d", "layers": [10, 18], "ready": False}]


def test_self_heal_replans_when_the_same_node_moves_to_a_different_range():
    """Change detection is on the ASSIGNMENTS, not the node ids: the same node re-planned onto
    a different range is a new plan, and any ready progress against the old one is void."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    c.self_heal(nodes, _serving28(), 0, apply)
    assert c.heal_status()["plan"] == [{"node_id": "d", "layers": [10, 18], "ready": False}]
    c.mark_ready("d")

    # c comes back and b leaves: the gap d must fill is now 19-27, not 10-18.
    moved = [NL("a", 0, 9, head=True), NL("c", 10, 18),
            NL("b", 19, 27, status="offline"), NL("d", 99, 99)]
    st = c.self_heal(moved, _serving28(), 1, apply)
    assert st["plan"] == [{"node_id": "d", "layers": [19, 27], "ready": False}]
    assert st["ready_count"] == 0                     # stale readiness discarded
    assert calls == []                                # and no cutover onto the old range


def test_self_heal_does_not_run_while_preparing():
    c = mig.MigrationController()
    s = _serving28()
    c.update(_trio28(), {"model_id": "meta/8b", "layers": 32}, s, 0, lambda *a: None)
    assert c.phase == "preparing"
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    st = c.self_heal(nodes, s, 1, apply)
    assert st["healing"] is False and calls == []      # a real migration always wins


def test_self_heal_clears_stale_plan_when_real_migration_starts():
    c = mig.MigrationController()
    s = _serving28()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    c.self_heal(nodes, s, 0, apply)
    assert c.heal_status()["healing"] is True
    c.update(nodes, {"model_id": "meta/8b", "layers": 32}, s, 1, lambda *a: None)
    assert c.heal_status()["healing"] is False
    assert c.heal_plan == [] and c.heal_ready == set() and c.heal_target is None


def test_self_heal_replans_when_surplus_node_drops_offline():
    c = mig.MigrationController()
    s = _serving28()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    c.self_heal(nodes, s, 0, apply)
    assert c.heal_status()["plan"][0]["node_id"] == "d"

    # d drops offline before ever reporting ready; e shows up as a fresh surplus candidate
    nodes2 = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
             NL("b", 19, 27), NL("d", 99, 99, status="offline"), NL("e", 88, 88)]
    st = c.self_heal(nodes2, s, 1, apply)
    assert st["plan"] == [{"node_id": "e", "layers": [10, 18], "ready": False}]
    assert st["ready_count"] == 0


def test_no_replan_when_healing_progress_unchanged():
    """Mirrors test_no_replan_when_nothing_changed for the tier-migration state machine: a
    plain repeated tick with nothing actually different must not reset ready progress. Uses a
    2-node gap so marking ONE surplus node ready leaves the plan partially ready (not an
    immediate cutover), matching the original test's spirit of observing progress preserved
    across an unchanged tick."""
    c = mig.MigrationController()
    s = _serving28()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18), NL("b", 19, 27, status="offline"),
            NL("d", 99, 99), NL("e", 88, 88)]
    st0 = c.self_heal(nodes, s, 0, apply)
    assert st0["plan_size"] == 2                        # both surplus nodes split the gap
    first_node = st0["plan"][0]["node_id"]
    c.mark_ready(first_node)
    st = c.self_heal(nodes, s, 1, apply)                 # identical tick
    assert st["ready_count"] == 1                        # progress preserved, not reset
    assert calls == []                                   # not fully ready -> no cutover yet


def test_assignment_for_and_mark_ready_expose_heal_target():
    c = mig.MigrationController()
    s = _serving28()
    calls, apply = _heal_apply()
    nodes = [NL("a", 0, 9, head=True), NL("c", 10, 18, status="offline"),
            NL("b", 19, 27), NL("d", 99, 99)]
    c.self_heal(nodes, s, 0, apply)
    asg = c.assignment_for("d")
    assert asg["migrating"] is True
    assert asg["model_id"] == s["model_id"] and asg["total_layers"] == s["layers"]
    assert asg["layer_start"] == 10 and asg["layer_end"] == 18
    assert c.mark_ready("d") is True


# --------------------------------------------------------------------------- #
# memory awareness: a plan must never hand a node more layers than it can hold
#
# Observed live 2026-08-07: the network auto-promoted to Qwen2.5-7B-Instruct once three nodes
# were eligible and the even split handed 9-10 layers (~4.5 GB of weights) to machines with 8 GB
# of TOTAL RAM. The tier gate qualified the promotion on AGGREGATE capacity ("3 nodes · 20 GB"),
# which says nothing about whether one machine can hold one slice.
#
# Arithmetic these cases depend on (balancer.max_layers_for, RAM_OS_RESERVE_GB=3, headroom=0.75).
#
# GB_7B is the tier table's figure and is on an fp16 basis (2 bytes/param). Nodes actually
# store weights at fp32 (common.WEIGHT_DTYPE defaults to fp32), so balancer.effective_gb_per_
# layer charges 0.932/layer, and the RAM figures below are what it takes to hold these slices
# ON A REAL NODE:
#   effective 0.932/layer -> 13 GB node: (13-3)*0.75/0.932 =  8 layers
#                            21 GB node: (21-3)*0.75/0.932 = 14 layers
#                             4 GB node: (4-3)*0.75/0.932  =  1 layer (floored at 1)
#
# These were 8/12/4 GB, sized against the fp16 figure -- which made every "these machines can
# hold it" fixture claim something untrue of a real machine. That is the same mistake as [P31]
# in miniature: an assumption about the runtime, written down as fact, that the runtime
# contradicts. The numbers grew; none of the LOGIC under test changed.
# --------------------------------------------------------------------------- #
GB_7B = 0.466          # one Qwen2.5-7B layer at fp16, from coordinator/model_tiers.py


def NR(node_id, ls, le, ram, head=False, status="online", eligible=True):
    """Like NL(), plus the reported total RAM that makes the node's memory cap real.

    """
    n = NL(node_id, ls, le, head=head, status=status, eligible=eligible)
    n["ram_gb"] = ram
    return n


def _mixed_trio():
    """Three machines that can between them hold the 7B, 28 layers to place.

    Sized so each node's CAP is 8/8/14 layers at the real fp32 footprint. It was 8/8/12 GB and
    described as "the live shape" — but at fp32 those machines hold 4/4/7 layers, 15 of 28, so
    the fixture was asserting a partition that cannot exist on the hardware it named. The live
    trio genuinely cannot hold the 7B; that is a fact about the fleet, not a property of the
    planner, and pinning it here would only have hidden it."""
    return [NR("a", 0, 9, 13, head=True), NR("b", 10, 18, 13), NR("c", 19, 27, 21)]


def test_plan_fits_the_slice_to_the_machine():
    """The regression. An even 28/3 split is 10/9/9; the 8 GB machines can hold 8."""
    plan = mig.plan_migration(_mixed_trio(), 28, gb_per_layer=GB_7B)
    assert plan == [
        {"node_id": "a", "layer_start": 0, "layer_end": 7},     # 8 layers, at its cap
        {"node_id": "b", "layer_start": 8, "layer_end": 15},    # 8 layers, at its cap
        {"node_id": "c", "layer_start": 16, "layer_end": 27},   # 12, and it could hold 14
    ]
    # the un-memory-aware plan is what shipped, and what OOM-killed the 8 GB machines
    assert [p["layer_end"] for p in mig.plan_migration(_mixed_trio(), 28)] == [9, 18, 27]


def test_plan_never_exceeds_any_node_cap_and_still_covers_the_model():
    for layers in (12, 20, 28, 30):
        plan = mig.plan_migration(_mixed_trio(), layers, gb_per_layer=GB_7B)
        caps = {"a": 8, "b": 8, "c": 14}
        covered = []
        for p in plan:
            got = p["layer_end"] - p["layer_start"] + 1
            assert got <= caps[p["node_id"]], (layers, p)
            covered += list(range(p["layer_start"], p["layer_end"] + 1))
        assert covered == list(range(layers)), (layers, plan)   # contiguous, no gap, no overlap


def test_plan_is_unchanged_when_the_footprint_is_unknown():
    """A model with no measured gb_per_layer (an env-injected tier) must behave exactly as
    before -- an unknown footprint is not a licence to refuse to serve."""
    assert mig.plan_migration(_trio(), 30, gb_per_layer=None) == mig.plan_migration(_trio(), 30)
    # ...and so must a node that reports no RAM at all, even when the footprint IS known
    assert mig.plan_migration(_trio(), 30, gb_per_layer=GB_7B) == mig.plan_migration(_trio(), 30)


def test_plan_can_wake_a_surplus_node_under_memory_pressure():
    """More nodes than an even split needs: the extras normally get nothing, but a machine with
    room is exactly where a layer that does not fit elsewhere belongs."""
    nodes = [NR("a", 0, 13, 4, head=True), NR("b", 14, 27, 4), NR("d", 99, 99, 16)]
    plan = mig.plan_migration(nodes, 6, gb_per_layer=GB_7B)     # even split is 2/2/2
    by = {p["node_id"]: p["layer_end"] - p["layer_start"] + 1 for p in plan}
    assert by["a"] == 1 and by["b"] == 1                        # 4 GB machines hold one each
    assert by["d"] == 4                                         # the rest go where they fit


def test_partition_shortfall_reports_what_cannot_be_held():
    tiny = [NR("a", 0, 9, 4, head=True), NR("b", 10, 18, 4), NR("c", 19, 27, 4)]
    assert mig.partition_shortfall(tiny, 28, GB_7B) == 25       # 3 nodes x 1 layer of 28
    assert mig.partition_shortfall(_mixed_trio(), 28, GB_7B) == 0
    assert mig.partition_shortfall(tiny, 28, None) == 0         # unknown footprint -> no claim


def test_partition_shortfall_counts_only_online_eligible_nodes():
    nodes = [NR("a", 0, 9, 4, head=True),
             NR("big", 10, 27, 64, status="offline"),           # would cover everything
             NR("big2", 10, 27, 64, eligible=False)]            # ...if it were allowed to serve
    assert mig.partition_shortfall(nodes, 28, GB_7B) == 27
    assert mig.partition_shortfall([dict(n, status="online", eligible=True) for n in nodes],
                                   28, GB_7B) == 0


# --------------------------------------------------------------------------- #
# the migration gate: refuse a target no per-node partition can hold
# --------------------------------------------------------------------------- #
def _tiny_trio():
    """Aggregate 12 GB across three nodes -- nowhere near enough for a 28-layer 7B."""
    return [NR("a", 0, 9, 4, head=True), NR("b", 10, 18, 4), NR("c", 19, 27, 4)]


def test_refuses_a_target_the_network_cannot_hold():
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    st = c.update(_tiny_trio(), tgt, s, 0, apply)
    assert st["phase"] == "steady"                       # never even starts preparing
    assert st["plan"] == [] and st["plan_size"] == 0     # nobody is told to download anything
    assert s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"  # and serving is untouched
    assert st["blocked"] == {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28,
                             "reason": "capacity", "capacity_shortfall": 25}


def test_migrates_when_every_node_can_hold_its_slice():
    """The gate is about capacity, not about caution: the same target on machines that fit
    goes through, on the memory-fitted split."""
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    st = c.update(_mixed_trio(), tgt, s, 0, apply)
    assert st["phase"] == "preparing" and st["blocked"] is None
    assert [p["layers"] for p in st["plan"]] == [[0, 7], [8, 15], [16, 27]]
    for nid in ("a", "b", "c"):
        assert c.mark_ready(nid)
    st2 = c.update(_mixed_trio(), tgt, s, 1, apply)
    assert st2["phase"] == "steady" and s["model_id"] == "Qwen/Qwen2.5-7B-Instruct"


def test_an_unknown_footprint_never_blocks_a_migration():
    s, apply = _serving()
    c = mig.MigrationController()
    st = c.update(_tiny_trio(), {"model_id": "meta/8b", "layers": 32}, s, 0, apply)
    assert st["phase"] == "preparing" and st["blocked"] is None


def test_in_flight_preparation_aborts_when_the_machines_that_fit_leave():
    """Capacity can vanish mid-preparing. Preparing on regardless means every remaining node
    downloads several GB it cannot load, and cutover moves the whole network onto a model that
    OOM-kills its own pipeline."""
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    c.update(_mixed_trio(), tgt, s, 0, apply)
    c.mark_ready("a")
    assert c.status()["phase"] == "preparing"

    shrunk = [NR("a", 0, 9, 4, head=True), NR("b", 10, 18, 4)]     # the 12 GB machine left
    st = c.update(shrunk, tgt, s, 1, apply)
    assert st["phase"] == "steady" and st["plan"] == []
    assert st["blocked"]["capacity_shortfall"] == 26
    assert s["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"           # still serving, unchanged

    # and it resumes by itself when the capacity comes back -- blocked is a fact about now
    st2 = c.update(_mixed_trio(), tgt, s, 2, apply)
    assert st2["phase"] == "preparing" and st2["blocked"] is None


def test_a_blocked_migration_does_not_stop_the_network_healing_itself():
    """Why `blocked` is a field and not a phase. self_heal() runs only while phase == "steady",
    so parking a blocked migration in a phase of its own would leave a network that is both
    unable to grow AND unable to repair -- for exactly as long as the unservable tier stays
    qualified, which is forever."""
    s, apply = _serving()
    c = mig.MigrationController()
    tgt = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    nodes = [NR("a", 0, 9, 4, head=True), NR("c", 10, 18, 4, status="offline"),
             NR("b", 19, 27, 4), NR("d", 99, 99, 4)]              # gap at 10-18, d is idle
    st = c.update(nodes, tgt, {"model_id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 28},
                  0, apply)
    assert st["blocked"] is not None and st["phase"] == "steady"

    calls, heal_apply = _heal_apply()
    heal = c.self_heal(nodes, _serving28(), 1, heal_apply)
    assert heal["healing"] is True
    assert heal["plan"] == [{"node_id": "d", "layers": [10, 18], "ready": False}]


def test_self_heal_resplit_reports_a_shortfall_instead_of_hiding_it():
    """A gap means not one request can complete, so a re-split the survivors cannot hold is
    still better than no plan -- but the number has to reach the operator, because the real
    fix is a tier demotion onto a smaller model, which this machine does not perform."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    serving = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    nodes = [NR("a", 0, 9, 4, head=True), NR("c", 10, 18, 4, status="offline"),
             NR("b", 19, 27, 4)]
    st = c.self_heal(nodes, serving, 0, apply)
    assert st["mode"] == "resplit"
    assert st["capacity_shortfall"] == 26                 # 2 nodes x 1 layer against 28
    covered = []
    for p in st["plan"]:
        covered += list(range(p["layers"][0], p["layers"][1] + 1))
    assert covered == list(range(28))                     # the gap is still closed meanwhile


def test_self_heal_surplus_plan_respects_memory_too():
    """The gap is 9 layers and the only idle machine that can take them is the big one."""
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    serving = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    nodes = [NR("a", 0, 9, 16, head=True), NR("c", 10, 18, 16, status="offline"),
             NR("b", 19, 27, 16), NR("small", 88, 88, 4), NR("big", 99, 99, 16)]
    st = c.self_heal(nodes, serving, 0, apply)
    assert st["mode"] == "surplus"
    by = {p["node_id"]: p["layers"] for p in st["plan"]}
    assert by["small"] == [10, 10]                        # one layer, its whole capacity
    assert by["big"] == [11, 18]                          # the other eight
    assert not any(p["node_id"] in ("a", "b") for p in st["plan"])   # working stages untouched


def test_heal_shortfall_clears_after_a_completed_heal():
    c = mig.MigrationController()
    calls, apply = _heal_apply()
    serving = {"model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "gb_per_layer": GB_7B}
    nodes = [NR("a", 0, 9, 4, head=True), NR("c", 10, 18, 4, status="offline"),
             NR("b", 19, 27, 4)]
    c.self_heal(nodes, serving, 0, apply)
    assert c.heal_status()["capacity_shortfall"] == 26
    assert c.mark_ready("a") and c.mark_ready("b")
    st = c.self_heal(nodes, serving, 1, apply)
    assert st["healing"] is False and st["capacity_shortfall"] == 0 and len(calls) == 1


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
