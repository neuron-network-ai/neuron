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
