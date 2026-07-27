"""coordinator/test_migration.py — rolling migration state machine (Build 3).

Run:  python -m coordinator.test_migration      (from repo root)
"""
from coordinator import migration as mig


def N(node_id, ls=0, head=False, status="online", eligible=True):
    return {"node_id": node_id, "layer_start": ls, "status": status,
            "eligible": eligible, "head_ms": 38 if head else 0}


def _trio():
    return [N("a", 0, head=True), N("c", 10), N("b", 19)]


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
