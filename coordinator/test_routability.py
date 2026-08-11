"""coordinator/test_routability.py — coverage is not routability (PROBLEMS.md [P32]).

A roster can cover every layer and still be unusable. `router._walk` advances by
`max(layer_end)` from each cursor, so a node holding a range that SPANS another node's wins the
walk and swallows it: two eligible nodes on 0-27 and 0-9 cover all 28 layers and produce a
ONE-stage chain, which `node_a.coord_get_chain` refuses outright.

That state was live on 2026-08-09 for hours. `/status` reported `total_layers_covered: 28/28`,
`uncovered_layers: []` and `network_healthy: true` throughout; `self_heal` returns early unless
a layer is uncovered, so the repair path was blind to it for the same reason; and every chat
failed AFTER the coordinator had taken a wallet hold, so the failure cost NRN each time.

These tests pin the distinction itself, not the incident: that `chain_shape` reports stages
independently of coverage, and that `network_healthy` means "a request can complete".

Run:  python -m coordinator.test_routability     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, main, models, router

models.init_db()

N = config.TOTAL_LAYERS          # 28


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def _reg(node_id, ls, le, trusted=True):
    models.register_node(node_id, "1.1.1.1", 50999, ls, le, 8, 16,
                         f"tok-{node_id}", ms_per_layer=10,
                         head_ms=(38 if ls == 0 else 0), trusted=trusted)


def _roster():
    return models.list_nodes()


def _offline(node_id):
    """Backdate the heartbeat, do NOT write the `status` column: `_node_dict` DERIVES status
    from last_seen (models.py:235), so setting the column is a no-op and a test that did it
    would silently be asserting nothing."""
    with models._db() as c:
        c.execute("UPDATE nodes SET last_seen=? WHERE node_id=?",
                  (0.0, node_id))


# --------------------------------------------------------------------------- #
# chain_shape — the description
# --------------------------------------------------------------------------- #
def test_three_stages_is_routable():
    _clear()
    _reg("a", 0, 9); _reg("c", 10, 18); _reg("b", 19, 27)
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 3, s
    assert s["ranges"] == [[0, 9], [10, 18], [19, 27]], s
    assert s["routable"] is True, s


def test_two_stages_is_routable_not_degraded():
    """Two is the ordinary shape of a three-machine network with one machine away --
    node_a.coord_get_chain accepts it explicitly, so it must not read as a fault."""
    _clear()
    _reg("a", 0, 9); _reg("b", 10, 27)
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 2 and s["routable"] is True, s


def test_the_p32_shape_is_covered_but_not_routable():
    """THE regression this file exists for: the exact live 2026-08-09 roster."""
    _clear()
    _reg("pavilion", 0, 27)          # the whole model
    _reg("driver", 0, 9)             # the machine you chat from
    s = router.chain_shape(_roster(), N)
    # every layer IS covered ...
    covered = set()
    for n in _roster():
        covered.update(range(n["layer_start"], n["layer_end"] + 1))
    assert covered == set(range(N)), "premise: coverage is complete"
    # ... and the chain is still unusable
    assert s["stages"] == 1, s
    assert s["ranges"] == [[0, 27]], s
    assert s["routable"] is False, s


def test_single_node_holding_everything_is_not_routable():
    _clear()
    _reg("solo", 0, 27)
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 1 and s["routable"] is False, s


def test_a_gap_is_not_routable_even_with_a_plausible_stage_count():
    """A chain that stops at a hole can still have 2 stages. Stage count alone would call that
    routable, which is why `missing` is checked as well as the count, not instead of it."""
    _clear()
    _reg("a", 0, 9); _reg("c", 10, 18)        # 19..27 uncovered
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 2, s
    assert s["routable"] is False, s


def test_more_stages_than_the_driver_accepts():
    _clear()
    for i, (lo, hi) in enumerate([(0, 6), (7, 13), (14, 20), (21, 27)]):
        _reg(f"n{i}", lo, hi)
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 4, s
    assert s["routable"] is False, s          # PIPELINE_STAGES is 3


def test_offline_nodes_are_excluded():
    """Routing only ever uses eligible online nodes, so the description must use the same set --
    otherwise an offline node's stale wide range would make a healthy roster look broken. This
    matters here specifically: [P27]'s two office PCs are offline holding a stale 0-13."""
    _clear()
    _reg("a", 0, 9); _reg("b", 10, 27)
    _reg("ghost", 0, 27)                      # would swallow the chain if counted
    _offline("ghost")
    assert [n["status"] for n in _roster() if n["node_id"] == "ghost"] == ["offline"]
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 2 and s["routable"] is True, s


def test_probationary_nodes_are_excluded():
    """A probationary node receives no live requests, so it cannot be part of the chain -- and
    must not be allowed to swallow it in the description either."""
    _clear()
    _reg("a", 0, 9); _reg("b", 10, 27)
    _reg("newcomer", 0, 27, trusted=False)    # open-join, unverified
    assert [n["eligible"] for n in _roster() if n["node_id"] == "newcomer"] == [False]
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 2 and s["routable"] is True, s


def test_shape_is_deterministic_across_calls():
    """It DESCRIBES a roster rather than routing a request; two calls a second apart that
    disagree are not a description."""
    _clear()
    _reg("a", 0, 9); _reg("r1", 10, 27); _reg("r2", 10, 27)   # replicas of stage 2
    first = router.chain_shape(_roster(), N)
    for _ in range(20):
        assert router.chain_shape(_roster(), N) == first
    assert first["routable"] is True, first


# --------------------------------------------------------------------------- #
# _network_summary — the report
# --------------------------------------------------------------------------- #
def test_summary_reports_routability_alongside_coverage():
    _clear()
    _reg("a", 0, 9); _reg("c", 10, 18); _reg("b", 19, 27)
    net, _ = main._network_summary()
    assert net["stages"] == 3
    assert net["routable"] is True
    assert net["chain_ranges"] == [[0, 9], [10, 18], [19, 27]]
    assert net["network_healthy"] is True


def test_summary_is_not_healthy_when_covered_but_unroutable():
    """The whole point. Before this, every assertion below except the last two passed while
    chat was dead -- and the last two are the ones a human reads."""
    _clear()
    _reg("pavilion", 0, 27); _reg("driver", 0, 9)
    net, _ = main._network_summary()
    # the facts that made it look fine
    assert net["total_layers_covered"] == net["total_layers"] == N
    assert net["uncovered_layers"] == []
    # the facts that say it is not
    assert net["stages"] == 1
    assert net["routable"] is False
    assert net["network_healthy"] is False


def test_summary_still_catches_a_plain_gap():
    """The old coverage check must not be lost in the new one."""
    _clear()
    _reg("a", 0, 9); _reg("c", 10, 18)
    net, _ = main._network_summary()
    assert net["total_layers_covered"] < net["total_layers"]
    assert net["uncovered_layers"] == [[19, 27]], net["uncovered_layers"]
    assert net["routable"] is False
    assert net["network_healthy"] is False


def test_empty_network_is_not_healthy():
    _clear()
    net, _ = main._network_summary()
    assert net["stages"] == 0
    assert net["routable"] is False
    assert net["network_healthy"] is False


def test_stage1_must_be_exactly_the_drivers_shard():
    """Live 2026-08-10: a halted 7B migration left [[0,16],[17,23],[24,27]] — three stages,
    28/28 covered, reported routable, and node_a would have refused every chain because stage 1
    was 17 layers wide instead of the driver's fixed 10."""
    _clear()
    s1 = config.DRIVER_STAGE1_LAYERS
    _reg("driver", 0, 16); _reg("mid", 17, 23); _reg("tail", 24, N - 1)
    s = router.chain_shape(_roster(), N)
    assert s["stages"] == 3, s
    assert s["stage1_ok"] is False, s
    assert s["routable"] is False, "a legal stage count over full coverage is not enough"
    assert s["expected_stage1"] == [0, s1 - 1]
    # ...and the correctly-shaped split is routable
    _clear()
    _reg("driver", 0, s1 - 1); _reg("mid", s1, 18); _reg("tail", 19, N - 1)
    s2 = router.chain_shape(_roster(), N)
    assert s2["stage1_ok"] is True and s2["routable"] is True, s2


def test_summary_reports_stage1_mismatch():
    _clear()
    _reg("driver", 0, 16); _reg("tail", 17, N - 1)
    net, _ = main._network_summary()
    assert net["total_layers_covered"] == net["total_layers"]
    assert net["stage1_ok"] is False
    assert net["routable"] is False and net["network_healthy"] is False


def test_min_and_max_stage_bounds_come_from_config():
    """Both halves of the shape rule are named in config, so neither can drift alone."""
    assert config.MIN_PIPELINE_STAGES == 2
    assert config.PIPELINE_STAGES == 3
    assert config.MIN_PIPELINE_STAGES <= config.PIPELINE_STAGES


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
