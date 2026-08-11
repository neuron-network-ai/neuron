"""coordinator/test_placement_ownership.py — the coordinator owns placement (PROBLEMS.md [P32]).

A re-registration REPORTS what a node believes it holds. It does not decide.

Before this, `register_node` overwrote `layer_start`/`layer_end` with whatever the node sent,
while the four fields on the lines below it (`ms_per_layer`, `head_ms`, `platform`,
`hw_fingerprint`) were COALESCEd. So `neuron fix` wrote a split to the coordinator and the node's
own `config.json` silently won it back on the next restart, reconnect or relay-ticket refresh --
collapsing the chain to one stage while every layer stayed covered. Live 2026-08-09: the split
was applied and verified, and had reverted ~4 hours later.

Safe to ignore the echo because the node ALREADY follows the coordinator: `/node/{id}/slice-info`
returns this row's range (main.py), and `agent.setup()` serves whatever that returns. Ignoring
the claim removes the one path by which the two could disagree; it does not create one.

Run:  python -m coordinator.test_placement_ownership     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, main, models, router

models.init_db()

N = config.TOTAL_LAYERS


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def _register(node_id, ls, le, token="tok", trusted=True):
    """A registration exactly as agent.register() makes it: the node's OWN config range."""
    return models.register_node(node_id, "1.1.1.1", 50999, ls, le, 8, 16,
                                token, ms_per_layer=10,
                                head_ms=(38 if ls == 0 else 0), trusted=trusted)


# --------------------------------------------------------------------------- #
# The core rule
# --------------------------------------------------------------------------- #
def test_first_registration_establishes_the_range():
    """A node the coordinator has never seen has no assignment to protect, so its claim IS the
    placement. Deleting a node and letting it rejoin is how a wiped machine comes back."""
    _clear()
    _register("newnode", 0, 13)
    n = models.get_node("newnode")
    assert (n["layer_start"], n["layer_end"]) == (0, 13)
    assert n["placement_drift"] is False


def test_reregistration_does_not_move_an_assigned_node():
    """THE regression. The node re-asserts its stale config; the assignment survives."""
    _clear()
    _register("pavilion", 0, 27)              # joined holding the whole model
    models.update_layers("pavilion", 10, 27)  # operator ran `neuron fix`
    _register("pavilion", 0, 27)              # restart / reconnect / relay-ticket refresh
    n = models.get_node("pavilion")
    assert (n["layer_start"], n["layer_end"]) == (10, 27), "the pin must survive"


def test_the_claim_is_recorded_not_discarded():
    """Ignoring the claim silently would trade one invisible fact for another."""
    _clear()
    _register("pavilion", 0, 27)
    models.update_layers("pavilion", 10, 27)
    _register("pavilion", 0, 27)
    n = models.get_node("pavilion")
    assert (n["reported_layer_start"], n["reported_layer_end"]) == (0, 27)
    assert n["placement_drift"] is True


def test_no_drift_when_the_node_agrees():
    _clear()
    _register("good", 0, 9)
    _register("good", 0, 9)
    n = models.get_node("good")
    assert n["placement_drift"] is False


def test_drift_clears_once_the_node_reports_the_assigned_range():
    """What a fixed agent (or a corrected local config) looks like: the claim catches up."""
    _clear()
    _register("pavilion", 0, 27)
    models.update_layers("pavilion", 10, 27)
    _register("pavilion", 0, 27)
    assert models.get_node("pavilion")["placement_drift"] is True
    _register("pavilion", 10, 27)             # agent now persists what it was told
    n = models.get_node("pavilion")
    assert n["placement_drift"] is False
    assert (n["layer_start"], n["layer_end"]) == (10, 27)


def test_rows_predating_this_change_are_not_reported_as_drift():
    """reported_* is NULL until a node re-registers. NULL means "no claim seen", which must not
    read as a mismatch -- every live node would have lit up on the first deploy."""
    _clear()
    _register("old", 0, 9)
    with models._db() as c:
        c.execute("UPDATE nodes SET reported_layer_start=NULL, reported_layer_end=NULL "
                  "WHERE node_id='old'")
    n = models.get_node("old")
    assert n["reported_layer_start"] is None
    assert n["placement_drift"] is False


# --------------------------------------------------------------------------- #
# The authoritative paths must still work
# --------------------------------------------------------------------------- #
def test_update_layers_still_moves_a_node():
    """/network/layers, migration cutover and self-heal all go through this. If placement
    ownership moved to the coordinator and the coordinator then could not place, that would be
    strictly worse than the bug."""
    _clear()
    _register("n", 0, 27)
    models.update_layers("n", 14, 27)
    assert models.get_node("n")["assigned_layers"] == [14, 27]


def test_other_registration_fields_still_update():
    """Only the layer range is protected. An address or port change must still land, or a node
    that moved network would become unreachable forever."""
    _clear()
    _register("n", 0, 9)
    models.register_node("n", "9.9.9.9", 51999, 0, 9, 16, 32, "tok2",
                         ms_per_layer=5, trusted=True)
    n = models.get_node("n")
    assert n["tailscale_ip"] == "9.9.9.9" and n["port"] == 51999
    assert n["cores"] == 16 and n["ram_gb"] == 32
    assert n["ms_per_layer"] == 5


# --------------------------------------------------------------------------- #
# End to end: the live 2026-08-09 sequence
# --------------------------------------------------------------------------- #
def test_the_live_sequence_no_longer_reverts():
    """Exactly what happened: pin, then a re-registration, then check routability.

    Previously this ended at 1 stage with network_healthy True. It now ends where the pin put
    it, and the drift is visible instead of the damage.
    """
    _clear()
    _register("driver", 0, 9)
    _register("pavilion", 0, 27)
    # before the pin: covered, but a 1-stage chain
    assert router.chain_shape(models.list_nodes(), N)["stages"] == 1
    # `neuron fix`
    models.update_layers("pavilion", 10, 27)
    shape = router.chain_shape(models.list_nodes(), N)
    assert shape["stages"] == 2 and shape["routable"] is True
    # ...and now every way a node re-registers
    _register("pavilion", 0, 27)
    _register("driver", 0, 9)
    shape = router.chain_shape(models.list_nodes(), N)
    assert shape["stages"] == 2, "the pin reverted -- [P32] is back"
    assert shape["routable"] is True
    net, _ = main._network_summary()
    assert net["network_healthy"] is True
    assert models.get_node("pavilion")["placement_drift"] is True


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
