"""coordinator/test_serving_model.py — the serving-model seam (Build 2c).

The coordinator has ONE authoritative "serving model" (model_id + layer count). Placement,
slice-info, routing and coverage all track it, so pointing the network at a different model
is a single change — distinct from the TierController's target tier.

Run:  python -m coordinator.test_serving_model     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, main, migration, models, router

models.init_db()


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def _reg(node_id, ls, le, ram=16):
    models.register_node(node_id, "1.1.1.1", 50999, ls, le, 8, ram,
                         f"tok-{node_id}", ms_per_layer=10,
                         head_ms=(38 if ls == 0 else 0), trusted=True)


def _reset_serving():
    main.set_serving_model(config.MODEL_ID, config.TOTAL_LAYERS)


def test_serving_default_is_config_floor():
    _reset_serving()
    assert main.serving_model() == {"model_id": config.MODEL_ID,
                                    "layers": config.TOTAL_LAYERS}


def test_set_and_reset_serving():
    main.set_serving_model("foo/bar-8b", 32)
    assert main.serving_model()["layers"] == 32
    assert main.serving_model()["model_id"] == "foo/bar-8b"
    _reset_serving()
    assert main.serving_model()["layers"] == config.TOTAL_LAYERS


def test_placement_includes_serving_model():
    _reset_serving(); _clear()
    _reg("a", 0, 9); _reg("c", 10, 18)          # gap at 19..27
    p = main.node_placement()
    assert p["model_id"] == config.MODEL_ID
    assert p["total_layers"] == config.TOTAL_LAYERS
    assert p["role"] == "fill-gap" and p["layer_start"] == 19 and p["layer_end"] == 27


def test_serving_layers_drive_coverage():
    _reset_serving(); _clear()
    _reg("a", 0, 9); _reg("c", 10, 18); _reg("b", 19, 27)   # full 0..27
    net, _ = main._network_summary()
    assert net["total_layers"] == 28 and net["network_healthy"] is True
    main.set_serving_model("big/32-layer", 32)              # now 32 layers are required
    net2, _ = main._network_summary()
    assert net2["total_layers"] == 32 and net2["network_healthy"] is False
    _reset_serving()


def test_build_chain_respects_total():
    _clear()
    _reg("a", 0, 9); _reg("c", 10, 18)                      # cover 0..18 only
    ch, miss = router.build_chain(total=19)                 # need 0..18 -> complete
    assert miss == [] and [n["node_id"] for n in ch] == ["a", "c"]
    ch2, miss2 = router.build_chain(total=28)               # need 0..27 -> gap
    assert (19, 27) in miss2


def test_cutover_persists_node_layer_ranges():
    """The node-side follow-up (Build 3): apply_migration_cutover must not only flip the
    served model_id, it must ALSO repartition each node's registered layer_start/end to the
    migration plan — a node reloads onto the NEW range as part of reporting ready, so the
    coordinator's routing/placement have to agree with that range once cutover happens,
    not keep the pre-migration split."""
    _reset_serving(); _clear()
    _reg("a", 0, 9); _reg("c", 10, 18); _reg("b", 19, 27)     # old 28-layer split

    ctrl = migration.MigrationController()
    main._migration.__dict__.update(ctrl.__dict__)             # drive the module singleton
    tgt = {"model_id": "meta/8b", "layers": 30}
    main._migration.update(models.list_nodes(), tgt, main.serving_model(), 0,
                           main.apply_migration_cutover)
    for nid in ("a", "c", "b"):
        assert main._migration.mark_ready(nid)
    st = main._migration.update(models.list_nodes(), tgt, main.serving_model(), 1,
                                main.apply_migration_cutover)
    assert st["phase"] == "steady"
    assert main.serving_model() == {"model_id": "meta/8b", "layers": 30}
    # 30/3 = 10 each, driver ("a", head_ms set) first
    assert [models.get_node(n)["layer_start"] for n in ("a", "c", "b")] == [0, 10, 20]
    assert [models.get_node(n)["layer_end"] for n in ("a", "c", "b")] == [9, 19, 29]
    _reset_serving()


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
