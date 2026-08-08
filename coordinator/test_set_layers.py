"""coordinator/test_set_layers.py — run: python -m coordinator.test_set_layers

POST /network/layers pins an explicit split, overriding the balancer.

It exists because the balancer optimises for stage time and produces splits no client can
route: the driver holds a fixed shard (neuron_driver.S1, layers 0..S1-1) and rejects any chain
whose first stage is not exactly that. A speed-optimal 0-12/13-27 split is unusable by a driver
built for 0-9, and the alternatives were re-downloading the driver's shard or re-rolling the
balancer until it agreed.

The tests that matter are the REFUSALS. This endpoint writes routing state for the whole
network; a half-applied or incoherent split leaves a network no chain can be built from, which
is exactly the failure it is supposed to prevent.
"""
import os
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_setl_"), "s.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, models, router  # noqa: E402
from coordinator.main import (RegisterBody, SetLayersBody, network_set_layers,  # noqa: E402
                              register)

SECRET = config.REGISTRATION_SECRET
N = config.TOTAL_LAYERS          # 28
ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def status_of(fn, *a, **kw):
    try:
        return fn(*a, **kw), None
    except HTTPException as e:
        return None, e.status_code


def layers_of(node_id):
    n = models.get_node(node_id)
    return [n["layer_start"], n["layer_end"]]


def main():
    models.init_db()
    for nid in ("drv", "mid", "tail", "spare"):
        register(RegisterBody(node_id=nid, tailscale_ip="127.0.0.1", port=50000 + len(nid),
                              layer_start=0, layer_end=N - 1, cores=4, ram_gb=8),
                 x_register_secret=SECRET)

    # ---- the case this was built for: pin a split the driver can actually route ---- #
    out, _ = status_of(network_set_layers,
                       SetLayersBody(layers={"drv": [0, 9], "tail": [10, N - 1]}), True)
    check("an explicit contiguous split is applied", out and out["status"] == "set")
    check("...and the driver's stage is exactly what S1 expects", layers_of("drv") == [0, 9])
    check("...and the tail covers the rest", layers_of("tail") == [10, N - 1])
    check("stages are reported back", out["stages"] == [[0, 9], [10, N - 1]])

    chain, missing = router.build_chain(total=N)
    check("the network routes end to end after pinning", missing == [])

    # ---- a node not named is left alone (that is how replicas survive a pin) ------- #
    check("an unnamed node keeps its range", layers_of("mid") == [0, N - 1])

    # ...but that cuts both ways, and it is the trap to know about: a node left spanning the
    # WHOLE model wins the chain walk (build_chain advances to the farthest layer_end from each
    # cursor), so it collapses the pinned stages back into a 1-stage chain. Pinning is only
    # meaningful if every ONLINE node is given a stage. This is the live 2026-08-07 layout:
    # two machines both on 0-27, producing a 1-stage chain the driver cannot route.
    check("an unpinned node spanning the whole model collapses the chain to 1 stage",
          len(chain) == 1)

    out, _ = status_of(network_set_layers,
                       SetLayersBody(layers={"drv": [0, 9], "tail": [10, N - 1],
                                             "mid": [10, N - 1], "spare": [0, 9]}), True)
    chain, missing = router.build_chain(total=N)
    check("once EVERY online node has a stage, the chain is 2 stages",
          missing == [] and len(chain) == 2)

    # ---- replicas: two nodes may share one range without that reading as overlap --- #
    out, _ = status_of(network_set_layers,
                       SetLayersBody(layers={"drv": [0, 9], "tail": [10, N - 1],
                                             "spare": [0, 9]}), True)
    check("two nodes on the SAME stage is allowed (they are replicas)",
          out and out["stages"] == [[0, 9], [10, N - 1]])
    check("...and the replica really got that range", layers_of("spare") == [0, 9])

    # ---- refusals: nothing is written unless the WHOLE split is coherent ---------- #
    before = layers_of("drv")

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [0, 9], "tail": [12, N - 1]}), True)
    check("a GAP between stages is refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [0, 12], "tail": [10, N - 1]}), True)
    check("OVERLAPPING stages are refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [1, 9], "tail": [10, N - 1]}), True)
    check("a split that does not start at layer 0 is refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [0, 9], "tail": [10, N - 5]}), True)
    check("a split that does not reach the last layer is refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [0, 9], "tail": [10, N + 5]}), True)
    check("a range past the end of the model is refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [9, 0], "tail": [10, N - 1]}), True)
    check("a backwards range is refused", code == 400)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"nope": [0, N - 1]}), True)
    check("an unknown node id is refused", code == 404)

    _, code = status_of(network_set_layers,
                        SetLayersBody(layers={"drv": [0]}), True)
    check("a malformed range is refused", code == 400)

    check("NOTHING was written by any refused call — the split is all-or-nothing",
          layers_of("drv") == before)

    # ---- an empty body is a no-op, not a wipe ------------------------------------- #
    out, _ = status_of(network_set_layers, SetLayersBody(layers={}), True)
    check("an empty body changes nothing", out and out["assignments"] == []
          and layers_of("drv") == before)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
