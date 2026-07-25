"""Auto-placement tests (Session 20) — run: python -m coordinator.test_placement

A joining node asks the coordinator where to fit, so a stranger never picks layer numbers:
fill a coverage gap if the chain is incomplete; otherwise replicate the last segment (the
segment proof-of-compute can verify) to add throughput. Uses a throwaway DB + the real
router/endpoint code — no HTTP server.
"""
import os
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_place_"), "p.db")

from coordinator import config, models, router  # noqa: E402
from coordinator.main import RegisterBody, node_placement, register  # noqa: E402

SECRET = config.REGISTRATION_SECRET
S1, S2, N = 10, 19, config.TOTAL_LAYERS
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def reg(node_id, ls, le):
    register(RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=50000 + le,
                          layer_start=ls, layer_end=le, cores=4, ram_gb=8),
             x_register_secret=SECRET)


def main():
    models.init_db()

    # empty network -> the whole range is a gap
    p = router.suggest_placement()
    check("empty network -> fill-gap over all layers",
          p["role"] == "fill-gap" and p["layer_start"] == 0 and p["layer_end"] == N - 1)

    # partial chain (driver + middle up, last missing) -> fill the last gap
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    p = router.suggest_placement()
    check("missing last segment -> fill-gap 19-27",
          p["role"] == "fill-gap" and [p["layer_start"], p["layer_end"]] == [S2, N - 1])

    # complete the chain -> next node should REPLICATE the last segment
    reg("last-b", S2, N - 1)
    p = router.suggest_placement()
    check("complete chain -> replica of last segment",
          p["role"] == "replica-last" and [p["layer_start"], p["layer_end"]] == [S2, N - 1])

    # a probationary node covering the gap does NOT count as coverage -> still a gap
    for n in models.list_nodes():         # reset (init_db is CREATE IF NOT EXISTS, not a wipe)
        models.delete_node(n["node_id"])
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    register(RegisterBody(node_id="stranger-last", tailscale_ip="127.0.0.1", port=51000,
                          layer_start=S2, layer_end=N - 1, cores=4, ram_gb=8),
             x_register_secret=None)   # probationary
    p = router.suggest_placement()
    check("probationary node doesn't fill the gap", p["role"] == "fill-gap")
    models.record_attestation("stranger-last", True)   # verify it
    p = router.suggest_placement()
    check("after verification, chain complete -> replica-last", p["role"] == "replica-last")

    # endpoint returns total_layers + the placement fields
    out = node_placement()
    check("endpoint includes total_layers + role",
          out["total_layers"] == N and "role" in out and "layer_start" in out)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
