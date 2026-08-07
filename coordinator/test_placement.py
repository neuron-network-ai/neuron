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

    # complete the chain -> next node replicates the LEAST-replicated stage. Every stage has
    # exactly one node here, so the tie breaks to the earliest: the front of the pipeline is
    # traversed by every request, so a shortfall there throttles everything behind it.
    reg("last-b", S2, N - 1)
    p = router.suggest_placement()
    check("complete chain -> replica of the scarcest (here: first) segment",
          p["role"] == "replica-balance" and [p["layer_start"], p["layer_end"]] == [0, S1 - 1])

    # A probationary node is invisible to ROUTING but visible to PLACEMENT. It cannot serve a
    # request until it is verified -- so the chain is still incomplete -- yet it has already been
    # given that range and has already downloaded that slice, so the next joiner must not be
    # handed the same one.
    for n in models.list_nodes():         # reset (init_db is CREATE IF NOT EXISTS, not a wipe)
        models.delete_node(n["node_id"])
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    register(RegisterBody(node_id="stranger-last", tailscale_ip="127.0.0.1", port=51000,
                          layer_start=S2, layer_end=N - 1, cores=4, ram_gb=8),
             x_register_secret=None)   # probationary
    _chain, missing = router.build_chain()
    check("probationary node does not count as routable coverage", missing == [(S2, N - 1)])
    p = router.suggest_placement()
    check("...but its range is taken, so the next joiner is not sent there too",
          p["role"] == "replica-balance" and [p["layer_start"], p["layer_end"]] != [S2, N - 1])
    models.record_attestation("stranger-last", True)   # verify it
    check("after verification the chain routes end to end", router.build_chain()[1] == [])
    p = router.suggest_placement()
    check("after verification, chain complete -> replica-balance",
          p["role"] == "replica-balance")

    # THE HERD ([P25], live 2026-08-07). Three strangers join a network whose only eligible node
    # is the trusted one on 14-20. Placement used to reason over eligible nodes ONLY, so each
    # newcomer was invisible to the next and all three were told "the first gap is 0-13": three
    # copies of one slice, layers 21-27 covered by nobody, DEGRADED forever.
    for n in models.list_nodes():
        models.delete_node(n["node_id"])
    reg("trusted-mid", 14, 20)
    got = []
    for i in range(3):
        p = router.suggest_placement()
        got.append((p["layer_start"], p["layer_end"]))
        register(RegisterBody(node_id=f"stranger{i}", tailscale_ip="127.0.0.1", port=52000 + i,
                              layer_start=p["layer_start"], layer_end=p["layer_end"],
                              cores=4, ram_gb=8),
                 x_register_secret=None)              # every one of them probationary
    check("strangers close both gaps before anyone duplicates a slice",
          set(got[:2]) == {(0, 13), (21, N - 1)})
    check("only once every layer is claimed does the next one replicate",
          got[2] in {(0, 13), (14, 20), (21, N - 1)})
    covered = set()
    for n in models.list_nodes():
        covered |= set(range(n["layer_start"], n["layer_end"] + 1))
    check("and between them the whole model is claimed", covered == set(range(N)))

    # `exclude` = "where would I go if I weren't already here?" -- the question a node re-asks
    # before deciding to move. It must be self-stabilising: a node whose range is needed is told
    # to stay, or every restart would start a migration.
    for n in models.list_nodes():
        models.delete_node(n["node_id"])
    reg("node_a", 0, S1 - 1)
    reg("node_b", S1, S2 - 1)
    reg("node_c", S2, N - 1)
    p = router.suggest_placement(exclude="node_b")
    check("excluding a node whose range is needed hands the same range back",
          [p["layer_start"], p["layer_end"]] == [S1, S2 - 1])
    reg("node_b2", S1, S2 - 1)                        # now node_b is a redundant duplicate
    p = router.suggest_placement(exclude="node_b")
    check("excluding a redundant node advises a different range",
          [p["layer_start"], p["layer_end"]] != [S1, S2 - 1])
    models.delete_node("node_b2")

    # endpoint returns total_layers + the placement fields
    out = node_placement()
    check("endpoint includes total_layers + role",
          out["total_layers"] == N and "role" in out and "layer_start" in out)
    check("endpoint passes exclude through",
          node_placement(exclude="node_a")["layer_start"] == 0)

    # ---- balanced replication is what turns added machines into concurrency ---- #
    # Regression for a real throughput bug: placement used to always advise "replicate the LAST
    # segment", so every machine after the third piled onto the tail while the single node
    # holding 0-9 and the single node holding 10-18 stayed un-replicated. Since a request must
    # traverse every stage -- and node_server.py's module-level compute_lock serialises each
    # machine's forward pass -- ten machines still delivered ONE machine's throughput.
    for n in models.list_nodes():
        models.delete_node(n["node_id"])
    reg("node_a", 0, S1 - 1)
    reg("node_b", S1, S2 - 1)
    reg("node_c", S2, N - 1)
    for i in range(4, 11):                      # seven more machines join, following advice
        p = router.suggest_placement()
        reg(f"pc{i}", p["layer_start"], p["layer_end"])

    from collections import Counter
    layout = Counter((n["layer_start"], n["layer_end"]) for n in models.list_nodes())
    check("10 machines spread across ALL stages, not piled on the tail", len(layout) == 3)
    check("every stage is replicated (no single-copy bottleneck stage)",
          min(layout.values()) >= 3)
    check("stage depth differs by at most one machine",
          max(layout.values()) - min(layout.values()) <= 1)

    # with a balanced layout the router can actually build distinct parallel chains
    chains = {tuple(n["node_id"] for n in router.build_chain()[0]) for _ in range(60)}
    check("router builds many distinct parallel chains (was exactly 1 shared bottleneck)",
          len(chains) > 1)
    busiest = Counter()
    for _ in range(60):
        for n in router.build_chain()[0]:
            busiest[n["node_id"]] += 1
    check("no single node appears in every chain (the old node_a 10/10 bottleneck)",
          max(busiest.values()) < 60)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
