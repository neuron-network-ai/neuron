"""Replica-routing tests (Session 18) — run: python -m coordinator.test_replica

A 4th node placed as a REPLICA of an existing segment must: be selectable so load spreads
across replicas; never break chain completeness (each chain still driver->middle->last);
be skippable deterministically (injected picker); only participate once eligible (a
probationary replica is never routed); and earn only when it is the one chosen for a request.
Uses a throwaway DB and the real router/ledger/models code — no HTTP server.
"""
import os
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_PROBATION_MIN_PASSES"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_rep_"), "rep.db")

from coordinator import config, ledger, models, router  # noqa: E402
from coordinator.main import RegisterBody, register  # noqa: E402

SECRET = config.REGISTRATION_SECRET
S1, S2, N = 10, 19, config.TOTAL_LAYERS  # driver 0-9, middle 10-18, last 19-27
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def reg(node_id, ls, le, secret=SECRET):
    register(RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=50000 + le,
                          layer_start=ls, layer_end=le, cores=4, ram_gb=8),
             x_register_secret=secret)


def last_of(chain):
    return chain[-1]["node_id"]


def main():
    models.init_db()
    # base pipeline + a 4th node replicating the LAST segment (19-27)
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    reg("last-b", S2, N - 1)
    reg("last-4th", S2, N - 1)     # <-- the 4th node: a replica of the last segment

    # 1) both last-segment replicas get chosen across many calls (load spreads)
    seen = set()
    for _ in range(300):
        chain, missing = router.build_chain()
        assert not missing and len(chain) == 3, (missing, len(chain))
        seen.add(last_of(chain))
    check("both replicas selected across calls", seen == {"last-b", "last-4th"})
    check("every chain complete + 3-stage (300x)", True)  # asserted in the loop above

    # 2) deterministic picker can force either replica (predictable routing / tests)
    lo = router.build_chain(pick=lambda xs: sorted(xs, key=lambda n: n["node_id"])[0])[0]
    hi = router.build_chain(pick=lambda xs: sorted(xs, key=lambda n: n["node_id"])[-1])[0]
    check("picker forces the low replica", last_of(lo) == "last-4th")   # 'last-4th' < 'last-b'
    check("picker forces the high replica", last_of(hi) == "last-b")

    # 3) earnings follow the chosen replica; the other earns nothing that request
    # ledger.distribute() no longer exists (fixed-supply settle() replaced unconditional
    # minting) -- fund __escrow__ directly as a test-only shortcut; see
    # coordinator/test_wallet_settlement.py for the invariant-preserving hold->settle tests.
    b0 = models.get_ledger("last-b")["balance"]
    f0 = models.get_ledger("last-4th")["balance"]
    chosen = router.build_chain(pick=lambda xs: sorted(xs, key=lambda n: n["node_id"])[0])[0]
    models.credit(config.ESCROW_LEDGER_ID, 1.0)
    ledger.settle("rep-req-1", "rep-test-wallet", 1.0, prompt_tokens=0, completion_tokens=1000,
                 plan_nodes=chosen)   # routes to last-4th
    check("chosen replica earned", models.get_ledger("last-4th")["balance"] > f0)
    check("unchosen replica did not earn", models.get_ledger("last-b")["balance"] == b0)

    # 4) a PROBATIONARY replica (open-join, no secret) is never routed until verified
    reg("last-stranger", S2, N - 1, secret=None)
    picks = {last_of(router.build_chain()[0]) for _ in range(300)}
    check("probationary replica never selected", "last-stranger" not in picks)
    models.record_attestation("last-stranger", True)      # verify it
    picks2 = {last_of(router.build_chain()[0]) for _ in range(300)}
    check("verified replica now selectable", "last-stranger" in picks2)

    # 5) removing all base replicas but keeping the stranger still yields a complete chain
    models.delete_node("last-b")
    models.delete_node("last-4th")
    chain, missing = router.build_chain()
    check("chain still complete via the stranger alone",
          not missing and last_of(chain) == "last-stranger")

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
