"""coordinator/test_escrow_conservation.py — run: python -m coordinator.test_escrow_conservation

One property, checked after every shape of settlement:

    __escrow__ holds exactly the sum of the holds still in flight
    (and therefore exactly 0 when nothing is in flight)

Escrow is a staging area, never a destination. The live ledger disproved that -- it held
0.056001 NRN with zero holds in state 'held' -- because ledger.settle() paid node shares inside
`if total_le > 0` and neither paid nor refunded the 90% pool when no planned node was eligible,
and because independently-rounded shares did not add up to the pool.

Neither leak was visible in `test_wallet_settlement.py`, which checks escrow drains on the happy
path. The cases below are the ones that were never exercised: nobody eligible, some eligible,
a split that does not divide evenly, a transfer that fails, and repeated settlements where a
per-settlement residue would only show up in the accumulation.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-escrow-"), "t.db")

from coordinator import config, genesis, ledger, models      # noqa: E402

ok = fail = 0
_req = [0]


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def node(node_id, lo, hi, eligible=True, head=False):
    return {"node_id": node_id, "layer_start": lo, "layer_end": hi,
            "eligible": eligible, "head_ms": 38.0 if head else 0.0}


def escrow():
    return models.get_ledger(config.ESCROW_LEDGER_ID)["balance"]


# Balances are SQLite REALs, so a hold-then-settle round trip leaves ~1e-15 of float residue in
# escrow: `balance = balance - amount` cannot undo `balance = balance + amount` exactly. That is
# rounding, not stranded NRN -- eight orders of magnitude below the 1e-6 the supply invariant
# tolerates, and below the 1e-9 that ledger.settle() and prune_test_accounts.py treat as drift.
# What must NOT survive is anything larger, which is what every check below is really asserting.
DUST = 1e-9


def escrow_clear(expected=0.0):
    """True if escrow holds `expected` (the live holds) and nothing more."""
    balance, held = models.escrow_state()
    return abs(balance - expected) < DUST and abs(held - expected) < DUST


def supply_ok():
    return models.supply_snapshot()["invariant_ok"]


def run_one(wallet, hold_amount, plan, completion_tokens=1000, prompt_tokens=0):
    """hold -> settle, returning (breakdown, escrow after, wallet after)."""
    _req[0] += 1
    rid = f"esc-req-{_req[0]}"
    assert models.hold(rid, wallet, hold_amount), "hold failed -- fund the wallet first"
    bd = ledger.settle(rid, wallet, hold_amount, prompt_tokens, completion_tokens, plan)
    return bd, escrow(), models.get_ledger(wallet)["balance"]


def fund(wallet, amount):
    models.ensure_account(wallet, "wallet")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, wallet, amount)


def main():
    models.init_db()
    genesis.seed_genesis()
    genesis.verify_invariant()

    print("\n-- baseline")
    check("escrow starts at 0", escrow() == 0.0)
    bal, held = models.escrow_state()
    check("escrow_state agrees with nothing in flight", (bal, held) == (0.0, 0.0))

    trio = [node("esc-a", 0, 9, head=True), node("esc-b", 10, 18), node("esc-c", 19, 27)]

    print("\n-- normal settlement (every node eligible)")
    fund("w-normal", 50.0)
    bd, esc, wal = run_one("w-normal", 5.0, trio)
    check("escrow is empty afterwards", escrow_clear(), f"escrow={esc}")
    check("all three nodes were paid", len([k for k in bd if k.startswith("esc-")]) == 3, str(bd))
    nodes_paid = round(sum(v for k, v in bd.items() if k.startswith("esc-")), 6)
    charged = round(5.0 - bd.get("__refund__", 0.0), 6)
    check("node shares + fee account for every NRN charged",
          round(nodes_paid + bd["__coordinator__"], 6) == charged,
          f"nodes {nodes_paid} + fee {bd['__coordinator__']} != charged {charged}")
    check("the shares are exactly the 90% pool, not 90% minus rounding",
          nodes_paid == round(charged * (1 - config.COORDINATOR_FEE), 6),
          f"{nodes_paid} vs {round(charged * (1 - config.COORDINATOR_FEE), 6)}")
    check("the supply invariant holds", supply_ok())

    print("\n-- partial: one node ineligible (flagged or probationary)")
    fund("w-partial", 50.0)
    plan = [node("esc-a", 0, 9, head=True), node("esc-b", 10, 18, eligible=False),
            node("esc-c", 19, 27)]
    bd, esc, wal = run_one("w-partial", 5.0, plan)
    check("escrow is empty afterwards", escrow_clear(), f"escrow={esc}")
    check("the ineligible node earned nothing", "esc-b" not in bd, str(bd))
    check("the eligible nodes still split the whole pool",
          round(bd.get("esc-a", 0) + bd.get("esc-c", 0), 6)
          == round(bd["__coordinator__"] * 9, 6),
          f"nodes={bd.get('esc-a', 0) + bd.get('esc-c', 0)} fee={bd['__coordinator__']}")
    check("the supply invariant holds", supply_ok())

    print("\n-- the leak: NO node eligible")
    fund("w-none", 50.0)
    plan = [node("esc-a", 0, 9, eligible=False), node("esc-b", 10, 18, eligible=False),
            node("esc-c", 19, 27, eligible=False)]
    before = models.get_ledger("w-none")["balance"]
    bd, esc, wal = run_one("w-none", 5.0, plan)
    check("escrow is empty afterwards -- the pool did NOT stay behind", escrow_clear(),
          f"escrow={esc}")
    check("no node was paid", not any(k.startswith("esc-") for k in bd), str(bd))
    check("the payer got the undelivered pool back",
          round(wal, 6) == round(before - bd.get("__coordinator__", 0.0), 6),
          f"wallet {before} -> {wal}, fee {bd.get('__coordinator__')}")
    check("the refund covers the whole hold minus the fee",
          round(bd["__refund__"] + bd.get("__coordinator__", 0.0), 6) == 5.0, str(bd))
    check("the supply invariant holds", supply_ok())

    print("\n-- an empty plan settles cleanly too")
    fund("w-empty", 50.0)
    before = models.get_ledger("w-empty")["balance"]
    bd, esc, wal = run_one("w-empty", 2.0, [])
    check("escrow is empty afterwards", escrow_clear(), f"escrow={esc}")
    check("everything but the fee came back", round(wal, 6)
          == round(before - bd.get("__coordinator__", 0.0), 6))
    check("the supply invariant holds", supply_ok())

    print("\n-- a plan node that no longer exists (get_node returned None)")
    fund("w-gone", 50.0)
    bd, esc, wal = run_one("w-gone", 2.0, [None, node("esc-a", 0, 27, head=True), None])
    check("escrow is empty afterwards", escrow_clear(), f"escrow={esc}")
    check("the surviving node was paid", "esc-a" in bd, str(bd))
    check("the supply invariant holds", supply_ok())

    print("\n-- splits that do not divide evenly")
    fund("w-odd", 200.0)
    seven = [node(f"esc-n{i}", i * 4, i * 4 + 3, head=(i == 0)) for i in range(7)]
    for i, hold_amount in enumerate((0.000003, 0.07, 0.11, 1.23, 3.33, 7.77)):
        bd, esc, wal = run_one("w-odd", hold_amount, seven, completion_tokens=13 + i)
        check(f"escrow empty after an uneven split of {hold_amount}", escrow_clear(),
              f"escrow={esc} breakdown={bd}")
    check("the supply invariant holds", supply_ok())

    print("\n-- repeated settlements (a per-settlement residue would accumulate here)")
    fund("w-many", 100.0)
    for i in range(25):
        run_one("w-many", 0.137, trio, completion_tokens=7 + i)
    check("escrow is still empty after 25 settlements", escrow_clear(),
          f"escrow={escrow()}")
    bal, held = models.escrow_state()
    check("escrow_state still agrees", abs(bal - held) < DUST, f"{bal} vs {held}")
    check("the supply invariant holds", supply_ok())

    print("\n-- concurrency: escrow is NOT swept to zero under a live hold")
    fund("w-live", 50.0)
    fund("w-other", 50.0)
    models.hold("esc-inflight", "w-other", 4.0)          # left in flight on purpose
    bd, esc, wal = run_one("w-live", 5.0, trio)
    check("the in-flight hold is untouched by another request settling", escrow_clear(4.0),
          f"escrow={esc}, expected the 4.0 still in flight")
    bal, held = models.escrow_state()
    check("escrow equals the live holds, not zero",
          abs(bal - 4.0) < DUST and abs(held - 4.0) < DUST, f"{bal} vs {held}")
    models.release_hold("esc-inflight")
    check("and drains to 0 when that one is released", escrow_clear(), f"escrow={escrow()}")
    check("the supply invariant holds", supply_ok())

    print("\n-- a failed node transfer still returns the money to the payer")
    fund("w-failtx", 50.0)
    real_transfer = models.transfer

    def flaky(from_id, to_id, amount, count_request=False):
        if to_id == "esc-b":            # this node's payout never lands
            return False
        return real_transfer(from_id, to_id, amount, count_request)

    models.transfer = flaky
    try:
        before = models.get_ledger("w-failtx")["balance"]
        bd, esc, wal = run_one("w-failtx", 5.0, trio)
    finally:
        models.transfer = real_transfer
    check("escrow is empty even when a payout fails", escrow_clear(), f"escrow={esc}")
    check("the failed node is absent from the breakdown", "esc-b" not in bd, str(bd))
    check("its share was refunded to the payer, not stranded",
          round(before - wal, 6)
          == round(sum(v for k, v in bd.items() if k != "__refund__"), 6),
          f"wallet paid {round(before - wal, 6)}, breakdown {bd}")
    check("the supply invariant holds", supply_ok())

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
