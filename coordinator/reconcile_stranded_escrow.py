"""coordinator/reconcile_stranded_escrow.py — return NRN stranded in __escrow__ to its payer.

    python coordinator/reconcile_stranded_escrow.py --db <snapshot.db>            # dry run
    python coordinator/reconcile_stranded_escrow.py --db <snapshot.db> --execute  # apply

A **one-time** repair, not a routine. `ledger.settle()` used to leave money behind in two ways
(the whole 90% node pool when no planned node was eligible, and a sub-cent residue from rounding
each share independently); both are fixed and `coordinator/test_escrow_conservation.py` holds
them fixed. What that fix cannot do is undo what already happened, and the live ledger carries
**0.056001 NRN** in `__escrow__` against zero holds in state `held`.

Escrow is a staging area, never a destination: every NRN in it should be backed by a live hold.
Anything above that total is money a request took from a wallet and never gave back, so it goes
back to the wallet that paid it.

Attribution is by weight of evidence, and the script prints that evidence rather than asserting
it. Per-request payout records were never stored, so which settlement stranded which fraction
cannot be recovered — but the wallet that paid for almost all of the settled volume is the
overwhelmingly likely payer, and it is inferred from the `holds` table rather than hardcoded.
`--to` overrides. On the live ledger this barely matters financially: the inferred wallet is
itself a `prune_test_accounts.py` target, so the NRN reaches `__ecosystem__` either way. It
matters as a principle — undelivered money returns to whoever paid, and the ledger should say
so.

Once this has run, `prune_test_accounts.py` stops refusing (its escrow check is what surfaced
this in the first place).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sqlite3
import sys
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "neuron.db")
DEFAULT_LOG = os.path.join(HERE, "reconcile_log.json")

TOTAL_SUPPLY = 1_000_000_000
TOLERANCE = Decimal("0.000001")          # matches models.supply_snapshot
ESCROW = "__escrow__"
# Below this, a difference is float residue from REAL arithmetic, not stranded NRN. A
# hold -> settle round trip cannot return escrow to exactly its previous bit pattern.
DUST = 1e-9


def read_all(con):
    con.row_factory = sqlite3.Row
    ledger = [dict(r) for r in con.execute(
        "SELECT node_id, balance, account_type FROM ledger ORDER BY node_id")]
    try:
        holds = [dict(r) for r in con.execute(
            "SELECT request_id, wallet_id, amount, status FROM holds")]
    except sqlite3.OperationalError:
        holds = []
    return ledger, holds


def supply(ledger):
    return sum(Decimal(repr(float(r["balance"]))) for r in ledger)


def invariant_ok(total):
    return abs(total - Decimal(TOTAL_SUPPLY)) < TOLERANCE


def escrow_state(ledger, holds):
    balance = next((float(r["balance"]) for r in ledger if r["node_id"] == ESCROW), 0.0)
    live = sum(float(h["amount"]) for h in holds if h["status"] == "held")
    return balance, live


def infer_payer(holds):
    """(wallet, evidence rows). The wallet with the most settled volume is the one whose
    settlements had the most opportunity to strand money."""
    volume = {}
    for h in holds:
        if h["status"] == "settled":
            volume[h["wallet_id"]] = round(volume.get(h["wallet_id"], 0.0)
                                           + float(h["amount"]), 6)
    rows = sorted(volume.items(), key=lambda kv: -kv[1])
    return (rows[0][0] if rows else None), rows


def main(argv=None):
    p = argparse.ArgumentParser(description="Return stranded escrow NRN to its payer.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"ledger to operate on (default {DEFAULT_DB})")
    p.add_argument("--log", default=DEFAULT_LOG)
    p.add_argument("--to", help="wallet to credit (default: inferred from settled holds)")
    p.add_argument("--execute", action="store_true",
                   help="actually apply the transfer (default: dry run)")
    p.add_argument("--backup", action="store_true",
                   help="copy the db to <db>.pre-reconcile-<ts> before applying")
    args = p.parse_args(argv)

    if not os.path.exists(args.db):
        raise SystemExit(f"no ledger at {args.db}")

    # Read-only for the whole planning phase: a dry run physically cannot write.
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        ledger, holds = read_all(con)
    finally:
        con.close()

    before = supply(ledger)
    balance, live = escrow_state(ledger, holds)
    stranded = round(balance - live, 6)
    inferred, evidence = infer_payer(holds)
    recipient = args.to or inferred

    print(f"ledger        : {args.db}")
    print(f"mode          : {'EXECUTE' if args.execute else 'DRY RUN (nothing will change)'}")
    print(f"sum(balance)  : {before:,} NRN "
          f"({'invariant holds' if invariant_ok(before) else 'INVARIANT BROKEN'})")
    print(f"\n{ESCROW} holds : {balance:.6f} NRN")
    print(f"live holds     : {live:.6f} NRN "
          f"({sum(1 for h in holds if h['status'] == 'held')} requests in flight)")
    print(f"stranded       : {stranded:.6f} NRN")

    if evidence:
        print("\nsettled hold volume by wallet (the attribution evidence):")
        total_vol = sum(v for _, v in evidence) or 1.0
        for wallet, vol in evidence:
            mark = "->" if wallet == recipient else "  "
            print(f"  {mark} {wallet:<40} {vol:>10.6f} NRN  ({vol / total_vol * 100:5.1f}%)")

    # ---------------------------------------------------------------------------- refusals
    if not invariant_ok(before):
        print(f"\nREFUSED: the ledger does not sum to {TOTAL_SUPPLY:,} to begin with "
              f"({before}). Reconciling on top of a broken supply would bake the error in.")
        return 2
    if balance < live - DUST:
        print(f"\nREFUSED: escrow holds LESS than the {live:.6f} NRN of live holds. That is a "
              f"different and worse problem than stranding -- money backing in-flight requests "
              f"is missing. Do not paper over it with a credit.")
        return 2
    if stranded <= DUST:
        print(f"\nNothing to reconcile: escrow already matches its live holds "
              f"(difference {stranded:.9f} is float residue, not NRN).")
        return 0
    if not recipient:
        print("\nREFUSED: no settled holds to infer a payer from, and no --to given. "
              "Name the wallet explicitly.")
        return 2
    if not any(r["node_id"] == recipient for r in ledger):
        print(f"\nREFUSED: {recipient} has no ledger row. Crediting it would create an "
              f"account out of a repair, which is not what a repair is for.")
        return 2

    print(f"\nWILL CREDIT   : {stranded:.6f} NRN  ->  {recipient}")
    print(f"  reason      : money taken from this wallet by a settlement and never returned "
          f"(escrow leak, fixed in ledger.settle)")
    if not args.execute:
        print("\nDRY RUN -- nothing was changed. Re-run with --execute to apply.")
        applied_at = None
    else:
        if args.backup:
            dest = f"{args.db}.pre-reconcile-{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.copy2(args.db, dest)
            print(f"\nbackup        : {dest}")
        con = sqlite3.connect(args.db)
        con.row_factory = sqlite3.Row
        try:
            with con:                                    # one transaction: all or nothing
                cur = con.execute(
                    "UPDATE ledger SET balance=? WHERE node_id=? AND balance=?",
                    (live, ESCROW, balance))
                if cur.rowcount != 1:
                    raise RuntimeError(
                        f"{ESCROW} changed underneath us (balance is no longer {balance}) -- "
                        f"rolling back, nothing was applied")
                # Set escrow to exactly the live-hold total and hand the difference over, so
                # the result is exact rather than the outcome of two float subtractions.
                row = con.execute("SELECT balance FROM ledger WHERE node_id=?",
                                  (recipient,)).fetchone()
                credited = float(Decimal(repr(float(row["balance"]))) + Decimal(repr(stranded)))
                con.execute("UPDATE ledger SET balance=? WHERE node_id=?",
                            (credited, recipient))
            applied_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"\ncredited {stranded:.6f} NRN to {recipient}")
        except Exception as exc:                                    # noqa: BLE001
            con.close()
            print(f"\nFAILED, rolled back -- nothing was changed: {exc}")
            return 1
        con.close()

    # ------------------------------------------------------------------- verify and record
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        ledger_after, holds_after = read_all(con)
    finally:
        con.close()
    after = supply(ledger_after)
    drift = after - before
    bal_after, live_after = escrow_state(ledger_after, holds_after)
    recipient_after = next((float(r["balance"]) for r in ledger_after
                            if r["node_id"] == recipient), 0.0)
    budget = Decimal(repr(math.ulp(recipient_after) if recipient_after else 1.0))

    print(f"\n{ESCROW} now   : {bal_after:.6f} NRN  (live holds {live_after:.6f})")
    print(f"sum(balance)  : {after:,} NRN "
          f"({'invariant holds' if invariant_ok(after) else 'INVARIANT BROKEN'})")
    print(f"supply drift  : {drift:+} NRN "
          f"(budget {budget} = one rounding of the credited balance)")

    escrow_ok = abs(bal_after - live_after) < DUST
    ok = invariant_ok(after) and abs(drift) <= budget and (escrow_ok or not args.execute)
    if args.execute:
        print(f"escrow matches its live holds: {'yes' if escrow_ok else 'NO'}")

    log = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": not args.execute,
        "db": os.path.abspath(args.db),
        "escrow_before": balance,
        "live_holds_before": live,
        "stranded_nrn": stranded,
        "recipient": recipient,
        "recipient_inferred": args.to is None,
        "attribution_evidence": [{"wallet_id": w, "settled_volume_nrn": v}
                                 for w, v in evidence],
        "applied_at": applied_at,
        "applied": bool(applied_at),
        "reason": "escrow leak in ledger.settle -- money taken from this wallet by a "
                  "settlement and never returned",
        "escrow_after": bal_after,
        "live_holds_after": live_after,
        "escrow_matches_live_holds_after": escrow_ok,
        "supply_before": str(before),
        "supply_after": str(after),
        "supply_drift": str(drift),
        "supply_drift_budget": str(budget),
        "invariant_before": invariant_ok(before),
        "invariant_after": invariant_ok(after),
    }
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"log           : {args.log}")

    if not ok:
        print("\nPOST-CHECK FAILED -- investigate before doing anything else with this ledger.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
