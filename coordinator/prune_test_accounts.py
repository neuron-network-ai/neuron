"""coordinator/prune_test_accounts.py — return development balances to __ecosystem__.

    python coordinator/prune_test_accounts.py --db <snapshot.db>              # dry run
    python coordinator/prune_test_accounts.py --db <snapshot.db> --execute    # apply

Every account in the live ledger was created by us. No stranger has ever run a node, so every
balance is either real compute done on real hardware (the dev trio), fees the coordinator
actually earned, or a faucet grant handed to a test identity during security and wallet work.
The last group would become real, transferable tokens the moment the ledger moves on-chain
(blockchain/MIGRATION_PLAN.md blocker 2), so it goes back where it came from first.

**Back to __ecosystem__ is not an arbitrary destination.** Every prune target below is a ~25 NRN
faucet grant, and `models.wallet_for_oauth` funds the faucet out of the 150M ecosystem bucket.
Returning them there restores the bucket they were drawn from. Node earnings would be a
different question -- those came out of `__emission_pool__` -- which is exactly why node_a/b/c
are on the keep list rather than swept along with everything else.

What it will not do:
  * touch anything not named. An account that matches no rule is KEPT and reported loudly, so
    an unrecognised name is a decision someone makes, not a balance that quietly disappears.
  * run while `__escrow__` holds anything. A non-zero escrow means requests are in flight and
    their settlement is mid-air; pruning underneath that would miscount the result.
  * apply anything without `--execute`. Dry run is the default and prints the full plan first.

On the invariant: `SUM(balance)` is checked before and after against 1,000,000,000, using the
same 1e-6 tolerance as `models.supply_snapshot`. Exact equality is not achievable -- balances
are SQLite REALs and the live ledger already sums to 999,999,999.99999999999999719 -- so this
also asserts the stricter thing that actually matters here: that the total does not MOVE across
the operation.
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
DEFAULT_LOG = os.path.join(HERE, "prune_log.json")

TOTAL_SUPPLY = 1_000_000_000
TOLERANCE = Decimal("0.000001")          # matches models.supply_snapshot
DESTINATION = "__ecosystem__"
ESCROW = "__escrow__"

# System rows. Never a source: the four allocation buckets ARE the supply, and __escrow__ is
# bookkeeping for in-flight payments.
GENESIS = {"__emission_pool__", "__founder__", "__ecosystem__", "__liquidity__", "__escrow__"}

# Real work on real hardware, and fees actually earned. These keep their NRN.
KEEP = {"node_a", "node_b", "node_c", "__coordinator__"}

# Test identities created during security and wallet development.
PRUNE_NAMES = {"attacker-demo-1", "attacker-demo-2", "probe-only", "live-verify-wallet"}
PRUNE_PREFIXES = (
    ("node_a-cli-", "CLI test wallet from wallet-settlement development"),
    ("w_", "faucet-funded test wallet (OAuth wallet development)"),
)


def classify(account_id, extra_prune=(), extra_keep=()):
    """(disposition, reason) for one account. Explicit flags win over the built-in rules so a
    one-off decision does not need a code change."""
    if account_id in extra_keep:
        return "keep", "kept by --keep-also"
    if account_id in extra_prune:
        return "prune", "pruned by --prune-also"
    if account_id in GENESIS:
        return "keep", "genesis bucket -- part of the supply, never a source"
    if account_id in KEEP:
        return "keep", "real compute on real hardware, or fees actually earned"
    if account_id in PRUNE_NAMES:
        return "prune", "test identity created during security development"
    for prefix, why in PRUNE_PREFIXES:
        if account_id.startswith(prefix):
            return "prune", why
    return "unclassified", "matches no rule -- decide before executing"


def read_ledger(con):
    cols = {r["name"] for r in con.execute("PRAGMA table_info(ledger)")}
    if not cols:
        raise SystemExit("this database has no `ledger` table -- wrong file?")
    acct = "account_type" if "account_type" in cols else "'node' AS account_type"
    rows = con.execute(
        f"SELECT node_id, balance, {acct} FROM ledger ORDER BY node_id").fetchall()
    return [dict(r) for r in rows]


def supply(rows):
    """Exact Decimal total, built from each float's shortest round-tripping repr."""
    return sum(Decimal(repr(float(r["balance"]))) for r in rows)


def invariant_ok(total):
    return abs(total - Decimal(TOTAL_SUPPLY)) < TOLERANCE


def build_plan(rows, extra_prune=(), extra_keep=()):
    plan = []
    for r in rows:
        disposition, reason = classify(r["node_id"], extra_prune, extra_keep)
        plan.append({
            "account_id": r["node_id"],
            "account_type": r["account_type"],
            "balance_nrn": float(r["balance"]),
            "disposition": disposition,
            "reason": reason,
        })
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Return development/test balances to __ecosystem__.")
    p.add_argument("--db", default=DEFAULT_DB, help=f"ledger to operate on (default {DEFAULT_DB})")
    p.add_argument("--log", default=DEFAULT_LOG)
    p.add_argument("--execute", action="store_true",
                   help="actually apply the transfers (default: dry run)")
    p.add_argument("--prune-also", action="append", default=[], metavar="ACCOUNT",
                   help="prune this account too (repeatable) -- for the unclassified list")
    p.add_argument("--keep-also", action="append", default=[], metavar="ACCOUNT",
                   help="keep this account (repeatable), overriding the prune rules")
    p.add_argument("--backup", action="store_true",
                   help="copy the db to <db>.pre-prune-<ts> before applying")
    args = p.parse_args(argv)

    if not os.path.exists(args.db):
        raise SystemExit(f"no ledger at {args.db}")
    extra_prune, extra_keep = set(args.prune_also), set(args.keep_also)
    overlap = extra_prune & extra_keep
    if overlap:
        raise SystemExit(f"--prune-also and --keep-also both name: {', '.join(sorted(overlap))}")

    # Read-only for the whole planning phase, so a dry run physically cannot write.
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = read_ledger(con)
    finally:
        con.close()

    before = supply(rows)
    plan = build_plan(rows, extra_prune, extra_keep)
    to_prune = [e for e in plan if e["disposition"] == "prune" and e["balance_nrn"] > 0]
    empty = [e for e in plan if e["disposition"] == "prune" and e["balance_nrn"] <= 0]
    unclassified = [e for e in plan if e["disposition"] == "unclassified"]
    kept = [e for e in plan if e["disposition"] == "keep"]
    moving = sum(Decimal(repr(e["balance_nrn"])) for e in to_prune)
    escrow = next((float(r["balance"]) for r in rows if r["node_id"] == ESCROW), 0.0)

    # ------------------------------------------------------------------ the plan, up front
    print(f"ledger        : {args.db}")
    print(f"accounts      : {len(rows)}")
    print(f"sum(balance)  : {before:,} NRN "
          f"({'invariant holds' if invariant_ok(before) else 'INVARIANT BROKEN'})")
    print(f"destination   : {DESTINATION}")
    print(f"mode          : {'EXECUTE' if args.execute else 'DRY RUN (nothing will change)'}")

    print(f"\nWILL PRUNE ({len(to_prune)} accounts, {moving:,} NRN -> {DESTINATION})")
    if not to_prune:
        print("    (nothing)")
    for e in to_prune:
        print(f"    - {e['account_id']:<36} {e['balance_nrn']:>14,.6f}  {e['reason']}")
    if empty:
        print(f"\nalready empty ({len(empty)}): "
              + ", ".join(e["account_id"] for e in empty))

    print(f"\nWILL KEEP ({len(kept)} accounts)")
    for e in kept:
        print(f"    = {e['account_id']:<36} {e['balance_nrn']:>14,.6f}  {e['reason']}")

    if unclassified:
        print(f"\nUNCLASSIFIED ({len(unclassified)}) -- KEPT, because a balance that matches no "
              f"rule is a decision, not a default:")
        for e in unclassified:
            print(f"    ? {e['account_id']:<36} {e['balance_nrn']:>14,.6f}  ({e['account_type']})")
        print("    Add --prune-also <account> to sweep one, or --keep-also to silence it.")

    # ------------------------------------------------------------------------- refusals
    if not invariant_ok(before):
        print(f"\nREFUSED: the ledger does not sum to {TOTAL_SUPPLY:,} before we start "
              f"({before}). Fix or re-snapshot first -- pruning a broken ledger would bake "
              f"the error in.")
        return 2
    if abs(escrow) > 1e-9:
        print(f"\nREFUSED: {ESCROW} holds {escrow} NRN, so requests are in flight and their "
              f"settlement is mid-air. Quiesce the coordinator and re-run.")
        return 2

    if not args.execute:
        applied_at = None
        print(f"\nDRY RUN -- nothing was changed. Re-run with --execute to apply.")
    else:
        if args.backup:
            dest = f"{args.db}.pre-prune-{time.strftime('%Y%m%d-%H%M%S')}"
            shutil.copy2(args.db, dest)
            print(f"\nbackup        : {dest}")
        applied_at = time.time()
        con = sqlite3.connect(args.db)
        con.row_factory = sqlite3.Row
        try:
            with con:                                   # one transaction: all or nothing
                con.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (DESTINATION,))
                for e in to_prune:
                    # Zero the source outright rather than subtracting its own balance from
                    # itself: float arithmetic can leave a 1e-15 crumb behind, and a ledger
                    # full of crumbs is how an invariant starts drifting.
                    cur = con.execute(
                        "UPDATE ledger SET balance=0 WHERE node_id=? AND balance=?",
                        (e["account_id"], e["balance_nrn"]))
                    if cur.rowcount != 1:
                        raise RuntimeError(
                            f"{e['account_id']} changed underneath us (balance is no longer "
                            f"{e['balance_nrn']}) -- rolling back, nothing was applied")
                    e["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                # Credit the destination ONCE, from a total summed in Decimal.
                # `balance = balance + x` per account looked equivalent and was not: each
                # addition into a ~1.5e8 balance rounds to the nearest representable double
                # (~3e-8 apart there), so eight of them drifted the supply by 3e-8. Summing
                # exactly and writing once leaves the single unavoidable rounding of one
                # float addition, which is the best a REAL column can do.
                row = con.execute("SELECT balance FROM ledger WHERE node_id=?",
                                  (DESTINATION,)).fetchone()
                credited = float(Decimal(repr(float(row["balance"]))) + moving)
                con.execute("UPDATE ledger SET balance=? WHERE node_id=?",
                            (credited, DESTINATION))
            print(f"\napplied {len(to_prune)} transfers")
        except Exception as exc:                                    # noqa: BLE001
            con.close()
            print(f"\nFAILED, rolled back -- nothing was changed: {exc}")
            return 1
        con.close()

    # -------------------------------------------------------------- verify and record
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        after_rows = read_ledger(con)
    finally:
        con.close()
    after = supply(after_rows)
    drift = after - before

    eco_before = next((e["balance_nrn"] for e in plan if e["account_id"] == DESTINATION), 0.0)
    eco_after = next((float(r["balance"]) for r in after_rows if r["node_id"] == DESTINATION), 0.0)
    # The only error a correct run can introduce is rounding ONE float addition into the
    # destination balance. Anything larger means NRN was created or destroyed, not rounded.
    budget = Decimal(repr(math.ulp(eco_after) if eco_after else 1.0))

    print(f"\nsum(balance)  : {after:,} NRN "
          f"({'invariant holds' if invariant_ok(after) else 'INVARIANT BROKEN'})")
    print(f"supply drift  : {drift:+} NRN "
          f"(budget {budget} = one rounding of the destination balance)")
    print(f"{DESTINATION:<14}: {eco_before:,.6f} -> {eco_after:,.6f}")
    if args.execute:
        shortfall = Decimal(150_000_000) - Decimal(repr(eco_after))
        if shortfall > 0:
            print(f"    still {shortfall:,} NRN below its 150,000,000 allocation -- that is "
                  f"faucet NRN these identities SPENT on inference, which is now node "
                  f"earnings and correctly stays with the nodes.")

    ok = invariant_ok(after) and abs(drift) <= budget
    log = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dry_run": not args.execute,
        "db": os.path.abspath(args.db),
        "destination": DESTINATION,
        "supply_before": str(before),
        "supply_after": str(after),
        "supply_drift": str(drift),
        "supply_drift_budget": str(budget),
        "invariant_before": invariant_ok(before),
        "invariant_after": invariant_ok(after),
        "transfers": [{
            "account_id": e["account_id"],
            "account_type": e["account_type"],
            "amount_nrn": e["balance_nrn"],
            "to": DESTINATION,
            "reason": e["reason"],
            "timestamp": e.get("applied_at"),
            "applied": bool(e.get("applied_at")),
        } for e in to_prune],
        "kept": [{"account_id": e["account_id"], "balance_nrn": e["balance_nrn"],
                  "reason": e["reason"]} for e in kept],
        "unclassified": [{"account_id": e["account_id"], "balance_nrn": e["balance_nrn"],
                          "account_type": e["account_type"]} for e in unclassified],
        "totals": {
            "pruned_accounts": len(to_prune),
            "pruned_nrn": float(moving),
            "already_empty": len(empty),
            "kept_accounts": len(kept),
            "unclassified_accounts": len(unclassified),
        },
    }
    with open(args.log, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
        f.write("\n")
    print(f"log           : {args.log}")

    if not ok:
        print("\nINVARIANT CHECK FAILED AFTER THE OPERATION -- investigate before doing "
              "anything else with this ledger.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
