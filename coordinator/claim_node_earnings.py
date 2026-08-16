"""coordinator/claim_node_earnings.py — move a node account's balance to a signed-in wallet.

    python coordinator/claim_node_earnings.py --node node-c-pavilion --wallet w_<hex>
    python coordinator/claim_node_earnings.py --node node-c-pavilion --wallet w_<hex> --execute

Runs on the coordinator VM, where the DB is. Dry run by default; `--execute` applies.

**Why this script exists.** A node account and a wallet are both rows in `ledger`, but nothing
has ever connected them. `register_node` creates `INSERT OR IGNORE INTO ledger (node_id)` at
`account_type='node'` — no email, no login, no owner — while a wallet is minted by
`wallet_for_oauth` at `account_type='wallet'` and is the only account a person can actually
spend from. So a volunteer's earnings accumulate against a machine name they cannot sign in as,
and there is no endpoint anywhere that moves them. `models.transfer` is the primitive; nothing
exposes it. That is the gap this closes, and it is exactly the compute-barter case TOKENOMICS
§11 argues is the honest pitch: the laptop pays its owner's own AI bill.

**Re-runnable on purpose.** It sweeps whatever balance is present, so it is not a one-shot: the
node keeps earning availability emission every hour and this can be run again. That is a
mitigation, not a fix — the durable fix is an owner link recorded at registration, so earnings
land somewhere spendable without an operator remembering to sweep.

**What it refuses to do**, because each of these strands money worse than leaving it alone:
  * pay into an account that does not exist — `transfer`'s `INSERT OR IGNORE` would happily
    create a row for a typo'd wallet id, and nobody can sign in as a typo;
  * pay into anything that is not `account_type='wallet'` (use `--force-type` deliberately);
  * run at all if the 1,000,000,000 supply invariant is already broken — a sweep across an
    unbalanced ledger buries the discrepancy under a second movement.

**On `total_earned`:** this matches `models.transfer` exactly — the recipient's `total_earned`
rises, the sender's does not fall. So `network_stats.total_nrn_distributed` (which sums
`total_earned` over `account_type='node'`) does NOT drop when a node is swept, which is correct:
the network really did distribute that NRN for compute, and a later transfer does not un-earn
it. Wallets are excluded from that sum, so no double count.

The ledger has no transactions table, so a sweep leaves no trace in the DB beyond the new
balances. Every applied run therefore appends to `claim_log.json` next to the database.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "neuron.db")
DEFAULT_LOG = os.path.join(HERE, "claim_log.json")

TOTAL_SUPPLY = 1_000_000_000
TOLERANCE = Decimal("0.000001")          # matches models.supply_snapshot
DUST = 1e-9                              # below this a "balance" is float residue, not money


def read_ledger(con):
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(
        "SELECT node_id, balance, total_earned, account_type FROM ledger ORDER BY node_id")]


def supply(ledger):
    return sum(Decimal(repr(float(r["balance"]))) for r in ledger)


def invariant_ok(total):
    return abs(total - Decimal(TOTAL_SUPPLY)) < TOLERANCE


def find(ledger, account_id):
    return next((r for r in ledger if r["node_id"] == account_id), None)


def plan(ledger, node_ids, wallet_id, amount=None, force_type=False):
    """Decide what would move. Pure — no DB writes — so the arithmetic is checkable before
    anything is applied, and `--execute` runs the same function again on fresh rows."""
    problems, moves = [], []

    dest = find(ledger, wallet_id)
    if dest is None:
        problems.append(
            f"destination '{wallet_id}' has no ledger row. A wallet is created by signing in "
            f"(wallet_for_oauth); paying a name that has never logged in strands the NRN "
            f"somewhere nobody can authenticate as.")
    elif dest["account_type"] != "wallet" and not force_type:
        problems.append(
            f"destination '{wallet_id}' is account_type='{dest['account_type']}', not 'wallet'. "
            f"Pass --force-type if that is genuinely intended.")

    for nid in node_ids:
        src = find(ledger, nid)
        if src is None:
            problems.append(f"source '{nid}' has no ledger row — nothing has ever been credited "
                            f"to that name.")
            continue
        if src["account_type"] != "node" and not force_type:
            problems.append(f"source '{nid}' is account_type='{src['account_type']}', not "
                            f"'node'. Pass --force-type if that is genuinely intended.")
            continue
        if nid == wallet_id:
            problems.append(f"source and destination are both '{nid}'.")
            continue
        available = float(src["balance"])
        take = available if amount is None else min(float(amount), available)
        if take <= DUST:
            moves.append({"from": nid, "amount": 0.0, "from_balance": available,
                          "reason": "balance is zero (or float dust) — nothing to sweep"})
            continue
        if amount is not None and float(amount) > available:
            problems.append(f"'{nid}' holds {available:.6f} NRN, less than the {float(amount):.6f} "
                            f"requested. A partial sweep must be asked for explicitly.")
            continue
        moves.append({"from": nid, "amount": take, "from_balance": available, "reason": None})

    return moves, problems, dest


def apply_moves(con, moves, wallet_id, now):
    """Debit-then-credit in ONE transaction, mirroring models.transfer: the credit only runs if
    the debit's WHERE clause actually matched, so a losing debit leaves both sides untouched."""
    applied = []
    for m in moves:
        if m["amount"] <= DUST:
            continue
        cur = con.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (m["amount"], m["from"], m["amount"]))
        if cur.rowcount == 0:
            m["reason"] = "debit did not match — balance changed under us; skipped"
            continue
        con.execute("INSERT OR IGNORE INTO ledger (node_id, account_type) VALUES (?, 'wallet')",
                    (wallet_id,))
        con.execute("UPDATE ledger SET balance=balance+?, total_earned=total_earned+? "
                    "WHERE node_id=?", (m["amount"], m["amount"], wallet_id))
        applied.append({"from": m["from"], "to": wallet_id, "amount": m["amount"], "at": now})
    return applied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--node", action="append", required=True, metavar="NODE_ID",
                    help="node account to sweep (repeatable)")
    ap.add_argument("--wallet", required=True, metavar="WALLET_ID",
                    help="destination wallet (w_<32 hex>, minted by signing in)")
    ap.add_argument("--amount", type=float, default=None,
                    help="sweep this much from EACH node instead of the full balance")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--execute", action="store_true", help="apply (default is a dry run)")
    ap.add_argument("--force-type", action="store_true",
                    help="allow account types other than node -> wallet")
    ap.add_argument("--no-backup", action="store_true", help="skip the pre-write DB copy")
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no database at {args.db}", file=sys.stderr)
        return 2

    con = sqlite3.connect(args.db)
    try:
        ledger = read_ledger(con)
        before = supply(ledger)
        print(f"ledger: {len(ledger)} accounts, supply {before}")
        if not invariant_ok(before):
            print(f"REFUSING: supply is {before}, not {TOTAL_SUPPLY}. Fix the invariant first — "
                  f"a sweep across an unbalanced ledger hides the discrepancy.", file=sys.stderr)
            return 3
        print("supply invariant OK\n")

        moves, problems, dest = plan(ledger, args.node, args.wallet, args.amount, args.force_type)
        for p in problems:
            print("PROBLEM:", p, file=sys.stderr)
        if problems:
            return 4

        total = sum(m["amount"] for m in moves)
        print(f"destination {args.wallet}: balance {float(dest['balance']):.6f} NRN")
        for m in moves:
            if m["amount"] <= DUST:
                print(f"  {m['from']:<28} -> nothing ({m['reason']})")
            else:
                print(f"  {m['from']:<28} -> {m['amount']:.6f} NRN "
                      f"(of {m['from_balance']:.6f} held)")
        print(f"\ntotal to move: {total:.6f} NRN")
        if not total:
            print("nothing to do.")
            return 0

        if not args.execute:
            print(f"\nDRY RUN — nothing written. Re-run with --execute to apply.")
            print(f"after:  {args.wallet} would hold "
                  f"{float(dest['balance']) + total:.6f} NRN")
            return 0

        if not args.no_backup:
            backup = f"{args.db}.bak_claim_{int(time.time())}"
            shutil.copy2(args.db, backup)
            print(f"\nbacked up -> {backup}")

        now = time.time()
        with con:
            applied = apply_moves(con, moves, args.wallet, now)

        after_ledger = read_ledger(con)
        after = supply(after_ledger)
        if not invariant_ok(after):
            print(f"INVARIANT BROKEN after write: {after}. Restore from the backup above.",
                  file=sys.stderr)
            return 5

        dest_after = find(after_ledger, args.wallet)
        moved = sum(a["amount"] for a in applied)
        print(f"moved {moved:.6f} NRN in {len(applied)} transfer(s)")
        print(f"{args.wallet} now holds {float(dest_after['balance']):.6f} NRN")
        print(f"supply still {after} — invariant OK")

        record = {"at": now, "wallet": args.wallet, "transfers": applied,
                  "total": moved, "supply_after": str(after)}
        existing = []
        if os.path.exists(args.log):
            try:
                existing = json.load(open(args.log, encoding="utf-8"))
            except (ValueError, OSError):
                existing = []
        existing.append(record)
        with open(args.log, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
        print(f"logged -> {args.log}")
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
