"""coordinator/test_reconcile_stranded_escrow.py — run:
python -m coordinator.test_reconcile_stranded_escrow

A one-time repair still gets tested like a payment path, because it is one. The cases that
matter are the ones where running it would make things worse than leaving the money stranded:

  * escrow holding LESS than its live holds -- money backing in-flight requests is missing,
    which is a different and worse bug, and crediting someone would paper over it;
  * a live hold being mistaken for stranded NRN and paid out from under a request in flight;
  * running twice and paying twice.
"""
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinator import reconcile_stranded_escrow as rec     # noqa: E402

ok = fail = 0

WALLET = "w_d35c84ddd33ea857d74c29db22cd76a9"
OTHER = "node_a-cli-ee8b499f40a0"
# Shaped like the live ledger, with 0.056001 stranded in escrow. Sums to exactly 1e9.
LEDGER = [
    ("__emission_pool__", 600_000_099.609073, "bucket"),
    ("__founder__",       200_000_000.0,      "bucket"),
    ("__ecosystem__",     149_999_800.0,      "bucket"),
    ("__liquidity__",      50_000_000.0,      "bucket"),
    ("__escrow__",                  0.056001, "bucket"),
    ("node_a",                      9.011876, "node"),
    ("node_b",                      6.082124, "node"),
    ("node_c",                      8.107126, "node"),
    ("__coordinator__",             2.8678,   "node"),
    (WALLET,                       24.295,    "wallet"),
    (OTHER,                        24.971,    "wallet"),
    ("w_ef7ca46713e2bc4d0d627b69fe4aa660", 25.0, "wallet"),
]
HOLDS = [
    ("h1", OTHER,  0.029, "settled"),
    ("h2", WALLET, 0.159, "settled"),
    ("h3", WALLET, 0.195, "settled"),
    ("h4", WALLET, 0.330, "settled"),
    ("h5", WALLET, 0.160, "released"),
]


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def make_db(ledger=None, holds=None):
    path = os.path.join(tempfile.mkdtemp(prefix="neuron-reconcile-"), "neuron.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE ledger (
        node_id TEXT PRIMARY KEY, balance REAL NOT NULL DEFAULT 0,
        total_earned REAL NOT NULL DEFAULT 0, requests_served INTEGER NOT NULL DEFAULT 0,
        account_type TEXT NOT NULL DEFAULT 'node')""")
    con.execute("""CREATE TABLE holds (
        request_id TEXT PRIMARY KEY, wallet_id TEXT NOT NULL, amount REAL NOT NULL,
        created_at REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'held')""")
    con.executemany("INSERT INTO ledger (node_id, balance, account_type) VALUES (?,?,?)",
                    ledger if ledger is not None else LEDGER)
    con.executemany("INSERT INTO holds (request_id, wallet_id, amount, status) VALUES (?,?,?,?)",
                    holds if holds is not None else HOLDS)
    con.commit()
    con.close()
    return path


def balances(path):
    con = sqlite3.connect(path)
    out = {r[0]: r[1] for r in con.execute("SELECT node_id, balance FROM ledger")}
    con.close()
    return out


def run(db, *extra):
    log = os.path.join(os.path.dirname(db), "reconcile_log.json")
    code = rec.main(["--db", db, "--log", log, *extra])
    payload = {}
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            payload = json.load(f)
    return code, payload


def main():
    from decimal import Decimal
    print("\n-- the fixture")
    total = sum(Decimal(repr(b)) for _, b, _ in LEDGER)
    check("sums to exactly 1,000,000,000 NRN", total == Decimal(1_000_000_000), f"got {total}")

    print("\n-- dry run")
    db = make_db()
    before_bytes = open(db, "rb").read()
    code, log = run(db)
    check("dry run exits 0", code == 0)
    check("the database file is byte-for-byte unchanged", open(db, "rb").read() == before_bytes)
    check("it found exactly the stranded amount", log["stranded_nrn"] == 0.056001)
    check("it inferred the wallet with the most settled volume",
          log["recipient"] == WALLET and log["recipient_inferred"] is True)
    check("the attribution evidence names both wallets that settled",
          {e["wallet_id"] for e in log["attribution_evidence"]} == {WALLET, OTHER})
    check("nothing is marked applied", log["applied"] is False)
    check("the log carries the reason", "escrow leak" in log["reason"])

    print("\n-- execute")
    code, log = run(db, "--execute")
    b = balances(db)
    check("execute exits 0", code == 0)
    check("escrow is now empty", b["__escrow__"] == 0.0, f"escrow={b['__escrow__']}")
    check("the payer was credited exactly the stranded amount",
          abs(b[WALLET] - (24.295 + 0.056001)) < 1e-9, f"{b[WALLET]}")
    check("no other wallet was touched",
          (b[OTHER], b["w_ef7ca46713e2bc4d0d627b69fe4aa660"]) == (24.971, 25.0))
    check("no node or bucket was touched",
          (b["node_a"], b["__ecosystem__"], b["__emission_pool__"])
          == (9.011876, 149_999_800.0, 600_000_099.609073))
    check("the supply invariant still holds", log["invariant_after"] is True)
    check("the supply moved by no more than one float rounding",
          abs(Decimal(log["supply_drift"])) <= Decimal(log["supply_drift_budget"]),
          f"{log['supply_drift']} vs {log['supply_drift_budget']}")
    check("escrow matches its live holds afterwards",
          log["escrow_matches_live_holds_after"] is True)
    check("it is timestamped", bool(log["applied_at"]) and log["applied"] is True)

    print("\n-- it does not pay twice")
    snapshot = balances(db)
    code, log = run(db, "--execute")
    check("a second run reports nothing to reconcile", code == 0)
    check("and moves nothing", balances(db) == snapshot)

    print("\n-- live holds are not stranded NRN")
    # 4.0 in escrow, all of it backing a request still in flight (funded out of __founder__
    # so the fixture still sums to 1e9).
    def with_escrow(amount):
        delta = round(amount - 0.056001, 6)
        return [(n, amount if n == "__escrow__"
                 else (round(v - delta, 6) if n == "__founder__" else v), t)
                for n, v, t in LEDGER]

    ledger = with_escrow(4.0)
    holds = HOLDS + [("h-live", WALLET, 4.0, "held")]
    db2 = make_db(ledger, holds)
    snapshot = open(db2, "rb").read()
    code, log = run(db2, "--execute")
    check("an in-flight hold is not mistaken for stranded NRN", code == 0)
    check("and nothing was written", open(db2, "rb").read() == snapshot)
    check("the log says there was nothing to do", log == {} or log.get("applied") is not True)

    print("\n-- escrow SHORT of its live holds is refused, not patched")
    db3 = make_db(with_escrow(1.0), HOLDS + [("h-live", WALLET, 4.0, "held")])
    snapshot = open(db3, "rb").read()
    code, _ = run(db3, "--execute")
    check("refused when escrow holds less than its live holds", code == 2)
    check("and nothing was written", open(db3, "rb").read() == snapshot)

    print("\n-- other refusals")
    broken = [(n, v + (0.5 if n == "__founder__" else 0), t) for n, v, t in LEDGER]
    db4 = make_db(broken)
    snapshot = open(db4, "rb").read()
    code, _ = run(db4, "--execute")
    check("a ledger that does not sum to 1B is refused", code == 2)
    check("and nothing was written", open(db4, "rb").read() == snapshot)

    db5 = make_db()
    snapshot = open(db5, "rb").read()
    code, _ = run(db5, "--execute", "--to", "w_does_not_exist")
    check("crediting a wallet with no ledger row is refused", code == 2)
    check("and nothing was written", open(db5, "rb").read() == snapshot)

    print("\n-- --to overrides the inference")
    db6 = make_db()
    code, log = run(db6, "--execute", "--to", OTHER)
    check("the named wallet is credited instead",
          abs(balances(db6)[OTHER] - (24.971 + 0.056001)) < 1e-9)
    check("and the log records that it was not inferred",
          log["recipient_inferred"] is False and log["recipient"] == OTHER)

    print("\n-- backup")
    db7 = make_db()
    run(db7, "--execute", "--backup")
    backups = [f for f in os.listdir(os.path.dirname(db7)) if ".pre-reconcile-" in f]
    check("--backup leaves a pre-reconcile copy", len(backups) == 1)
    if backups:
        copy = os.path.join(os.path.dirname(db7), backups[0])
        check("and the copy still shows the stranded escrow",
              balances(copy)["__escrow__"] == 0.056001)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
