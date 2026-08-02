"""coordinator/test_prune_test_accounts.py — run: python -m coordinator.test_prune_test_accounts

A script that moves money gets tested for what it REFUSES to do, not just what it does. The
cases below are the ones where a mistake is expensive and quiet:

  * a dry run that is not actually dry (this one operates on the real live ledger, so "it only
    printed" has to be true at the file level, not just in intent);
  * an account nobody classified being swept along with the obvious test wallets;
  * the supply invariant surviving the operation -- both that it still holds afterwards, and
    the stricter property that the total did not MOVE, which a tolerance-based check alone
    would happily hide;
  * a partial apply. If any single transfer fails, none of them may land.
"""
import json
import os
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinator import prune_test_accounts as prune     # noqa: E402

ok = fail = 0

# Shaped like the live ledger: the four buckets + escrow, the dev trio, the coordinator's fee
# row, and the eight faucet-funded test identities. Sums to exactly 1,000,000,000.
LEDGER = [
    ("__emission_pool__", 599_999_971.555973, "bucket"),
    ("__founder__",       200_000_000.0,      "bucket"),
    ("__ecosystem__",     149_999_800.0,      "bucket"),
    ("__liquidity__",      50_000_000.0,      "bucket"),
    ("__escrow__",                  0.0,      "bucket"),
    ("node_a",                      9.011876, "node"),
    ("node_b",                      6.082124, "node"),
    ("node_c",                      8.107126, "node"),
    ("__coordinator__",             2.8678,   "node"),
    ("attacker-demo-1",            25.0,      "wallet"),
    ("attacker-demo-2",            25.0,      "wallet"),
    ("probe-only",                 25.0,      "wallet"),
    ("live-verify-wallet",         25.0,      "wallet"),
    ("node_a-cli-e79214baa6df",    25.0,      "wallet"),
    ("node_a-cli-ee8b499f40a0",    24.971,    "wallet"),
    ("w_d35c84ddd33ea857d74c29db22cd76a9", 24.295, "wallet"),
    ("w_ef7ca46713e2bc4d0d627b69fe4aa660", 25.0,   "wallet"),
    ("agent-optinovate",            2.025002, "node"),   # dev install of the packaged agent
    ("stranger-test-win",           0.208607, "node"),   # rehearsal of the stranger join path
    ("node-b-optiplex",             0.187746, "node"),   # the OptiPlex under an earlier id
    ("node-c-pavilion",             0.187746, "node"),   # the Pavilion under an earlier id
    ("unknown-node-xyz",            0.5,      "node"),   # matches no rule -- must be reported
    ("w_spent_out",                 0.0,      "wallet"),  # prune target, already empty
]
PRUNED = ["attacker-demo-1", "attacker-demo-2", "probe-only", "live-verify-wallet",
          "node_a-cli-e79214baa6df", "node_a-cli-ee8b499f40a0",
          "w_d35c84ddd33ea857d74c29db22cd76a9", "w_ef7ca46713e2bc4d0d627b69fe4aa660",
          "stranger-test-win", "agent-optinovate"]
KEPT = ["node_a", "node_b", "node_c", "__coordinator__",
        "node-b-optiplex", "node-c-pavilion"]


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def make_db(rows=None, extra=None):
    path = os.path.join(tempfile.mkdtemp(prefix="neuron-prune-"), "neuron.db")
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE ledger (
        node_id TEXT PRIMARY KEY, balance REAL NOT NULL DEFAULT 0,
        total_earned REAL NOT NULL DEFAULT 0, requests_served INTEGER NOT NULL DEFAULT 0,
        account_type TEXT NOT NULL DEFAULT 'node')""")
    con.executemany("INSERT INTO ledger (node_id, balance, account_type) VALUES (?,?,?)",
                    rows if rows is not None else LEDGER)
    for row in (extra or []):
        con.execute("INSERT INTO ledger (node_id, balance, account_type) VALUES (?,?,?)", row)
    con.commit()
    con.close()
    return path


def balances(path):
    con = sqlite3.connect(path)
    out = {r[0]: r[1] for r in con.execute("SELECT node_id, balance FROM ledger")}
    con.close()
    return out


def run(db, *extra):
    log = os.path.join(os.path.dirname(db), "prune_log.json")
    code = prune.main(["--db", db, "--log", log, *extra])
    payload = {}
    if os.path.exists(log):
        with open(log, encoding="utf-8") as f:
            payload = json.load(f)
    return code, payload


def main():
    print("\n-- the test ledger")
    db = make_db()
    from decimal import Decimal
    total = sum(Decimal(repr(b)) for _, b, _ in LEDGER)
    check("sums to exactly 1,000,000,000 NRN", total == Decimal(1_000_000_000), f"got {total}")

    print("\n-- dry run is actually dry")
    before_bytes = open(db, "rb").read()
    code, log = run(db)
    check("dry run exits 0", code == 0)
    check("the database file is byte-for-byte unchanged", open(db, "rb").read() == before_bytes)
    check("the log records it as a dry run", log["dry_run"] is True)
    check("it planned every test identity", sorted(t["account_id"] for t in log["transfers"])
          == sorted(PRUNED), str([t["account_id"] for t in log["transfers"]]))
    check("no transfer is marked applied", all(not t["applied"] for t in log["transfers"]))
    check("every planned transfer carries a reason",
          all(t["reason"] and t["to"] == "__ecosystem__" for t in log["transfers"]))

    print("\n-- classification")
    check("real hardware and earned fees are kept -- including the older machine ids",
          set(KEPT) <= {k["account_id"] for k in log["kept"]})
    check("genesis buckets are kept",
          {"__emission_pool__", "__founder__", "__ecosystem__", "__liquidity__"}
          <= {k["account_id"] for k in log["kept"]})
    check("accounts matching no rule are reported, not swept",
          sorted(u["account_id"] for u in log["unclassified"]) == ["unknown-node-xyz"])
    check("an already-empty prune target is not a transfer",
          "w_spent_out" not in {t["account_id"] for t in log["transfers"]}
          and log["totals"]["already_empty"] == 1)
    check("prefix rules match both families",
          {"node_a-cli-e79214baa6df", "w_d35c84ddd33ea857d74c29db22cd76a9"}
          <= {t["account_id"] for t in log["transfers"]})
    check("node_a is NOT caught by the node_a-cli- prefix",
          "node_a" not in {t["account_id"] for t in log["transfers"]})

    print("\n-- execute")
    code, log = run(db, "--execute")
    b = balances(db)
    check("execute exits 0", code == 0)
    check("every test identity is now zero", all(b[a] == 0 for a in PRUNED),
          str({a: b[a] for a in PRUNED if b[a] != 0}))
    check("the dev trio is untouched",
          (b["node_a"], b["node_b"], b["node_c"]) == (9.011876, 6.082124, 8.107126))
    check("the same machines under their older ids are untouched",
          (b["node-b-optiplex"], b["node-c-pavilion"]) == (0.187746, 0.187746))
    check("the coordinator's fees are untouched", b["__coordinator__"] == 2.8678)
    check("unclassified accounts are untouched", b["unknown-node-xyz"] == 0.5)
    check("the two dev-artifact nodes WERE pruned",
          (b["agent-optinovate"], b["stranger-test-win"]) == (0, 0))
    check("the other genesis buckets are untouched",
          (b["__emission_pool__"], b["__founder__"], b["__liquidity__"])
          == (599_999_971.555973, 200_000_000.0, 50_000_000.0))
    starting = {n: v for n, v, _ in LEDGER}
    moved = sum(starting[a] for a in PRUNED)
    import math
    want = float(Decimal("149999800.0") + sum(Decimal(repr(starting[a])) for a in PRUNED))
    check("__ecosystem__ grew by exactly what was pruned",
          abs(b["__ecosystem__"] - want) <= math.ulp(want),
          f"{b['__ecosystem__']} vs {want}")
    check("the supply invariant still holds", log["invariant_after"] is True)
    # Not "drift == 0": one float addition into a ~1.5e8 balance cannot be exact. It must be
    # within a single rounding of that balance -- anything more means NRN was created or
    # destroyed rather than rounded. (Crediting per-account instead of once drifted 3e-8.)
    check("the supply moved by no more than one float rounding",
          abs(Decimal(log["supply_drift"])) <= Decimal(log["supply_drift_budget"]),
          f"drift {log['supply_drift']} vs budget {log['supply_drift_budget']}")
    check("every applied transfer has a timestamp",
          all(t["applied"] and t["timestamp"] for t in log["transfers"]))

    print("\n-- running it twice")
    code, log2 = run(db, "--execute")
    check("a second run finds nothing left to prune",
          code == 0 and log2["transfers"] == [])
    check("and moves nothing", balances(db) == b)

    print("\n-- explicit overrides")
    db2 = make_db()
    code, log = run(db2, "--execute", "--prune-also", "unknown-node-xyz")
    check("--prune-also sweeps an unclassified account",
          balances(db2)["unknown-node-xyz"] == 0
          and "unknown-node-xyz" in {t["account_id"] for t in log["transfers"]})
    check("and the reason says why it was included",
          any(t["reason"] == "pruned by --prune-also"
              for t in log["transfers"] if t["account_id"] == "unknown-node-xyz"))
    db3 = make_db()
    code, log = run(db3, "--execute", "--keep-also", "probe-only")
    check("--keep-also protects a default prune target",
          balances(db3)["probe-only"] == 25.0
          and "probe-only" not in {t["account_id"] for t in log["transfers"]})
    try:
        prune.main(["--db", db3, "--prune-also", "dup", "--keep-also", "dup"])
        check("naming an account in both lists is refused", False)
    except SystemExit as e:
        check("naming an account in both lists is refused", "both name" in str(e))

    print("\n-- refusals")
    broken = [(n, v + (0.5 if n == "__founder__" else 0), t) for n, v, t in LEDGER]
    db4 = make_db(broken)
    snapshot = open(db4, "rb").read()
    code, _ = run(db4, "--execute")
    check("a ledger that does not sum to 1B is refused", code == 2)
    check("and nothing was written", open(db4, "rb").read() == snapshot)

    held = [(n, 3.0 if n == "__escrow__" else (v - 3.0 if n == "__founder__" else v), t)
            for n, v, t in LEDGER]
    db5 = make_db(held)
    snapshot = open(db5, "rb").read()
    code, _ = run(db5, "--execute")
    check("a non-zero escrow is refused (requests in flight)", code == 2)
    check("and nothing was written", open(db5, "rb").read() == snapshot)

    print("\n-- a failing transfer must not half-apply")
    db6 = make_db()
    original = balances(db6)

    real_connect = sqlite3.connect
    calls = {"n": 0}

    class FailingConn:
        """Lets the first two writes through, then fails -- the shape of a mid-run error."""
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a):
            if sql.startswith("UPDATE ledger SET balance=0"):
                calls["n"] += 1
                if calls["n"] > 2:
                    raise sqlite3.OperationalError("simulated disk failure")
            return self._inner.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def __enter__(self):
            self._inner.__enter__()
            return self

        def __exit__(self, *exc):
            return self._inner.__exit__(*exc)

    def patched(path, *a, **kw):
        conn = real_connect(path, *a, **kw)
        return conn if "mode=ro" in str(path) else FailingConn(conn)

    sqlite3.connect = patched
    try:
        code, _ = run(db6, "--execute")
    finally:
        sqlite3.connect = real_connect
    check("a mid-run failure exits non-zero", code == 1)
    check("and rolls back completely -- no partial prune", balances(db6) == original,
          str({k: (original[k], balances(db6)[k])
               for k in original if original[k] != balances(db6)[k]}))

    print("\n-- backup flag")
    db7 = make_db()
    run(db7, "--execute", "--backup")
    backups = [f for f in os.listdir(os.path.dirname(db7)) if ".pre-prune-" in f]
    check("--backup leaves a pre-prune copy", len(backups) == 1)
    if backups:
        copy = os.path.join(os.path.dirname(db7), backups[0])
        check("and the copy still holds the un-pruned balances",
              balances(copy)["probe-only"] == 25.0)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
