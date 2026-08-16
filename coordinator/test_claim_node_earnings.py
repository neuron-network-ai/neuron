"""coordinator/test_claim_node_earnings.py — run: python -m coordinator.test_claim_node_earnings

`claim_node_earnings.py` moves real money between real accounts on the live ledger, and the
ledger has no transactions table to reconstruct a mistake from. So the cases below are weighted
toward what it must REFUSE, not what it does on the happy path.

The refusals matter more than the transfers because every one of them fails in the same
direction: `models.transfer` ends with `INSERT OR IGNORE INTO ledger (node_id)`, so paying a
destination that does not exist does not error — it silently creates the account and succeeds.
A typo'd wallet id would therefore report "moved 213.50 NRN" and put it somewhere nobody can
ever sign in as, which is strictly worse than leaving it on the node.

Also pinned here: a swept node's `total_earned` must NOT fall. `network_stats` sums
`total_earned` over `account_type='node'` to report what the network has distributed for
compute, and a later transfer does not un-earn it.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-claim-"), "t.db")

from coordinator import claim_node_earnings as claim              # noqa: E402
from coordinator import config, genesis, models                   # noqa: E402

DB = os.environ["NEURON_DB"]
NODE = "test-node-pavilion"
NODE2 = "test-node-driver"
WALLET = "w_" + "a" * 32
EARNED = 213.4954

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def balance(account_id):
    row = models.get_ledger(account_id)
    return None if row is None else float(row["balance"])


def earned(account_id):
    row = models.get_ledger(account_id)
    return None if row is None else float(row["total_earned"])


def run(*argv):
    """Invoke the script exactly as an operator would, returning its exit code."""
    return claim.main(list(argv) + ["--db", DB, "--log", DB + ".log", "--no-backup"])


def setup():
    models.init_db()
    genesis.seed_genesis()
    # Earn the way a node really earns: a transfer OUT of the emission pool, so the 1e9
    # invariant holds from the start and the fixture cannot pass by breaking it.
    models.transfer(config.GENESIS_BUCKETS_EMISSION_ID, NODE, EARNED)
    models.transfer(config.GENESIS_BUCKETS_EMISSION_ID, NODE2, 12.5)
    models.ensure_account(WALLET, "wallet")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, WALLET, 25.0)   # the faucet


def main():
    setup()
    print("\n-- fixture --")
    check("node holds its earnings", abs(balance(NODE) - EARNED) < 1e-9)
    check("wallet holds only the faucet", abs(balance(WALLET) - 25.0) < 1e-9)
    check("supply invariant holds before anything moves",
          models.supply_snapshot()["invariant_ok"])

    print("\n-- it refuses what would strand the money --")
    code = run("--node", NODE, "--wallet", "w_typo_nobody_can_sign_in_as", "--execute")
    check("refuses a destination with no ledger row", code == 4)
    check("...and moved nothing", abs(balance(NODE) - EARNED) < 1e-9)
    check("...and did NOT create the typo'd account", models.get_ledger(
        "w_typo_nobody_can_sign_in_as") is None,
        "transfer()'s INSERT OR IGNORE is exactly the trap this check exists for")

    code = run("--node", NODE, "--wallet", NODE2, "--execute")
    check("refuses a destination that is a node, not a wallet", code == 4)
    check("...and moved nothing", abs(balance(NODE2) - 12.5) < 1e-9)

    code = run("--node", "node-that-never-existed", "--wallet", WALLET, "--execute")
    check("refuses a source with no ledger row", code == 4)

    code = run("--node", NODE, "--wallet", NODE, "--execute")
    check("refuses sweeping an account into itself", code == 4)

    code = run("--node", NODE, "--wallet", WALLET, "--amount", "9999", "--execute")
    check("refuses --amount larger than the balance", code == 4,
          "a partial sweep must be asked for, never silently substituted")
    check("...and moved nothing", abs(balance(NODE) - EARNED) < 1e-9)

    print("\n-- dry run is genuinely dry --")
    code = run("--node", NODE, "--wallet", WALLET)
    check("dry run exits 0", code == 0)
    check("dry run moved nothing from the node", abs(balance(NODE) - EARNED) < 1e-9)
    check("dry run moved nothing to the wallet", abs(balance(WALLET) - 25.0) < 1e-9)

    print("\n-- the sweep --")
    before_earned = earned(NODE)
    code = run("--node", NODE, "--wallet", WALLET, "--execute")
    check("sweep exits 0", code == 0)
    check("node balance is now zero", abs(balance(NODE)) < 1e-9)
    check("wallet gained exactly the balance", abs(balance(WALLET) - (25.0 + EARNED)) < 1e-9)
    check("supply invariant still holds", models.supply_snapshot()["invariant_ok"])
    check("the node's total_earned did NOT fall", abs(earned(NODE) - before_earned) < 1e-9,
          "network_stats reports lifetime compute earnings; a transfer does not un-earn them")
    check("total_nrn_distributed is unchanged by the sweep",
          abs(models.network_stats()["total_nrn_distributed"] - (EARNED + 12.5)) < 1e-3)

    print("\n-- re-running is safe --")
    code = run("--node", NODE, "--wallet", WALLET, "--execute")
    check("a second sweep of an empty node exits 0", code == 0)
    check("...and the wallet is unchanged", abs(balance(WALLET) - (25.0 + EARNED)) < 1e-9)

    print("\n-- multiple nodes in one run --")
    code = run("--node", NODE, "--node", NODE2, "--wallet", WALLET, "--execute")
    check("sweeps every named node", code == 0 and abs(balance(NODE2)) < 1e-9)
    check("wallet holds the lot", abs(balance(WALLET) - (25.0 + EARNED + 12.5)) < 1e-9)
    check("supply invariant survives a multi-node sweep",
          models.supply_snapshot()["invariant_ok"])

    print("\n-- it will not run on a broken ledger --")
    with models._db() as c:                       # deliberately break the invariant
        c.execute("UPDATE ledger SET balance=balance+1000 WHERE node_id=?", (NODE2,))
    check("invariant is now broken (fixture)", not models.supply_snapshot()["invariant_ok"])
    code = run("--node", NODE2, "--wallet", WALLET, "--execute")
    check("refuses to sweep across an unbalanced ledger", code == 3,
          "a second movement over a discrepancy buries it")
    check("...and moved nothing", abs(balance(NODE2) - 1000.0) < 1e-9)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
