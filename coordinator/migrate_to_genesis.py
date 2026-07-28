"""coordinator/migrate_to_genesis.py — run ONCE, manually, before deploying the fixed-supply
ledger (Workstream B) to a live coordinator that already has real node earnings under the old
unconditional-mint model. genesis.seed_genesis() is idempotent (safe to re-run — no-ops if
already seeded), but this script is meant as a deliberate, observed one-time operation you run
and read the output of, not something that happens silently on a routine restart.

Run:  python -m coordinator.migrate_to_genesis
"""
from coordinator import genesis, models


def main():
    models.init_db()
    seeded = genesis.seed_genesis()
    genesis.verify_invariant()
    snap = models.supply_snapshot()

    print("Genesis buckets seeded." if seeded else "Genesis buckets already existed -- no-op.")
    for bucket, balance in snap["buckets"].items():
        print(f"  {bucket:24s} {balance:>18,.6f}")
    print(f"  {'TOTAL SUPPLY':24s} {snap['total_supply']:>18,.6f}")
    print(f"  invariant_ok = {snap['invariant_ok']}")


if __name__ == "__main__":
    main()
