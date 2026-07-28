"""coordinator/genesis.py — one-time seeding of the fixed-supply genesis buckets
(TOKENOMICS.md §11.3). Idempotent — safe to call on every coordinator startup.

Approved reconciliation approach (founder decision, 2026-07-28): BACKDATE
__emission_pool__ by whatever node earnings already exist under the old unconditional-mint
model, rather than resetting existing balances. Nobody's already-earned NRN disappears.
"""
from coordinator import config, models


def seed_genesis():
    """If the genesis buckets already exist, this is a no-op (checked via __emission_pool__'s
    presence — all 4 buckets + __escrow__ are always seeded together in one transaction, so
    that row's existence is a reliable proxy for "genesis already ran"). On first run: seeds
    all 4 allocation buckets + __escrow__, backdating __emission_pool__ by whatever's already
    in `total_earned` across existing node rows, so the SUM(balance)==1e9 invariant holds from
    the very first read and no pre-existing node balance is disturbed. Returns True if this
    call actually seeded (vs. no-op)."""
    with models._db() as c:
        already = c.execute("SELECT 1 FROM ledger WHERE node_id=?",
                            (config.GENESIS_BUCKETS_EMISSION_ID,)).fetchone()
        if already:
            return False
        already_minted = (c.execute(
            "SELECT COALESCE(SUM(total_earned),0) AS s FROM ledger "
            "WHERE account_type='node'").fetchone())["s"]
        emission_seed = config.GENESIS_TOTAL_SUPPLY * 0.6 - already_minted
        if emission_seed < 0:
            raise RuntimeError(
                f"genesis seeding failed: already-minted node earnings ({already_minted}) "
                f"exceed the 600M emission pool allocation -- needs manual reconciliation, "
                f"refusing to silently under-seed")
        buckets = {
            config.GENESIS_BUCKETS_EMISSION_ID: emission_seed,
            config.GENESIS_BUCKETS_FOUNDER_ID: config.GENESIS_TOTAL_SUPPLY * 0.2,
            config.GENESIS_BUCKETS_ECOSYSTEM_ID: config.GENESIS_TOTAL_SUPPLY * 0.15,
            config.GENESIS_BUCKETS_LIQUIDITY_ID: config.GENESIS_TOTAL_SUPPLY * 0.05,
        }
        for bucket_id, amount in buckets.items():
            c.execute(
                "INSERT INTO ledger (node_id, balance, account_type) VALUES (?,?,'bucket')",
                (bucket_id, amount))
        c.execute(
            "INSERT OR IGNORE INTO ledger (node_id, balance, account_type) VALUES (?,0,'bucket')",
            (config.ESCROW_LEDGER_ID,))
        return True


def verify_invariant():
    """Raise loudly if SUM(ledger.balance) != 1,000,000,000. Meant to be called right after
    seeding, before the app is allowed to serve — a broken invariant should fail startup, not
    surface later as a silent accounting mismatch."""
    snap = models.supply_snapshot()
    if not snap["invariant_ok"]:
        raise RuntimeError(
            f"GENESIS INVARIANT BROKEN: total supply = {snap['total_supply']}, "
            f"expected exactly {config.GENESIS_TOTAL_SUPPLY}")
