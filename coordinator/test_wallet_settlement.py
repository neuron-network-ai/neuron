"""coordinator/test_wallet_settlement.py — fixed-supply ledger Phase 0 (Workstream B).
Run: python -m coordinator.test_wallet_settlement

Covers the wallet/escrow/hold/faucet/OAuth-identity primitives (coordinator/models.py) and
genesis seeding (coordinator/genesis.py). settle()/quote() coverage (coordinator/ledger.py)
is appended once that rewrite lands. Uses a throwaway DB — no HTTP server.
"""
import os
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_wallet_"), "w.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, genesis, ledger, models  # noqa: E402
from coordinator.main import (InferBody, ViolationBody, WalletOAuthBody,  # noqa: E402
                              infer, wallet_oauth, wallet_violation)

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    models.init_db()

    # ---- genesis: fresh DB seeds exactly 1e9, split across the 4 buckets + escrow=0 ---- #
    seeded = genesis.seed_genesis()
    check("genesis seeds on a fresh DB", seeded)
    snap = models.supply_snapshot()
    check("total supply is exactly 1,000,000,000", snap["total_supply"] == 1_000_000_000)
    check("invariant_ok", snap["invariant_ok"])
    check("emission pool = 600M on a fresh DB", snap["buckets"]["__emission_pool__"] == 600_000_000)
    check("escrow starts at 0", snap["buckets"][config.ESCROW_LEDGER_ID] == 0)
    check("re-seeding is a no-op", genesis.seed_genesis() is False)

    # ---- transfer: happy path + insufficient-balance fails closed (no partial effect) ---- #
    check("transfer moves funds", models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, "w1", 25.0))
    check("w1 has the transferred amount", models.get_ledger("w1")["balance"] == 25.0)
    before_w1 = models.get_ledger("w1")["balance"]
    check("insufficient transfer returns False", not models.transfer("w1", "w2", 1000.0))
    check("failed transfer leaves sender untouched", models.get_ledger("w1")["balance"] == before_w1)
    check("failed transfer never creates the recipient", models.get_ledger("w2") is None)
    check("zero-amount transfer is a no-op success", models.transfer("w1", "w2", 0))

    # ---- hold/release: money moves to escrow and back, invariant holds throughout ---- #
    check("hold succeeds and debits the wallet", models.hold("req-1", "w1", 5.0))
    check("wallet debited by hold amount", models.get_ledger("w1")["balance"] == 20.0)
    check("escrow credited by hold amount",
          models.get_ledger(config.ESCROW_LEDGER_ID)["balance"] == 5.0)
    check("invariant holds while a request is in-flight (money in escrow, not lost)",
          models.supply_snapshot()["invariant_ok"])
    check("release returns the hold to the wallet", models.release_hold("req-1"))
    check("wallet restored after release", models.get_ledger("w1")["balance"] == 25.0)
    check("escrow drained after release",
          models.get_ledger(config.ESCROW_LEDGER_ID)["balance"] == 0.0)
    check("double release is a safe no-op", not models.release_hold("req-1"))
    check("hold on insufficient balance fails, no escrow leak",
          not models.hold("req-broke", "w1", 999.0)
          and models.get_ledger(config.ESCROW_LEDGER_ID)["balance"] == 0.0)

    # ---- TTL sweep: an abandoned hold gets returned automatically ---- #
    check("stale hold created", models.hold("req-stale", "w1", 3.0))
    with models._db() as c:
        c.execute("UPDATE holds SET created_at=0 WHERE request_id='req-stale'")
    released = models.release_stale_holds(ttl_s=1)
    check("TTL sweep releases exactly the stale hold", released == ["req-stale"])
    check("wallet restored after TTL release", models.get_ledger("w1")["balance"] == 25.0)
    check("a fresh hold survives the same sweep (not stale)", models.hold("req-fresh", "w1", 1.0)
          and models.release_stale_holds(ttl_s=600) == [])
    models.release_hold("req-fresh")

    # ---- faucet: exactly once per wallet ---- #
    check("faucet grants the configured amount to a fresh wallet",
          models.claim_faucet("faucet-w1", config.FAUCET_AMOUNT_NRN)
          and models.get_ledger("faucet-w1")["balance"] == config.FAUCET_AMOUNT_NRN)
    check("faucet is one-time per wallet", not models.claim_faucet("faucet-w1", config.FAUCET_AMOUNT_NRN))

    # ---- OAuth identity -> wallet mapping: stable, faucet ships with the first login ---- #
    w, is_new = models.wallet_for_oauth("google", "sub-123", "a@b.com")
    check("first OAuth login creates a new wallet", is_new)
    check("new wallet auto-receives the faucet", models.get_ledger(w)["balance"] == config.FAUCET_AMOUNT_NRN)
    w2, is_new2 = models.wallet_for_oauth("google", "sub-123", "a@b.com")
    check("repeat login returns the SAME wallet_id", w2 == w)
    check("repeat login does not re-grant the faucet", not is_new2)
    w3, _ = models.wallet_for_oauth("github", "sub-123", "a@b.com")
    check("same external_id on a DIFFERENT provider gets a DIFFERENT wallet",
          w3 != w)  # provider+external_id is the composite key, not external_id alone

    # ---- invariant holds after everything above ---- #
    final = models.supply_snapshot()
    check("SUM(balance) still exactly 1e9 after all operations", final["invariant_ok"])

    settle_tests()

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


def N(node_id, ls, le, head=False, eligible=True):
    return {"node_id": node_id, "layer_start": ls, "layer_end": le,
           "head_ms": 38.3 if head else 0, "eligible": eligible}


def settle_tests():
    # ---- layer_equivalents: self-adjusting weight, head bonus only for the head-holder ---- #
    check("layer_equivalents: plain node = layer count", ledger.layer_equivalents(N("a", 0, 9)) == 10)
    check("layer_equivalents: head-holder gets the bonus",
          ledger.layer_equivalents(N("a", 0, 9, head=True)) == 10 + config.HEAD_BONUS_LE)

    # ---- quote: worst-case estimate from max_tokens ---- #
    q = ledger.quote(max_tokens=100, input_tokens_estimate=20)
    check("quote matches the documented formula",
          q == round((20 * config.INPUT_WEIGHT + 100) / 1000.0 * config.PRICE_PER_1K_WEIGHTED, 6))

    # ---- settle: full lifecycle, real numbers, LE-proportional split + head bonus ---- #
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, "settle-wallet", 10.0)
    hold_amount = ledger.quote(max_tokens=100, input_tokens_estimate=20)   # 0.12
    check("hold succeeds for the quoted amount", models.hold("settle-req-1", "settle-wallet", hold_amount))
    before_escrow = models.get_ledger(config.ESCROW_LEDGER_ID)["balance"]

    plan = [N("node_a", 0, 9, head=True), N("node_b", 19, 27)]   # LE = 15, 9 -> total 24
    breakdown = ledger.settle("settle-req-1", "settle-wallet", hold_amount,
                              prompt_tokens=20, completion_tokens=50, plan_nodes=plan)
    # weighted = 50 + 20*1.0 = 70 -> actual = 0.07; pool = 0.063; fee = 0.007; refund = 0.05
    check("node_a (head holder) paid its LE-proportional share",
          breakdown["node_a"] == 0.039375)
    check("node_b paid its LE-proportional share", breakdown["node_b"] == 0.023625)
    check("node payouts sum to exactly the 90% pool",
          round(breakdown["node_a"] + breakdown["node_b"], 6) == 0.063)
    check("coordinator fee is exactly 10% of actual cost", breakdown[config.COORDINATOR_LEDGER_ID] == 0.007)
    check("unused hold is refunded to the wallet", breakdown["__refund__"] == 0.05)
    check("wallet balance reflects hold minus actual cost (not minus the full hold)",
          round(models.get_ledger("settle-wallet")["balance"], 6) == round(10.0 - 0.07, 6))
    check("escrow fully drained back to its pre-hold level after settlement",
          models.get_ledger(config.ESCROW_LEDGER_ID)["balance"] == before_escrow - hold_amount)
    check("settled hold is marked settled, not left held",
          models.get_hold("settle-req-1")["status"] == "settled")
    check("node_a's requests_served counter incremented",
          models.get_ledger("node_a")["requests_served"] == 1)

    # ---- settle: an ineligible node in the plan earns nothing, doesn't break the split ---- #
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, "settle-wallet-2", 10.0)
    hold2 = ledger.quote(max_tokens=100, input_tokens_estimate=0)
    models.hold("settle-req-2", "settle-wallet-2", hold2)
    plan2 = [N("node_c", 0, 9), N("node_probation", 10, 18, eligible=False)]
    bd2 = ledger.settle("settle-req-2", "settle-wallet-2", hold2,
                        prompt_tokens=0, completion_tokens=100, plan_nodes=plan2)
    check("ineligible node in the plan earns nothing", "node_probation" not in bd2)
    check("eligible node gets the FULL pool (ineligible node's share isn't stranded)",
          bd2["node_c"] == round(0.1 * 0.9, 6))

    # ---- settle: actual cost is capped at the hold amount even if usage implausibly exceeds it ---- #
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, "settle-wallet-3", 10.0)
    tiny_hold = 0.01
    models.hold("settle-req-3", "settle-wallet-3", tiny_hold)
    bd3 = ledger.settle("settle-req-3", "settle-wallet-3", tiny_hold,
                        prompt_tokens=0, completion_tokens=100000, plan_nodes=[N("node_d", 0, 27)])
    check("actual cost never exceeds the held amount",
          round(bd3["node_d"] + bd3[config.COORDINATOR_LEDGER_ID], 6) == tiny_hold)
    check("no refund when the full hold was consumed", "__refund__" not in bd3)

    check("invariant still holds after the full settle test suite", models.supply_snapshot()["invariant_ok"])

    wallet_oauth_endpoint_tests()
    network_stats_scoping_test()
    moderation_violation_tests()


def network_stats_scoping_test():
    """Found live on the production deploy, TWICE, in the same stat:
    (1) total_nrn_distributed jumped by exactly a faucet grant amount -- it summed
        total_earned for every non-coordinator ledger row, which now includes wallets.
    (2) the first fix (account_type='node' alone) OVER-corrected: __coordinator__'s row was
        never reclassified off the schema default, so it's STILL account_type='node' -- the
        fee it earns started leaking back in. Needs BOTH account_type='node' AND the explicit
        __coordinator__ exclusion."""
    before = models.network_stats()["total_nrn_distributed"]
    models.claim_faucet("stats-scoping-wallet", 25.0)
    after_faucet = models.network_stats()["total_nrn_distributed"]
    check("faucet grant to a wallet does NOT move total_nrn_distributed", after_faucet == before)

    models.credit(config.COORDINATOR_LEDGER_ID, 4.0, count_request=False)
    after_fee = models.network_stats()["total_nrn_distributed"]
    check("a coordinator fee credit does NOT move total_nrn_distributed", after_fee == after_faucet)

    models.credit("node_c", 3.0, count_request=True)   # simulate a node earning (account_type='node' by default)
    after_node = models.network_stats()["total_nrn_distributed"]
    check("a real node earning DOES move total_nrn_distributed",
          round(after_node - after_fee, 6) == 3.0)


def wallet_oauth_endpoint_tests():
    # ---- POST /wallet/oauth: gated by the shared secret, wraps wallet_for_oauth ---- #
    try:
        wallet_oauth(WalletOAuthBody(provider="google", external_id="ep-1", email="e@p.com"),
                    x_wallet_link_secret=None)
        check("missing X-Wallet-Link-Secret rejected (401)", False)
    except HTTPException as e:
        check("missing X-Wallet-Link-Secret rejected (401)", e.status_code == 401)

    try:
        wallet_oauth(WalletOAuthBody(provider="google", external_id="ep-1", email="e@p.com"),
                    x_wallet_link_secret="wrong-secret")
        check("wrong X-Wallet-Link-Secret rejected (401)", False)
    except HTTPException as e:
        check("wrong X-Wallet-Link-Secret rejected (401)", e.status_code == 401)

    resp = wallet_oauth(WalletOAuthBody(provider="google", external_id="ep-1", email="e@p.com"),
                        x_wallet_link_secret=config.WALLET_LINK_SECRET)
    check("correct secret creates a wallet", resp["is_new"] is True)
    check("new wallet has the faucet balance",
          models.get_ledger(resp["wallet_id"])["balance"] == config.FAUCET_AMOUNT_NRN)

    resp2 = wallet_oauth(WalletOAuthBody(provider="google", external_id="ep-1", email="e@p.com"),
                         x_wallet_link_secret=config.WALLET_LINK_SECRET)
    check("repeat call for the same identity returns the same wallet, not new",
          resp2["wallet_id"] == resp["wallet_id"] and resp2["is_new"] is False)


def moderation_violation_tests():
    # ---- models.record_violation / wallet_moderation_status ---- #
    check("unknown wallet has no violations, not banned",
          models.wallet_moderation_status("never-seen-wallet") == {"violation_count": 0, "banned": False})

    r1 = models.record_violation("w-offender", "in", "weapons_cbrn")
    check("1st violation recorded, not yet banned", r1 == {"violation_count": 1, "banned": False})
    r2 = models.record_violation("w-offender", "out", "self_harm")
    check("2nd violation recorded, still below threshold",
          r2 == {"violation_count": 2, "banned": False})
    check("threshold is 3 by default", config.MODERATION_BAN_THRESHOLD == 3)
    r3 = models.record_violation("w-offender", "in", "weapons_cbrn")
    check("3rd violation crosses the threshold -> banned", r3 == {"violation_count": 3, "banned": True})
    status = models.wallet_moderation_status("w-offender")
    check("wallet_moderation_status reflects the ban", status == {"violation_count": 3, "banned": True})

    r4 = models.record_violation("w-offender", "in", "weapons_cbrn")
    check("a banned wallet's count keeps climbing, ban stays True",
          r4 == {"violation_count": 4, "banned": True})

    check("a single-violation wallet is NOT banned (no false-positive lockout)",
          not models.record_violation("w-oneoff", "in", "self_harm")["banned"])

    # ---- POST /wallet/{id}/violation: same shared-secret gate as /wallet/oauth ---- #
    try:
        wallet_violation("w-offender", ViolationBody(direction="in", category="x"),
                         x_wallet_link_secret=None)
        check("violation endpoint: missing secret rejected (401)", False)
    except HTTPException as e:
        check("violation endpoint: missing secret rejected (401)", e.status_code == 401)

    try:
        wallet_violation("w-offender", ViolationBody(direction="in", category="x"),
                         x_wallet_link_secret="wrong")
        check("violation endpoint: wrong secret rejected (401)", False)
    except HTTPException as e:
        check("violation endpoint: wrong secret rejected (401)", e.status_code == 401)

    before = models.get_ledger("w-fresh-endpoint")
    check("wallet doesn't exist yet", before is None)
    resp = wallet_violation("w-fresh-endpoint", ViolationBody(direction="out", category="cat"),
                            x_wallet_link_secret=config.WALLET_LINK_SECRET)
    check("violation endpoint creates the wallet row on first violation (ensure_account)",
          resp == {"wallet_id": "w-fresh-endpoint", "violation_count": 1, "banned": False})

    # ---- /infer refuses a banned wallet BEFORE any hold/chain-building happens ---- #
    # These wallets are minted through wallet_for_oauth rather than ensure_account: /infer now
    # also refuses any wallet with no Google/GitHub login behind it, so an invented wallet id
    # would 403 for the WRONG reason and the ban assertion below would pass vacuously.
    banned_wallet, _ = models.wallet_for_oauth("test", "banned-user", "banned@example.com")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, banned_wallet, 100.0)
    for _ in range(config.MODERATION_BAN_THRESHOLD):
        models.record_violation(banned_wallet, "in", "x")
    check("setup: wallet is now banned", models.wallet_moderation_status(banned_wallet)["banned"])
    balance_before = models.get_ledger(banned_wallet)["balance"]
    try:
        infer(InferBody(prompt="hello", max_tokens=10, wallet_id=banned_wallet))
        check("/infer rejects a banned wallet (403)", False)
    except HTTPException as e:
        check("/infer rejects a banned wallet (403)",
              e.status_code == 403 and "blocked for repeated" in str(e.detail))
    check("a rejected banned-wallet request never touches its balance (no hold attempted)",
          models.get_ledger(banned_wallet)["balance"] == balance_before)

    # ---- /infer also refuses a wallet with no login behind it (the anonymous-access gate) ---- #
    models.ensure_account("w-no-login", "wallet")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, "w-no-login", 100.0)
    try:
        infer(InferBody(prompt="hello", max_tokens=10, wallet_id="w-no-login"))
        check("/infer rejects a funded wallet that never logged in", False)
    except HTTPException as e:
        check("/infer rejects a funded wallet that never logged in",
              e.status_code == 403 and "not linked" in str(e.detail))

    # a well-funded, logged-in, NOT-banned wallet clears both gates (whatever it fails on next --
    # an empty test network -- is unrelated to moderation, proving the gate isn't over-firing)
    clean_wallet, _ = models.wallet_for_oauth("test", "clean-user", "clean@example.com")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, clean_wallet, 100.0)
    try:
        infer(InferBody(prompt="hello", max_tokens=10, wallet_id=clean_wallet))
        check("clean wallet passes the moderation gate", True)
    except HTTPException as e:
        check("clean wallet passes the moderation gate (fails later, not with 403)",
              e.status_code != 403)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
