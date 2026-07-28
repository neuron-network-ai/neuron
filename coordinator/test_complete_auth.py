"""Completion-auth tests ([P12]) — run: python -m coordinator.test_complete_auth

/infer/{id}/complete must be safe on an open, stranger-reachable coordinator:
  - /infer issues a per-request complete_token and records the chain it chose;
  - /complete requires that token (wrong/missing -> 401, no credit, still pending);
  - settlement pays the RECORDED plan, never the caller-supplied node_ids, so a
    completion cannot redirect NRN to an arbitrary node or to the unchosen replica;
  - tokens_generated is clamped to max_tokens; 404/409 still hold.
Uses a throwaway DB and the real endpoint/ledger/models code — no HTTP server.
"""
import os
import tempfile
import threading

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_p12_"), "p12.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, genesis, models  # noqa: E402
from coordinator.main import CompleteBody, InferBody, RegisterBody, complete, infer, register  # noqa: E402

SECRET = config.REGISTRATION_SECRET
S1, S2, N = 10, 19, config.TOTAL_LAYERS
WALLET = "test-wallet"
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def reg(node_id, ls, le):
    register(RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=50000 + le,
                          layer_start=ls, layer_end=le, cores=4, ram_gb=8),
             x_register_secret=SECRET)


def bal(nid):
    row = models.get_ledger(nid)
    return row["balance"] if row else None


def main():
    global WALLET
    models.init_db()
    genesis.seed_genesis()
    # /infer now refuses any wallet that isn't backed by a real Google/GitHub login (that gate
    # is what makes a ban un-evadable -- see coordinator/test_identity_gate.py), so this test's
    # spending wallet has to be a genuine OAuth-minted one rather than an invented string.
    WALLET, _ = models.wallet_for_oauth("test", "complete-auth-user", "test@example.com")
    models.transfer(config.GENESIS_BUCKETS_ECOSYSTEM_ID, WALLET, 100.0)   # fund the test wallet
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    reg("last-x", S2, N - 1)
    reg("last-y", S2, N - 1)     # replica of the last segment

    # ---- /infer issues a token and records the chosen chain ----
    out = infer(InferBody(prompt="hi", max_tokens=50, wallet_id=WALLET))
    rid, token = out["request_id"], out["complete_token"]
    chain_ids = [c["node_id"] for c in out["chain"]]
    check("/infer returns a complete_token", bool(token))
    req = models.get_request(rid)
    import json
    check("plan persisted = returned chain", json.loads(req["plan_node_ids"]) == chain_ids)
    chosen_last = chain_ids[-1]
    other_last = "last-y" if chosen_last == "last-x" else "last-x"

    # ---- wrong token -> 401, nothing credited, still pending ----
    before = {n: bal(n) for n in ("driver-a", "middle-c", "last-x", "last-y")}
    try:
        complete(rid, CompleteBody(tokens_generated=10, duration_ms=100,
                                   node_ids=chain_ids, complete_token="WRONG"))
        check("wrong token rejected (401)", False)
    except HTTPException as e:
        check("wrong token rejected (401)", e.status_code == 401)
    try:
        complete(rid, CompleteBody(tokens_generated=10, duration_ms=100, node_ids=chain_ids))
        check("missing token rejected (401)", False)
    except HTTPException as e:
        check("missing token rejected (401)", e.status_code == 401)
    check("no credit after failed auth", all(bal(n) == before[n] for n in before))
    check("request still pending after failed auth", models.get_request(rid)["status"] == "pending")

    # ---- correct token + a LIE about node_ids -> pays the recorded plan only ----
    # caller claims only the UNCHOSEN replica served; settlement must ignore that.
    complete(rid, CompleteBody(tokens_generated=999, duration_ms=100,
                               node_ids=[other_last, "ghost-node"], complete_token=token))
    check("chosen replica in plan earned", bal(chosen_last) > (before[chosen_last] or 0))
    check("unchosen replica (claimed by caller) earned nothing",
          bal(other_last) == before[other_last])
    check("ghost node never credited", bal("ghost-node") is None)
    check("driver + middle (in plan) earned", bal("driver-a") > 0 and bal("middle-c") > 0)
    check("tokens_generated clamped to max_tokens",
          models.get_request(rid)["tokens_generated"] == 50)

    # ---- double completion -> 409 ----
    try:
        complete(rid, CompleteBody(tokens_generated=10, duration_ms=100,
                                   node_ids=chain_ids, complete_token=token))
        check("double completion rejected (409)", False)
    except HTTPException as e:
        check("double completion rejected (409)", e.status_code == 409)

    # ---- concurrent double-completion (the real race, not the sequential check above) ----
    # /complete used to check status='pending' with a plain read, THEN unconditionally
    # distribute() — so N concurrent calls racing the SAME token could all pass the read
    # before any of them committed, each triggering its own distribute() and multiplying the
    # payout. The fix gates distribute() on complete_request()'s own atomic UPDATE ... WHERE
    # status='pending' return value (only one caller can ever win it), not the earlier read.
    out2 = infer(InferBody(prompt="race", max_tokens=50, wallet_id=WALLET))
    rid2, token2 = out2["request_id"], out2["complete_token"]
    chain2_ids = [c["node_id"] for c in out2["chain"]]
    before2 = {n: bal(n) or 0 for n in
              ("driver-a", "middle-c", "last-x", "last-y", config.COORDINATOR_LEDGER_ID)}
    results, lock = [], threading.Lock()

    def racer():
        try:
            r = complete(rid2, CompleteBody(tokens_generated=10, duration_ms=100,
                                            node_ids=chain2_ids, complete_token=token2))
            outcome = ("ok", r)
        except HTTPException as e:
            outcome = ("err", e.status_code)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = [r for kind, r in results if kind == "ok"]
    losses = [r for kind, r in results if kind == "err"]
    check("exactly one racer wins the completion", len(wins) == 1)
    check("every other racer gets 409, not a second payout", losses == [409] * 7)
    reward_once = wins[0]["rewards"] if wins else {}
    for nid, amount in reward_once.items():
        if nid == "__refund__":
            continue   # not a ledger account (goes back to the paying wallet) -- skip
        got = bal(nid) - before2.get(nid, 0)
        check(f"{nid} credited exactly the single-completion amount (no multiplication)",
              abs(got - amount) < 1e-9)

    # ---- prompt privacy: the coordinator stores a LENGTH, never the prompt text ----
    real_prompt = "how do I bake a sourdough loaf"
    out3 = infer(InferBody(prompt=real_prompt, max_tokens=50, wallet_id=WALLET))
    rid3, token3 = out3["request_id"], out3["complete_token"]
    chain3_ids = [c["node_id"] for c in out3["chain"]]
    stored = models.get_request(rid3)
    check("stored request has no prompt text (column stays NULL)", stored.get("prompt") is None)
    check("stored request records the correct LENGTH instead",
          stored.get("prompt_len") == len(real_prompt))
    # a driver that (lies and) over-reports prompt_tokens gets clamped using prompt_len --
    # not the removed prompt text -- as the ceiling. Verify by checking the SETTLED COST
    # actually charged matches the clamped value, not the inflated 530 the caller claimed.
    from coordinator import config as _cfg
    wallet_before = bal(WALLET)          # AFTER /infer's hold already left the wallet
    complete(rid3, CompleteBody(tokens_generated=10, duration_ms=100, node_ids=chain3_ids,
                                complete_token=token3, prompt_tokens=len(real_prompt) + 500))
    refund = bal(WALLET) - wallet_before  # settle() refunds hold_amount - actual_cost back
    actual_cost = out3["hold_amount"] - refund
    expected_weighted = 10 + len(real_prompt) * _cfg.INPUT_WEIGHT   # clamped prompt_tokens
    expected_cost = expected_weighted / 1000 * _cfg.PRICE_PER_1K_WEIGHTED
    check("prompt_tokens is clamped to prompt_len, not the inflated 530 the caller reported",
          abs(actual_cost - expected_cost) < 1e-9)

    # ---- unknown request -> 404 ----
    try:
        complete("no-such-id", CompleteBody(tokens_generated=1, duration_ms=1,
                                            node_ids=[], complete_token="x"))
        check("unknown request rejected (404)", False)
    except HTTPException as e:
        check("unknown request rejected (404)", e.status_code == 404)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
