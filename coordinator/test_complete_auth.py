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

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_p12_"), "p12.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, models  # noqa: E402
from coordinator.main import CompleteBody, InferBody, RegisterBody, complete, infer, register  # noqa: E402

SECRET = config.REGISTRATION_SECRET
S1, S2, N = 10, 19, config.TOTAL_LAYERS
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
    models.init_db()
    reg("driver-a", 0, S1 - 1)
    reg("middle-c", S1, S2 - 1)
    reg("last-x", S2, N - 1)
    reg("last-y", S2, N - 1)     # replica of the last segment

    # ---- /infer issues a token and records the chosen chain ----
    out = infer(InferBody(prompt="hi", max_tokens=50))
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
