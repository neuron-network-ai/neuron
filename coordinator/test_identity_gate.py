"""coordinator/test_identity_gate.py — run: python -m coordinator.test_identity_gate

Verified identity is the abuse control. The keyword filter (safety/moderation.py) is trivially
evadable -- "how to build a bomb" is blocked but "I want to make a bomb" is not -- so blocking
content was never going to be the thing that stops misuse. What stops it is that every request
is tied to a real Google/GitHub account which can be banned.

That only works if a wallet CANNOT exist without a login. It could:

    POST /wallet/faucet {"wallet_id": "abuse-1"}   -> wallet created + 25 NRN, no login at all
    use "abuse-1" as the API bearer key            -> full anonymous model access
    get banned -> mint "abuse-2"                   -> unlimited, free, instant ban reset

/wallet/faucet was ungated (unlike its two sibling endpoints, which both check
X-Wallet-Link-Secret) and models.claim_faucet CREATES the ledger row for any string it's
handed. These tests pin that hole shut and pin the operator's ban lever open.

Uses an isolated temp DB -- never touches coordinator/neuron.db.
"""
import os
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_identity_test_"), "t.db")
os.environ["NEURON_WALLET_LINK_SECRET"] = "test-secret"

from fastapi.testclient import TestClient          # noqa: E402
from coordinator import config                     # noqa: E402
from coordinator.main import app                   # noqa: E402

SECRET = {"X-Wallet-Link-Secret": "test-secret"}
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    # guard against ever pointing this at the real dev/production DB
    assert "neuron_identity_test_" in config.DB_PATH, f"NOT ISOLATED: {config.DB_PATH}"

    with TestClient(app) as c:      # context manager -> lifespan runs -> init_db()
        # ---- the bypass: a wallet must not be creatable without a login ---- #
        check("faucet with no operator secret -> 401",
              c.post("/wallet/faucet", json={"wallet_id": "abuse-1"}).status_code == 401)
        check("faucet with secret but no linked login -> 403",
              c.post("/wallet/faucet", json={"wallet_id": "abuse-1"},
                     headers=SECRET).status_code == 403)
        check("a refused faucet call leaves NO ledger row behind",
              c.get("/wallet/abuse-1").status_code == 404)
        check("/infer refuses a wallet with no login behind it -> 403",
              c.post("/infer", json={"prompt": "x", "max_tokens": 5,
                                     "wallet_id": "abuse-1"}).status_code == 403)

        # ---- privileged endpoints are all gated ---- #
        check("admin identity list without secret -> 401",
              c.get("/admin/identities").status_code == 401)
        check("activity lookup without secret -> 401",
              c.get("/wallet/whatever/activity").status_code == 401)
        check("ban without secret -> 401",
              c.post("/wallet/whatever/ban").status_code == 401)

        # ---- banning a typo'd id must fail loudly, not mint a ghost account ---- #
        check("ban of an unknown wallet -> 404",
              c.post("/wallet/nope/ban", headers=SECRET).status_code == 404)
        check("...and created no ledger row", c.get("/wallet/nope").status_code == 404)

        # ---- the real login path still works end to end ---- #
        r = c.post("/wallet/oauth", json={"provider": "google", "external_id": "u1",
                                          "email": "a@b.com", "email_verified": True},
                   headers=SECRET)
        wallet = r.json()["wallet_id"]
        check("a real login mints a wallet", r.status_code == 200 and wallet.startswith("w_"))
        check("signup auto-claims the faucet",
              c.get(f"/wallet/{wallet}").json()["balance"] == config.FAUCET_AMOUNT_NRN)
        check("faucet is one-shot -> 409",
              c.post("/wallet/faucet", json={"wallet_id": wallet},
                     headers=SECRET).status_code == 409)

        # ---- the operator's ban lever ---- #
        check("ban a real identity -> 200",
              c.post(f"/wallet/{wallet}/ban", headers=SECRET).status_code == 200)
        check("a banned identity is refused at /infer -> 403",
              c.post("/infer", json={"prompt": "x", "max_tokens": 5,
                                     "wallet_id": wallet}).status_code == 403)
        check("a ban survives a faucet re-claim attempt",
              c.post("/wallet/faucet", json={"wallet_id": wallet},
                     headers=SECRET).status_code == 409
              and c.get(f"/wallet/{wallet}").json()["moderation_banned"] is True)
        check("unban -> 200",
              c.post(f"/wallet/{wallet}/unban", headers=SECRET).status_code == 200)
        check("unbanned identity is no longer refused for being banned",
              "blocked for repeated" not in
              c.post("/infer", json={"prompt": "x", "max_tokens": 5,
                                     "wallet_id": wallet}).text)

        # ---- what the operator sees ---- #
        ids = c.get("/admin/identities", headers=SECRET).json()["identities"]
        check("exactly one identity, with the provider's verified-email flag",
              len(ids) == 1 and ids[0]["email_verified"] == 1 and ids[0]["email"] == "a@b.com")
        check("banned_only filter excludes an active identity",
              c.get("/admin/identities?banned_only=true",
                    headers=SECRET).json()["identities"] == [])
        act = c.get(f"/wallet/{wallet}/activity", headers=SECRET).json()
        check("activity lookup returns the identity + its history",
              act["identity"]["email"] == "a@b.com" and "requests" in act)
        check("activity never exposes prompt text",
              all("prompt" not in k or k == "prompt_len" for q in act["requests"] for k in q))

        # ---- the admin page is a shell; all data comes from gated endpoints ---- #
        page = c.get("/admin")
        check("admin page loads", page.status_code == 200 and "identity review" in page.text)
        check("admin page embeds no identity data", "a@b.com" not in page.text)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
