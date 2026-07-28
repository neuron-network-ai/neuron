"""ui/test_oauth.py — Google/GitHub login wiring (Workstream B). No real OAuth app
credentials exist in this dev environment, so this covers what's testable without them: the
"not configured -> clear 501, not a crash" path, session read/clear, and _link_wallet()'s
call to the coordinator (mocked HTTP -- proves the shared-secret header and payload shape are
correct without needing a live coordinator). Run: python -m ui.test_oauth
"""
import base64
import json

import itsdangerous
from fastapi.testclient import TestClient

import ui.app as ui_app
import ui.oauth as oauth_module
from ui.app import app

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _session_cookie(data):
    """Hand-sign a session cookie the same way starlette.middleware.sessions.SessionMiddleware
    does (see ui/test_moderation_gate.py for the same helper) -- simulates a logged-in
    request without a real OAuth round-trip."""
    signer = itsdangerous.TimestampSigner(str(ui_app.SESSION_SECRET))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def main():
    client = TestClient(app)

    # -- neither provider has real credentials in this dev environment -- #
    check("google not configured in this env (by construction of the test)",
          "google" not in oauth_module._PROVIDERS_CONFIGURED)
    check("github not configured in this env (by construction of the test)",
          "github" not in oauth_module._PROVIDERS_CONFIGURED)

    r = client.get("/auth/login/google", follow_redirects=False)
    check("unconfigured google login returns 501, not a crash", r.status_code == 501)
    check("501 explains what's missing", "NEURON_GOOGLE_CLIENT_ID" in r.json()["detail"])

    r2 = client.get("/auth/login/github", follow_redirects=False)
    check("unconfigured github login returns 501, not a crash", r2.status_code == 501)

    r3 = client.get("/auth/login/not-a-real-provider", follow_redirects=False)
    check("unknown provider returns 404", r3.status_code == 404)

    # -- /auth/me with no session -- #
    r4 = client.get("/auth/me")
    check("no session -> wallet_id is null", r4.json()["wallet_id"] is None)
    check("/auth/me reports which providers ARE configured (none, here)",
          r4.json()["providers_configured"] == [])

    # -- _link_wallet() calls the coordinator with the shared secret + correct payload -- #
    calls = []

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"wallet_id": "w_fake123", "is_new": True}

    real_post = oauth_module.requests.post

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return FakeResp()

    oauth_module.requests.post = fake_post
    try:
        result = oauth_module._link_wallet("google", "sub-abc", "a@b.com")
        check("_link_wallet returns the coordinator's response", result["wallet_id"] == "w_fake123")
        url, body, headers = calls[0]
        check("_link_wallet posts to the coordinator's /wallet/oauth",
              url == f"{oauth_module.COORDINATOR}/wallet/oauth")
        check("_link_wallet sends the shared secret header",
              headers.get("X-Wallet-Link-Secret") == oauth_module.WALLET_LINK_SECRET)
        check("_link_wallet sends provider/external_id/email",
              body == {"provider": "google", "external_id": "sub-abc", "email": "a@b.com"})
    finally:
        oauth_module.requests.post = real_post

    # -- /wallet/balance proxy: no session -> logged_in:false, no coordinator call -- #
    r6 = client.get("/wallet/balance")
    check("no session -> logged_in false", r6.json() == {"logged_in": False})

    # -- /wallet/balance proxy: with a session, proxies the coordinator (mocked) -- #
    real_get = ui_app.requests.get

    class FakeBalResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"balance": 12.5, "total_earned": 30.0}

    def fake_get(url, timeout=None):
        return FakeBalResp()

    ui_app.requests.get = fake_get
    client.cookies.set("session", _session_cookie({"wallet_id": "w_test", "email": "a@b.com"}))
    try:
        r7 = client.get("/wallet/balance")
        body = r7.json()
        check("logged-in balance proxy returns the coordinator's numbers",
              body["logged_in"] and body["balance"] == 12.5 and body["total_earned"] == 30.0)
        check("balance proxy includes the wallet_id and email from the session",
              body["wallet_id"] == "w_test" and body["email"] == "a@b.com")
    finally:
        ui_app.requests.get = real_get
        client.cookies.set("session", "")

    # -- logout clears the session -- #
    client.cookies.set("session", "not-a-real-signed-cookie")   # bad signature -> empty session
    r5 = client.post("/auth/logout")
    check("logout returns ok", r5.json()["status"] == "logged out")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
