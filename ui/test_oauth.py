"""ui/test_oauth.py — the agent's half of sign-in. Run: python -m ui.test_oauth

Sign-in is delegated to the coordinator (coordinator/auth.py). This file used to run the OAuth
dance itself, which meant a Google/GitHub client SECRET had to exist in this process -- i.e. on
every stranger's PC once each installed agent started serving its own Chat UI. A secret shipped
in a binary is not a secret, and the alternative (each user creating their own Google Cloud
project) is not a product.

So what is tested here is the loopback half:

  * this process holds no client secret and no provider credentials at all;
  * /auth/login bounces to the coordinator, telling it which local port to return to;
  * /auth/adopt redeems the coordinator's ONE-TIME code server-to-server and puts the wallet in
    the session -- the wallet_id itself never travels in a browser URL;
  * an unavailable/unconfigured coordinator degrades to a clear message, never a crash.

Coordinator HTTP is mocked throughout; no network.
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
    signer = itsdangerous.TimestampSigner(str(ui_app.SESSION_SECRET))
    return signer.sign(base64.b64encode(json.dumps(data).encode())).decode()


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise oauth_module.requests.RequestException(f"HTTP {self.status_code}")


def main():
    client = TestClient(app)
    real_get, real_post = oauth_module.requests.get, oauth_module.requests.post

    # ---- the agent carries no OAuth secrets of its own ---- #
    src = open(oauth_module.__file__, encoding="utf-8").read()
    check("agent code holds no client_secret at all",
          "CLIENT_SECRET" not in src and "client_secret" not in src)
    check("no local provider registry to fall out of sync with the coordinator",
          not hasattr(oauth_module, "_PROVIDERS_CONFIGURED"))

    # ---- providers come from the coordinator, and an outage degrades quietly ---- #
    oauth_module.requests.get = lambda *a, **k: FakeResp({"providers": ["google", "github"]})
    check("providers are read from the coordinator",
          oauth_module.providers_configured() == ["google", "github"])

    def boom(*a, **k):
        raise oauth_module.requests.RequestException("unreachable")
    oauth_module.requests.get = boom
    check("unreachable coordinator -> no providers, no crash",
          oauth_module.providers_configured() == [])

    r = client.get("/auth/login/google", follow_redirects=False)
    check("login with no reachable coordinator -> clear 501, not a crash", r.status_code == 501)

    # ---- starting a login bounces to the coordinator with our loopback port ---- #
    oauth_module.requests.get = lambda *a, **k: FakeResp({"providers": ["google"]})
    r = client.get("/auth/login/google", follow_redirects=False)
    loc = r.headers.get("location", "")
    check("login redirects to the coordinator", r.status_code in (302, 307)
          and loc.startswith(f"{oauth_module.COORDINATOR}/auth/login/google"))
    check("login tells the coordinator which loopback port to return to",
          f"port={oauth_module.LOCAL_PORT}" in loc)
    r = client.get("/auth/login/myspace", follow_redirects=False)
    check("unknown provider returns 404", r.status_code == 404)

    # ---- adopt: redeem the one-time code, wallet lands in the session ---- #
    posted = {}

    def fake_post(url, json=None, timeout=None, **k):
        posted["url"], posted["body"] = url, json
        return FakeResp({"wallet_id": "w_real123", "email": "a@b.com"})
    oauth_module.requests.post = fake_post
    r = client.get("/auth/adopt?code=one-time-abc", follow_redirects=False)
    check("adopt redeems the code against the coordinator",
          posted["url"].endswith("/auth/exchange") and posted["body"] == {"code": "one-time-abc"})
    check("adopt redirects back into the chat page", r.headers.get("location") == "/")
    me = client.get("/auth/me").json()
    check("the wallet is now in this browser's session", me["wallet_id"] == "w_real123")
    check("the email is carried through for display", me["email"] == "a@b.com")

    # ---- a bad/expired code fails cleanly and grants nothing ---- #
    client2 = TestClient(app)
    oauth_module.requests.post = lambda *a, **k: FakeResp({"detail": "invalid"}, status=400)
    r = client2.get("/auth/adopt?code=stale", follow_redirects=False)
    check("an expired/invalid code -> 502, not a session", r.status_code == 502)
    check("...and no wallet was granted", client2.get("/auth/me").json()["wallet_id"] is None)
    oauth_module.requests.post = fake_post

    # ---- balance proxy + logout still behave ---- #
    r = TestClient(app).get("/wallet/balance")
    check("no session -> logged_in false", r.json() == {"logged_in": False})

    paid = TestClient(app)
    paid.cookies.set("session", _session_cookie({"wallet_id": "w_real123", "email": "a@b.com"}))
    ui_app.requests.get = lambda *a, **k: FakeResp({"balance": 25.0, "total_earned": 25.0})
    try:
        b = paid.get("/wallet/balance").json()
        check("logged-in balance proxy returns the coordinator's numbers",
              b["balance"] == 25.0 and b["logged_in"] is True)
        check("balance proxy includes the wallet_id and email from the session",
              b["wallet_id"] == "w_real123" and b["email"] == "a@b.com")
    finally:
        ui_app.requests.get = real_get

    logout = paid.post("/auth/logout")
    check("logout returns ok", logout.json()["status"] == "logged out")
    # Assert the SERVER told the browser to drop the cookie. Re-reading /auth/me here would
    # test httpx's cookie jar instead: it does not evict a cookie that was injected by hand
    # (as this test does to fake a login), even when the response expires it correctly.
    check("logout expires the session cookie in the browser",
          "expires=Thu, 01 Jan 1970" in logout.headers.get("set-cookie", ""))

    oauth_module.requests.get, oauth_module.requests.post = real_get, real_post
    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
