"""ui/test_payout_ui.py — run: python -m ui.test_payout_ui

The Chat UI proxies payout binding so a person can set the on-chain destination for their
earnings from the page they already use. Before this, the only route was a coordinator endpoint
nobody browses to, and an account with no bound address is silently skipped as `unmapped` by
blockchain/migrate_ledger.py — a discoverability gap with money on the other side of it.

What these hold:

  * **The wallet id comes from the SESSION, never from the request.** This is the whole reason
    the proxy exists rather than letting the page call the coordinator: the coordinator's own
    wallet endpoints are bearer-authorized, so a page that learned somebody's wallet id could
    start a binding against it. Through here it cannot — there is no parameter to carry one.
  * **No session means 401**, before anything is proxied anywhere.
  * **A coordinator 400 is passed through verbatim.** payout.py writes those to be read by the
    person who caused them ("sign with the key for the address you are claiming"); replacing
    them with a generic failure throws away the only thing that says what went wrong.
  * **An unreachable coordinator is a 502, not a traceback.**
"""
import base64
import json
import sys

import itsdangerous
from fastapi.testclient import TestClient

import ui.app as ui_app
from ui.app import app

WALLET = "w_" + "c" * 32
ADDR = "0x29772e94d9D31287C10032Fd9dF2b3C4E9af7ff2"
ok = fail = 0


def check(name, cond, detail=""):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"\n        {detail}" if not cond and detail else ""))


def _session_cookie(data):
    signer = itsdangerous.TimestampSigner(str(ui_app.SESSION_SECRET))
    return signer.sign(base64.b64encode(json.dumps(data).encode())).decode()


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise ui_app.requests.RequestException(f"HTTP {self.status_code}")


def main():
    real_get, real_post = ui_app.requests.get, ui_app.requests.post
    seen = {}

    client = TestClient(app)              # no `with` — avoids the lifespan model load
    client.cookies.set("session", _session_cookie({"wallet_id": WALLET, "email": "x@y.z"}))
    anon = TestClient(app)

    try:
        print("\n-- without a session, nothing is proxied")
        calls = []
        ui_app.requests.get = lambda *a, **k: calls.append(a) or FakeResponse({})
        ui_app.requests.post = lambda *a, **k: calls.append(a) or FakeResponse({})
        r = anon.get("/wallet/payout/challenge?address=" + ADDR)
        check("challenge without a session -> 401", r.status_code == 401)
        r = anon.post("/wallet/payout/bind",
                      json={"address": ADDR, "nonce": "n", "signature": "0xsig"})
        check("bind without a session -> 401", r.status_code == 401)
        check("...and the coordinator was never called", calls == [],
              f"called {calls}")
        r = anon.get("/wallet/payout")
        check("reading the address without a session says so, and does not error",
              r.status_code == 200 and r.json() == {"logged_in": False})

        print("\n-- the wallet id comes from the session, not the caller")
        def fake_get(url, **kw):
            seen["url"] = url
            seen["params"] = kw.get("params")
            return FakeResponse({"wallet_id": WALLET, "nonce": "abc123",
                                 "address": ADDR, "message": "NEURON payout address binding\n…",
                                 "expires_in_seconds": 600.0})
        ui_app.requests.get = fake_get
        r = client.get("/wallet/payout/challenge?address=" + ADDR)
        check("challenge succeeds for a logged-in session", r.status_code == 200)
        check("...against the SESSION's wallet", WALLET in seen["url"],
              seen.get("url", ""))
        check("...and passes the address through", seen["params"] == {"address": ADDR})
        check("the signing message is returned to the page", "message" in r.json())
        # There is no request parameter that could name a different wallet -- the only input is
        # the address being claimed. This asserts the shape, which is what makes it true.
        r = client.get("/wallet/payout/challenge?address=" + ADDR + "&wallet_id=w_someoneelse")
        check("an injected wallet_id parameter is ignored entirely",
              r.status_code == 200 and WALLET in seen["url"] and "someoneelse" not in seen["url"])

        print("\n-- a refusal explains itself")
        ui_app.requests.get = lambda *a, **k: FakeResponse(
            {"detail": "address failed its EIP-55 checksum -- it was mistyped or truncated"}, 400)
        r = client.get("/wallet/payout/challenge?address=0xbad")
        check("a bad address is a 400", r.status_code == 400)
        check("...carrying the coordinator's own words", "EIP-55" in r.json()["error"],
              r.json().get("error", ""))

        ui_app.requests.post = lambda *a, **k: FakeResponse(
            {"detail": "signature is valid but was made by 0xother, not the address being "
                       "bound (0x2977) -- sign with the key for the address you are claiming"},
            400)
        r = client.post("/wallet/payout/bind",
                        json={"address": ADDR, "nonce": "n", "signature": "0xsig"})
        check("a wrong-key signature is a 400", r.status_code == 400)
        check("...and says which key signed it",
              "sign with the key for the address you are claiming" in r.json()["error"])

        print("\n-- a successful binding")
        def fake_post(url, **kw):
            seen["post_url"] = url
            seen["body"] = kw.get("json")
            return FakeResponse({"wallet_id": WALLET, "payout_address": ADDR,
                                 "previous_address": None, "rebound": False})
        ui_app.requests.post = fake_post
        r = client.post("/wallet/payout/bind",
                        json={"address": ADDR, "nonce": "abc123", "signature": "0xsig"})
        check("binds", r.status_code == 200 and r.json()["payout_address"] == ADDR)
        check("...posted to the session's wallet", WALLET in seen["post_url"])
        check("...forwarding nonce and signature", seen["body"]["nonce"] == "abc123"
              and seen["body"]["signature"] == "0xsig")

        print("\n-- the coordinator being down is not a traceback")
        def boom(*a, **k):
            raise ui_app.requests.RequestException("connection refused")
        ui_app.requests.get = boom
        ui_app.requests.post = boom
        r = client.get("/wallet/payout/challenge?address=" + ADDR)
        check("challenge -> 502", r.status_code == 502)
        r = client.post("/wallet/payout/bind",
                        json={"address": ADDR, "nonce": "n", "signature": "0xsig"})
        check("bind -> 502", r.status_code == 502)
        r = client.get("/wallet/payout")
        check("reading the address degrades to an error field, still 200",
              r.status_code == 200 and "error" in r.json())
    finally:
        ui_app.requests.get, ui_app.requests.post = real_get, real_post

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
