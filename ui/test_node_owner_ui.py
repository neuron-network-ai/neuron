"""ui/test_node_owner_ui.py — run: python -m ui.test_node_owner_ui

The Chat UI is the one place where both halves of [P39] are in the same process: the agent
knows its own node_id, and the session knows who just signed in. These routes turn that into
a recorded owner.

What the tests pin, in order of what would hurt most if it broke:

  * the node_token never reaches the browser. It is read from the environment the agent put
    it in and used only server-side, exactly as the wallet routes already do.
  * the owner comes from the SESSION, never from the request body. Otherwise a page could
    nominate somebody else's wallet as the owner of this machine's earnings.
  * a driver-only machine (no node_id) gets `is_node: false` rather than an error or a prompt
    for something it does not have.
  * a coordinator that is down degrades to a reported error, not a traceback — a broken
    ownership prompt must never take the chat down with it.
"""
import os
import sys
import tempfile

os.environ.setdefault("NEURON_SESSION_SECRET", "test-secret-not-a-real-one")

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def main():
    from ui import app as uiapp

    calls = []

    def fake_get(url, **kw):
        calls.append(("GET", url, kw))
        if "/payout-address" in url:
            return FakeResp(200, {"node_id": "node-x", "payout_address": None,
                                  "owner_wallet_id": None})
        return FakeResp(200, {"nonce": "n1", "message": "node: node-x ..."})

    def fake_post(url, **kw):
        calls.append(("POST", url, kw))
        return FakeResp(200, {"payout_address": "0xabc", "owner_wallet_id": "w_alice"})

    uiapp.requests.get, uiapp.requests.post = fake_get, fake_post
    uiapp.NODE_ID, uiapp.NODE_TOKEN = "node-x", "tok-secret"

    class Req:
        def __init__(self, session):
            self.session = session

    signed_in = Req({"wallet_id": "w_alice"})
    anon = Req({})

    print("\n-- it only offers the prompt when there is something to record")
    out = uiapp.node_owner(signed_in)
    check("a node with no owner and a signed-in user needs one", out["needs_owner"] is True)
    check("...and reports its node id", out["node_id"] == "node-x")
    check("a signed-out user is not prompted", uiapp.node_owner(anon)["needs_owner"] is False)

    calls.clear()
    uiapp.NODE_ID = None
    out = uiapp.node_owner(signed_in)
    check("a driver-only machine reports is_node false", out["is_node"] is False)
    check("...and asks the coordinator nothing", calls == [])
    uiapp.NODE_ID = "node-x"

    print("\n-- the node token stays server-side")
    calls.clear()
    uiapp.node_payout_challenge("0xAddr", signed_in)
    hdrs = calls[-1][2].get("headers", {})
    check("the challenge is proxied with the node token", hdrs.get("X-Node-Token") == "tok-secret")
    body = uiapp.node_owner(signed_in)
    check("the token is never in a response to the browser",
          "tok-secret" not in repr(body), repr(body))

    print("\n-- the owner comes from the session, not the request")
    calls.clear()
    b = uiapp.NodeBindBody(address="0xAddr", nonce="n1", signature="0xsig")
    uiapp.node_payout_bind(b, signed_in)
    sent = calls[-1][2]["json"]
    check("bind injects the session's wallet as owner", sent["owner_wallet_id"] == "w_alice")
    check("...and the body has no way to override it",
          "owner_wallet_id" not in uiapp.NodeBindBody.model_fields,
          "a body field here would let a page nominate somebody else's wallet")
    check("...and it is sent with the node token",
          calls[-1][2]["headers"]["X-Node-Token"] == "tok-secret")

    print("\n-- signing in is required")
    r = uiapp.node_payout_bind(b, anon)
    check("bind refuses an anonymous session", r.status_code == 401)
    r = uiapp.node_payout_challenge("0xAddr", anon)
    check("challenge refuses an anonymous session", r.status_code == 401)

    print("\n-- a refusal from the coordinator is passed through, not flattened")
    uiapp.requests.post = lambda url, **kw: FakeResp(
        400, {"detail": "sign with the key for the address you are claiming"})
    r = uiapp.node_payout_bind(b, signed_in)
    check("a 400 keeps the coordinator's wording", r.status_code == 400)
    check("...which is the only thing that tells someone what they got wrong",
          b"key for the address" in bytes(r.body))

    print("\n-- a coordinator outage does not take the chat down")
    import requests as real_requests

    def boom(*a, **kw):
        raise real_requests.RequestException("connection refused")

    uiapp.requests.get = boom
    out = uiapp.node_owner(signed_in)
    check("node_owner reports the error rather than raising", "error" in out)
    r = uiapp.node_payout_challenge("0xAddr", signed_in)
    check("challenge returns 502 rather than raising", r.status_code == 502)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
