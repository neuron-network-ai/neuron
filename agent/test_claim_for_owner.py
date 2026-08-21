"""agent/test_claim_for_owner.py — run: python -m agent.test_claim_for_owner

**A headless node could not be claimed at all, and said nothing about it.**

Claiming was a browser action: open the Chat UI on the machine, sign in, click. A node running
with `local_chat: false` has no page to click — so it registers, serves, earns, and the money
accrues to an account no wallet owns. Nothing anywhere says so, and the operator's only signal
is a machine quietly missing from "My machines". Absence again.

Live on 2026-08-21: `agent-optiplex-server-ce473b` ran headless for a day and accumulated
**28.87 NRN** belonging to nobody. It was found by comparing the wallet's node list against the
coordinator's roster, which is not something an operator would think to do.

`payout_key.claim_for_owner` is that claim with no browser in it, driving `agent.py --claim`.
What it must keep from [P53] is the part that protects somebody else's money:

  * re-bind the address ALREADY bound, signed by the key this machine holds, carrying
    `owner_wallet_id`. The payout address never moves; only ownership is recorded.
  * a bound address this machine has no key for is REFUSED. That is a real address change and
    must keep needing the incumbent key — the requirement that stops a copied `node_token`
    redirecting earnings. An account-claim that could override it would be the exact bypass the
    control exists to prevent.
  * **the refusing path must not mint a key.** Reading the key to decide whether we CAN claim
    would create one as a side effect, including when the answer is going to be no — and a key
    nobody knows exists is [P53]'s other half.
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from agent import payout_key

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class _Resp:
    def __init__(self, payload=None, status=200):
        self._p = payload or {}
        self.status_code = status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise payout_key.requests.HTTPError(response=self)


def run(state_dir, bound, posted, get_fail=False):
    """Drive claim_for_owner against a scripted coordinator. `posted` collects the bind body."""
    real_get, real_post = payout_key.requests.get, payout_key.requests.post

    def fake_get(url, **kw):
        if get_fail:
            raise payout_key.requests.ConnectionError("down")
        if url.endswith("/payout-address"):
            return _Resp({"payout_address": bound})
        if "/payout-challenge" in url:
            return _Resp({"nonce": "n1", "message": "sign me"})
        raise AssertionError(url)

    def fake_post(url, json=None, **kw):
        posted.append(json)
        return _Resp({"payout_address": json["address"],
                      "owner_wallet_id": json.get("owner_wallet_id")})

    payout_key.requests.get, payout_key.requests.post = fake_get, fake_post
    try:
        return payout_key.claim_for_owner("http://c", "node-1", "tok", state_dir, "w_me")
    finally:
        payout_key.requests.get, payout_key.requests.post = real_get, real_post


def main():
    if payout_key._eth() is None:
        print("  SKIP  eth-account is not installed, so no key can be signed with")
        print("\n0 passed, 0 failed")
        return 0

    print("-- the ordinary case: this machine holds the key that is bound")
    with tempfile.TemporaryDirectory() as d:
        addr, _priv = payout_key.load_or_create(d)
        posted = []
        status, payload = run(d, addr, posted)
        check("claims successfully", status == 200, f"{status} {payload}")
        check("...carrying owner_wallet_id, which is the whole point",
              posted and posted[0].get("owner_wallet_id") == "w_me", str(posted))
        check("...re-binding the SAME address, so the payout never moves",
              posted[0]["address"].lower() == addr.lower())
        check("...and signing it", bool(posted[0].get("signature")))
        check("the caller is told what it now owns",
              payload.get("owner_wallet_id") == "w_me"
              and payload.get("payout_address", "").lower() == addr.lower(), str(payload))

    print("\n-- a node bound to an address this machine has NO key for")
    with tempfile.TemporaryDirectory() as d:
        posted = []
        status, payload = run(d, "0xSomebodyElsesAddress", posted)
        check("refused with 409", status == 409, f"{status} {payload}")
        check("...saying which address it pays out to", "0xSomebodyElsesAddress" in payload["error"])
        check("...and nothing was bound", not posted)
        check("**no key was minted on the refusing path**",
              not os.path.exists(payout_key.key_path(d)),
              "reading the key to decide whether we CAN claim creates one as a side effect — "
              "a key nobody knows exists is [P53]'s other half")

    print("\n-- a node with nothing bound yet mints and binds")
    with tempfile.TemporaryDirectory() as d:
        posted = []
        status, payload = run(d, None, posted)
        check("claims successfully", status == 200, f"{status} {payload}")
        check("...and a key now exists here", os.path.exists(payout_key.key_path(d)))
        check("...bound to the address it just made",
              posted and posted[0]["address"] == json.load(
                  open(payout_key.key_path(d)))["address"])

    print("\n-- a coordinator that cannot be reached is 502, never a silent success")
    with tempfile.TemporaryDirectory() as d:
        payout_key.load_or_create(d)
        status, payload = run(d, None, [], get_fail=True)
        check("reports 502 rather than claiming nothing quietly", status == 502, str(payload))
        check("...and says so", "coordinator" in payload.get("error", ""))

    print("\n-- owner_of separates 'nobody owns it' from 'could not ask'")
    real_get = payout_key.requests.get
    try:
        payout_key.requests.get = lambda *a, **k: _Resp({"owner_wallet_id": "w_x"})
        check("reports the owner when there is one",
              payout_key.owner_of("http://c", "n", "t") == "w_x")

        def boom(*a, **k):
            raise payout_key.requests.ConnectionError("down")

        payout_key.requests.get = boom
        check("...and never raises when the coordinator is down",
              payout_key.owner_of("http://c", "n", "t") is None)
    finally:
        payout_key.requests.get = real_get

    print("\n-- the agent offers the headless path and warns when it is unclaimed")
    src = open(os.path.join(HERE, "agent.py"), encoding="utf-8").read()
    check("`--claim WALLET_ID` exists", '"--claim"' in src)
    check("...and calls the shared rule rather than a second copy of it",
          "payout_key as _pk" in src and "_pk.claim_for_owner(" in src)
    check("an unclaimed node says so on every registration",
          "THIS NODE IS UNCLAIMED" in src and "_warn_if_unclaimed" in src)
    check("...and stays quiet when it merely could not ask",
          "could not ask: say nothing rather than guess" in src,
          "[P28]: reporting one event as another is its own bug")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
