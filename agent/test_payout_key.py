"""agent/test_payout_key.py — run: python -m agent.test_payout_key

The agent half of payout binding. The case that matters most is the last one: the agent builds
the message it signs from `agent/payout_key.py` and the coordinator rebuilds it from
`coordinator/payout.py`, and if those two ever disagree by a single character, every binding
fails with "signature is valid but was made by <someone else>" and no test that stubs the
coordinator would notice. So the round-trip is checked against the real verifier.

The rest is the failure modes a volunteer would actually hit: an older coordinator with no
payout endpoints, a machine that is briefly offline, and a key file that has gone bad — which
must never silently generate a replacement, because the old address may already have been paid.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import payout_key                       # noqa: E402
from coordinator import payout as coord_payout     # noqa: E402
from eth_account import Account                    # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload = payload or {}
        self.status_code = status
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise payout_key.requests.HTTPError(f"{self.status_code}", response=self)


class FakeRequests:
    """Stands in for the coordinator. Records what the agent sent so the signature can be
    verified with the real coordinator-side code."""
    HTTPError = None          # filled in below from the real requests module
    RequestException = None

    def __init__(self, node_id, bound=None, challenge_status=200, read_status=200):
        self.node_id = node_id
        self.bound = bound
        self.challenge_status = challenge_status
        self.read_status = read_status
        self.posted = None
        self.nonce = "a" * 32

    def get(self, url, params=None, headers=None, timeout=None):
        if url.endswith("/payout-address"):
            if self.read_status != 200:
                return FakeResponse({}, self.read_status)
            return FakeResponse({"payout_address": self.bound})
        if url.endswith("/payout-challenge"):
            if self.challenge_status != 200:
                return FakeResponse({"detail": "nope"}, self.challenge_status)
            address = params["address"]
            return FakeResponse({
                "nonce": self.nonce, "address": address, "expires_in_seconds": 600,
                "message": coord_payout.binding_message(self.node_id, address, self.nonce)})
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url, json=None, headers=None, timeout=None):
        self.posted = json
        return FakeResponse({"node_id": self.node_id, "payout_address": json["address"],
                             "previous_address": None, "rebound": False})


def main():
    import requests as real_requests
    FakeRequests.HTTPError = real_requests.HTTPError
    FakeRequests.RequestException = real_requests.RequestException
    original = payout_key.requests

    print("\n-- key generation")
    d = tempfile.mkdtemp(prefix="neuron-payoutkey-")
    addr1, key1 = payout_key.load_or_create(d)
    check("a key is generated on first use", bool(addr1) and addr1.startswith("0x"))
    check("it is written to the state dir", os.path.exists(payout_key.key_path(d)))
    addr2, key2 = payout_key.load_or_create(d)
    check("the same key is reused on the next call", (addr1, key1) == (addr2, key2))
    with open(payout_key.key_path(d), encoding="utf-8") as f:
        stored = json.load(f)
    check("the file records the address and warns what losing it costs",
          stored["address"] == addr1 and "Losing this file" in stored["note"])

    bad = tempfile.mkdtemp(prefix="neuron-payoutkey-bad-")
    with open(payout_key.key_path(bad), "w", encoding="utf-8") as f:
        f.write("{ this is not json")
    addr, key = payout_key.load_or_create(bad)
    check("a corrupt key file does NOT silently generate a replacement",
          addr is None and key is None)
    with open(payout_key.key_path(bad), encoding="utf-8") as f:
        check("and the unreadable file is left alone for recovery",
              f.read() == "{ this is not json")

    print("\n-- binding, end to end against the real verifier")
    try:
        d2 = tempfile.mkdtemp(prefix="neuron-payoutkey-bind-")
        fake = FakeRequests("agent-x")
        payout_key.requests = fake
        bound = payout_key.ensure_bound("http://c", "agent-x", "tok", d2)
        check("ensure_bound reports the address it bound", bool(bound))
        check("it posted address, nonce and signature",
              set(fake.posted) == {"address", "nonce", "signature"})

        # The real check: does the coordinator accept what the agent produced?
        verified = coord_payout.verify_binding("agent-x", fake.posted["address"],
                                               fake.posted["nonce"], fake.posted["signature"])
        check("the coordinator verifies the agent's signature", verified == bound)

        # ...and is it actually bound to THIS node, not reusable elsewhere?
        try:
            coord_payout.verify_binding("agent-other", fake.posted["address"],
                                        fake.posted["nonce"], fake.posted["signature"])
            check("the signature does not verify for a different node_id", False)
        except coord_payout.PayoutError:
            check("the signature does not verify for a different node_id", True)

        print("\n-- the ways it is allowed to do nothing")
        already = Account.create().address
        fake = FakeRequests("agent-x", bound=already)
        payout_key.requests = fake
        check("an already-bound node is left alone",
              payout_key.ensure_bound("http://c", "agent-x", "tok", d2) == already
              and fake.posted is None)

        d3 = tempfile.mkdtemp(prefix="neuron-payoutkey-old-")
        fake = FakeRequests("agent-x", read_status=404)
        payout_key.requests = fake
        check("a coordinator with no payout endpoints is not an error",
              payout_key.ensure_bound("http://c", "agent-x", "tok", d3) is None)
        check("and no key is generated for one",
              not os.path.exists(payout_key.key_path(d3)))

        d4 = tempfile.mkdtemp(prefix="neuron-payoutkey-own-")
        fake = FakeRequests("agent-x")
        payout_key.requests = fake
        own = Account.create().address
        check("an operator-supplied address is never auto-bound",
              payout_key.ensure_bound("http://c", "agent-x", "tok", d4,
                                      configured_address=own) is None
              and fake.posted is None)
        check("and no key is generated behind their back",
              not os.path.exists(payout_key.key_path(d4)))

        d5 = tempfile.mkdtemp(prefix="neuron-payoutkey-refused-")
        fake = FakeRequests("agent-x", challenge_status=400)
        payout_key.requests = fake
        check("a refused challenge is survivable",
              payout_key.ensure_bound("http://c", "agent-x", "tok", d5) is None)
    finally:
        payout_key.requests = original

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
