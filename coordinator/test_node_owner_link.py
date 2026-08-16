"""coordinator/test_node_owner_link.py — run: python -m coordinator.test_node_owner_link

A node's earnings accrue to an account whose entire credential is the `node_token` in one
`config.json` ([P39]). Lose the file, lose the money; and `blockchain/migrate_ledger.py`
skips any account with no bound payout address as `unmapped`. `claim_node_earnings.py`
sweeps a balance across, but it is a sweep, not an owner link — somebody has to remember to
run it, and it says nothing about who the node belongs to.

This is the owner link: the wallet a person actually signs into, recorded against the node.

Two design decisions the tests below pin, because both are easy to get wrong later:

  * it is recorded ONLY as part of a successful payout binding. A copied node_token is
    therefore not enough to move it — rebinding already needs the incumbent key. Making it
    a standalone endpoint would have handed a stolen token the power to redirect ownership.
  * it never leaves `_node_dict`. `list_nodes` feeds the public /node/list, the public
    dashboard and the router; a node->person map is exactly the correlation that private
    balances and private payout addresses exist to prevent.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-owner-"), "t.db")

from coordinator import config, models, payout        # noqa: E402
from coordinator import main as coord                 # noqa: E402
from fastapi import HTTPException                     # noqa: E402
from eth_account import Account                       # noqa: E402
from eth_account.messages import encode_defunct       # noqa: E402

NODE, TOKEN = "node-owned", "tok-node-owned"
ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def expect_status(label, fn, status, needle=None):
    try:
        fn()
    except HTTPException as e:
        check(label, e.status_code == status and (needle is None or needle in str(e.detail)),
              f"status {e.status_code}, detail {str(e.detail)!r}")
        return
    check(label, False, "no error raised")


def sign(acct, node_id, address, nonce):
    msg = payout.binding_message(node_id, address, nonce, label="node")
    return acct.sign_message(encode_defunct(text=msg)).signature.hex()


def bind(address, nonce, signature, owner=None, old_signature=None, token=TOKEN):
    body = coord.PayoutBindBody(address=address, nonce=nonce, signature=signature,
                                old_signature=old_signature, owner_wallet_id=owner)
    return coord.bind_payout_address(NODE, body, x_node_token=token)


def main():
    models.init_db()
    models.register_node(NODE, "127.0.0.1", 51070, 0, 9, 4, 8.0, TOKEN)
    alice = Account.create()                       # the person at the keyboard
    mallory = Account.create()                     # someone who copied the node_token
    wallet, _ = models.wallet_for_oauth("google", "alice-ext-id", email="a@example.com")
    other, _ = models.wallet_for_oauth("github", "bob-ext-id", email="b@example.com")

    print("\n-- the migration")
    with models._db() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(nodes)").fetchall()}
    check("nodes has owner_wallet_id", "owner_wallet_id" in cols)
    check("a node starts with no owner", models.get_node_owner(NODE) is None)

    print("\n-- recording an owner, as part of a binding")
    ch = coord.payout_challenge(NODE, address=alice.address, x_node_token=TOKEN)
    out = bind(alice.address, ch["nonce"], sign(alice, NODE, alice.address, ch["nonce"]),
               owner=wallet)
    check("the binding succeeds", bool(out.get("payout_address")))
    check("...and the owner is recorded", models.get_node_owner(NODE) == wallet)
    check("...and it is echoed back to the caller", out.get("owner_wallet_id") == wallet)

    print("\n-- it refuses an owner that is not a real login")
    ch = coord.payout_challenge(NODE, address=alice.address, x_node_token=TOKEN)
    sig = sign(alice, NODE, alice.address, ch["nonce"])
    old = sign(alice, NODE, alice.address, ch["nonce"])
    expect_status("an invented wallet id -> 400",
                  lambda: bind(alice.address, ch["nonce"], sig, owner="w_" + "0" * 32,
                               old_signature=old),
                  400, "Google/GitHub")
    check("...and the recorded owner is unchanged", models.get_node_owner(NODE) == wallet)
    check("...and no account was manufactured for it",
          models.get_ledger("w_" + "0" * 32) is None,
          "set_node_owner would happily write a phantom without the is_oauth_wallet gate")

    print("\n-- a copied node_token cannot move the owner")
    ch = coord.payout_challenge(NODE, address=mallory.address, x_node_token=TOKEN)
    expect_status("rebinding without the incumbent key -> 400",
                  lambda: bind(mallory.address, ch["nonce"],
                               sign(mallory, NODE, mallory.address, ch["nonce"]),
                               owner=other),
                  400)
    check("...so the owner still points at the original wallet",
          models.get_node_owner(NODE) == wallet,
          "this is why the owner rides on the binding instead of its own endpoint")

    print("\n-- the owner is never public")
    listed = models.list_nodes()
    check("list_nodes does not carry it",
          all("owner_wallet_id" not in n for n in listed),
          "dropped in _node_dict, so /node/list, /dashboard and the router cannot leak it")
    public = coord.node_list(x_register_secret=None)
    blob = repr(public)
    check("the public /node/list does not contain the wallet id", wallet not in blob)
    operator = coord.node_list(x_register_secret=config.REGISTRATION_SECRET)
    check("...and neither does the operator view", wallet not in repr(operator),
          "it is available deliberately via get_node_owner(), not incidentally via the roster")

    print("\n-- reading it back is behind the node's own token")
    seen = coord.read_payout_address(NODE, x_node_token=TOKEN)
    check("the node can read its own owner", seen["owner_wallet_id"] == wallet)
    expect_status("a wrong token cannot", lambda: coord.read_payout_address(NODE,
                  x_node_token="nope"), 401)

    print("\n-- an older agent that sends no owner still works")
    models.register_node("node-legacy", "127.0.0.1", 51071, 10, 19, 4, 8.0, "tok-legacy")
    ch = coord.payout_challenge("node-legacy", address=alice.address,
                                x_node_token="tok-legacy")
    body = coord.PayoutBindBody(address=alice.address, nonce=ch["nonce"],
                                signature=sign(alice, "node-legacy", alice.address,
                                               ch["nonce"]))
    out = coord.bind_payout_address("node-legacy", body, x_node_token="tok-legacy")
    check("binding without owner_wallet_id succeeds", bool(out.get("payout_address")))
    check("...and leaves the owner unset rather than guessing",
          models.get_node_owner("node-legacy") is None)

    print("\n-- emission pays the owner, not the machine  (phase 3)")
    # Driving a real payable slot needs a heartbeat, an attendance row and a proof-of-compute
    # pass inside it -- all covered by test_emission_slots. What is NEW here is one line: who
    # the reward is transferred TO. So the surrounding machinery is stubbed and only that is
    # exercised, which is also the only part that can regress independently.
    from coordinator import emission

    slot = models.slot_start_for(1_800_000_000.0)
    real_unpaid, real_plan = models.unpaid_attendance, emission.plan_slot
    real_settle, real_transfer = models.settle_attendance, models.transfer
    paid_to = []

    def stub_for(node_id):
        models.unpaid_attendance = lambda before: [{"node_id": node_id, "slot_start": slot}]
        emission.plan_slot = lambda rows, total, now=None: [
            {"node_id": node_id, "slot_start": slot, "reward": 1.0, "replicas": 1,
             "multiplier": 1.0, "attended_frac": 1.0, "reason": None}]
        models.settle_attendance = lambda *a, **kw: True
        models.transfer = lambda src, dst, amt: (paid_to.append(dst), True)[1]

    try:
        models.register_node("node-unowned", "127.0.0.1", 51072, 20, 27, 4, 8.0, "tok-unowned")

        stub_for(NODE)
        emission.close_slots(config.TOTAL_LAYERS, now=slot + config.SLOT_SECONDS * 1.5,
                             log=lambda *_: None)
        check("an owned node's reward goes to the owner's wallet", paid_to == [wallet],
              f"paid to {paid_to}")

        paid_to.clear()
        stub_for("node-unowned")
        emission.close_slots(config.TOTAL_LAYERS, now=slot + config.SLOT_SECONDS * 1.5,
                             log=lambda *_: None)
        check("a node with no owner still earns into its own account",
              paid_to == ["node-unowned"], f"paid to {paid_to}")
    finally:
        models.unpaid_attendance, emission.plan_slot = real_unpaid, real_plan
        models.settle_attendance, models.transfer = real_settle, real_transfer

    check("the attendance row is still claimed against the NODE",
          "settle_attendance(entry[\"node_id\"]" in
          open("coordinator/emission.py", encoding="utf-8").read(),
          "settling elsewhere would let one owner's two nodes settle each other's hours")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
