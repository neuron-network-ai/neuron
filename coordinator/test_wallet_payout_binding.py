"""coordinator/test_wallet_payout_binding.py — run: python -m coordinator.test_wallet_payout_binding

Nodes could bind an on-chain payout address; wallets could not. That gap had a cost that only
showed up once node earnings started being swept into wallets: `blockchain/migrate_ledger.py`
marks any account with no bound address `unmapped` and skips it, so moving NRN somewhere
spendable moved it somewhere unmappable. Spend it now and keep it later were mutually
exclusive, and nothing said so.

The cases below are the node suite's, re-asked of the wallet surface — a wrong key, a replayed
nonce, a rebinding without the incumbent key's consent — plus the two that are specific to
having two account kinds sharing one signing protocol:

  * a signature made for `node: X` must not bind `wallet: X`, and vice versa. The label is
    inside the signed text precisely so the two cannot be traded.
  * binding must refuse an id that is not already a wallet. `set_payout_address` ends in
    INSERT OR IGNORE, so without that check this endpoint manufactures accounts.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-wpayout-"), "t.db")

from coordinator import config, models, payout        # noqa: E402
from coordinator import main as coord                 # noqa: E402
from fastapi import HTTPException                     # noqa: E402
from eth_account import Account                       # noqa: E402
from eth_account.messages import encode_defunct       # noqa: E402

WALLET = "w_" + "b" * 32
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
        detail = str(e.detail)
        check(label, e.status_code == status and (needle is None or needle in detail),
              f"status {e.status_code}, detail {detail!r}")
        return
    check(label, False, "no error raised")


def sign(acct, account_id, address, nonce, label="wallet"):
    msg = payout.binding_message(account_id, address, nonce, label=label)
    return acct.sign_message(encode_defunct(text=msg)).signature.hex()


def challenge(wallet_id=WALLET, address=None):
    return coord.wallet_payout_challenge(wallet_id, address=address)


def bind(wallet_id, address, nonce, signature, old_signature=None, secret=None):
    body = coord.PayoutBindBody(address=address, nonce=nonce, signature=signature,
                                old_signature=old_signature)
    return coord.bind_wallet_payout_address(wallet_id, body, x_register_secret=secret)


def main():
    models.init_db()
    alice = Account.create()        # the person who signed in
    mallory = Account.create()      # someone who would like their NRN
    models.ensure_account(WALLET, "wallet")
    models.register_node("node-w", "127.0.0.1", 51050, 0, 9, 4, 8.0, "tok-node-w")

    print("\n-- it refuses accounts that are not wallets")
    expect_status("unknown wallet id -> 404", lambda: challenge("w_" + "0" * 32), 404,
                  "unknown wallet")
    expect_status("a NODE id on the wallet endpoint -> 404",
                  lambda: challenge("node-w"), 404, "unknown wallet")
    check("...and no ledger row was manufactured for the unknown id",
          models.get_ledger("w_" + "0" * 32) is None,
          "set_payout_address ends in INSERT OR IGNORE -- this is the check that stops it")

    print("\n-- the challenge")
    out = challenge(address="0x29772e94d9D31287C10032Fd9dF2b3C4E9af7ff2")
    check("issues a nonce", bool(out.get("nonce")))
    check("carries its own TTL", out["expires_in_seconds"] == config.PAYOUT_CHALLENGE_TTL)
    check("returns the checksummed address",
          out["address"] == "0x29772e94d9D31287C10032Fd9dF2b3C4E9af7ff2")
    check("the signing prompt says wallet, not node", "wallet: " + WALLET in out["message"],
          out["message"])
    check("...and never claims to move money",
          "transfers no funds" in out["message"])
    expect_status("a mistyped address is caught before signing",
                  lambda: challenge(address="0xdeadbeef"), 400, "40-hex")

    print("\n-- an honest binding")
    addr = alice.address
    nonce = challenge()["nonce"]
    res = bind(WALLET, addr, nonce, sign(alice, WALLET, addr, nonce))
    check("binds", res["payout_address"] == addr)
    check("reports the wallet, not a node", res.get("wallet_id") == WALLET and
          "node_id" not in res)
    check("stored and readable back",
          coord.read_wallet_payout_address(WALLET)["payout_address"] == addr)
    check("appears in the migration address book",
          models.payout_addresses().get(WALLET) == addr,
          "this is what stops migrate_ledger.py marking it unmapped")

    print("\n-- signatures that must not work")
    nonce = challenge()["nonce"]
    expect_status("a signature by another key is rejected",
                  lambda: bind(WALLET, addr, nonce, sign(mallory, WALLET, addr, nonce)),
                  400, "not the address being bound")

    nonce = challenge()["nonce"]
    node_sig = sign(alice, WALLET, addr, nonce, label="node")
    expect_status("a signature made for `node:` cannot bind a wallet",
                  lambda: bind(WALLET, addr, nonce, node_sig), 400,
                  "not the address being bound")

    nonce = challenge()["nonce"]
    good = sign(alice, WALLET, addr, nonce)
    bind(WALLET, addr, nonce, good)
    expect_status("a nonce cannot be replayed", lambda: bind(WALLET, addr, nonce, good),
                  400, "no challenge issued")

    print("\n-- rebinding needs the incumbent key")
    new_addr = mallory.address
    nonce = challenge()["nonce"]
    expect_status("rebinding without old_signature is refused",
                  lambda: bind(WALLET, new_addr, nonce, sign(mallory, WALLET, new_addr, nonce)),
                  400, "already pays out to")
    check("...and the original address still stands",
          coord.read_wallet_payout_address(WALLET)["payout_address"] == addr)

    nonce = challenge()["nonce"]
    res = bind(WALLET, new_addr, nonce, sign(mallory, WALLET, new_addr, nonce),
               old_signature=sign(alice, WALLET, new_addr, nonce))
    check("rebinding with the incumbent key's consent works",
          res["payout_address"] == new_addr and res["rebound"])

    nonce = challenge()["nonce"]
    res = bind(WALLET, addr, nonce, sign(alice, WALLET, addr, nonce),
               secret=config.REGISTRATION_SECRET)
    check("the register secret is the lost-key recovery path",
          res["payout_address"] == addr)

    print("\n-- the node surface is untouched")
    msg = payout.binding_message("node-w", addr, "n123")
    check("a node's signed text is byte-identical to before",
          msg.splitlines()[1] == f"node: node-w", msg)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
