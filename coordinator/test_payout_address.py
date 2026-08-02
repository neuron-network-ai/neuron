"""coordinator/test_payout_address.py — run: python -m coordinator.test_payout_address

Payout binding exists because `ledger.node_id` is a hostname-ish string and an on-chain ledger
needs a key. A column on its own would have been worse than nothing: it would look like the
problem was solved while letting anyone who can authenticate as a node write any address into
it, including someone else's.

So the cases worth testing are the ways a proof of control gets faked:
a signature made by the wrong key, a signature lifted from a different node, a nonce replayed
after the fact, and — the one that actually matters for a stolen `node_token` — changing an
address that someone else already proved they own.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-payout-"), "t.db")

from coordinator import config, models, payout      # noqa: E402
from coordinator import main as coord               # noqa: E402
from fastapi import HTTPException                   # noqa: E402
from eth_account import Account                     # noqa: E402
from eth_account.messages import encode_defunct     # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def expect_400(label, fn, needle=None):
    try:
        fn()
    except HTTPException as e:
        detail = str(e.detail)
        check(label, e.status_code == 400 and (needle is None or needle in detail),
              f"status {e.status_code}, detail {detail!r}")
        return
    except payout.PayoutError as e:
        check(label, needle is None or needle in str(e), str(e))
        return
    check(label, False, "no error raised")


def sign(acct, node_id, address, nonce):
    msg = payout.binding_message(node_id, address, nonce)
    return acct.sign_message(encode_defunct(text=msg)).signature.hex()


def register(node_id, port):
    body = coord.RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=port,
                              layer_start=0, layer_end=9, cores=4, ram_gb=8.0)
    return coord.register(body, x_register_secret=config.REGISTRATION_SECRET)["node_token"]


def bind(node_id, token, address, nonce, signature, old_signature=None, secret=None):
    body = coord.PayoutBindBody(address=address, nonce=nonce, signature=signature,
                                old_signature=old_signature)
    return coord.bind_payout_address(node_id, body, x_node_token=token,
                                     x_register_secret=secret)


def main():
    models.init_db()
    alice = Account.create()          # the honest operator of node-1
    mallory = Account.create()        # someone who wants node-1's earnings
    tok1 = register("node-1", 51001)
    tok2 = register("node-2", 51002)

    print("\n-- the challenge")
    try:
        coord.payout_challenge("node-1", x_node_token="wrong-token")
        check("a challenge needs the node's own token", False)
    except HTTPException as e:
        check("a challenge needs the node's own token", e.status_code == 401)
    ch = coord.payout_challenge("node-1", address=alice.address, x_node_token=tok1)
    check("the challenge returns a nonce", bool(ch.get("nonce")))
    check("and the exact message to sign, naming node and address",
          "node: node-1" in ch["message"] and alice.address in ch["message"]
          and ch["nonce"] in ch["message"])
    check("the message says signing moves no money",
          "transfers no funds" in ch["message"])
    expect_400("a malformed address is refused at challenge time",
               lambda: coord.payout_challenge("node-1", address="not-an-address",
                                              x_node_token=tok1),
               "40-hex")

    print("\n-- binding with a real signature")
    ch = coord.payout_challenge("node-1", address=alice.address, x_node_token=tok1)
    out = bind("node-1", tok1, alice.address, ch["nonce"],
               sign(alice, "node-1", alice.address, ch["nonce"]))
    check("a valid signature binds the address", out["payout_address"] == alice.address)
    check("it is not recorded as a rebind", out["rebound"] is False)
    check("the ledger row holds it",
          models.get_payout_address("node-1")["payout_address"] == alice.address)
    check("the node can read its own binding back",
          coord.read_payout_address("node-1", x_node_token=tok1)["payout_address"]
          == alice.address)
    try:
        coord.read_payout_address("node-1", x_node_token=tok2)
        check("another node cannot read it", False)
    except HTTPException as e:
        check("another node cannot read it", e.status_code == 401)

    print("\n-- forging control of an address")
    ch = coord.payout_challenge("node-2", address=alice.address, x_node_token=tok2)
    expect_400("signing with the wrong key is refused",
               lambda: bind("node-2", tok2, alice.address, ch["nonce"],
                            sign(mallory, "node-2", alice.address, ch["nonce"])),
               "not the address being bound")
    check("and nothing was bound", models.get_payout_address("node-2") is None)

    # The signature below is perfectly valid -- for node-1. It must not bind node-2.
    ch1 = coord.payout_challenge("node-1", address=alice.address, x_node_token=tok1)
    lifted = sign(alice, "node-1", alice.address, ch1["nonce"])
    ch2 = coord.payout_challenge("node-2", address=alice.address, x_node_token=tok2)
    expect_400("a signature lifted from another node does not transfer",
               lambda: bind("node-2", tok2, alice.address, ch2["nonce"], lifted))
    check("still nothing bound on node-2", models.get_payout_address("node-2") is None)

    print("\n-- nonce discipline")
    ch = coord.payout_challenge("node-2", address=mallory.address, x_node_token=tok2)
    sig = sign(mallory, "node-2", mallory.address, ch["nonce"])
    bind("node-2", tok2, mallory.address, ch["nonce"], sig)
    expect_400("the same nonce cannot be used twice",
               lambda: bind("node-2", tok2, mallory.address, ch["nonce"], sig),
               "no challenge issued")
    ch = coord.payout_challenge("node-2", address=mallory.address, x_node_token=tok2)
    expect_400("a nonce that was never issued is refused",
               lambda: bind("node-2", tok2, mallory.address, "deadbeef" * 4,
                            sign(mallory, "node-2", mallory.address, "deadbeef" * 4)),
               "does not match")
    ch = coord.payout_challenge("node-2", address=mallory.address, x_node_token=tok2)
    sig = sign(mallory, "node-2", mallory.address, ch["nonce"])
    old_ttl, config.PAYOUT_CHALLENGE_TTL = config.PAYOUT_CHALLENGE_TTL, -1
    expect_400("an expired challenge is refused",
               lambda: bind("node-2", tok2, mallory.address, ch["nonce"], sig), "expired")
    config.PAYOUT_CHALLENGE_TTL = old_ttl

    print("\n-- rebinding: the control that survives a stolen node_token")
    # Mallory has node-1's token (copied off disk) and a key of her own. That is exactly the
    # attack the old-signature requirement exists to stop.
    ch = coord.payout_challenge("node-1", address=mallory.address, x_node_token=tok1)
    expect_400("a stolen token alone cannot redirect earnings",
               lambda: bind("node-1", tok1, mallory.address, ch["nonce"],
                            sign(mallory, "node-1", mallory.address, ch["nonce"])),
               "already pays out to")
    check("the original address is untouched",
          models.get_payout_address("node-1")["payout_address"] == alice.address)

    ch = coord.payout_challenge("node-1", address=mallory.address, x_node_token=tok1)
    expect_400("an old_signature from the wrong key does not authorise the change",
               lambda: bind("node-1", tok1, mallory.address, ch["nonce"],
                            sign(mallory, "node-1", mallory.address, ch["nonce"]),
                            old_signature=sign(mallory, "node-1", mallory.address,
                                               ch["nonce"])),
               "not the currently bound address")

    alice2 = Account.create()
    ch = coord.payout_challenge("node-1", address=alice2.address, x_node_token=tok1)
    out = bind("node-1", tok1, alice2.address, ch["nonce"],
               sign(alice2, "node-1", alice2.address, ch["nonce"]),
               old_signature=sign(alice, "node-1", alice2.address, ch["nonce"]))
    check("the real operator can move their address with the old key",
          out["payout_address"] == alice2.address and out["rebound"] is True)
    check("and the previous address is reported", out["previous_address"] == alice.address)

    lost = Account.create()
    ch = coord.payout_challenge("node-1", address=lost.address, x_node_token=tok1)
    out = bind("node-1", tok1, lost.address, ch["nonce"],
               sign(lost, "node-1", lost.address, ch["nonce"]),
               secret=config.REGISTRATION_SECRET)
    check("the operator can rebind a lost key with the register secret",
          out["payout_address"] == lost.address)

    print("\n-- address hygiene")
    node3 = register("node-3", 51003)
    lower = alice.address.lower()
    ch = coord.payout_challenge("node-3", address=lower, x_node_token=node3)
    check("a lowercase address is accepted and checksummed", ch["address"] == alice.address)
    out = bind("node-3", node3, lower, ch["nonce"], sign(alice, "node-3", alice.address,
                                                         ch["nonce"]))
    check("and stored checksummed", out["payout_address"] == alice.address)
    expect_400("the zero address is refused",
               lambda: payout.normalize_address("0x" + "0" * 40), "zero address")
    bad = alice.address[:10].lower() + alice.address[10:]        # break EIP-55, keep the hex
    if bad != alice.address:
        expect_400("a mistyped mixed-case address fails its checksum",
                   lambda: payout.normalize_address(bad), "EIP-55")

    print("\n-- privacy")
    pub = models.list_nodes()
    check("payout addresses are not in the public node list",
          all("payout_address" not in n for n in pub))
    dash = coord.dashboard()
    check("nor on the public dashboard", alice.address not in dash and lost.address not in dash)
    own = coord.node_dashboard("node-1", token=tok1)
    check("but a node sees its own on its private dashboard", lost.address in own)

    print("\n-- the migration reads bindings, not a hand-edited file")
    book = models.payout_addresses()
    check("payout_addresses() returns every bound account",
          book.get("node-1") == lost.address and book.get("node-3") == alice.address)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
