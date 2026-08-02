"""coordinator/payout.py — proving that whoever runs a node controls the EVM address its
earnings will be sent to.

blockchain/MIGRATION_PLAN.md blocker 1: `ledger.node_id` is a string like
`agent-optinovate-447583`, so when the ledger moves on-chain there is nowhere to send anyone's
NRN. A column alone is not enough. Anybody who can authenticate as a node could otherwise write
*any* address into it, and the coordinator would have no way to tell an operator claiming their
own wallet from an attacker pointing a stranger's earnings at their own.

So a binding is a signature, not a claim. The node signs a message naming itself, the address
and a server-issued nonce; the coordinator recovers the signer and requires it to equal the
address being claimed.

What this does and does not protect against, stated plainly because the difference matters:

  * It proves the binder holds the private key for the address. A typo, or an address copied
    from someone else (to grief them, or to burn the earnings), cannot be bound.
  * It binds to ONE node_id. A signature captured from node A cannot bind node B, because the
    node_id is inside the signed text.
  * It is single-use and short-lived. A signature captured off the wire cannot be replayed
    later to undo a rebinding.
  * It does NOT, on its own, stop someone holding a stolen `node_token` from binding an address
    they control. That is why REBINDING an already-bound account additionally requires a
    signature from the currently bound address (see `require_rebind_authority`): stealing the
    bearer token is then not enough to redirect earnings — you also need the original
    operator's key. An operator who has genuinely lost that key needs the register secret.

The message is written to be read by a human, because on a real wallet it appears in a signing
prompt and "sign this hex blob" is how people get robbed.
"""
from __future__ import annotations

import re

from coordinator import config, models

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")


class PayoutError(ValueError):
    """A binding that could not be verified. The message is safe to return to the caller."""


def _eth():
    """eth_account is imported lazily so the coordinator still starts (and every unrelated
    endpoint still works) on a host where it is not installed -- only payout binding fails,
    and it fails with an instruction rather than an ImportError at startup."""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
    except ImportError as exc:                                      # pragma: no cover
        raise PayoutError(
            "payout binding needs the `eth-account` package on the coordinator "
            "(pip install eth-account)") from exc
    return Account, encode_defunct


def normalize_address(address):
    """Validate and checksum. Stored checksummed so two spellings of one address can never
    look like two different payees."""
    if not isinstance(address, str) or not _ADDRESS_RE.match(address.strip()):
        raise PayoutError("address must be a 0x-prefixed 40-hex-character EVM address")
    address = address.strip()
    if int(address, 16) == 0:
        raise PayoutError("refusing to bind the zero address")
    Account, _ = _eth()
    from eth_utils import to_checksum_address
    checksummed = to_checksum_address(address)
    # A mixed-case address is an EIP-55 checksum and a mismatch means a corrupted paste, not a
    # style choice. All-lower and all-upper carry no checksum, so they are accepted as typed.
    body = address[2:]
    if body != body.lower() and body != body.upper() and address != checksummed:
        raise PayoutError("address failed its EIP-55 checksum -- it was mistyped or truncated")
    return checksummed


def binding_message(node_id, address, nonce):
    """The exact text that gets signed. Both sides build it from the same three inputs, so
    there is nothing to transmit and nothing to disagree about.

    Every field is load-bearing: the node_id binds the signature to one node, the address is
    what is being authorised, and the nonce makes it single-use. The closing sentence is there
    because this shows up in a wallet prompt and the reader deserves to know that signing it
    moves no money.
    """
    return (
        "NEURON payout address binding\n"
        f"node: {node_id}\n"
        f"address: {address}\n"
        f"nonce: {nonce}\n"
        "\n"
        "Signing this proves you control the address above and authorises NEURON to send "
        "this node's NRN earnings there. It transfers no funds and grants no spending power."
    )


def recover_signer(node_id, address, nonce, signature):
    """Who signed `binding_message(node_id, address, nonce)`? Checksummed, or PayoutError."""
    if not isinstance(signature, str) or not signature.strip():
        raise PayoutError("signature is required")
    Account, encode_defunct = _eth()
    message = binding_message(node_id, address, nonce)
    try:
        return Account.recover_message(encode_defunct(text=message), signature=signature.strip())
    except Exception as exc:                                        # noqa: BLE001
        # eth_account raises a family of ValueError/eth_keys errors for malformed input; the
        # caller only needs to know the signature did not verify, not which layer objected.
        raise PayoutError(f"signature could not be recovered ({type(exc).__name__})") from exc


def verify_binding(node_id, address, nonce, signature):
    """Full check for one claimed address. Returns the checksummed address to store."""
    address = normalize_address(address)
    signer = recover_signer(node_id, address, nonce, signature)
    if signer.lower() != address.lower():
        raise PayoutError(
            f"signature is valid but was made by {signer}, not the address being bound "
            f"({address}) -- sign with the key for the address you are claiming")
    return address


def require_rebind_authority(node_id, current_address, new_address, nonce,
                             old_signature, operator_override=False):
    """Changing an address that is already bound needs the OLD key's consent.

    This is the control that makes a stolen `node_token` insufficient to redirect earnings.
    `operator_override` (register-secret gated) is the recovery path for an operator who has
    genuinely lost the old key -- deliberately a human decision, since it is also exactly what
    an attacker would ask for.
    """
    if not current_address or current_address.lower() == new_address.lower():
        return
    if operator_override:
        return
    if not old_signature:
        raise PayoutError(
            f"this node already pays out to {current_address}. Changing it needs "
            f"`old_signature`: the same message signed by that address's key. If the key is "
            f"lost, the operator must rebind with the register secret.")
    signer = recover_signer(node_id, new_address, nonce, old_signature)
    if signer.lower() != current_address.lower():
        raise PayoutError(
            f"old_signature was made by {signer}, not the currently bound address "
            f"({current_address})")


def bind(node_id, address, nonce, signature, old_signature=None, operator_override=False,
         account_type="node"):
    """Verify and store. Raises PayoutError with a caller-safe message on any failure.

    The nonce is consumed FIRST, before any signature is looked at, so a wrong guess costs the
    challenge -- an attacker cannot sit on one nonce and grind signatures against it.
    """
    bad = models.consume_payout_challenge(node_id, nonce, config.PAYOUT_CHALLENGE_TTL)
    if bad:
        raise PayoutError(bad)
    address = verify_binding(node_id, address, nonce, signature)
    existing = models.get_payout_address(node_id)
    current = existing["payout_address"] if existing else None
    require_rebind_authority(node_id, current, address, nonce, old_signature, operator_override)
    models.set_payout_address(node_id, address, account_type=account_type)
    return {"node_id": node_id, "payout_address": address,
            "previous_address": current, "rebound": bool(current and current != address)}
