"""agent/payout_key.py — the node's own EVM key, and the binding handshake.

A volunteer running a laptop node does not have MetaMask, and telling them to install a wallet
before they can earn anything would lose most of them at step one. So the agent generates a
key itself, keeps it next to its config, and binds the address automatically — the operator
does nothing and still ends up with an address they fully control, because it is an ordinary
secp256k1 key they can import into any wallet later.

Operators who DO have a wallet are not forced to use ours: set `payout_address` in config.json
to your own address and the agent stops generating one. Binding it then needs a signature made
in your wallet, which `agent/bind_payout.py` walks through.

The key is written with 0600 permissions where the OS supports it. That is a real limit worth
stating plainly rather than burying: this is a hot key on a volunteer's machine, protected by
file permissions and nothing else. It is appropriate for a payout destination that only ever
RECEIVES; it is not appropriate for holding meaningful value, and the operator should move
earnings to a wallet they control properly once there is anything worth moving.
"""
from __future__ import annotations

import json
import logging
import os
import stat

import requests

log = logging.getLogger("neuron.agent")

KEY_FILENAME = "payout_key.json"


def _eth():
    try:
        from eth_account import Account
        return Account
    except ImportError:                                             # pragma: no cover
        return None


def key_path(state_dir):
    return os.path.join(state_dir, KEY_FILENAME)


def load_or_create(state_dir):
    """The node's payout keypair, generated on first use. Returns (address, private_key) or
    (None, None) if eth-account is unavailable -- a missing optional dependency must not stop
    a node from serving inference, it only defers binding."""
    Account = _eth()
    if Account is None:
        log.info("eth-account not installed — payout address binding skipped "
                 "(the node still serves and still earns; the balance just has no on-chain "
                 "destination yet)")
        return None, None

    path = key_path(state_dir)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            acct = Account.from_key(data["private_key"])
            return acct.address, data["private_key"]
        except (OSError, ValueError, KeyError) as exc:
            # Refuse to silently generate a second key: that would strand whatever was already
            # bound (and possibly already paid) at an address whose key we just abandoned.
            log.error("payout key at %s is unreadable (%s) — NOT generating a replacement. "
                      "Move it aside deliberately if you mean to rebind.", path, exc)
            return None, None

    acct = Account.create()
    os.makedirs(state_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"address": acct.address,
                   "private_key": acct.key.hex(),
                   "note": "NEURON payout key. Import into any EVM wallet to control these "
                           "earnings. Losing this file means losing access to whatever has "
                           "been paid to this address."}, f, indent=2)
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)                 # 0600; no-op on some FSes
    except OSError:
        pass
    log.info("generated a payout key — earnings will be bound to %s (key stored at %s)",
             acct.address, path)
    return acct.address, acct.key.hex()


def sign_binding(private_key, message):
    Account = _eth()
    from eth_account.messages import encode_defunct
    signed = Account.sign_message(encode_defunct(text=message), private_key=private_key)
    return signed.signature.hex()


def current_binding(base, node_id, node_token, timeout=15):
    r = requests.get(f"{base}/node/{node_id}/payout-address",
                     headers={"X-Node-Token": node_token}, timeout=timeout)
    r.raise_for_status()
    return r.json().get("payout_address")


def owner_of(base, node_id, node_token, timeout=15):
    """The wallet recorded as this node's owner, or None. Never raises.

    None means EITHER "nobody owns it" or "we could not ask", and the caller must not treat
    those alike -- see how `agent.py` words its warning.
    """
    try:
        r = requests.get(f"{base}/node/{node_id}/payout-address",
                         headers={"X-Node-Token": node_token}, timeout=timeout)
        r.raise_for_status()
        return r.json().get("owner_wallet_id")
    except (requests.RequestException, ValueError):
        return None


def claim_for_owner(base, node_id, node_token, state_dir, wallet_id, timeout=12):
    """Record `wallet_id` as this node's owner. Returns (status_code, payload).

    **This is [P53]'s claim, extracted so there is ONE of it.** `ui/app.py`'s `/node/claim`
    used to carry the whole rule inline, which was fine while a browser on the machine was the
    only way to claim -- and that assumption is exactly what stranded the OptiPlex: a headless
    node (`local_chat: false`) has no page to click, so it earned into an account nobody owned
    and nothing ever said so. `agent.py --claim` needs the identical rule, and two copies of a
    rule about who owns the money is not a risk worth taking.

    The rule, unchanged: re-bind the address that is ALREADY bound, signed here by the key this
    machine holds, carrying `owner_wallet_id`. `require_rebind_authority` exempts a bind to the
    same address, so no `old_signature` and no register secret. The payout address never moves;
    only ownership is recorded.

    **What it still refuses.** A bound address this machine has no key for is a real address
    CHANGE and must keep needing the incumbent key -- that requirement is what stops a copied
    `node_token` redirecting somebody's earnings, and letting an account-claim override it
    would hand an attacker the exact bypass the control exists to prevent.
    """
    try:
        # The BINDING first, because it decides everything after it: with nothing bound this
        # machine may mint a key and bind it, and with something bound it may only re-bind the
        # very same address. Reading the key first would mint one even on the refusing path.
        bound = current_binding(base, node_id, node_token, timeout)
    except (requests.RequestException, ValueError) as e:
        return 502, {"error": f"could not reach the coordinator: {e.__class__.__name__}"}

    address, private_key = (load_or_create(state_dir) if not bound
                            else _existing_key(state_dir))
    if not address:
        return 409, {"error": (
            f"this node pays out to {bound}, and this machine holds no key at all — nothing "
            f"here can sign for it. Use the browser-wallet claim, or ask the operator to "
            f"rebind with the register secret." if bound else
            "this machine could not create a payout key, so it has nothing to sign with.")}
    if bound and bound.lower() != address.lower():
        return 409, {"error": (
            f"this node pays out to {bound}, which is not an address this machine holds a key "
            f"for. Changing it needs a signature from that address — use the browser-wallet "
            f"claim.")}

    try:
        r = requests.get(f"{base}/node/{node_id}/payout-challenge",
                         params={"address": address},
                         headers={"X-Node-Token": node_token}, timeout=timeout)
        if r.status_code == 400:
            return 400, {"error": r.json().get("detail", "bad address")}
        r.raise_for_status()
        ch = r.json()
        signature = sign_binding(private_key, ch["message"])
        if not signature.startswith("0x"):
            signature = "0x" + signature
        r = requests.post(f"{base}/node/{node_id}/payout-address",
                          json={"address": address, "nonce": ch["nonce"],
                                "signature": signature,
                                # From the caller's SESSION or CLI argument, never from a page:
                                # the browser must not be able to nominate somebody else.
                                "owner_wallet_id": wallet_id},
                          headers={"X-Node-Token": node_token}, timeout=timeout)
        if r.status_code == 400:
            return 400, {"error": r.json().get("detail", "claim refused")}
        r.raise_for_status()
        out = r.json()
    except (requests.RequestException, ValueError) as e:
        return 502, {"error": f"could not reach the coordinator: {e.__class__.__name__}"}

    log.info("node %s claimed by %s (payout address unchanged: %s)", node_id, wallet_id, address)
    return 200, {"node_id": node_id,
                 "owner_wallet_id": out.get("owner_wallet_id", wallet_id),
                 "payout_address": out.get("payout_address", address),
                 "rebound": False}


def _existing_key(state_dir):
    """The key this machine already holds, WITHOUT creating one. `load_or_create` does what it
    says, so reading the key to decide whether we CAN claim would mint one as a side effect --
    including on the path that goes on to refuse. A key generated by a refusal is a key nobody
    knows exists ([P53])."""
    if not os.path.exists(key_path(state_dir)):
        return None, None
    return load_or_create(state_dir)


def ensure_bound(base, node_id, node_token, state_dir, configured_address=None, timeout=15):
    """Bind this node's payout address if it is not bound already. Returns the bound address,
    or None if binding did not happen (for any reason -- this is best-effort by design).

    Never raises: a coordinator that is briefly unreachable, or an older coordinator with no
    payout endpoints at all, must not stop a node from serving.
    """
    try:
        already = current_binding(base, node_id, node_token, timeout)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            log.debug("coordinator has no payout endpoints — skipping payout binding")
            return None
        log.debug("could not read payout binding: %s", exc)
        return None
    except requests.RequestException as exc:
        log.debug("could not read payout binding: %s", exc)
        return None

    if already:
        return already

    if configured_address:
        # The operator supplied their own address; we hold no key for it and must not invent
        # one. Tell them exactly how to finish, rather than logging a shrug.
        log.info("payout_address %s is set in config but not bound yet — the coordinator "
                 "needs a signature from that address. Run: python -m agent.bind_payout "
                 "--address %s", configured_address, configured_address)
        return None

    address, private_key = load_or_create(state_dir)
    if not address:
        return None

    try:
        r = requests.get(f"{base}/node/{node_id}/payout-challenge",
                         params={"address": address},
                         headers={"X-Node-Token": node_token}, timeout=timeout)
        r.raise_for_status()
        ch = r.json()
        signature = sign_binding(private_key, ch["message"])
        if not signature.startswith("0x"):
            signature = "0x" + signature
        r = requests.post(f"{base}/node/{node_id}/payout-address",
                          json={"address": address, "nonce": ch["nonce"],
                                "signature": signature},
                          headers={"X-Node-Token": node_token}, timeout=timeout)
        r.raise_for_status()
        log.info("payout address bound: %s — NRN earned by this node has an on-chain "
                 "destination", address)
        return address
    except requests.HTTPError as exc:
        detail = ""
        try:
            detail = exc.response.json().get("detail", "")
        except Exception:                                           # noqa: BLE001
            detail = exc.response.text[:200] if exc.response is not None else ""
        log.warning("payout binding refused: %s", detail or exc)
    except requests.RequestException as exc:
        log.debug("payout binding deferred (coordinator unreachable): %s", exc)
    return None
