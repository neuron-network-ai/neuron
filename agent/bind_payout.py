"""agent/bind_payout.py — bind a payout address by hand.

    python -m agent.bind_payout                          # show what is bound now
    python -m agent.bind_payout --address 0xYourWallet   # print the message to sign
    python -m agent.bind_payout --address 0x... --signature 0x...     # submit it

Most operators never need this: the agent generates a key and binds it automatically. This is
for the operator who wants their NRN paid to a wallet they already control — a hardware wallet,
an exchange-independent address, an address they use for everything else.

The flow is deliberately two steps. Run it with `--address` and it prints the exact text to
paste into your wallet's "sign message" box; sign it there, come back with `--signature`. The
private key never touches this machine, which is the entire point of doing it this way.

Changing an address that is already bound also needs `--old-signature`: the SAME message signed
by the address currently on file. That is what stops someone who has copied your node_token off
your disk from redirecting your earnings — they would need your old key too. If you have lost
it, the operator can rebind for you with the register secret.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

DEFAULT_CONFIG = os.path.join(HERE, "config.json")


def main(argv=None):
    p = argparse.ArgumentParser(description="Bind this node's on-chain payout address.")
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"default {DEFAULT_CONFIG}")
    p.add_argument("--address", help="the EVM address to pay this node's NRN to")
    p.add_argument("--signature", help="`message` signed by --address")
    p.add_argument("--old-signature",
                   help="the same message signed by the address currently bound "
                        "(only needed when changing an existing binding)")
    args = p.parse_args(argv)

    if not os.path.exists(args.config):
        raise SystemExit(f"no config at {args.config} -- run the agent once first")
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg["coordinator"].rstrip("/")
    node_id, token = cfg.get("node_id"), cfg.get("node_token")
    if not node_id or not token:
        raise SystemExit("this config has no node_id/node_token yet -- run the agent once so "
                         "it registers, then come back")
    headers = {"X-Node-Token": token}

    current = requests.get(f"{base}/node/{node_id}/payout-address",
                           headers=headers, timeout=15).json().get("payout_address")
    print(f"node    : {node_id}")
    print(f"bound to: {current or 'nothing yet'}")

    if not args.address:
        if not current:
            print("\nPass --address 0xYourWallet to start binding one.")
        return 0

    r = requests.get(f"{base}/node/{node_id}/payout-challenge",
                     params={"address": args.address}, headers=headers, timeout=15)
    if r.status_code == 400:
        raise SystemExit(f"coordinator rejected that address: {r.json().get('detail')}")
    r.raise_for_status()
    ch = r.json()

    if not args.signature:
        print("\nSign EXACTLY this text with the key for that address, then re-run with "
              "--signature:\n")
        print("-" * 72)
        print(ch["message"])
        print("-" * 72)
        print(f"\n(the nonce expires in {int(ch['expires_in_seconds'])}s -- re-running this "
              f"command issues a fresh one, so sign and submit in one sitting)")
        if current and current.lower() != ch["address"].lower():
            print(f"\nThis CHANGES an existing binding ({current}), so you will also need "
                  f"--old-signature: the same text signed by that address.")
        return 0

    body = {"address": ch["address"], "nonce": ch["nonce"], "signature": args.signature}
    if args.old_signature:
        body["old_signature"] = args.old_signature
    r = requests.post(f"{base}/node/{node_id}/payout-address", json=body,
                      headers=headers, timeout=15)
    if r.status_code == 400:
        raise SystemExit(f"refused: {r.json().get('detail')}")
    r.raise_for_status()
    out = r.json()
    print(f"\nbound   : {out['payout_address']}"
          + (f"  (was {out['previous_address']})" if out.get("rebound") else ""))

    # Persist it so the agent stops trying to generate its own key on the next start.
    cfg["payout_address"] = out["payout_address"]
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
