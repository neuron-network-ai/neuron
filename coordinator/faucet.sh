#!/usr/bin/env bash
# coordinator/faucet.sh — claim the one-time NRN grant for a wallet, so it can pay for a request.
#
#   ./coordinator/faucet.sh <wallet_id>
#   ./coordinator/faucet.sh <wallet_id> --balance   # just report the balance, grant nothing
#
# Why this exists: /infer places a HOLD before it dispatches (0.158 NRN for a short prompt), so a
# wallet with no balance gets a 402 and no chain is ever built. On 2026-08-07 that surfaced in the
# chat as "a node dropped out while loading its slice" and cost an hour of restarting a machine
# that was working perfectly. An empty wallet is now the first thing to rule out, and this is how.
#
# The secret is never typed, printed, or stored on this machine. /wallet/faucet is gated by
# X-Wallet-Link-Secret, which lives ONLY in the coordinator's systemd environment on the VM -- so
# the curl runs THERE, reading it straight out of `systemctl show`. Nothing sensitive crosses the
# wire and nothing lands in shell history.
#
# It also cannot fund an arbitrary string: the endpoint requires the wallet to have come from a
# real Google/GitHub login (models.is_oauth_wallet). That gate is the reason the faucet stopped
# being a way to mint unlimited clean-record wallets, so this script deliberately does not work
# around it -- a 403 here means "log in first", not "try harder".
set -euo pipefail

WALLET="${1:-}"
MODE="${2:-}"
HOST="${NEURON_DEPLOY_HOST:-ubuntu@150.230.22.250}"
KEY="${NEURON_DEPLOY_KEY:-$HOME/.ssh/oracle_coordinator}"
LOCAL_URL="${NEURON_COORDINATOR_LOCAL:-http://127.0.0.1:8001}"

if [ -z "$WALLET" ]; then
  cat <<'USAGE'
usage: ./coordinator/faucet.sh <wallet_id> [--balance]

Get your wallet id from http://localhost:8080/wallet/balance while signed in.
If that says {"logged_in":false}, sign in on the chat page first -- the faucet only
funds wallets created by a real Google/GitHub login.
USAGE
  exit 2
fi

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "balance before"
"${SSH[@]}" "curl -fsS -m 20 '$LOCAL_URL/wallet/$WALLET'" || echo "   (no ledger row yet for $WALLET)"
echo

if [ "$MODE" = "--balance" ]; then
  say "balance only — nothing granted"
  exit 0
fi

say "claiming the faucet grant"
# -f so a 401/403/409 is a loud failure rather than an empty line. The one-liner form of this
# swallowed exactly those, which is how "it looked like it worked and nothing happened" happens.
if ! "${SSH[@]}" "curl -fsS -m 20 -X POST '$LOCAL_URL/wallet/faucet' \
      -H 'Content-Type: application/json' \
      -H \"X-Wallet-Link-Secret: \$(systemctl show neuron-coordinator -p Environment \
          | tr ' ' '\n' | sed -n 's/^NEURON_WALLET_LINK_SECRET=//p')\" \
      -d '{\"wallet_id\":\"$WALLET\"}'"; then
  echo
  echo "FAILED. What the status codes mean:"
  echo "  401  the coordinator's NEURON_WALLET_LINK_SECRET is not set, or systemctl show did not"
  echo "       return it -- check: systemctl show neuron-coordinator -p Environment"
  echo "  403  this wallet did not come from a Google/GitHub login. Sign in on the chat page and"
  echo "       use the wallet id from /wallet/balance."
  echo "  409  already claimed for this wallet. The grant is one-time; check the balance above."
  exit 1
fi
echo

say "balance after"
"${SSH[@]}" "curl -fsS -m 20 '$LOCAL_URL/wallet/$WALLET'"
echo
echo "   Now reload the chat page and send a prompt. With NEURON_FORCE_NETWORK=1 the request"
echo "   goes over the node chain, so the reply should name a real node instead of 'this"
echo "   machine' and report a non-zero NRN cost."
