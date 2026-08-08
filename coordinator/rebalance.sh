#!/usr/bin/env bash
# coordinator/rebalance.sh — re-split the model across whoever is online, right now.
#
#   ./coordinator/rebalance.sh              # apply the balanced split
#   ./coordinator/rebalance.sh --dry-run    # show the split it WOULD apply, change nothing
#
# Exists because the one-liner form of this needs `$(...)` command substitution to read the
# operator secret out of .env.coordinator, and that is Git Bash syntax. Pasted into cmd.exe it
# silently sends the literal text `$(grep ...)` as the header, the coordinator answers 401, and
# `curl -s` swallows it — so it looks like it worked and nothing happened. A script cannot be
# pasted into the wrong shell by accident.
#
# The secret is read from the file and never printed. It never has to be typed or known.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVFILE="${NEURON_ENV_FILE:-$here/.env.coordinator}"
COORD="${NEURON_COORDINATOR:-https://neuronnet.duckdns.org}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

if [ ! -f "$ENVFILE" ]; then
  echo "no $ENVFILE — set NEURON_REGISTER_SECRET in the environment, or NEURON_ENV_FILE to its file"
  exit 2
fi
SECRET="${NEURON_REGISTER_SECRET:-$(grep '^NEURON_REGISTER_SECRET=' "$ENVFILE" | cut -d= -f2- | tr -d '"'"'"' \r')}"
if [ -z "$SECRET" ]; then
  echo "NEURON_REGISTER_SECRET is empty in $ENVFILE"
  exit 2
fi

say "before"
curl -fsS -m 20 "$COORD/status"; echo

say "the split the balancer recommends"
# Advisory only — /network/plan computes but never applies. Shows what --dry-run would do.
curl -fsS -m 20 "$COORD/network/plan"; echo

if [ -n "$DRY" ]; then
  say "dry run — nothing applied"
  exit 0
fi

say "applying it"
# -f so a 401 is a loud failure and not an empty line, which is exactly how the one-liner
# version of this failed silently.
if ! curl -fsS -m 30 -X POST -H "X-Register-Secret: $SECRET" "$COORD/network/rebalance"; then
  echo
  echo "REBALANCE FAILED. A 401 here means the secret in $ENVFILE does not match the one the"
  echo "coordinator is running with (systemd: Environment=NEURON_REGISTER_SECRET=...)."
  exit 1
fi
echo

say "after"
curl -fsS -m 20 "$COORD/status"; echo
echo "   Each node picks up its new range on its next restart. Because a node reuses a slice"
echo "   that already CONTAINS its new range, this costs no download on a machine that has"
echo "   been through a re-split before."
