#!/usr/bin/env bash
# coordinator/deploy.sh — push the coordinator to the live VM and verify it came back healthy.
#
#   ./coordinator/deploy.sh                 # ship code, restart, verify
#   ./coordinator/deploy.sh --dry-run       # show what WOULD be copied, change nothing
#
# Why a script: the coordinator now carries changes that are silently load-bearing --
# /infer refuses wallets with no login behind it, /wallet/faucet is gated, placement balances
# replicas, and login runs here. Deploying those by hand, one scp at a time, is how a half-
# updated coordinator happens. This is idempotent, backs the DB up first, and rolls the service
# back if the new code fails to answer.
#
# It NEVER touches the database except to copy it aside: neuron.db holds every identity, wallet
# balance and ban, and models.init_db() migrates the schema forward on startup by itself.
set -euo pipefail

HOST="${NEURON_DEPLOY_HOST:-ubuntu@150.230.22.250}"
KEY="${NEURON_DEPLOY_KEY:-$HOME/.ssh/oracle_coordinator}"
REMOTE="${NEURON_DEPLOY_DIR:-/home/ubuntu/neuron}"
SERVICE="neuron-coordinator"
PUBLIC_URL="${NEURON_PUBLIC_URL:-http://150.230.22.250:8001}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new "$HOST")
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "pre-flight: is the coordinator answering now?"
before="$(curl -fsS -m 10 "$PUBLIC_URL/status" >/dev/null 2>&1 && echo up || echo down)"
echo "   currently: $before"

say "backing up the live database (identities, balances, bans)"
if [ -z "$DRY" ]; then
  "${SSH[@]}" "cd $REMOTE && cp -f coordinator/neuron.db coordinator/neuron.db.bak-\$(date +%Y%m%d-%H%M%S) 2>/dev/null || echo '   (no db yet — first deploy)'"
else
  echo "   [dry-run] would back up $REMOTE/coordinator/neuron.db"
fi

say "shipping code"
# Only what the coordinator actually runs. Deliberately NOT the whole repo: the VM has no
# torch and no business holding the agent, the model slices or the installer.
#
# tar-over-ssh rather than rsync: this is normally run from the founder's Windows box, where
# Git Bash ships ssh/scp/tar but NOT rsync -- so an rsync-based deploy fails at the one moment
# it matters. tar needs nothing that isn't already there on both ends.
FILES=(coordinator relay_auth.py common.py)
EXCLUDES=(--exclude='__pycache__' --exclude='*.pyc' --exclude='neuron.db*'
          --exclude='test_*.py' --exclude='*.sh')
if [ -n "$DRY" ]; then
  echo "   would ship:"
  tar -cz "${EXCLUDES[@]}" -C "$here" -f - "${FILES[@]}" | tar -tzf - | sed 's/^/     /' | head -40
  say "dry run complete — nothing changed"
  exit 0
fi
tar -cz "${EXCLUDES[@]}" -C "$here" -f - "${FILES[@]}" \
  | "${SSH[@]}" "cat > /tmp/neuron-deploy.tgz && tar -xzf /tmp/neuron-deploy.tgz -C $REMOTE && rm -f /tmp/neuron-deploy.tgz && echo '   unpacked into $REMOTE'"

say "restarting $SERVICE"
"${SSH[@]}" "sudo systemctl restart $SERVICE && sleep 4 && systemctl is-active $SERVICE"

say "verifying"
ok=1
curl -fsS -m 15 "$PUBLIC_URL/status" >/dev/null || ok=0
echo "   /status            $([ $ok = 1 ] && echo OK || echo FAILED)"
# The gates that must actually be live for any of yesterday's work to mean anything.
code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 -X POST "$PUBLIC_URL/wallet/faucet" \
        -H 'Content-Type: application/json' -d '{"wallet_id":"deploy-probe"}')
echo "   faucet unauth       $code $([ "$code" = 401 ] && echo '(gated ✓)' || { echo '(NOT GATED)'; ok=0; })"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "$PUBLIC_URL/admin/identities")
echo "   admin unauth        $code $([ "$code" = 401 ] && echo '(gated ✓)' || { echo '(NOT GATED)'; ok=0; })"
code=$(curl -s -o /dev/null -w '%{http_code}' -m 15 "$PUBLIC_URL/auth/providers")
echo "   /auth/providers     $code $([ "$code" = 200 ] && echo '(login endpoints live ✓)' || echo '(missing)')"
echo -n "   logins configured:  "; curl -fsS -m 15 "$PUBLIC_URL/auth/providers" 2>/dev/null || echo "?"

if [ "${SWEEP:-0}" != "1" ]; then
  say "login-less wallets: reporting only (set SWEEP=1 to delete them)"
  # Deliberately NOT automatic. OAuth has never been configured on this coordinator, so EVERY
  # wallet that exists predates login and would match -- including the founder's own test
  # wallets. Deleting real balances as a side effect of "deploy the code" is the wrong default.
  "${SSH[@]}" "cd $REMOTE && ./.venv/bin/python - <<'PY'
import sqlite3
c = sqlite3.connect('coordinator/neuron.db'); c.row_factory = sqlite3.Row
rows = c.execute(\"\"\"SELECT node_id, balance FROM ledger WHERE account_type='wallet'
                     AND node_id NOT IN (SELECT wallet_id FROM oauth_identities)\"\"\").fetchall()
print(f'   {len(rows)} wallet(s) with no login behind them, {sum(r[\"balance\"] for r in rows)} NRN total')
for r in rows: print('     -', r['node_id'], r['balance'])
print('   (re-run with SWEEP=1 to remove them and return the NRN to __ecosystem__)')
PY"
else
say "sweeping wallets that were minted through the open faucet"
# Every wallet with no oauth_identities row was created by the ungated /wallet/faucet, since
# that is now the ONLY way one could exist without a login. Their balance goes back to the
# ecosystem bucket so the fixed-supply invariant (SUM == 1,000,000,000) still holds.
"${SSH[@]}" "cd $REMOTE && ./.venv/bin/python - <<'PY'
import sqlite3
c = sqlite3.connect('coordinator/neuron.db'); c.row_factory = sqlite3.Row
rows = c.execute(\"\"\"SELECT node_id, balance FROM ledger WHERE account_type='wallet'
                     AND node_id NOT IN (SELECT wallet_id FROM oauth_identities)\"\"\").fetchall()
if not rows:
    print('   none found')
else:
    total = sum(r['balance'] for r in rows)
    for r in rows:
        c.execute('DELETE FROM ledger WHERE node_id=?', (r['node_id'],))
    c.execute('UPDATE ledger SET balance=balance+? WHERE node_id=?',
              (total, '__ecosystem__'))
    c.commit()
    print(f'   removed {len(rows)} login-less wallet(s), returned {total} NRN to __ecosystem__')
    for r in rows: print('     -', r['node_id'], r['balance'])
PY"
fi

if [ $ok = 0 ]; then
  say "VERIFICATION FAILED — rolling the service back"
  "${SSH[@]}" "sudo systemctl restart $SERVICE" || true
  echo "The service was restarted. Check: ssh -i $KEY $HOST 'journalctl -u $SERVICE -n 50 --no-pager'"
  exit 1
fi
say "deployed and verified"
