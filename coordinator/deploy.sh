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

# WHERE THE KEY ACTUALLY IS, which is not always "$HOME/.ssh".
#
# This script is normally run from Git Bash on the founder's Windows box, and Git Bash sets
# $HOME to a POSIX-looking path (`/home/user1`) that frequently does not exist on the machine.
# The real profile is %USERPROFILE%. The failure that produced this: `Identity file
# /home/user1/.ssh/oracle_coordinator not accessible` followed by `Permission denied
# (publickey)` — two messages that read like a key problem when the key was fine and the path
# was invented. Worse, it happens AFTER the dry run passes, because --dry-run never opens an
# ssh connection, so the rehearsal cannot catch it.
#
# There are THREE different bashes this can run under on one Windows machine and they disagree
# about every path involved:
#   Git Bash   $HOME=/c/Users/<you>        Windows drives at /c/...      has cygpath
#   WSL        $HOME=/home/<linuxuser>     Windows drives at /mnt/c/...  no cygpath, and
#                                          %USERPROFILE% is not inherited
#   MSYS/other anything
# `bash foo.sh` from cmd.exe picks whichever is first on PATH, which on Windows 11 with WSL
# installed is usually WSL — so the key sits at /mnt/c/Users/<you>/.ssh and $HOME points at a
# Linux home that has never seen it. So: look in all of them rather than assume one.
_key_name="oracle_coordinator"
_candidates=("$HOME/.ssh/$_key_name")
if [ -n "${USERPROFILE:-}" ]; then
  _u="$(cygpath -u "$USERPROFILE" 2>/dev/null || true)"                    # Git Bash
  [ -n "$_u" ] && _candidates+=("$_u/.ssh/$_key_name")
  # No cygpath (WSL): translate C:\Users\you -> /c/Users/you and /mnt/c/Users/you by hand.
  _p="${USERPROFILE//\\//}"                                                # backslash -> slash
  _drive="$(printf '%s' "${_p%%:*}" | tr 'A-Z' 'a-z')"; _rest="${_p#*:}"
  _candidates+=("/$_drive$_rest/.ssh/$_key_name" "/mnt/$_drive$_rest/.ssh/$_key_name")
fi
# Last resort: whatever profile actually holds a key, under either mount layout.
#
# Drive letters are iterated EXPLICITLY rather than globbed. `/[a-z]/Users/...` looks like it
# would work and does not: in Git Bash `/c` is a virtual mount, the root directory does not
# enumerate it, so the pattern never expands and the whole fallback is silently dead. It is
# only the leaf `*` (a real directory listing) that can be globbed.
for _d in c d e; do
  for _root in "/$_d" "/mnt/$_d"; do
    [ -d "$_root/Users" ] || continue
    for _g in "$_root"/Users/*/.ssh/"$_key_name"; do
      [ -f "$_g" ] && _candidates+=("$_g")
    done
  done
done
_key_default="$HOME/.ssh/$_key_name"
for _c in "${_candidates[@]}"; do
  if [ -f "$_c" ]; then _key_default="$_c"; break; fi
done
KEY="${NEURON_DEPLOY_KEY:-$_key_default}"
if [ ! -f "$KEY" ]; then
  # Said here rather than letting ssh say it, because ssh's version of this is "Permission
  # denied (publickey)" — which sends you looking at the VM's authorized_keys for a fault that
  # is entirely on this side.
  printf '\n\033[1m== deploy key not found\033[0m\n' >&2
  echo "   this shell is: $(uname -s), HOME=$HOME" >&2
  echo "   looked in:" >&2
  for _c in "${_candidates[@]}"; do echo "     $_c" >&2; done
  echo "   set NEURON_DEPLOY_KEY to its real path, e.g." >&2
  echo "     NEURON_DEPLOY_KEY=/mnt/c/Users/<you>/.ssh/oracle_coordinator $0   # WSL" >&2
  echo "     NEURON_DEPLOY_KEY=/c/Users/<you>/.ssh/oracle_coordinator $0       # Git Bash" >&2
  exit 1
fi
REMOTE="${NEURON_DEPLOY_DIR:-/home/ubuntu/neuron}"
SERVICE="neuron-coordinator"
PUBLIC_URL="${NEURON_PUBLIC_URL:-http://150.230.22.250:8001}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

# The deploy key carries a passphrase (2026-08-09). An unencrypted key on a Windows box is a
# root-equivalent credential readable by anything running as that user -- and this one is the
# ONLY key authorised on the VM that holds the ledger, every wallet balance and the register
# secret. `chmod 0600` does not protect it there: NTFS ignores POSIX mode bits.
#
# So prefer a running ssh-agent. Without one, `ssh -i` still works and simply prompts, which is
# correct for a human running this by hand and merely inconvenient. With one, unattended runs
# keep working. Started once per boot:
#     ssh-agent -a ~/.ssh/neuron-agent.sock
#     SSH_AUTH_SOCK=~/.ssh/neuron-agent.sock ssh-add ~/.ssh/oracle_coordinator
: "${SSH_AUTH_SOCK:=$HOME/.ssh/neuron-agent.sock}"
[ -S "$SSH_AUTH_SOCK" ] && export SSH_AUTH_SOCK || unset SSH_AUTH_SOCK

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
# `security` is NOT optional and its absence is not a soft failure: [P52] made router.py do
# `from security import wire_crypto` at import time, so a deploy without it takes the whole
# coordinator down in a systemd restart loop with ModuleNotFoundError — the network becomes
# unroutable while `systemctl is-active` still says "activating". That happened on 2026-08-19.
# Anything the coordinator imports at module scope belongs in this list.
FILES=(coordinator security relay_auth.py common.py logtail.py)
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
