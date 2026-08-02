"""coordinator/backup.py — consistent snapshots of the coordinator database.

WHY THIS EXISTS. The coordinator holds every NRN balance, every node identity and token, and
every wallet-to-login mapping, in ONE SQLite file on ONE disk. As of 2026-08-02 there was no
backup of it anywhere: no copy, no cron job, nothing. The host is an Oracle *Always Free* VM,
which Oracle documents as reclaimable when idle — so the most likely way NEURON loses its coin
is not an attacker but a housekeeping job at a cloud provider.

Uses sqlite3's ONLINE BACKUP API rather than copying the file. A plain `cp` of a live SQLite
database in WAL mode can capture a torn page set — the copy looks fine and fails to open later,
which is the worst kind of backup. The backup API takes a transactionally consistent snapshot
while the coordinator keeps serving.

    python -m coordinator.backup                  # one snapshot into ./backups
    python -m coordinator.backup --keep 48        # rotate, keeping the newest 48
    python -m coordinator.backup --verify-only <f>  # check a snapshot really opens

Off-box copies are the caller's job (scp/rsync/object storage) — this module deliberately has
no credentials and no network access.
"""
import argparse
import glob
import os
import sqlite3
import sys
import time

from coordinator import config

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backups")


def snapshot(dest_dir=DEFAULT_DIR, db_path=None):
    """Write a consistent snapshot. Returns its path."""
    db_path = db_path or config.DB_PATH
    os.makedirs(dest_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.join(dest_dir, f"neuron-{stamp}.db")
    src = sqlite3.connect(db_path)
    try:
        dst = sqlite3.connect(out)
        try:
            src.backup(dst)          # online backup API: consistent, no service interruption
        finally:
            dst.close()
    finally:
        src.close()
    return out


def verify(path):
    """Open the snapshot and confirm it is a usable database, not just a file of the right size.

    A backup nobody has restored is a hope, not a backup. This runs SQLite's own integrity
    check and then asserts the ledger invariant the live system maintains
    (SUM(balance) == GENESIS_TOTAL_SUPPLY), so a snapshot that is readable but economically
    nonsense is still reported as bad.
    """
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    try:
        if c.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            return False, "integrity_check failed"
        nodes = c.execute("SELECT COUNT(*) n FROM nodes").fetchone()["n"]
        row = c.execute("SELECT COALESCE(SUM(balance),0) t FROM ledger").fetchone()
        total = round(row["t"], 4)
        if not c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ledger'"
                         ).fetchone():
            return False, "no ledger table"
        # Genesis may not have been seeded yet on a brand-new database; only enforce the
        # invariant once there is money in it at all.
        if total > 0 and abs(total - config.GENESIS_TOTAL_SUPPLY) > 0.01:
            return False, (f"supply invariant broken: SUM(balance)={total} != "
                           f"{config.GENESIS_TOTAL_SUPPLY}")
        return True, f"ok — {nodes} node(s), SUM(balance)={total}"
    except sqlite3.DatabaseError as e:
        return False, f"unreadable: {e}"
    finally:
        c.close()


def rotate(dest_dir=DEFAULT_DIR, keep=48):
    """Delete all but the newest `keep` snapshots. Returns how many were removed."""
    files = sorted(glob.glob(os.path.join(dest_dir, "neuron-*.db")))
    doomed = files[:-keep] if keep > 0 else []
    for f in doomed:
        try:
            os.remove(f)
        except OSError:
            pass
    return len(doomed)


def main():
    ap = argparse.ArgumentParser(description="Snapshot the coordinator database.")
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--keep", type=int, default=48, help="how many snapshots to retain")
    ap.add_argument("--verify-only", metavar="FILE", help="check an existing snapshot and exit")
    args = ap.parse_args()

    if args.verify_only:
        good, msg = verify(args.verify_only)
        print(f"{'OK  ' if good else 'BAD '} {args.verify_only}: {msg}")
        return 0 if good else 1

    path = snapshot(args.dir)
    good, msg = verify(path)
    if not good:
        # A snapshot that does not verify is worse than none, because it looks like safety.
        os.remove(path)
        print(f"BACKUP FAILED and was discarded: {msg}", file=sys.stderr)
        return 1
    removed = rotate(args.dir, args.keep)
    size = os.path.getsize(path)
    print(f"backup ok: {path} ({size/1024:.0f} KB) — {msg}"
          + (f"; rotated out {removed}" if removed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
