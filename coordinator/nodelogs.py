"""
coordinator/nodelogs.py — get a node's log off that node and in front of an operator.

Until now a NEURON node's log existed only on the machine running it (`agent/agent.log`), which
means the one thing needed to diagnose a broken node was reachable only by the person sitting at
it. That is how [P24] happened: a stranger's node was misbehaving for three days and the entire
diagnostic path was "please open this file and paste it to me". The coordinator's own log has the
same shape of problem, one `journalctl` away behind SSH.

**Nodes push; the coordinator never pulls.** Volunteer machines are behind NAT — that is the
whole reason the relay exists — so there is nothing to connect to. Instead the coordinator raises
a per-node `want_logs` flag, the node notices it on its next heartbeat (already a 30 s round
trip), uploads a tail, and the flag clears. No new inbound path, no new port, and a node that is
offline simply uploads whenever it comes back.

**A log from someone else's computer is their data, not ours.** Everything here follows from
that:

  * only ever a TAIL, capped at MAX_BYTES -- enough to see a failure, not a transcript of
    somebody's day;
  * `redact()` runs coordinator-side on arrival AND node-side before sending, because the
    node cannot be sure the coordinator is honest and the coordinator cannot be sure the node
    redacted. Neither end is trusted to be the only one that scrubs;
  * secrets never land in the database even briefly -- node tokens, the registration secret,
    relay tickets, payout keys, bearer headers;
  * a Windows home directory names a real person (`C:\\Users\\firstname`), so paths are
    collapsed to `~`;
  * one row per node, replaced on upload, so this cannot grow into an archive of a volunteer's
    activity nobody asked for.

Reading requires the operator secret. Uploading requires that node's own token.
"""
import time

import logtail
from coordinator import models

# The rules themselves live in logtail.py at the repo root, shared with the agent so the two ends
# cannot drift on what counts as a secret. Re-exported here so callers have one import.
MAX_BYTES = logtail.MAX_BYTES
redact = logtail.redact
tail_bytes = logtail.tail_bytes

SCHEMA = """
CREATE TABLE IF NOT EXISTS node_logs (
    node_id     TEXT PRIMARY KEY,
    uploaded_at REAL NOT NULL,
    lines       INTEGER NOT NULL DEFAULT 0,
    body        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS node_log_requests (
    node_id      TEXT PRIMARY KEY,
    requested_at REAL NOT NULL
);
"""

def init():
    """Create the tables. Safe to call repeatedly; called from models.init_db()'s caller."""
    with models._db() as c:                                    # noqa: SLF001
        c.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# requesting
# --------------------------------------------------------------------------- #
def request(node_ids):
    """Ask these nodes to upload on their next heartbeat. Returns the ids marked."""
    now = time.time()
    ids = [n for n in node_ids if n]
    with models._db() as c:                                    # noqa: SLF001
        for nid in ids:
            c.execute("INSERT INTO node_log_requests (node_id, requested_at) VALUES (?,?) "
                      "ON CONFLICT(node_id) DO UPDATE SET requested_at=excluded.requested_at",
                      (nid, now))
    return ids


def wanted(node_id):
    """Is an upload outstanding for this node? Read on every heartbeat, so it stays a single
    indexed primary-key lookup and nothing more."""
    with models._db() as c:                                    # noqa: SLF001
        row = c.execute("SELECT 1 FROM node_log_requests WHERE node_id=?", (node_id,)).fetchone()
    return row is not None


def clear(node_id):
    with models._db() as c:                                    # noqa: SLF001
        c.execute("DELETE FROM node_log_requests WHERE node_id=?", (node_id,))


def pending():
    with models._db() as c:                                    # noqa: SLF001
        return [r["node_id"] for r in
                c.execute("SELECT node_id FROM node_log_requests ORDER BY node_id").fetchall()]


# --------------------------------------------------------------------------- #
# storing / reading
# --------------------------------------------------------------------------- #
def store(node_id, body):
    """Record an upload. Redacts and caps AGAIN here: the node already did both, but a node is
    not a trusted place to enforce a rule that protects the node's own operator. Clears the
    request so the node stops uploading. Returns what was stored."""
    body = redact(tail_bytes(body or ""))
    now = time.time()
    with models._db() as c:                                    # noqa: SLF001
        c.execute("INSERT INTO node_logs (node_id, uploaded_at, lines, body) VALUES (?,?,?,?) "
                  "ON CONFLICT(node_id) DO UPDATE SET uploaded_at=excluded.uploaded_at, "
                  "lines=excluded.lines, body=excluded.body",
                  (node_id, now, body.count("\n"), body))
        c.execute("DELETE FROM node_log_requests WHERE node_id=?", (node_id,))
    return {"node_id": node_id, "uploaded_at": now, "bytes": len(body.encode("utf-8")),
            "lines": body.count("\n")}


def get(node_id):
    with models._db() as c:                                    # noqa: SLF001
        row = c.execute("SELECT * FROM node_logs WHERE node_id=?", (node_id,)).fetchone()
    return dict(row) if row else None


def summary():
    """What we hold, without the bodies -- so an operator can see who has answered a request
    and who is still silent, which is itself the diagnosis when a node is dead."""
    with models._db() as c:                                    # noqa: SLF001
        rows = c.execute("SELECT node_id, uploaded_at, lines, LENGTH(body) AS bytes "
                         "FROM node_logs ORDER BY uploaded_at DESC").fetchall()
    return [dict(r) for r in rows]
