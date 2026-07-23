"""NEURON coordinator — SQLite storage layer.

Plain `sqlite3` (no ORM, no setup). Every call opens a short-lived connection in
WAL mode with a busy timeout, which is safe under FastAPI's threadpool. Three
tables: nodes, ledger, requests. Node online/offline status is computed from
`last_seen` on read so it is always accurate between background sweeps.
"""
import contextlib
import json
import sqlite3
import time

from coordinator import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id       TEXT PRIMARY KEY,
    tailscale_ip  TEXT NOT NULL,
    port          INTEGER NOT NULL,
    layer_start   INTEGER NOT NULL,
    layer_end     INTEGER NOT NULL,
    cores         INTEGER,
    ram_gb        REAL,
    node_token    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'online',
    last_seen     REAL NOT NULL,
    registered_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    node_id         TEXT PRIMARY KEY,
    balance         REAL NOT NULL DEFAULT 0,
    total_earned    REAL NOT NULL DEFAULT 0,
    requests_served INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS requests (
    request_id       TEXT PRIMARY KEY,
    prompt           TEXT,
    max_tokens       INTEGER,
    status           TEXT NOT NULL DEFAULT 'pending',
    created_at       REAL NOT NULL,
    completed_at     REAL,
    tokens_generated INTEGER,
    duration_ms      INTEGER,
    node_ids         TEXT
);
"""


@contextlib.contextmanager
def _db():
    conn = sqlite3.connect(config.DB_PATH, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _db() as c:
        c.executescript(SCHEMA)


def _status(last_seen, now=None):
    now = now if now is not None else time.time()
    return "online" if (now - last_seen) <= config.HEARTBEAT_TIMEOUT_S else "offline"


def _node_dict(row, now=None):
    d = dict(row)
    d["status"] = _status(d["last_seen"], now)
    d["assigned_layers"] = [d["layer_start"], d["layer_end"]]
    return d


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def register_node(node_id, tailscale_ip, port, layer_start, layer_end, cores,
                  ram_gb, token):
    now = time.time()
    with _db() as c:
        c.execute(
            """INSERT INTO nodes (node_id, tailscale_ip, port, layer_start, layer_end,
                                  cores, ram_gb, node_token, status, last_seen, registered_at)
               VALUES (?,?,?,?,?,?,?,?, 'online', ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   tailscale_ip=excluded.tailscale_ip, port=excluded.port,
                   layer_start=excluded.layer_start, layer_end=excluded.layer_end,
                   cores=excluded.cores, ram_gb=excluded.ram_gb,
                   node_token=excluded.node_token, status='online', last_seen=excluded.last_seen""",
            (node_id, tailscale_ip, port, layer_start, layer_end, cores, ram_gb,
             token, now, now),
        )
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (node_id,))


def get_node(node_id, now=None):
    with _db() as c:
        row = c.execute("SELECT * FROM nodes WHERE node_id=?", (node_id,)).fetchone()
    return _node_dict(row, now) if row else None


def list_nodes(now=None):
    now = now if now is not None else time.time()
    with _db() as c:
        rows = c.execute("SELECT * FROM nodes ORDER BY layer_start").fetchall()
    return [_node_dict(r, now) for r in rows]


def online_nodes(now=None):
    return [n for n in list_nodes(now) if n["status"] == "online"]


def delete_node(node_id):
    with _db() as c:
        cur = c.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
        return cur.rowcount > 0


def touch_node(node_id):
    """Heartbeat: refresh last_seen and mark online. Returns True if node exists."""
    now = time.time()
    with _db() as c:
        cur = c.execute(
            "UPDATE nodes SET last_seen=?, status='online' WHERE node_id=?",
            (now, node_id),
        )
        return cur.rowcount > 0


def set_stored_status(node_id, status):
    with _db() as c:
        c.execute("UPDATE nodes SET status=? WHERE node_id=?", (status, node_id))


def sweep(now=None):
    """Mark stale online nodes offline. Returns node_ids that just went offline
    (for logging). Nodes that resumed pinging were set 'online' by touch_node()."""
    now = now if now is not None else time.time()
    cutoff = now - config.HEARTBEAT_TIMEOUT_S
    with _db() as c:
        newly = [r["node_id"] for r in c.execute(
            "SELECT node_id FROM nodes WHERE status='online' AND last_seen < ?",
            (cutoff,)).fetchall()]
        if newly:
            c.execute("UPDATE nodes SET status='offline' "
                      "WHERE status='online' AND last_seen < ?", (cutoff,))
    return newly


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def credit(node_id, amount, count_request=True):
    with _db() as c:
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (node_id,))
        c.execute(
            "UPDATE ledger SET balance=balance+?, total_earned=total_earned+?, "
            "requests_served=requests_served+? WHERE node_id=?",
            (amount, amount, 1 if count_request else 0, node_id),
        )


def get_ledger(node_id):
    with _db() as c:
        row = c.execute("SELECT * FROM ledger WHERE node_id=?", (node_id,)).fetchone()
    return dict(row) if row else None


def node_ledgers():
    """Real-node ledgers only (excludes the coordinator's fee row)."""
    with _db() as c:
        rows = c.execute(
            "SELECT * FROM ledger WHERE node_id != ?", (config.COORDINATOR_LEDGER_ID,)
        ).fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def create_request(request_id, prompt, max_tokens):
    with _db() as c:
        c.execute(
            "INSERT INTO requests (request_id, prompt, max_tokens, status, created_at) "
            "VALUES (?,?,?, 'pending', ?)",
            (request_id, prompt, max_tokens, time.time()),
        )


def get_request(request_id):
    with _db() as c:
        row = c.execute(
            "SELECT * FROM requests WHERE request_id=?", (request_id,)
        ).fetchone()
    return dict(row) if row else None


def complete_request(request_id, tokens_generated, duration_ms, node_ids):
    with _db() as c:
        cur = c.execute(
            "UPDATE requests SET status='completed', completed_at=?, tokens_generated=?, "
            "duration_ms=?, node_ids=? WHERE request_id=? AND status='pending'",
            (time.time(), tokens_generated, duration_ms, json.dumps(node_ids), request_id),
        )
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Aggregate stats
# --------------------------------------------------------------------------- #
def network_stats():
    with _db() as c:
        req = c.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(tokens_generated),0) AS toks "
            "FROM requests WHERE status='completed'"
        ).fetchone()
        distributed = c.execute(
            "SELECT COALESCE(SUM(total_earned),0) AS d FROM ledger WHERE node_id != ?",
            (config.COORDINATOR_LEDGER_ID,),
        ).fetchone()
    return {
        "total_requests_served": req["n"],
        "total_tokens_generated": int(req["toks"]),
        "total_nrn_distributed": round(distributed["d"], 4),
    }
