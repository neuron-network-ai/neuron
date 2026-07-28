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
    ms_per_layer  REAL,
    head_ms       REAL,
    challenges_passed INTEGER NOT NULL DEFAULT 0,
    challenges_failed INTEGER NOT NULL DEFAULT 0,
    trusted       INTEGER NOT NULL DEFAULT 0,
    node_token    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'online',
    last_seen     REAL NOT NULL,
    registered_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger (
    node_id         TEXT PRIMARY KEY,
    balance         REAL NOT NULL DEFAULT 0,
    total_earned    REAL NOT NULL DEFAULT 0,
    requests_served INTEGER NOT NULL DEFAULT 0,
    account_type    TEXT NOT NULL DEFAULT 'node',
    faucet_claimed  INTEGER NOT NULL DEFAULT 0,
    locked_until    REAL
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
    node_ids         TEXT,
    plan_node_ids    TEXT,
    complete_token   TEXT,
    wallet_id        TEXT,
    hold_amount      REAL
);
CREATE TABLE IF NOT EXISTS holds (
    request_id  TEXT PRIMARY KEY,
    wallet_id   TEXT NOT NULL,
    amount      REAL NOT NULL,
    created_at  REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'held'
);
CREATE TABLE IF NOT EXISTS oauth_identities (
    provider     TEXT NOT NULL,
    external_id  TEXT NOT NULL,
    wallet_id    TEXT NOT NULL,
    email        TEXT,
    created_at   REAL NOT NULL,
    PRIMARY KEY (provider, external_id)
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
        # migration: add columns to a pre-existing nodes table
        cols = {r["name"] for r in c.execute("PRAGMA table_info(nodes)").fetchall()}
        for col in ("ms_per_layer", "head_ms"):            # S14
            if col not in cols:
                c.execute(f"ALTER TABLE nodes ADD COLUMN {col} REAL")
        for col in ("challenges_passed", "challenges_failed"):   # S16
            if col not in cols:
                c.execute(f"ALTER TABLE nodes ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
        if "trusted" not in cols:                                # S12 (open join)
            c.execute("ALTER TABLE nodes ADD COLUMN trusted INTEGER NOT NULL DEFAULT 0")
            # every pre-open-join node registered under the shared secret -> grandfather
            # them in as trusted so opening the door doesn't demote the live network.
            c.execute("UPDATE nodes SET trusted=1")
        # S18b ([P12]): record the routed chain + a per-request completion token so /complete
        # can be authenticated and settled from the coordinator's own plan, not caller input.
        rcols = {r["name"] for r in c.execute("PRAGMA table_info(requests)").fetchall()}
        for col in ("plan_node_ids", "complete_token", "wallet_id"):
            if col not in rcols:
                c.execute(f"ALTER TABLE requests ADD COLUMN {col} TEXT")
        if "hold_amount" not in rcols:
            c.execute("ALTER TABLE requests ADD COLUMN hold_amount REAL")
        # Fixed-supply ledger (Workstream B): every pre-existing ledger row (all real nodes
        # today) becomes account_type='node' via the column default -- zero data loss, zero
        # behavior change for existing rows. Buckets/wallets are new rows, not migrated ones.
        lcols = {r["name"] for r in c.execute("PRAGMA table_info(ledger)").fetchall()}
        if "account_type" not in lcols:
            c.execute("ALTER TABLE ledger ADD COLUMN account_type TEXT NOT NULL DEFAULT 'node'")
        if "faucet_claimed" not in lcols:
            c.execute("ALTER TABLE ledger ADD COLUMN faucet_claimed INTEGER NOT NULL DEFAULT 0")
        if "locked_until" not in lcols:
            c.execute("ALTER TABLE ledger ADD COLUMN locked_until REAL")


def _status(last_seen, now=None):
    now = now if now is not None else time.time()
    return "online" if (now - last_seen) <= config.HEARTBEAT_TIMEOUT_S else "offline"


def _node_dict(row, now=None):
    d = dict(row)
    d["status"] = _status(d["last_seen"], now)
    d["assigned_layers"] = [d["layer_start"], d["layer_end"]]
    # proof-of-compute reputation (Session 16)
    p, f = d.get("challenges_passed") or 0, d.get("challenges_failed") or 0
    total = p + f
    d["reputation"] = round(p / total, 3) if total else None
    d["flagged"] = (total >= config.REPUTATION_MIN_SAMPLES
                    and (p / total) < config.REPUTATION_THRESHOLD)
    # open join (Session 12): a node may serve live traffic and earn NRN only once it is
    # TRUSTED (registered with the secret) or has passed proof-of-compute enough times.
    # A fresh stranger is PROBATIONARY — reachable and challengeable, but not in production.
    d["trusted"] = bool(d.get("trusted"))
    passed = p >= config.PROBATION_MIN_PASSES
    d["eligible"] = (not d["flagged"]) and (d["trusted"] or passed)
    d["standing"] = ("flagged" if d["flagged"]
                     else "trusted" if d["trusted"]
                     else "verified" if passed
                     else "probationary")
    return d


def record_attestation(node_id, passed):
    """Log one proof-of-compute challenge result for a node. Returns True if it exists."""
    col = "challenges_passed" if passed else "challenges_failed"
    with _db() as c:
        cur = c.execute(f"UPDATE nodes SET {col} = {col} + 1 WHERE node_id=?", (node_id,))
        return cur.rowcount > 0


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def register_node(node_id, tailscale_ip, port, layer_start, layer_end, cores,
                  ram_gb, token, ms_per_layer=None, head_ms=None, trusted=False):
    now = time.time()
    with _db() as c:
        c.execute(
            """INSERT INTO nodes (node_id, tailscale_ip, port, layer_start, layer_end,
                                  cores, ram_gb, ms_per_layer, head_ms, trusted, node_token,
                                  status, last_seen, registered_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?, 'online', ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   tailscale_ip=excluded.tailscale_ip, port=excluded.port,
                   layer_start=excluded.layer_start, layer_end=excluded.layer_end,
                   cores=excluded.cores, ram_gb=excluded.ram_gb,
                   ms_per_layer=COALESCE(excluded.ms_per_layer, nodes.ms_per_layer),
                   head_ms=COALESCE(excluded.head_ms, nodes.head_ms),
                   trusted=excluded.trusted,
                   node_token=excluded.node_token, status='online', last_seen=excluded.last_seen""",
            (node_id, tailscale_ip, port, layer_start, layer_end, cores, ram_gb,
             ms_per_layer, head_ms, 1 if trusted else 0, token, now, now),
        )
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (node_id,))


def update_layers(node_id, layer_start, layer_end):
    """Reassign a node's layer range (used by the auto-balancer)."""
    with _db() as c:
        c.execute("UPDATE nodes SET layer_start=?, layer_end=? WHERE node_id=?",
                  (layer_start, layer_end, node_id))


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
# Fixed-supply wallet primitives (Workstream B / TOKENOMICS.md §11 Phase 0)
# --------------------------------------------------------------------------- #
def ensure_account(account_id, account_type="wallet"):
    """INSERT OR IGNORE a zero-balance row -- safe to call before any transfer touching an
    account that might not exist yet (a wallet's first-ever hold, e.g.)."""
    with _db() as c:
        c.execute("INSERT OR IGNORE INTO ledger (node_id, account_type) VALUES (?, ?)",
                  (account_id, account_type))


def debit(account_id, amount):
    """Conditional debit -- only succeeds if the balance actually covers `amount`. The
    WHERE clause is the whole safety property: a losing debit changes nothing (rowcount=0),
    so this is safe to call speculatively without a separate balance check first."""
    with _db() as c:
        cur = c.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (amount, account_id, amount))
        return cur.rowcount > 0


def transfer(from_id, to_id, amount, count_request=False):
    """Move `amount` from one account to another atomically (one connection, one commit).
    A losing debit (insufficient balance) leaves BOTH sides untouched -- the credit only
    runs if the debit's WHERE clause actually matched a row, so this never partially applies.
    `count_request` mirrors credit()'s own flag -- set it when `to_id` is a node being paid
    for serving a request (bumps its requests_served display stat), not for fees/refunds."""
    if amount <= 0:
        return True   # a zero/negative transfer is a no-op, not an error
    with _db() as c:
        cur = c.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (amount, from_id, amount))
        if cur.rowcount == 0:
            return False
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (to_id,))
        c.execute("UPDATE ledger SET balance=balance+?, total_earned=total_earned+?, "
                  "requests_served=requests_served+? WHERE node_id=?",
                  (amount, amount, 1 if count_request else 0, to_id))
        return True


def hold(request_id, wallet_id, amount):
    """Reserve `amount` from a wallet for an in-flight request -- a transfer INTO the
    bookkeeping-only __escrow__ account, so SUM(ledger.balance) never moves, only WHERE the
    money sits. Returns False (nothing held) on insufficient balance."""
    with _db() as c:
        cur = c.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (amount, wallet_id, amount))
        if cur.rowcount == 0:
            return False
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)",
                  (config.ESCROW_LEDGER_ID,))
        c.execute("UPDATE ledger SET balance=balance+? WHERE node_id=?",
                  (amount, config.ESCROW_LEDGER_ID))
        c.execute("INSERT INTO holds (request_id, wallet_id, amount, created_at, status) "
                  "VALUES (?,?,?,?, 'held')", (request_id, wallet_id, amount, time.time()))
        return True


def release_hold(request_id):
    """Return a still-`held` hold's full amount to its wallet (moderation-abort or a
    request that never got quoted a settlement). No-op if already settled/released/unknown."""
    with _db() as c:
        row = c.execute("SELECT * FROM holds WHERE request_id=? AND status='held'",
                        (request_id,)).fetchone()
        if not row:
            return False
        cur = c.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (row["amount"], config.ESCROW_LEDGER_ID, row["amount"]))
        if cur.rowcount == 0:
            return False   # escrow underfunded -- should never happen; fail closed, not silently
        c.execute("INSERT OR IGNORE INTO ledger (node_id) VALUES (?)", (row["wallet_id"],))
        c.execute("UPDATE ledger SET balance=balance+? WHERE node_id=?",
                  (row["amount"], row["wallet_id"]))
        c.execute("UPDATE holds SET status='released' WHERE request_id=?", (request_id,))
        return True


def release_stale_holds(ttl_s, now=None):
    """TTL sweep for abandoned/crashed requests (held but never completed or released).
    Meant to be piggybacked on the coordinator's existing health_loop. Returns the list of
    request_ids released, for logging."""
    now = now if now is not None else time.time()
    cutoff = now - ttl_s
    with _db() as c:
        stale = [r["request_id"] for r in c.execute(
            "SELECT request_id FROM holds WHERE status='held' AND created_at < ?",
            (cutoff,)).fetchall()]
    for rid in stale:
        release_hold(rid)
    return stale


def get_hold(request_id):
    with _db() as c:
        row = c.execute("SELECT * FROM holds WHERE request_id=?", (request_id,)).fetchone()
    return dict(row) if row else None


def mark_hold_settled(request_id):
    with _db() as c:
        c.execute("UPDATE holds SET status='settled' WHERE request_id=?", (request_id,))


def claim_faucet(wallet_id, amount):
    """One-time faucet grant per wallet_id. The UPDATE's WHERE clause (faucet_claimed=0) is
    the idempotency guard -- a second call for the same wallet always fails closed."""
    with _db() as c:
        c.execute("INSERT OR IGNORE INTO ledger (node_id, account_type) VALUES (?, 'wallet')",
                  (wallet_id,))
        cur = c.execute(
            "UPDATE ledger SET faucet_claimed=1 WHERE node_id=? AND faucet_claimed=0",
            (wallet_id,))
        if cur.rowcount == 0:
            return False
        c.execute(
            "UPDATE ledger SET balance=balance-? WHERE node_id=? AND balance>=?",
            (amount, config.GENESIS_BUCKETS_ECOSYSTEM_ID, amount))
        c.execute("UPDATE ledger SET balance=balance+?, total_earned=total_earned+? "
                  "WHERE node_id=?", (amount, amount, wallet_id))
        return True


def supply_snapshot():
    """Per-bucket balances + total + the invariant check -- the public /supply payload.
    Only genesis buckets + escrow are named individually; no wallet/node balances leak."""
    bucket_ids = (config.GENESIS_BUCKETS_EMISSION_ID, config.GENESIS_BUCKETS_FOUNDER_ID,
                 config.GENESIS_BUCKETS_ECOSYSTEM_ID, config.GENESIS_BUCKETS_LIQUIDITY_ID,
                 config.ESCROW_LEDGER_ID)
    with _db() as c:
        buckets = {}
        for bid in bucket_ids:
            row = c.execute("SELECT balance FROM ledger WHERE node_id=?", (bid,)).fetchone()
            buckets[bid] = row["balance"] if row else 0.0
        total = c.execute("SELECT COALESCE(SUM(balance),0) AS t FROM ledger").fetchone()["t"]
    return {"buckets": buckets, "total_supply": round(total, 6),
           "invariant_ok": abs(total - 1_000_000_000) < 1e-6}


def wallet_for_oauth(provider, external_id, email=None):
    """Look up (or create) the wallet_id for an OAuth identity. A brand-new wallet gets
    the faucet claimed automatically in the SAME call -- ships faucet+debit together, or the
    demo dies (per TOKENOMICS.md §11.6)."""
    now = time.time()
    with _db() as c:
        row = c.execute(
            "SELECT wallet_id FROM oauth_identities WHERE provider=? AND external_id=?",
            (provider, external_id)).fetchone()
        if row:
            return row["wallet_id"], False
        import secrets as _secrets
        wallet_id = "w_" + _secrets.token_hex(16)
        c.execute("INSERT INTO oauth_identities (provider, external_id, wallet_id, email, "
                 "created_at) VALUES (?,?,?,?,?)", (provider, external_id, wallet_id, email, now))
        c.execute("INSERT OR IGNORE INTO ledger (node_id, account_type) VALUES (?, 'wallet')",
                 (wallet_id,))
    claim_faucet(wallet_id, config.FAUCET_AMOUNT_NRN)
    return wallet_id, True


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def create_request(request_id, prompt, max_tokens, plan_node_ids=None, complete_token=None,
                   wallet_id=None, hold_amount=None):
    with _db() as c:
        c.execute(
            "INSERT INTO requests (request_id, prompt, max_tokens, status, created_at, "
            "plan_node_ids, complete_token, wallet_id, hold_amount) "
            "VALUES (?,?,?, 'pending', ?, ?, ?, ?, ?)",
            (request_id, prompt, max_tokens, time.time(),
             json.dumps(plan_node_ids) if plan_node_ids is not None else None,
             complete_token, wallet_id, hold_amount),
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
        # Workstream B found this live, TWICE: (1) excluding only __coordinator__ isn't
        # enough once wallets exist -- a wallet's total_earned includes faucet grants and
        # hold refunds (transfer() bumps total_earned on any credited recipient), not node
        # compute earnings -- so account_type='node' is also required. (2) account_type=
        # 'node' ALONE isn't enough either: __coordinator__'s ledger row was never
        # reclassified off the schema default, so it's STILL account_type='node' -- the
        # original query's explicit node_id exclusion was doing real work (excluding the
        # protocol fee from "distributed to compute nodes") and had to be kept, not dropped.
        distributed = c.execute(
            "SELECT COALESCE(SUM(total_earned),0) AS d FROM ledger "
            "WHERE account_type='node' AND node_id != ?",
            (config.COORDINATOR_LEDGER_ID,),
        ).fetchone()
    return {
        "total_requests_served": req["n"],
        "total_tokens_generated": int(req["toks"]),
        "total_nrn_distributed": round(distributed["d"], 4),
    }
