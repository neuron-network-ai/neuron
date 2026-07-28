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
    prompt_len       INTEGER,
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
    provider       TEXT NOT NULL,
    external_id    TEXT NOT NULL,
    wallet_id      TEXT NOT NULL,
    email          TEXT,
    email_verified INTEGER NOT NULL DEFAULT 0,
    created_at     REAL NOT NULL,
    last_seen      REAL,
    PRIMARY KEY (provider, external_id)
);
CREATE TABLE IF NOT EXISTS moderation_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    wallet_id   TEXT NOT NULL,
    direction   TEXT NOT NULL,
    category    TEXT,
    created_at  REAL NOT NULL
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
        # SAFETY.md gap closed: the coordinator used to store every request's FULL raw prompt
        # text forever (only ever needed for its LENGTH, to clamp a driver's self-reported
        # prompt_tokens in /complete against something at settlement time). Backfill prompt_len
        # for existing rows from their already-stored prompt, then stop writing prompt text at
        # all going forward (create_request() below now takes a length, not text) -- existing
        # rows' prompt text is left as-is here (a retroactive scrub is a separate, deliberate
        # decision, not bundled into a schema migration).
        if "prompt_len" not in rcols:
            c.execute("ALTER TABLE requests ADD COLUMN prompt_len INTEGER")
            c.execute("UPDATE requests SET prompt_len = LENGTH(prompt) "
                     "WHERE prompt_len IS NULL AND prompt IS NOT NULL")
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
        # Wallet-linked moderation escalation: a content-policy block is currently a one-off
        # (forgotten the moment the response is sent) -- these track repeated attempts by the
        # same wallet IDENTITY across separate requests, so they can actually be banned.
        if "violation_count" not in lcols:
            c.execute("ALTER TABLE ledger ADD COLUMN violation_count INTEGER NOT NULL DEFAULT 0")
        if "moderation_banned" not in lcols:
            c.execute("ALTER TABLE ledger ADD COLUMN moderation_banned INTEGER NOT NULL DEFAULT 0")
        # Verified identity (abuse accountability): a filter is always evadable, so the real
        # control is knowing WHO sent a request and being able to act on them. email_verified
        # distinguishes a real provider-verified address from a throwaway; last_seen shows
        # whether an identity is live or dormant when reviewing it.
        ocols = {r["name"] for r in c.execute("PRAGMA table_info(oauth_identities)").fetchall()}
        if "email_verified" not in ocols:
            c.execute("ALTER TABLE oauth_identities ADD COLUMN email_verified "
                     "INTEGER NOT NULL DEFAULT 0")
        if "last_seen" not in ocols:
            c.execute("ALTER TABLE oauth_identities ADD COLUMN last_seen REAL")


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


def wallet_for_oauth(provider, external_id, email=None, email_verified=False):
    """Look up (or create) the wallet_id for an OAuth identity. A brand-new wallet gets
    the faucet claimed automatically in the SAME call -- ships faucet+debit together, or the
    demo dies (per TOKENOMICS.md §11.6). Also stamps last_seen on every login (not just
    creation), so the admin view can tell a live identity from a dormant one."""
    now = time.time()
    with _db() as c:
        row = c.execute(
            "SELECT wallet_id FROM oauth_identities WHERE provider=? AND external_id=?",
            (provider, external_id)).fetchone()
        if row:
            c.execute("UPDATE oauth_identities SET last_seen=?, email=COALESCE(?, email), "
                     "email_verified=? WHERE provider=? AND external_id=?",
                     (now, email, int(bool(email_verified)), provider, external_id))
            return row["wallet_id"], False
        import secrets as _secrets
        wallet_id = "w_" + _secrets.token_hex(16)
        c.execute("INSERT INTO oauth_identities (provider, external_id, wallet_id, email, "
                 "email_verified, created_at, last_seen) VALUES (?,?,?,?,?,?,?)",
                 (provider, external_id, wallet_id, email, int(bool(email_verified)), now, now))
        c.execute("INSERT OR IGNORE INTO ledger (node_id, account_type) VALUES (?, 'wallet')",
                 (wallet_id,))
    claim_faucet(wallet_id, config.FAUCET_AMOUNT_NRN)
    return wallet_id, True


def is_oauth_wallet(wallet_id):
    """True only if this wallet was minted by a real Google/GitHub login. The gate that makes
    login MANDATORY rather than cosmetic: /infer, the API's bearer auth and the faucet all
    refuse a wallet_id that isn't backed by an identity here, so an attacker can't just invent
    a wallet string and use the network anonymously (which /wallet/faucet used to allow)."""
    if not wallet_id:
        return False
    with _db() as c:
        return c.execute("SELECT 1 FROM oauth_identities WHERE wallet_id=?",
                         (wallet_id,)).fetchone() is not None


# --------------------------------------------------------------------------- #
# Wallet-linked moderation escalation (content-safety + Workstream B combined)
# --------------------------------------------------------------------------- #
def record_violation(wallet_id, direction, category=None):
    """Record a moderation block against a wallet's IDENTITY (not just the one request it
    happened on), and escalate to a ban once config.MODERATION_BAN_THRESHOLD is reached.
    Only ever receives a category label from the caller (safety/moderation.py), never the
    actual text -- the coordinator staying blind to plaintext is preserved even here."""
    ensure_account(wallet_id, "wallet")
    with _db() as c:
        c.execute(
            "INSERT INTO moderation_events (wallet_id, direction, category, created_at) "
            "VALUES (?, ?, ?, ?)", (wallet_id, direction, category, time.time()))
        c.execute("UPDATE ledger SET violation_count=violation_count+1 WHERE node_id=?",
                  (wallet_id,))
        row = c.execute("SELECT violation_count, moderation_banned FROM ledger WHERE node_id=?",
                        (wallet_id,)).fetchone()
        banned = bool(row["moderation_banned"])
        if not banned and row["violation_count"] >= config.MODERATION_BAN_THRESHOLD:
            c.execute("UPDATE ledger SET moderation_banned=1 WHERE node_id=?", (wallet_id,))
            banned = True
    return {"violation_count": row["violation_count"], "banned": banned}


def wallet_moderation_status(wallet_id):
    """Cheap pre-flight check for /infer -- a banned wallet's requests are refused before
    any hold is even attempted."""
    row = get_ledger(wallet_id)
    if row is None:
        return {"violation_count": 0, "banned": False}
    return {"violation_count": row.get("violation_count", 0) or 0,
           "banned": bool(row.get("moderation_banned", 0))}


def set_ban(wallet_id, banned):
    """Ban/unban an identity BY HAND. The automatic MODERATION_BAN_THRESHOLD path only fires
    on violations the driver self-reports (safety/moderation.py runs on the user's own machine
    for a self-hosted install, so a stripped client never reports itself). This is the operator
    lever for everything the filter misses -- a jailbreak, an abuse report, anything spotted by
    a human. Enforcement is at /infer, which is server-side, so a ban bites even a modified
    client. Returns False for an unknown wallet -- deliberately does NOT create the row (a
    typo'd id must fail loudly, not silently mint a banned ghost account)."""
    if not is_oauth_wallet(wallet_id):
        return False
    ensure_account(wallet_id, "wallet")
    with _db() as c:
        cur = c.execute("UPDATE ledger SET moderation_banned=? WHERE node_id=?",
                        (1 if banned else 0, wallet_id))
        return cur.rowcount > 0


def list_identities(limit=200, banned_only=False):
    """Everyone who has ever logged in, newest first -- backs the admin review page."""
    where = "WHERE l.moderation_banned=1" if banned_only else ""
    with _db() as c:
        rows = c.execute(f"""
            SELECT o.provider, o.external_id, o.wallet_id, o.email, o.email_verified,
                   o.created_at, o.last_seen,
                   COALESCE(l.violation_count, 0)   AS violation_count,
                   COALESCE(l.moderation_banned, 0) AS moderation_banned,
                   COALESCE(l.balance, 0)           AS balance,
                   (SELECT COUNT(*) FROM requests r WHERE r.wallet_id = o.wallet_id)
                                                    AS request_count
            FROM oauth_identities o
            LEFT JOIN ledger l ON l.node_id = o.wallet_id
            {where}
            ORDER BY COALESCE(o.last_seen, o.created_at) DESC
            LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def wallet_activity(wallet_id, limit=100):
    """One identity's full reviewable history: who they are, their moderation events, and
    their recent requests. Requests carry prompt_len only -- never prompt text (SAFETY.md);
    this answers 'who did this and when', not 'what did they type'."""
    with _db() as c:
        identity = c.execute(
            "SELECT provider, external_id, wallet_id, email, email_verified, created_at, "
            "last_seen FROM oauth_identities WHERE wallet_id=?", (wallet_id,)).fetchone()
        events = c.execute(
            "SELECT direction, category, created_at FROM moderation_events "
            "WHERE wallet_id=? ORDER BY created_at DESC LIMIT ?", (wallet_id, limit)).fetchall()
        reqs = c.execute(
            "SELECT request_id, prompt_len, max_tokens, status, created_at, completed_at, "
            "tokens_generated FROM requests WHERE wallet_id=? ORDER BY created_at DESC LIMIT ?",
            (wallet_id, limit)).fetchall()
    return {"identity": dict(identity) if identity else None,
           "moderation_events": [dict(r) for r in events],
           "requests": [dict(r) for r in reqs],
           "status": wallet_moderation_status(wallet_id)}


def prune_old_requests(older_than_days=None):
    """Delete request rows past the retention window. `requests` is the ONLY unbounded table
    here -- identities and ledger rows grow with users (slowly, and must be kept forever since
    bans depend on them), but this one grows with TRAFFIC and would be ~1.25 GB/day at 1M
    users x 5 requests. Ban evidence survives pruning: moderation_events is never touched, and
    violation_count/moderation_banned live on the ledger row. Returns rows deleted."""
    days = config.REQUEST_RETENTION_DAYS if older_than_days is None else older_than_days
    if not days or days <= 0:
        return 0
    cutoff = time.time() - (days * 86400)
    with _db() as c:
        cur = c.execute("DELETE FROM requests WHERE created_at < ? AND status != 'pending'",
                        (cutoff,))
        return cur.rowcount


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def create_request(request_id, prompt_len, max_tokens, plan_node_ids=None, complete_token=None,
                   wallet_id=None, hold_amount=None):
    """Stores prompt_len (a character COUNT), never the prompt text itself -- the only
    persistent use was ever /complete's anti-cheat clamp (`min(reported prompt_tokens, actual
    length)`), which needs a number, not the content. The `prompt` TEXT column stays in the
    schema for pre-existing rows' history; new rows leave it NULL (see SAFETY.md)."""
    with _db() as c:
        c.execute(
            "INSERT INTO requests (request_id, prompt_len, max_tokens, status, created_at, "
            "plan_node_ids, complete_token, wallet_id, hold_amount) "
            "VALUES (?,?,?, 'pending', ?, ?, ?, ?, ?)",
            (request_id, prompt_len, max_tokens, time.time(),
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
