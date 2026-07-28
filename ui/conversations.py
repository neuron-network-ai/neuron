"""ui/conversations.py — driver-side conversation history (Chat UI real multi-turn memory).

WHY HERE, NOT THE COORDINATOR: the driver is already the one place in NEURON that handles
plaintext (see SAFETY.md) -- storing conversation history here adds no new party that sees
chat content. The coordinator stays exactly as blind as it already is; this is a private
SQLite file on the machine running ui/app.py, keyed by wallet_id, never sent anywhere else.

Plain sqlite3 (no ORM), same short-lived-connection-per-call pattern as
coordinator/models.py. Ownership is enforced by every read/write requiring the caller's
wallet_id to match the conversation's -- the same bearer-capability posture the coordinator
already uses for wallet_id itself.
"""
import contextlib
import os
import sqlite3
import time
import uuid

DB_PATH = os.environ.get("NEURON_CONVERSATIONS_DB",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversations.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    wallet_id   TEXT NOT NULL,
    title       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_wallet ON conversations(wallet_id);
"""


@contextlib.contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
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


def create_conversation(wallet_id, title):
    cid = uuid.uuid4().hex
    now = time.time()
    with _db() as c:
        c.execute("INSERT INTO conversations (id, wallet_id, title, created_at, updated_at) "
                 "VALUES (?,?,?,?,?)", (cid, wallet_id, title[:80], now, now))
    return cid


def list_conversations(wallet_id):
    with _db() as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE wallet_id=? ORDER BY updated_at DESC", (wallet_id,)).fetchall()
    return [dict(r) for r in rows]


def get_conversation(conversation_id, wallet_id):
    """Returns {id, title, messages: [{role, content, created_at}]} or None if the
    conversation doesn't exist OR doesn't belong to this wallet -- ownership check is baked
    in, not a separate step, so a caller can never accidentally skip it."""
    with _db() as c:
        conv = c.execute("SELECT id, title, created_at, updated_at FROM conversations "
                         "WHERE id=? AND wallet_id=?", (conversation_id, wallet_id)).fetchone()
        if conv is None:
            return None
        msgs = c.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id=? ORDER BY id ASC", (conversation_id,)).fetchall()
    return {**dict(conv), "messages": [dict(m) for m in msgs]}


def add_message(conversation_id, wallet_id, role, content):
    """Ownership-checked (via an UPDATE...WHERE that only touches a row this wallet owns)
    before the message insert, so a caller can't append to someone else's conversation even
    by guessing/forging a conversation_id."""
    now = time.time()
    with _db() as c:
        cur = c.execute("UPDATE conversations SET updated_at=? WHERE id=? AND wallet_id=?",
                        (now, conversation_id, wallet_id))
        if cur.rowcount == 0:
            return False
        c.execute("INSERT INTO messages (conversation_id, role, content, created_at) "
                 "VALUES (?,?,?,?)", (conversation_id, role, content, now))
    return True


def delete_conversation(conversation_id, wallet_id):
    with _db() as c:
        c.execute("DELETE FROM messages WHERE conversation_id=? AND conversation_id IN "
                 "(SELECT id FROM conversations WHERE id=? AND wallet_id=?)",
                 (conversation_id, conversation_id, wallet_id))
        cur = c.execute("DELETE FROM conversations WHERE id=? AND wallet_id=?",
                        (conversation_id, wallet_id))
        return cur.rowcount > 0


init_db()
