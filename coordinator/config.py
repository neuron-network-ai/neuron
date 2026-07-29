"""NEURON coordinator — settings.

Everything tunable lives here. Env vars override the defaults so you don't have to
edit code to change the DB location or the shared registration secret.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- storage ---------------------------------------------------------------- #
DB_PATH = os.environ.get("NEURON_DB", str(BASE_DIR / "neuron.db"))

# --- model / network shape -------------------------------------------------- #
TOTAL_LAYERS = int(os.environ.get("NEURON_TOTAL_LAYERS", "28"))   # Qwen2.5-1.5B
MODEL_ID = os.environ.get("NEURON_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

# --- economics (NRN = the network's coin) ----------------------------------- #
NRN_PER_REQUEST = 1.0     # total minted per completed request
COORDINATOR_FEE = 0.10    # coordinator always keeps 10%; nodes split the rest
#   -> a node holding L of TOTAL_LAYERS earns  (1 - FEE) * NRN * L / TOTAL_LAYERS
#   -> 47 completed requests => nodes share 47 * 0.9 = 42.3 NRN (matches spec)

# --- health checking -------------------------------------------------------- #
PING_INTERVAL_S = 30           # nodes are expected to ping this often
HEARTBEAT_TIMEOUT_S = 90       # mark offline if last_seen older than this
HEALTH_CHECK_INTERVAL_S = 60   # background sweep cadence

# --- security --------------------------------------------------------------- #
# Shared secret for the X-Register-Secret header. Presenting it marks a node TRUSTED
# (skips probation). Override in production.
REGISTRATION_SECRET = os.environ.get("NEURON_REGISTER_SECRET", "neuron-dev-secret")
TOKEN_BYTES = 24               # per-node auth token length (in bytes, hex-encoded)

# --- open join (Session 12 — the first stranger) ---------------------------- #
# When on, ANYONE can register a node without the shared secret — that is the whole
# point of an open network. A secret-less node joins as PROBATIONARY: it does NOT serve
# live traffic and earns NO NRN until a verifier confirms it computes correctly
# (proof-of-compute, Session 16). Presenting the valid secret still works and marks a
# node TRUSTED (the founder's own dev nodes register that way). Set NEURON_OPEN_JOIN=0
# to require the secret for every registration (fully private network).
OPEN_JOIN = os.environ.get("NEURON_OPEN_JOIN", "1") == "1"
# Proof-of-compute passes a probationary node needs before it may serve live traffic and earn.
PROBATION_MIN_PASSES = int(os.environ.get("NEURON_PROBATION_MIN_PASSES", "1"))

# Special ledger key that accumulates the coordinator's fee.
COORDINATOR_LEDGER_ID = "__coordinator__"

# --- agent auto-update (Session 9) ------------------------------------------ #
# Bump this when a new agent is published; agents poll /agent/version and update.
AGENT_VERSION = os.environ.get("NEURON_AGENT_VERSION", "0.3.0")

# --- stranger-NAT relay (Session 12) ---------------------------------------- #
# A node behind NAT registers with behind_nat=true; the coordinator assigns it a
# public port on the relay from the pool and stores its endpoint as the relay's, so
# node_a/node_c reach it via the relay. The agent auto-starts tunnel_client from the
# relay block returned at registration. Genericize RELAY_HOST before the repo goes
# public (see PROBLEMS.md [P11]).
RELAY_ENABLED = os.environ.get("NEURON_RELAY_ENABLED", "1") == "1"
RELAY_HOST = os.environ.get("NEURON_RELAY_HOST", "150.230.22.250")
RELAY_CONTROL_PORT = int(os.environ.get("NEURON_RELAY_CONTROL_PORT", "8010"))
RELAY_DATA_PORT = int(os.environ.get("NEURON_RELAY_DATA_PORT", "8011"))
RELAY_PORT_MIN = int(os.environ.get("NEURON_RELAY_PORT_MIN", "9000"))
RELAY_PORT_MAX = int(os.environ.get("NEURON_RELAY_PORT_MAX", "9100"))
# Shared with the relay process (relay.py --secret / NEURON_RELAY_SECRET on the relay host) so
# it can verify a node's tunnel registration ticket without a DB or calling back here. Override
# in production — see relay_auth.py and PROBLEMS.md for what this closes.
RELAY_SECRET = os.environ.get("NEURON_RELAY_SECRET", "neuron-relay-dev-secret")

# --- security (Session 16) -------------------------------------------------- #
# Proof-of-compute reputation: a node flagged once it has enough challenge samples and
# its pass-rate falls below the threshold -> excluded from routing, earns nothing.
REPUTATION_MIN_SAMPLES = int(os.environ.get("NEURON_REP_MIN_SAMPLES", "3"))
REPUTATION_THRESHOLD = float(os.environ.get("NEURON_REP_THRESHOLD", "0.6"))
# Basic per-IP rate limit (rough DDoS guard): N requests per window seconds.
RATE_LIMIT_MAX = int(os.environ.get("NEURON_RATE_LIMIT", "120"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("NEURON_RATE_WINDOW", "60"))

# --- fixed-supply ledger, Phase 0 (TOKENOMICS.md §11 — implemented same-session as safety) -- #
# 1 NRN = 1,000 "weighted tokens" (output + input*INPUT_WEIGHT) on the REFERENCE model
# (TOTAL_LAYERS above). INPUT_WEIGHT stays 1.0 until prefill cost is actually measured
# ([P13] in PROBLEMS.md) — undercharging input tokens before that is a farming surface.
PRICE_PER_1K_WEIGHTED = float(os.environ.get("NEURON_PRICE_PER_1K", "1.0"))
INPUT_WEIGHT = float(os.environ.get("NEURON_INPUT_WEIGHT", "1.0"))
# COORDINATOR_FEE (defined above, unchanged rate) now comes FROM the settled payment via
# settle(), never minted.
# The lm_head holder does meaningfully more work than a plain layer (~38ms vs ~9ms/layer,
# S14 benchmark data) — this many extra "layer-equivalents" get added to its share so the
# reward split reflects real compute cost, not just layer count.
HEAD_BONUS_LE = float(os.environ.get("NEURON_HEAD_BONUS_LE", "5.0"))
# A held request's escrow is returned to the wallet if never completed/released within this
# window (a crashed/abandoned request) — swept by the existing health_loop.
HOLD_TTL_S = int(os.environ.get("NEURON_HOLD_TTL_S", "600"))
# One-time grant per new wallet, from __ecosystem__ — MUST ship in the same release as the
# debit, or a new user can never spend anything (TOKENOMICS.md §11.6: "or the demo dies").
FAUCET_AMOUNT_NRN = float(os.environ.get("NEURON_FAUCET_AMOUNT", "25.0"))

# Genesis buckets — ledger rows, NOT config values that can silently drift the supply.
# sum() of the 4 allocation buckets is exactly 1,000,000,000; __escrow__ is bookkeeping-only
# (seeded at 0, holds in-flight payments, never counted as anyone's allocation).
GENESIS_BUCKETS_EMISSION_ID = "__emission_pool__"      # 600,000,000 — paid per device-hour donated
GENESIS_BUCKETS_FOUNDER_ID = "__founder__"             # 200,000,000 — vested, see TOKENOMICS.md §5
GENESIS_BUCKETS_ECOSYSTEM_ID = "__ecosystem__"         # 150,000,000 — grants + the faucet
GENESIS_BUCKETS_LIQUIDITY_ID = "__liquidity__"         #  50,000,000 — reserved for Phase 1
ESCROW_LEDGER_ID = "__escrow__"                        # 0 — in-flight held payments only
GENESIS_TOTAL_SUPPLY = 1_000_000_000

# Shared secret between the coordinator and a driver process (ui/app.py) so POST
# /wallet/oauth can be trusted: the driver holds the real OAuth client secret and has
# already verified the (provider, external_id) pair with Google/GitHub before calling
# this — without a shared secret, anyone on the internet could call this endpoint directly
# and squat a wallet under an external_id they don't own. Override in production.
WALLET_LINK_SECRET = os.environ.get("NEURON_WALLET_LINK_SECRET", "neuron-wallet-link-dev-secret")

# Wallet-linked moderation escalation: a wallet is banned from /infer once its recorded
# violation_count (coordinator/models.py::record_violation) reaches this many blocked
# requests. Deliberately > 1 -- a single false-positive keyword match (the blocklist is
# a cheap v1, see safety/moderation.py) shouldn't lock someone out immediately.
MODERATION_BAN_THRESHOLD = int(os.environ.get("NEURON_MODERATION_BAN_THRESHOLD", "3"))

# How long a completed request row is kept (models.py::prune_old_requests). `requests` is the
# only table that grows with TRAFFIC rather than with users -- at 1M users x 5 requests/day it
# would add ~1.25 GB/day, which no single-file SQLite on a 1 GB VM survives. Identities, ledger
# rows and moderation_events are NEVER pruned: bans depend on them and they grow slowly.
# 0 disables pruning entirely.
REQUEST_RETENTION_DAYS = int(os.environ.get("NEURON_REQUEST_RETENTION_DAYS", "90"))

# Login (coordinator/auth.py). Configured ONCE here for the whole network rather than on every
# installed agent: an OAuth client secret cannot live on a stranger's PC (it is extractable from
# any shipped binary, and this repo is public), and asking each user to create a Google Cloud
# project to send a chat message is not a product. The coordinator is a real server, so it can
# actually keep a secret -- and it is already the only thing that can mint a wallet.
GOOGLE_CLIENT_ID = os.environ.get("NEURON_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("NEURON_GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.environ.get("NEURON_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.environ.get("NEURON_GITHUB_CLIENT_SECRET")
# Where providers send the user back. MUST match the redirect URI registered with Google/GitHub
# exactly, so it has to be the coordinator's real public address, not a guess from the request.
PUBLIC_BASE_URL = os.environ.get("NEURON_PUBLIC_BASE_URL", "http://150.230.22.250:8001")
