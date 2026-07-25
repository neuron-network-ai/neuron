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
# Shared secret required in the X-Register-Secret header to register a node.
# Prevents random people registering fake nodes. Override in production.
REGISTRATION_SECRET = os.environ.get("NEURON_REGISTER_SECRET", "neuron-dev-secret")
TOKEN_BYTES = 24               # per-node auth token length (in bytes, hex-encoded)

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
# public.
RELAY_ENABLED = os.environ.get("NEURON_RELAY_ENABLED", "1") == "1"
RELAY_HOST = os.environ.get("NEURON_RELAY_HOST", "150.230.22.250")
RELAY_CONTROL_PORT = int(os.environ.get("NEURON_RELAY_CONTROL_PORT", "8010"))
RELAY_DATA_PORT = int(os.environ.get("NEURON_RELAY_DATA_PORT", "8011"))
RELAY_PORT_MIN = int(os.environ.get("NEURON_RELAY_PORT_MIN", "9000"))
RELAY_PORT_MAX = int(os.environ.get("NEURON_RELAY_PORT_MAX", "9100"))

# --- security (Session 16) -------------------------------------------------- #
# Proof-of-compute reputation: a node flagged once it has enough challenge samples and
# its pass-rate falls below the threshold -> excluded from routing, earns nothing.
REPUTATION_MIN_SAMPLES = int(os.environ.get("NEURON_REP_MIN_SAMPLES", "3"))
REPUTATION_THRESHOLD = float(os.environ.get("NEURON_REP_THRESHOLD", "0.6"))
# Basic per-IP rate limit (rough DDoS guard): N requests per window seconds.
RATE_LIMIT_MAX = int(os.environ.get("NEURON_RATE_LIMIT", "120"))
RATE_LIMIT_WINDOW_S = int(os.environ.get("NEURON_RATE_WINDOW", "60"))
