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
