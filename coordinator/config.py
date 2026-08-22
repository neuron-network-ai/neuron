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
# Peer verification: how many DISTINCT already-verified nodes must independently pass a
# newcomer's proof-of-compute before it is promoted. The point is to take the operator out of
# the loop entirely -- before this, a stranger could not earn until the founder's own PC ran
# the verifier, so the network's growth depended on one person being awake. 2 rather than 1
# because a single verifier could otherwise promote an unlimited number of its own sybils;
# with 2 an attacker needs two independently-verified machines to collude.
PEER_VERIFY_QUORUM = int(os.environ.get("NEURON_PEER_VERIFY_QUORUM", "2"))

# Special ledger key that accumulates the coordinator's fee.
COORDINATOR_LEDGER_ID = "__coordinator__"

# --- agent auto-update (Session 9) ------------------------------------------ #
# Bump this when a new agent is published; agents poll /agent/version and update.
# The coordinator's own public address, handed to every node on registration and on every
# heartbeat so they can follow it if it ever moves.
#
# Without this, the address is baked into each install's config.json and nothing can revise it:
# changing the public hostname would strand every node that already exists, permanently, and a
# new installer could not rescue them (agent.ensure_config only writes defaults when there is no
# config.json yet). The fix has to be in place BEFORE a move, not during one -- the old host is
# the only thing that can tell anyone where the new one is, and only while it is still up.
#
# Keep this pointed at the STABLE public name. The plan is to put a load balancer behind
# `neuronnet.duckdns.org` rather than move off it, precisely so this never has to change.
PUBLIC_URL = os.environ.get("NEURON_PUBLIC_URL", "https://neuronnet.duckdns.org")

# How many STAGES a routed chain may have. Not a tuning knob -- it is the shape the inference
# path is built out of: `node_a.py` embeds and runs layers 0..s1-1, `node_c.py` runs the middle
# and forwards, `node_b.py` runs the tail plus the final norm. Three programs, three roles, and
# `node_a.coord_get_chain` refuses anything else outright ("expected a 3-node chain, got N").
#
# Extra machines are meant to become REPLICAS of existing stages, not extra stages -- that is
# exactly what router.suggest_placement does, and how added machines turn into throughput rather
# than a deeper pipeline ([P16]). The planners have to honour the same rule: on 2026-08-07 a
# migration split the model across however many nodes happened to be eligible, cut over onto a
# ONE-stage plan, and left the network reporting 28/28 healthy while every chat failed on
# "expected a 3-node chain, got 1".
#
# Raising this requires generalising the driver first, not just this number.
PIPELINE_STAGES = int(os.environ.get("NEURON_PIPELINE_STAGES", "3"))
# The floor of that same shape, and the half nothing was checking. `node_a.coord_get_chain`
# accepts a chain of TWO or THREE stages -- two is not degraded, it is the ordinary shape of a
# three-machine network with one machine away. ONE is refused outright, and a roster can walk to
# one stage while every layer is covered: a node holding the whole model wins the chain walk
# from cursor 0 and swallows the stages below it. That was live on 2026-08-09 with
# `network_healthy: true` reported throughout -- see PROBLEMS.md [P32]. Named here, next to the
# ceiling, because the two are one rule and splitting them is what let half of it go unenforced.
MIN_PIPELINE_STAGES = int(os.environ.get("NEURON_MIN_PIPELINE_STAGES", "2"))

# How long auto-repair will stand down for a migration that is PREPARING before repairing the
# chain anyway.
#
# The stand-down itself is right: a repair that rewrote ranges mid-cutover would leave half the
# network on each model's partition, which is unrecoverable rather than merely unroutable. What
# was wrong is that it had no bound. `phase` only leaves "preparing" when EVERY planned node
# reports ready, so one node that never reports keeps repair switched off for as long as it
# stays away -- and on 2026-08-21 that meant a collapsed chain sat unroutable from 05:46 until
# a human ran pin_layers.sh, with `/status` saying `routable: false` and nothing anywhere
# saying why.
#
# migration.py already makes this exact argument about `blocked`: "a 'blocked' phase would
# silently disable gap healing ... the network would be both unable to grow AND unable to
# repair." It is just as true of a migration that is preparing and not progressing.
#
# 20 minutes is deliberately longer than a slice download needs on a home connection, because
# repairing UNDER a healthy migration is the failure this bound must not cause. Past it, an
# unroutable network serves nobody and a migration that still has not converged is not a reason
# to keep it that way.
REPAIR_STANDDOWN_S = float(os.environ.get("NEURON_REPAIR_STANDDOWN_S", "1200"))
# Layers the DRIVER holds: stage 1 is always exactly 0..DRIVER_STAGE1_LAYERS-1.
#
# Not a preference — `node_a.coord_get_chain` refuses any chain whose first stage is not its own
# shard, and that shard is fixed at `neuron_driver.S1`. A chain can have a legal stage count,
# cover every layer, and still be unroutable because stage 1 is the wrong width. Observed
# 2026-08-10: a migration left the chain at [[0,16],[17,23],[24,27]] — three stages, 28/28
# covered, reported routable, and every chat would still have been refused by the driver.
# Must match `neuron_driver.S1` and `pin_layers.sh`'s NEURON_S1.
DRIVER_STAGE1_LAYERS = int(os.environ.get("NEURON_S1", "10"))
# How many times slower than the FASTEST replica of the same segment a node may be and still be
# routed to. Beyond it, its share of traffic is zero rather than small.
#
# Weighted-random replica choice is right for "somewhat slower" -- a node at half the speed
# should still carry half the traffic. It is wrong for orders of magnitude: a chain runs at the
# speed of its slowest stage, so a node 188x slower than its peers turns the occasional request
# it wins into a visibly broken one. Live 2026-08-10: one node reported 4150 ms/layer against
# 8-22 ms for the rest (a measurement taken while it thrashed during a migration) and answers
# came back at 0.24 tok/s. That node is not slow, it is broken or its figure is stale.
REPLICA_SLOWDOWN_LIMIT = float(os.environ.get("NEURON_REPLICA_SLOWDOWN_LIMIT", "8.0"))
# How long a self-measured ms_per_layer is believed. After this it is treated as UNKNOWN and the
# node is scored at DEFAULT_MS_PER_LAYER again.
#
# It was measured once at agent startup and believed forever. So a reading taken while a machine
# thrashed under a migration -- 4150 ms/layer against 8-22 for its peers, live 2026-08-10 -- was
# permanent. Worse in combination with REPLICA_SLOWDOWN_LIMIT: an outlier is excluded from
# routing, therefore never serves, therefore is never re-measured. A stale bad number became a
# life sentence, and the guard I added is what made it one. Ageing the figure out is the way back
# in: an unknown node is scored at the default prior, which is exactly how a never-measured node
# is already treated.
MS_PER_LAYER_TTL_S = float(os.environ.get("NEURON_MS_PER_LAYER_TTL_S", "21600"))

# The latest RELEASED agent, which is not the same fact as the newest build in this repo and
# must never be raised to one that has not been published. Every node reads this daily,
# AGENT_DOWNLOAD_URL is derived from it, and the Chat UI's update notice links to it — so naming
# an unreleased version points the whole fleet at a 404 at once. It defaulted to 0.20.4 (built,
# never released) and production was correct only because a systemd drop-in pinned 0.20.3 over
# it. Bump this when a release is PUBLISHED, not when one is built; test_download_links.py
# refuses a value with no release notes.
AGENT_VERSION = os.environ.get("NEURON_AGENT_VERSION", "0.20.22")
# Where a node fetches that version, and the hash it must match before anything is run.
# The download is NOT served from here: this VM has 1 GB of RAM and the installer is ~200 MB,
# so the coordinator only advertises metadata and GitHub Releases does the bandwidth.
AGENT_DOWNLOAD_URL = os.environ.get(
    "NEURON_AGENT_DOWNLOAD_URL",
    f"https://github.com/neuron-network-ai/neuron/releases/download/v{AGENT_VERSION}"
    f"/NEURON-Setup-{AGENT_VERSION}.exe")
# SHA-256 of that file. Empty until a release is published and hashed -- and an EMPTY hash
# means no node will install anything, which is the correct failure direction: an unverified
# binary pushed to every volunteer's machine is the worst thing this project could ship.
AGENT_SHA256 = os.environ.get("NEURON_AGENT_SHA256",
                             "2c7820d063d16ed53039ec0a42e0ddfbce5beb36d7c7360654beb322bfdd2721")
# THE WAY BACK. An agent only ever moved forward, so a bad release could not be undone: the
# install ends in os._exit(0), the machine is behind a NAT in somebody's house, and "reinstall
# it" does not scale past the machines one person can name. Setting this to 1 -- together with
# an OLDER AGENT_VERSION and its matching SHA -- tells every node to install that older build
# deliberately. Off by default, and never inferred from the version alone: walking a fleet
# backwards must take an explicit act by an operator who knows they are doing it.
AGENT_ROLLBACK = os.environ.get("NEURON_AGENT_ROLLBACK", "").strip().lower() in ("1", "true", "yes")

# --- feedback relay (in-app feedback -> Discord) ---------------------------- #
# The webhook lives HERE and nowhere else. Putting it in the installed app would ship a
# write-credential for the project's own Discord to every volunteer's machine, where anyone
# could read it out of the bundle and post as the app. One copy, on one server we control.
#
# Empty means the relay is OFF and /feedback says so plainly, so the UI can fall back to the
# invite link instead of silently swallowing what somebody took the trouble to write.
DISCORD_WEBHOOK_URL = os.environ.get("NEURON_DISCORD_WEBHOOK", "").strip()
# The public invite, safe to ship: it is meant to be handed out.
DISCORD_INVITE = os.environ.get("NEURON_DISCORD_INVITE", "https://discord.gg/Cr5eRwCPV").strip()
# Discord rejects a message body over 2000 characters. Cut here rather than letting the post
# fail, and say it was cut, because a truncated report is worth more than a lost one.
FEEDBACK_MAX_CHARS = int(os.environ.get("NEURON_FEEDBACK_MAX_CHARS", "1800"))

# --------------------------------------------------------------------------- #
# Coordinator self-update (see coordinator/selfupdate.py)
# --------------------------------------------------------------------------- #
# What this build IS. Reported on /status so an update can be CONFIRMED live rather than
# assumed from "the process came back up". Bump it in the same commit as the change it ships.
#
# Its OWN number line, deliberately not the one in CHANGELOG.md. That one is the installer/agent
# version (AGENT_VERSION, updater.LOCAL_VERSION, neuron.iss AppVersion — all three must agree),
# it is at 0.18.0, and it moves when a volunteer's app changes. The coordinator is a different
# artifact on a different machine with a different release cadence: sharing the sequence would
# mean one number naming two things, and picking the next free value would announce an app
# release that never happened. Starts at 0.1.0, matching the version the FastAPI app has
# always reported. Coordinator releases are tagged `coordinator-v<version>` for the same reason.
COORDINATOR_VERSION = os.environ.get("NEURON_COORDINATOR_VERSION", "0.1.0")
# Where the updater reads what it SHOULD be running. Deliberately not an endpoint on this
# coordinator: a coordinator that is broken cannot serve its own manifest, and a bad deploy that
# could rewrite the manifest would remove the only signal saying it went wrong. A static file in
# the repo means publishing is a git push and the source of truth outlives the server.
COORDINATOR_MANIFEST_URL = os.environ.get(
    "NEURON_COORDINATOR_MANIFEST_URL",
    "https://raw.githubusercontent.com/neuron-network-ai/neuron/main/"
    "packaging/coordinator-latest.json")

# Origins allowed to read this API from a browser. The public landing page is served from
# GitHub Pages, which is a different origin, so it needs naming here to display live numbers.
# Comma-separated via NEURON_CORS_ORIGINS. Note an origin is scheme+host+port and never a path,
# so the entry below covers every page on that host, project sites included.
# Kept as an explicit list rather than "*" — see the middleware in main.py for why.
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "NEURON_CORS_ORIGINS",
    "https://neuron-network-ai.github.io,http://localhost:8080,http://127.0.0.1:8080",
).split(",") if o.strip()]

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

# --- availability emission (TOKENOMICS.md §11.4) ----------------------------- #
# Nodes have only ever earned by SERVING. Nothing paid a machine to be there, so coverage was
# whatever happened to be awake -- and the coordinator reacted by re-splitting layers, which
# costs every affected node a delete and a re-download. §11.4 already called for the fix and it
# was never built: emission paid PER DEVICE-HOUR, deliberately decoupled from traffic, because
# anything paying more than the spend for metered work can be farmed by generating your own
# traffic. This is that, plus the part §11.4 lacks -- paying for hours WHERE AND WHEN coverage
# is short, so availability is rostered rather than accidental.
#
# Paid by TRANSFER out of __emission_pool__, never minted: the fixed 1,000,000,000 supply
# invariant is not negotiable (see test_escrow_conservation).
SLOT_SECONDS = int(os.environ.get("NEURON_SLOT_SECONDS", "3600"))
# §11.4's figure: ~1 NRN per device-hour in era 0, halving per §4 as a RATE POLICY.
EMISSION_BASE_NRN_PER_HOUR = float(os.environ.get("NEURON_EMISSION_BASE", "1.0"))
# How much a thin slot may out-pay a well-covered one. Capped because the multiplier is a
# recruiting signal, not an auction: uncapped, a single slot with one eligible node would price
# itself arbitrarily high and drain the pool that has to last for years.
EMISSION_SCARCITY_MAX = float(os.environ.get("NEURON_EMISSION_SCARCITY_MAX", "3.0"))
# Replicas per block at which a slot is considered fully covered -- the point where the
# multiplier reaches 1.0. Below it the premium rises toward EMISSION_SCARCITY_MAX.
EMISSION_TARGET_REPLICAS = int(os.environ.get("NEURON_EMISSION_TARGET_REPLICAS", "3"))
# A second bound, on the whole network per day rather than per node. The multiplier reacts to
# scarcity, and scarcity is exactly what a mass outage looks like -- so the moment the pool is
# most at risk of being drained fast is the moment rates are highest. This is the backstop.
EMISSION_DAILY_CAP_NRN = float(os.environ.get("NEURON_EMISSION_DAILY_CAP", "5000.0"))
# Fraction of a slot a node must actually be present for before it earns anything for it.
# Prevents a node that appears for one heartbeat from being paid as though it held the hour.
SLOT_MIN_ATTENDANCE_FRAC = float(os.environ.get("NEURON_SLOT_MIN_ATTENDANCE", "0.5"))

# --- What emission pays FOR when the network did not check ([P47] cause 2) -------------------
# Emission pays for PROVEN work, not for OBSERVED work. Those were the same thing while the
# rule was "a challenge passed inside this slot", and the difference is entirely in the
# coordinator's hands rather than the node's: a node cannot make the verifier's rotation reach
# it, cannot keep the operator's PC awake, and cannot keep our DNS resolving.
#
# Measured, and it is not a corner case: `verify_service.log` shows the verifier was not running
# for 207 of the 390 hours of its own history (53%), and in the window [P47] measured, 57 of
# `node-c-pavilion`'s 64 unpaid hours are hours in which the verifier was either down (46) or
# unable to read the roster (11). That machine has passed 4,523 challenges. It earned nothing
# for those hours because WE were not watching.
#
# Rule 2 of emission.py is unchanged and non-negotiable: presence alone must never pay. Both
# constants below are bounded precisely so that no path to payment exists that a node can create,
# detect or exploit.

# How long a passing challenge stays EVIDENCE, in slots. A pass at 13:58 does not stop being
# true at 14:00. Covers rotation latency (one node per 60s cycle, so a 60-node roster is checked
# once an hour at best) and short verifier restarts, and it is node-independent: nothing a node
# does changes when it is challenged. Deliberately small -- this is the age of the evidence, and
# a proof two hours old is the most that can honestly be called current.
EMISSION_POC_VALID_SLOTS = int(os.environ.get("NEURON_EMISSION_POC_VALID_SLOTS", "2"))

# How many CONSECUTIVE unaudited slots the network will still pay for before it stops. An
# unaudited slot is one in which the coordinator heard nothing from any verifier at all -- our
# outage, recorded on our side, unforgeable by a node. Paying it is the network covering its own
# downtime rather than billing volunteers for it.
#
# Bounded rather than discounted, deliberately. Discounting an excused hour would be a penalty
# for our own failure, which is the thing being fixed; but "we could not check" stops being an
# excuse at some point and becomes "nobody has verified this network since yesterday". At that
# point the honest answer is that we do not know, and the payout log saying so is the signal.
EMISSION_MAX_UNAUDITED_SLOTS = int(os.environ.get("NEURON_EMISSION_MAX_UNAUDITED_SLOTS", "6"))

# Genesis buckets — ledger rows, NOT config values that can silently drift the supply.
# sum() of the 4 allocation buckets is exactly 1,000,000,000; __escrow__ is bookkeeping-only
# (seeded at 0, holds in-flight payments, never counted as anyone's allocation).
GENESIS_BUCKETS_EMISSION_ID = "__emission_pool__"      # 600,000,000 — paid per device-hour donated
GENESIS_BUCKETS_FOUNDER_ID = "__founder__"             # 200,000,000 — vested, see TOKENOMICS.md §5
GENESIS_BUCKETS_ECOSYSTEM_ID = "__ecosystem__"         # 150,000,000 — grants + the faucet
GENESIS_BUCKETS_LIQUIDITY_ID = "__liquidity__"         #  50,000,000 — reserved for Phase 1
ESCROW_LEDGER_ID = "__escrow__"                        # 0 — in-flight held payments only
GENESIS_TOTAL_SUPPLY = 1_000_000_000

# On-chain payout binding (coordinator/payout.py). How long a signing challenge stays valid.
# Short because the only thing that happens between issuing and using it is one HTTP round
# trip and a local signature; long enough that a human pasting the message into a wallet by
# hand is not racing a timer.
PAYOUT_CHALLENGE_TTL = float(os.environ.get("NEURON_PAYOUT_CHALLENGE_TTL", "600"))

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
# installed agent: an OAuth client secret cannot live on a stranger's PC (anyone holding the
# installer can extract it from the binary), and asking each user to create a Google Cloud
# project to send a chat message is not a product. The coordinator is a real server, so it can
# actually keep a secret -- and it is already the only thing that can mint a wallet.
GOOGLE_CLIENT_ID = os.environ.get("NEURON_GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("NEURON_GOOGLE_CLIENT_SECRET")
GITHUB_CLIENT_ID = os.environ.get("NEURON_GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.environ.get("NEURON_GITHUB_CLIENT_SECRET")
# Where providers send the user back. MUST match the redirect URI registered with Google/GitHub
# exactly, so it has to be the coordinator's real public address, not a guess from the request.
PUBLIC_BASE_URL = os.environ.get("NEURON_PUBLIC_BASE_URL", "https://neuronnet.duckdns.org")
