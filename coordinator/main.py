"""NEURON coordinator — FastAPI app (the brain of the network).

Run:  uvicorn coordinator.main:app --reload --port 8000   (from C:\\Users\\optin\\neuron)

Registry + health + routing + ledger + dashboard. Node-management calls are
token-gated (Part 6): registration needs the shared X-Register-Secret; a node's
own ping/delete need its X-Node-Token.
"""
import asyncio
import json
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coordinator import (auth, balancer, config, emission, genesis, ledger, migration,
                         model_registry, model_tiers, models, nodelogs, payout, router,
                         theme)
import relay_auth


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
class NodeLogBody(BaseModel):
    body: str = ""              # a redacted tail of the node's agent.log


class LogRequestBody(BaseModel):
    nodes: list[str] = []       # empty = every node currently known


class SetLayersBody(BaseModel):
    layers: dict[str, list[int]] = {}    # {node_id: [layer_start, layer_end]}, inclusive


class RegisterBody(BaseModel):
    node_id: str
    tailscale_ip: str
    port: int
    layer_start: int
    layer_end: int
    cores: int | None = None
    ram_gb: float | None = None
    behind_nat: bool = False       # if true, coordinator assigns a relay port (Session 12)
    ms_per_layer: float | None = None   # self-benchmark for auto-balancing (Session 14)
    head_ms: float | None = None        # driver's lm_head cost (Session 14)
    platform: str | None = None         # e.g. "Windows-11-10.0.26200"; with cores/ram_gb this
                                        # forms the coarse hardware signature a sybil signal
                                        # groups on. Optional: older agents omit it.
    has_gpu: bool = False               # NVIDIA GPU detected (torch.cuda, else nvidia-smi).
    gpu_vram_gb: float | None = None    # total VRAM. Recorded, and clamped by
                                        # `balancer.sane_vram_gb` below; it does NOT size a
                                        # slice while `balancer.GPU_EXECUTION` is off.
    gpu_name: str | None = None         # e.g. "NVIDIA GeForce RTX 4070". Operator-only in
                                        # /node/list — a card model is fingerprinting detail,
                                        # like `platform`.
    # Why a node is not on the version we published (0.20.2). Nothing reported the running build
    # before this, so a rollout could not be watched and a ROLLBACK could not be confirmed — the
    # recovery path added the same day fired blind. All three are optional and NULL means "an
    # agent too old to say", never "up to date".
    agent_version: str | None = None    # updater.LOCAL_VERSION of the running build
    auto_update: bool | None = None     # False here explains a node that never moves by itself
    update_check: str | None = None     # last check verdict: current / download-failed / …
    model_id: str | None = None         # the model this node is ACTUALLY serving. Reported so
                                        # a coordinator/node divergence is detectable at all --
                                        # without it, both 28-layer tiers validate identically
                                        # and a mismatch shows up only as a dropped socket.
    declared_slots: str | None = None   # UTC hours this machine is usually available, e.g.
                                        # "1,2,3,4,5". A DECLARATION, never a promise: nothing
                                        # is penalised for missing one, because a phone's
                                        # charger is not its owner's decision. It exists so the
                                        # coordinator can see tomorrow's 3am hole today.


class InferBody(BaseModel):
    prompt: str
    max_tokens: int = 200
    wallet_id: str                              # who pays -- required (Workstream B)
    prompt_tokens_estimate: int | None = None   # driver's rough estimate for the hold quote


class CompleteBody(BaseModel):
    tokens_generated: int
    duration_ms: int
    node_ids: list[str]
    complete_token: str | None = None   # [P12] token issued by /infer; required to settle
    prompt_tokens: int = 0              # driver's real tokenizer count (Workstream B)


class WalletFaucetBody(BaseModel):
    wallet_id: str


class WalletOAuthBody(BaseModel):
    provider: str
    external_id: str
    email: str | None = None
    email_verified: bool = False   # provider-asserted; distinguishes a throwaway address


class AttestBody(BaseModel):
    passed: bool
    max_err: float | None = None


class PayoutBindBody(BaseModel):
    address: str                          # the EVM address to be paid
    nonce: str                            # from GET /node/{id}/payout-challenge
    signature: str                        # binding_message signed by `address`
    old_signature: str | None = None      # required only when changing a bound address


class SetModelBody(BaseModel):
    model_id: str | None = None   # None clears the pin and restores capacity-driven tiering


class ViolationBody(BaseModel):
    direction: str               # "in" | "out"
    category: str | None = None  # a blocklist category label -- never the flagged text itself


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_register_secret(x_register_secret: str = Header(default=None)):
    if x_register_secret != config.REGISTRATION_SECRET:
        raise HTTPException(status_code=401, detail="invalid or missing X-Register-Secret")


def classify_registration(x_register_secret: str) -> bool:
    """Decide a registration's standing (Session 12 — open join).

    Returns True if the valid founder secret was presented -> the node is TRUSTED
    (skips probation). With OPEN_JOIN on, a missing/invalid secret is allowed and the
    node joins PROBATIONARY (False). With OPEN_JOIN off, a missing/invalid secret is
    rejected (fully private network — legacy behaviour)."""
    if x_register_secret == config.REGISTRATION_SECRET:
        return True
    if not config.OPEN_JOIN:
        raise HTTPException(status_code=401, detail="invalid or missing X-Register-Secret")
    return False


def require_node_token(node_id: str, x_node_token: str = Header(default=None)):
    node = models.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    if not x_node_token or x_node_token != node["node_token"]:
        raise HTTPException(status_code=401, detail="invalid or missing X-Node-Token")
    return node


# --------------------------------------------------------------------------- #
# Background health sweep
# --------------------------------------------------------------------------- #
# Capacity-driven model tier (auto-model-tiering): the network serves the biggest model
# its live online+eligible capacity can back, re-evaluated each sweep with hysteresis so
# node churn (a laptop sleeping) doesn't flap the active model. Guarded by a lock because
# the async sweep and the threadpool endpoints both advance it.
_tier_controller = model_tiers.TierController()
_tier_lock = threading.Lock()

# The model the network is CURRENTLY serving — what nodes load and what the router assembles
# a chain for. DISTINCT from the TierController's target tier (what the network's capacity
# QUALIFIES for): they converge when a migration moves nodes onto the target (Build 3). Defaults
# to the configured floor so today's behaviour is unchanged; a migration will set it.
#
# PERSISTED, and that is not a detail. This used to live only here, initialised to the config
# floor -- so any restart AFTER a tier migration silently reverted the coordinator's belief about
# which model the network runs, while every node carried on serving the new one. Both Qwen2.5
# tiers have 28 layers, so every range still validated and the chain still read as routable,
# while nodes fed each other activations from different models. The driver saw
# "socket closed mid-message" and users saw a RuntimeError. Live 2026-08-10, and invisible
# because every health signal was green.
_serving = {"model_id": config.MODEL_ID, "layers": config.TOTAL_LAYERS}
_serving_lock = threading.Lock()


def _load_serving():
    """Restore the serving model from the DB at import. Falls back to the configured floor when
    nothing was ever stored, which is the correct answer for a brand-new coordinator."""
    mid = models.get_setting("serving_model_id")
    layers = models.get_setting("serving_layers")
    if mid and layers:
        try:
            _serving["model_id"], _serving["layers"] = mid, int(layers)
        except (TypeError, ValueError):
            pass


def serving_model():
    with _serving_lock:
        return dict(_serving)


def set_serving_model(model_id, layers):
    """Point the network at a different model (used by migration, Build 3). Written through to
    the DB so a restart cannot forget it."""
    with _serving_lock:
        _serving["model_id"] = model_id
        _serving["layers"] = int(layers)
    models.set_setting("serving_model_id", model_id)
    models.set_setting("serving_layers", int(layers))


# Rolling model migration (Build 3): moves the network from the serving model to the target
# tier its capacity qualifies for, without dropping service. Advanced each health sweep.
_migration = migration.MigrationController()
_migration_lock = threading.Lock()


def apply_migration_cutover(model_id, layers):
    """Cutover callback: flip the serving model AND persist each planned node's new layer
    range (models.update_layers) in the same step. Without this, routing/placement would
    keep using each node's OLD range after the model (and thus the partition) changed —
    the node has already reloaded onto the NEW range by the time it reported ready."""
    for a in _migration.plan:
        models.update_layers(a["node_id"], a["layer_start"], a["layer_end"])
    set_serving_model(model_id, layers)


def apply_gap_heal(assignments):
    """Self-heal cutover callback: persist each surplus node's new layer range. Unlike a tier
    migration this never changes the serving model, so there's no set_serving_model() call --
    just closing a coverage gap within whatever model is already being served."""
    for a in assignments:
        models.update_layers(a["node_id"], a["layer_start"], a["layer_end"])


# Last routability verdict, so the sweep logs the TRANSITION rather than the state. An
# unroutable network is not self-correcting -- it would print every 60s forever, and a line that
# appears 1,440 times a day is one nobody reads (the heartbeat lesson, Session 55 half seven).
# None means "not yet evaluated", which is deliberately distinct from False.
_last_routable = None


async def health_loop():
    global _last_routable
    prune_due_at = 0.0
    while True:
        await asyncio.sleep(config.HEALTH_CHECK_INTERVAL_S)
        try:
            for node_id in models.sweep():
                print(f"[health] node '{node_id}' went OFFLINE "
                      f"(no ping in {config.HEARTBEAT_TIMEOUT_S}s)")
            # Is the roster still ROUTABLE? Nothing asked this before: self_heal returns early
            # unless a layer is uncovered (`if not missing: return`), so a fully-covered but
            # unroutable roster was invisible to the one thing that repairs placement, exactly
            # as it was invisible to /status. PROBLEMS.md [P32]. This does not MOVE anything --
            # repairing it means rewriting ranges the nodes themselves re-assert, which is the
            # open question in that entry. It makes a silent failure loud, which is the part
            # that has to exist either way.
            # Availability emission (TOKENOMICS.md §11.4). Piggybacks this sweep rather than
            # running its own timer -- it settles CLOSED slots only, so it needs to run
            # regularly, not punctually, and a second scheduler is a second thing that can
            # silently stop. Never raises into the sweep: an accounting failure must not stop
            # the health check that keeps the network routable.
            try:
                emission.close_slots(serving_model()["layers"])
            except Exception as e:
                print(f"[emission] sweep failed (nodes unaffected, slots stay unsettled "
                      f"and will retry): {type(e).__name__}: {e}")
            roster = models.list_nodes()
            sm_layers = serving_model()["layers"]
            shape = router.chain_shape(roster, sm_layers)
            # AUTO-REPAIR. Detection alone meant a human had to notice and run `neuron fix`
            # after every join or leave -- which on 2026-08-10 meant the chain sat unroutable
            # overnight while nobody was awake. The coordinator knows the one routable shape for
            # its roster; there is no reason for a person to type it in.
            #
            # Only safe because placement ownership shipped first ([P32]): before that a node
            # re-registering would overwrite this, and the sweep would have fought the roster
            # every 60s. Never touches a chain that already routes, so a healthy network is
            # never disturbed.
            # ...but NOT while a tier migration is in flight. A migration owns placement during
            # its own transition (migration.self_heal makes the same exception), and a repair
            # that rewrote ranges mid-cutover would fight it -- half the network on one model's
            # partition and half on the other's, which is unrecoverable rather than merely
            # unroutable.
            with _migration_lock:
                migrating = _migration.phase != "steady"
            if not shape["routable"] and migrating:
                print(f"[repair] chain is unroutable but a migration is {_migration.phase} — "
                      f"leaving placement to it")
            elif not shape["routable"]:
                plan = router.canonical_assignment(
                    roster, sm_layers, serving_model_id=serving_model()["model_id"])
                if plan:
                    for a in plan:
                        models.update_layers(a["node_id"], a["layer_start"], a["layer_end"])
                    shape = router.chain_shape(models.list_nodes(), sm_layers)
                    print(f"[repair] chain was unroutable — reassigned {len(plan)} node(s) to "
                          f"{shape['ranges']} ({shape['stages']} stage(s), "
                          f"routable={shape['routable']})")
                else:
                    print(f"[repair] chain is unroutable and cannot be fixed from this roster: "
                          f"{len(roster)} node(s) known, need at least "
                          f"{config.MIN_PIPELINE_STAGES} online and eligible")
            if shape["routable"] != _last_routable:
                if shape["routable"]:
                    print(f"[health] chain is routable again: {shape['stages']} stage(s) "
                          f"{shape['ranges']}")
                else:
                    print(f"[health] NETWORK NOT ROUTABLE: the chain walks to "
                          f"{shape['stages']} stage(s) {shape['ranges']}, and a driver accepts "
                          f"{config.MIN_PIPELINE_STAGES}-{config.PIPELINE_STAGES}. Every layer "
                          f"may still be covered -- coverage is not routability. Chats will "
                          f"fail AFTER a wallet hold is taken. Fix: "
                          f"./coordinator/pin_layers.sh --driver <the machine you chat from>")
                _last_routable = shape["routable"]
            # Retention: `requests` is the only table that grows with TRAFFIC rather than with
            # users, so it's the one that would actually kill a single-file SQLite DB (~1.25
            # GB/day at 1M users x 5 requests). Identities, ledger rows and moderation_events
            # are never pruned -- bans depend on them and they grow slowly. Piggybacks the
            # existing sweep rather than adding a second timer; hourly is plenty for a daily
            # cutoff.
            if config.REQUEST_RETENTION_DAYS > 0 and time.time() >= prune_due_at:
                prune_due_at = time.time() + 3600
                pruned = models.prune_old_requests()
                if pruned:
                    print(f"[retention] pruned {pruned} request row(s) older than "
                          f"{config.REQUEST_RETENTION_DAYS}d")
            for rid in models.release_stale_holds(config.HOLD_TTL_S):
                print(f"[ledger] released stale hold for request '{rid}' "
                      f"(crashed/abandoned, {config.HOLD_TTL_S}s TTL)")
            with _tier_lock:
                prev = _tier_controller.active()["name"]
                tier = _tier_controller.update(models.list_nodes(), time.time())
                if tier["name"] != prev:
                    print(f"[tier] network now qualifies for {prev} -> {tier['name']}")
            # advance any model migration toward the qualified target tier. gb_per_layer rides
            # along so the migration can refuse a target no single node could hold — the tier
            # gate qualifies on AGGREGATE RAM, which is not the same question.
            target = {"model_id": tier["model_id"], "layers": tier["layers"],
                      "gb_per_layer": tier.get("gb_per_layer")}
            # An operator PIN overrides the capacity ladder entirely. Which model the network
            # serves is a decision, not a consequence of how many machines happen to be awake --
            # and this is also the only way to move nodes onto a model REMOTELY. A volunteer's
            # PC is never reachable; "restart the agent" is not an instruction this product can
            # give. The migration handshake (prepare -> ready -> cutover) is the remote reload,
            # and until now nothing could aim it deliberately.
            pinned = models.get_setting("pinned_model_id")
            if pinned:
                pin = model_tiers.tier_for(pinned)
                if pin:
                    target = {"model_id": pin["model_id"], "layers": pin["layers"],
                              "gb_per_layer": pin.get("gb_per_layer")}
            serving = serving_model()
            serving["gb_per_layer"] = model_tiers.gb_per_layer_for(serving["model_id"])
            with _migration_lock:
                # self-heal first: closes a coverage gap in the CURRENTLY serving model using
                # idle surplus nodes, only while no real tier migration is in flight (it's a
                # no-op the instant update() below starts preparing one).
                healing_before = _migration.heal_status()["healing"]
                heal = _migration.self_heal(models.list_nodes(), serving, time.time(),
                                            apply_gap_heal)
                healing_after = heal["healing"]
                if healing_after and not healing_before:
                    what = ("re-splitting the model across the remaining node(s) — nothing was "
                            "idle to fill it" if heal["mode"] == "resplit"
                            else "reassigning idle node(s)")
                    print(f"[gap-heal] coverage gap detected, {what} "
                          f"(serving={serving['model_id']})")
                    if heal["capacity_shortfall"]:
                        print(f"[gap-heal] WARNING: the remaining node(s) cannot hold "
                              f"{serving['model_id']} — {heal['capacity_shortfall']} layer(s) "
                              f"over capacity; the tier controller should demote")
                elif healing_before and not healing_after:
                    print(f"[gap-heal] coverage restored (serving={serving['model_id']})")

                before = _migration.phase
                blocked_before = _migration.blocked
                _migration.update(models.list_nodes(), target, serving,
                                  time.time(), apply_migration_cutover)
                if _migration.phase != before:
                    print(f"[migration] {before} -> {_migration.phase} "
                          f"(serving={serving_model()['model_id']})")
                blocked = _migration.blocked
                if blocked and blocked != blocked_before:
                    print(f"[migration] NOT migrating to {blocked['model_id']}: no per-node "
                          f"split fits — {blocked['capacity_shortfall']} of "
                          f"{blocked['layers']} layer(s) have nowhere to live. Still serving "
                          f"{serving['model_id']}.")
        except Exception as e:  # never let the loop die
            print(f"[health] sweep error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.init_db()
    nodelogs.init()          # its own tables; kept out of models.SCHEMA so log collection can be
                             # removed in one file without a schema migration
    if genesis.seed_genesis():
        print("[coordinator] genesis buckets seeded (fixed-supply ledger, Phase 0)")
    genesis.verify_invariant()   # fail startup loudly rather than serve on a broken supply
    # BEFORE the health loop starts: the sweep assigns layer ranges against the serving model,
    # so a coordinator that has not yet remembered which model it serves would hand out ranges
    # for the wrong one. This is the line whose absence caused 2026-08-10.
    _load_serving()
    sm = serving_model()
    print(f"[coordinator] up | db={config.DB_PATH} | serving={sm['model_id']} "
          f"({sm['layers']} layers) | timeout={config.HEARTBEAT_TIMEOUT_S}s")
    task = asyncio.create_task(health_loop())
    try:
        yield
    finally:
        task.cancel()


# version from config, not a literal: the hardcoded "0.1" here and COORDINATOR_VERSION would
# drift apart on the first bump, which is exactly the trap CHANGELOG.md describes for the
# agent's three-place version.
app = FastAPI(title="NEURON Coordinator", version=config.COORDINATOR_VERSION, lifespan=lifespan)
# Login for the whole network lives here, not on each agent (coordinator/auth.py).
app.include_router(auth.router)

# --- CORS, so the public landing page can show live network numbers --------- #
# The GitHub Pages site is a different origin, so without this a browser fetches /status and
# then refuses to let the page read the reply. Deliberately narrow:
#
#   - **named origins, never `*`.** The data on /status is already public, but this middleware
#     applies to every route, and a wildcard invites any page on the internet to use a visitor's
#     browser as a client against endpoints that are not.
#   - **allow_credentials stays False.** Combined with a wildcard it is refused by browsers
#     anyway, and combined with named origins it would let a page send a visitor's cookies. The
#     coordinator's privileged endpoints authenticate with headers a web page has no way to know,
#     and that stays true only while nothing is sent automatically.
#   - **GET only.** Nothing on the landing page writes.
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
    max_age=3600,
)


# --- basic per-IP rate limit (Session 16 — rough DDoS guard) ---------------- #
import collections  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

_rate_hits = collections.defaultdict(collections.deque)


@app.middleware("http")
async def _rate_limit(request, call_next):
    ip = request.client.host if request.client else "?"
    now = time.time()
    dq = _rate_hits[ip]
    while dq and dq[0] < now - config.RATE_LIMIT_WINDOW_S:
        dq.popleft()
    if len(dq) >= config.RATE_LIMIT_MAX:
        return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
    dq.append(now)
    return await call_next(request)


# --------------------------------------------------------------------------- #
# Part 1 — Node registry
# --------------------------------------------------------------------------- #
def _assign_relay_port(node_id: str) -> int:
    """Give this NAT'd node a stable public port on the relay, from the pool."""
    existing = models.get_node(node_id)
    if existing and existing["tailscale_ip"] == config.RELAY_HOST \
            and config.RELAY_PORT_MIN <= existing["port"] <= config.RELAY_PORT_MAX:
        return existing["port"]                       # reuse on re-register
    used = {n["port"] for n in models.list_nodes() if n["tailscale_ip"] == config.RELAY_HOST}
    for p in range(config.RELAY_PORT_MIN, config.RELAY_PORT_MAX + 1):
        if p not in used:
            return p
    raise HTTPException(status_code=503, detail="relay port pool exhausted")


@app.post("/node/register")
def register(body: RegisterBody, x_register_secret: str = Header(default=None),
            x_node_token: str = Header(default=None)):
    trusted = classify_registration(x_register_secret)
    # open join: a secret-less registration must not hijack an existing node id that has
    # already earned standing (trusted OR verified via proof-of-compute) — NOT just trusted.
    # A verified-but-not-trusted node was previously unprotected: anyone could re-register its
    # id with no credential, inherit its standing, and get handed a FRESH node_token — locking
    # the real owner out of their own dashboard/balance (post-launch-audit fix). The one way
    # around the secret is presenting that exact node's CURRENT token, proving you already
    # control it (e.g. a legitimate re-register after losing local registration state).
    if not trusted:
        existing = models.get_node(body.node_id)
        if existing and existing["standing"] in ("trusted", "verified"):
            # isinstance guard: a direct (non-HTTP) caller that omits x_node_token gets
            # FastAPI's Header() sentinel object here, not a plain None — compare_digest
            # would TypeError on it, not just fail closed.
            owns_it = (isinstance(x_node_token, str)
                      and secrets.compare_digest(x_node_token, existing["node_token"]))
            if not owns_it:
                raise HTTPException(
                    status_code=409,
                    detail=f"node id '{body.node_id}' is already {existing['standing']}; "
                           f"re-registering it requires either the shared secret or that "
                           f"node's current X-Node-Token")
    token = secrets.token_hex(config.TOKEN_BYTES)
    tailscale_ip, port, relay_block = body.tailscale_ip, body.port, None
    if body.behind_nat and config.RELAY_ENABLED:
        relay_port = _assign_relay_port(body.node_id)
        tailscale_ip, port = config.RELAY_HOST, relay_port    # peers reach it via the relay
        ticket = relay_auth.make_ticket(config.RELAY_SECRET, body.node_id, relay_port)
        relay_block = {"host": config.RELAY_HOST, "control_port": config.RELAY_CONTROL_PORT,
                       "data_port": config.RELAY_DATA_PORT, "public_port": relay_port,
                       "ticket": ticket}
    known = models.get_node(body.node_id) is not None
    # `gpu_vram_gb` is the one field in this body that becomes a MEMORY BUDGET -- it is what
    # `balancer.max_layers_for` would size a slice from -- and under open join this endpoint
    # takes no credential. Bound it here, at the edge, rather than trusting every later reader.
    #
    # Clamped to None, never rejected. A 422 over a cosmetic hardware field would lock a
    # volunteer out of the network entirely, which is the failure class of [P24]: the machine
    # is fine, the operator can see nothing wrong, and it simply never joins. An unbelievable
    # VRAM figure costs nothing if it is dropped -- the node is then sized from its system RAM,
    # like every node is today.
    vram = balancer.sane_vram_gb(body.gpu_vram_gb) if body.has_gpu else None
    fingerprint = models.register_node(
        body.node_id, tailscale_ip, port, body.layer_start,
        body.layer_end, body.cores, body.ram_gb, token,
        ms_per_layer=body.ms_per_layer, head_ms=body.head_ms, trusted=trusted,
        platform=body.platform, has_gpu=body.has_gpu, gpu_vram_gb=vram,
        gpu_name=body.gpu_name,
        agent_version=body.agent_version, auto_update=body.auto_update,
        update_check=body.update_check)
    if body.declared_slots is not None:
        models.set_declared_slots(body.node_id, body.declared_slots)
    if body.model_id:
        models.set_reported_model(body.node_id, body.model_id)
    # Sybil SIGNAL, never a block. One machine registering several node_ids in a day is what a
    # sybil looks like -- and also what a legitimate operator running two nodes on a spare PC
    # looks like, and what two identical laptops look like, since the fingerprint is only
    # cores/RAM/OS. So it is recorded for the operator and nothing else happens. Blocking on a
    # signal this weak would lock out honest people to protect NRN that has no value yet; real
    # resistance arrives when faking it is worth something.
    if fingerprint and not known:
        siblings = models.fingerprint_siblings(fingerprint, body.node_id)
        if siblings:
            models.flag_sybil(
                "fingerprint_reuse", fingerprint, body.node_id,
                f"same hardware signature as {len(siblings)} other node(s) registered in the "
                f"last 24h: {', '.join(siblings[:5])}"
                + (" ..." if len(siblings) > 5 else ""))
    # Report the node's REAL standing, read back from the DB, not a binary guess from whether
    # this call carried the secret. A node that passed proof-of-compute is `verified`, and
    # `challenges_passed` survives re-registration — but this response used to say
    # "probationary" to anyone who re-registered without the secret, which every relayed node
    # does on a ticket refresh or a restart. The agent then logged "PROBATIONARY: ... will not
    # earn NRN" at a node that was verified, eligible and earning. For a stranger, being told
    # on every restart that they are not earning is the kind of thing that gets an agent
    # uninstalled.
    fresh = models.get_node(body.node_id) or {}
    standing = fresh.get("standing") or ("trusted" if trusted else "probationary")
    # A node re-asserting a stale config no longer overwrites its placement ([P32]) -- so say so,
    # here, at the one moment we know it happened. Not an error: the node is not misbehaving, it
    # is running an agent that has not been told what it is assigned. It will serve the right
    # range anyway (slice-info returns the assigned one), and this is what tells an operator the
    # config on that machine is stale before the next `neuron fix` gets quietly undone.
    if fresh.get("placement_drift"):
        print(f"[placement] node '{body.node_id}' registered claiming layers "
              f"{fresh['reported_layer_start']}-{fresh['reported_layer_end']}, but is assigned "
              f"{fresh['layer_start']}-{fresh['layer_end']} — keeping the assignment. Its local "
              f"config is stale; the node will serve the assigned range. PROBLEMS.md [P32]")
    resp = {
        "status": "registered",
        "standing": standing,
        # Read back from the DB, for exactly the reason `standing` above is: this used to echo
        # what the caller sent, which is now a DIFFERENT fact from what the node is assigned.
        # The agent logs this line on every start ("registered as X, assigned layers [a, b]"),
        # so echoing the request would have it confidently print the stale range it just failed
        # to impose.
        "assigned_layers": [fresh.get("layer_start", body.layer_start),
                            fresh.get("layer_end", body.layer_end)],
        "node_token": token,
        # Also here, not just on the heartbeat: a node that re-registers (relay ticket refresh,
        # restart, recovery) learns the current address immediately rather than waiting.
        "coordinator_url": config.PUBLIC_URL,
    }
    if standing == "probationary":
        resp["note"] = ("probationary — you are registered but will not receive live "
                        "requests or earn NRN until a verifier confirms your node with a "
                        "proof-of-compute challenge")
    if relay_block:
        resp["relay"] = relay_block         # agent auto-starts tunnel_client from this
    return resp


@app.get("/node/placement")
def node_placement(exclude: str = None):
    """Advise a joining node which slice to serve (zero-config open join, S20). No auth — a
    node calls this before it has a token; it is read-only and rate-limited by the middleware.
    Includes the serving model_id so the node downloads the right model's slice.

    `exclude` = a node id to leave out of the roster the advice is computed from. An ALREADY
    registered node passes its own id to ask "where would I go if I weren't here?" — the answer
    is its current range whenever that range is genuinely needed, so re-asking is safe and only
    moves a node whose slice is redundant. No auth needed for it either: it reveals nothing
    /node/list doesn't, and a caller passing someone else's id only gets worse advice for
    itself."""
    sm = serving_model()
    return {"total_layers": sm["layers"], "model_id": sm["model_id"],
            **router.suggest_placement(total=sm["layers"], exclude=exclude)}


@app.get("/node/list")
def node_list(x_register_secret: str = Header(default=None)):
    """Node roster. Public callers get health/standing info but NO addresses — node
    endpoints (IP:port) are infrastructure detail, visible only with the operator secret
    (the proof-of-compute verifier is the legitimate consumer). node_token never leaves.

    The hardware signature is operator-only for the same reason the addresses are: published
    against node ids it would let anyone group the roster by machine, which is precisely the
    correlation the private-earnings and private-address decisions exist to prevent. It is a
    review signal, not public information. `gpu_name` joins it: a specific card model is
    distinctive enough to correlate on, where the coarse `has_gpu`/`gpu_vram_gb` pair is no
    more identifying than the `cores`/`ram_gb` already published."""
    show_addr = x_register_secret == config.REGISTRATION_SECRET
    hidden = ({"node_token"} if show_addr
              else {"node_token", "tailscale_ip", "port", "hw_fingerprint", "platform",
                    "gpu_name"})
    nodes = [{k: v for k, v in n.items() if k not in hidden} for n in models.list_nodes()]
    return {"nodes": nodes}


@app.delete("/node/{node_id}")
def unregister(node_id: str, x_node_token: str = Header(default=None),
               x_register_secret: str = Header(default=None)):
    """Remove a node registration.

    Two ways in: the node's OWN token (how uninstall.py deregisters itself), or the operator's
    register secret. The second exists because the first cannot clear the mess that actually
    accumulates — dev nodes, abandoned test registrations and machines that were wiped without
    uninstalling all leave rows whose tokens nobody holds any more. They then sit on the public
    dashboard as permanently-offline entries, which makes a small live network look like a
    graveyard to the first stranger who looks at it.

    Only the `nodes` row goes. The node's LEDGER row is deliberately left alone: balances must
    keep summing to GENESIS_TOTAL_SUPPLY, so deleting one would break the supply invariant and
    quietly destroy NRN that node earned.
    """
    node = models.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    by_operator = (isinstance(x_register_secret, str)
                   and secrets.compare_digest(x_register_secret, config.REGISTRATION_SECRET))
    by_owner = (isinstance(x_node_token, str)
                and secrets.compare_digest(x_node_token, str(node["node_token"])))
    if not (by_operator or by_owner):
        raise HTTPException(status_code=401,
                            detail="this node's own X-Node-Token, or the operator's "
                                   "X-Register-Secret, is required")
    models.delete_node(node_id)
    return {"status": "unregistered", "node_id": node_id,
            "by": "operator" if by_operator else "node"}


# --------------------------------------------------------------------------- #
# Part 2 — Health check
# --------------------------------------------------------------------------- #
@app.get("/node/{node_id}/ping")
def ping(node_id: str, node=Depends(require_node_token)):
    """Heartbeat. Carries `coordinator_url` because this is the one call every live node makes
    continuously — it is how a change of address reaches the whole network within a heartbeat
    instead of never.

    It carries `standing` for the same reason. A node learned its standing exactly once, in the
    reply to its registration, and never again — so a node that joined probationary had no way
    to know it was still excluded from routing, and no way to know when it was promoted. [P24]:
    one sat that way for three days logging `heartbeat ok — active`. The fact was here the
    whole time; nothing was sending it."""
    models.touch_node(node_id)
    return {"status": "alive", "node_id": node_id, "last_seen": time.time(),
            "coordinator_url": config.PUBLIC_URL,
            "standing": node.get("standing", "trusted"),
            # And `want_logs` for a third variation on the same theme: the heartbeat is the only
            # channel that reaches a node behind NAT, so anything the coordinator needs to ASK a
            # node to do has to ride on it. Here it asks for a tail of agent.log — see
            # coordinator/nodelogs.py for why that is worth having and what is stripped from it.
            "want_logs": nodelogs.wanted(node_id)}


@app.post("/node/{node_id}/logs")
def node_logs_upload(node_id: str, body: NodeLogBody, _node=Depends(require_node_token)):
    """A node uploads a tail of its own log, because a heartbeat told it to.

    Authenticated with that node's OWN token, so a node can only ever submit its own log and
    nobody else can submit one on its behalf. Storing redacts and caps again — the agent already
    did both, but the rules protect the person running that machine and a node is not the right
    place to be the only enforcement of them."""
    return nodelogs.store(node_id, body.body)


@app.post("/admin/logs/request")
def admin_logs_request(body: LogRequestBody, _=Depends(require_register_secret)):
    """Ask nodes for a log tail. Empty list = every node currently known."""
    ids = body.nodes or [n["node_id"] for n in models.list_nodes()]
    return {"requested": nodelogs.request(ids)}


@app.get("/admin/logs")
def admin_logs_summary(_=Depends(require_register_secret)):
    """Who has a log on file and how old it is — no bodies. A node MISSING from a request it
    was asked for is itself the diagnosis: it is offline, or its agent is not running."""
    return nodelogs.summary()


@app.get("/admin/logs/{node_id}")
def admin_logs_get(node_id: str, _=Depends(require_register_secret)):
    """One node's stored log. Operator-gated: this is a file from somebody else's computer, and
    the fact that it has been scrubbed of credentials does not make it public."""
    return nodelogs.get(node_id) or {"node_id": node_id, "body": "", "uploaded_at": 0}


@app.get("/node/verify-assignment")
def verify_assignment(x_node_token: str = Header(default=None)):
    """Hand a VERIFIED node somebody to check (peer verification).

    This is what takes the operator out of the loop. Until now a newcomer could not earn until
    the founder personally ran security/proof_of_compute against it, so the network's ability
    to grow depended on one laptop being switched on. Now any already-verified node can pull an
    assignment and vouch, and PEER_VERIFY_QUORUM distinct vouches promote the newcomer.

    The caller authenticates with its OWN node token, which is also how we know it is verified
    and which id its vote belongs to. It gets the target's address here because addresses are
    otherwise private ([P11]) -- a verifier cannot challenge what it cannot reach.
    """
    me = models.node_by_token(x_node_token) if x_node_token else None
    if me is None:
        raise HTTPException(status_code=401, detail="X-Node-Token of a verified node required")
    if not me.get("eligible"):
        raise HTTPException(status_code=403,
                            detail="only a verified/trusted node may verify others")
    now = time.time()
    already = models.peer_targets_of(me["node_id"])
    for n in models.online_nodes(now):
        if n["node_id"] == me["node_id"] or n.get("standing") != "probationary":
            continue
        if n["node_id"] in already:      # one vote each; don't re-issue work already done
            continue
        sm = serving_model()
        return {"node_id": n["node_id"], "host": n["tailscale_ip"], "port": n["port"],
                "layer_start": n["layer_start"], "layer_end": n["layer_end"],
                "total_layers": sm["layers"], "model_id": sm["model_id"],
                "quorum": config.PEER_VERIFY_QUORUM}
    return {"node_id": None}


@app.post("/node/{node_id}/peer-attest")
def peer_attest(node_id: str, body: AttestBody, x_node_token: str = Header(default=None)):
    """A verified node's verdict on a probationary one. Authenticated by the VERIFIER's own
    node token, so every vote is attributable and one machine gets one vote per target."""
    me = models.node_by_token(x_node_token) if x_node_token else None
    if me is None:
        raise HTTPException(status_code=401, detail="X-Node-Token of a verified node required")
    if not me.get("eligible"):
        raise HTTPException(status_code=403, detail="only a verified/trusted node may attest")
    if me["node_id"] == node_id:
        raise HTTPException(status_code=400, detail="a node cannot verify itself")
    if models.get_node(node_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    models.record_peer_attestation(me["node_id"], node_id, body.passed,
                                   getattr(body, "max_err", None))
    passes, fails = models.peer_verdicts(node_id)
    n = models.get_node(node_id)
    print(f"[peer-verify] {me['node_id']} says {node_id} "
          f"{'PASSED' if body.passed else 'FAILED'} "
          f"({passes}/{config.PEER_VERIFY_QUORUM} distinct passes)")
    return {"node_id": node_id, "verifier": me["node_id"], "passed": body.passed,
            "distinct_passes": passes, "distinct_fails": fails,
            "quorum": config.PEER_VERIFY_QUORUM, "standing": n["standing"],
            "eligible": n["eligible"]}


@app.post("/node/{node_id}/attest")
def attest(node_id: str, body: AttestBody, _=Depends(require_register_secret)):
    """A trusted verifier reports a proof-of-compute result (Session 16). Failed
    challenges drop reputation; a flagged node is excluded from routing and earns nothing."""
    if models.get_node(node_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    models.record_attestation(node_id, body.passed)
    # A passing challenge also unlocks THIS slot's availability emission. Uptime alone must
    # never pay: a script that heartbeats and computes nothing would otherwise be the most
    # profitable node on the network. Recorded here because this is the only place the
    # coordinator learns that a node did real work at a known instant.
    if body.passed:
        models.mark_slot_poc(node_id)
    n = models.get_node(node_id)
    # `standing` was the one field missing, and the verifier's success line already read it with
    # a "verified" fallback -- so a node whose standing had NOT changed still logged as though
    # it had. It matters more now that a flagged node can be re-challenged: the caller needs to
    # know whether that pass actually lifted the flag.
    return {"node_id": node_id, "passed": body.passed, "reputation": n["reputation"],
            "flagged": n["flagged"], "standing": n["standing"], "eligible": n["eligible"],
            "challenges_passed": n["challenges_passed"],
            "challenges_failed": n["challenges_failed"]}


@app.post("/node/{node_id}/reputation-reset")
def reputation_reset(node_id: str, _=Depends(require_register_secret)):
    """Retire a node's challenge history (operator only).

    Exists because `flagged` had NO exit. It is derived from cumulative counters that only ever
    grow, the verifier skipped flagged nodes so they could never be re-challenged, and no
    endpoint cleared them — so the only way back was `DELETE /node/{id}`, which destroys the
    node's identity and token and makes a volunteer reinstall to escape a verdict.

    Re-challenging flagged nodes (verify_service) is the mechanism that makes this rare: a node
    that is actually fine now earns its own way back and needs no operator at all. This is for
    the case that mechanism cannot fix — evidence the network MANUFACTURED. On 2026-08-10 three
    honest nodes were failed for being challenged on layers they did not hold; their counters
    are a record of a coordinator bug, and no number of future passes makes those entries true.
    Deleting bad data is not the same as forgiving a bad node.

    Deliberately NOT self-serve: a node cannot clear its own reputation (that would make
    proof-of-compute advisory), and it does not touch peer attestations, which are other
    machines' testimony rather than ours to erase."""
    n = models.get_node(node_id)
    if n is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    before = {"reputation": n["reputation"], "standing": n["standing"],
              "challenges_passed": n["challenges_passed"],
              "challenges_failed": n["challenges_failed"]}
    models.reset_attestations(node_id)
    after = models.get_node(node_id)
    print(f"[attest] reputation reset for {node_id}: {before['challenges_passed']}/"
          f"{before['challenges_passed'] + before['challenges_failed']} cleared, "
          f"standing {before['standing']} -> {after['standing']}")
    return {"node_id": node_id, "before": before,
            "after": {"reputation": after["reputation"], "standing": after["standing"],
                      "eligible": after["eligible"]}}


# --- Session 8: tell a node exactly what to download before it downloads ----- #
@app.get("/node/{node_id}/slice-info")
def slice_info(node_id: str):
    node = models.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    from coordinator import sliceinfo
    try:
        sm = serving_model()
        return sliceinfo.slice_info(sm["model_id"], node["layer_start"],
                                    node["layer_end"], sm["layers"])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not read model header: {e}")


# --------------------------------------------------------------------------- #
# Part 3 — Request routing
# --------------------------------------------------------------------------- #
@app.post("/infer")
def infer(body: InferBody):
    # Login is enforced HERE, not in the UI. Every driver -- including a self-hosted one whose
    # owner has stripped the client-side moderation gate -- must call /infer to get a node
    # chain, so this is the one identity check a user cannot patch out of their own copy.
    if not models.is_oauth_wallet(body.wallet_id):
        raise HTTPException(status_code=403,
                            detail="this wallet is not linked to a verified Google/GitHub "
                                   "login; sign in to use the network")
    if models.wallet_moderation_status(body.wallet_id)["banned"]:
        raise HTTPException(status_code=403,
                            detail="this wallet is blocked for repeated content-policy "
                                   "violations (see SAFETY.md)")
    chain, missing = router.build_chain(total=serving_model()["layers"])
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"incomplete chain - missing layers {router.missing_str(missing)}",
        )
    request_id = str(uuid.uuid4())
    plan_node_ids = [n["node_id"] for n in chain]          # the chain WE chose (incl. replica)
    complete_token = secrets.token_hex(config.TOKEN_BYTES)  # only the caller who got this may complete
    # Fixed-supply ledger (Workstream B): hold the worst-case cost BEFORE dispatching anything.
    # A driver that doesn't know its real tokenizer count yet gets a cheap char/3 estimate --
    # only an upper bound is needed here, settle() charges the real metered cost afterward.
    est_input = (body.prompt_tokens_estimate if body.prompt_tokens_estimate is not None
                else max(1, len(body.prompt) // 3))
    hold_amount = ledger.quote(body.max_tokens, est_input)
    if not models.hold(request_id, body.wallet_id, hold_amount):
        raise HTTPException(status_code=402,
                            detail=f"insufficient NRN balance; this request needs "
                                   f"{hold_amount} NRN held")
    models.create_request(request_id, len(body.prompt), body.max_tokens, plan_node_ids,
                          complete_token, wallet_id=body.wallet_id, hold_amount=hold_amount)
    return {"chain": router.chain_public(chain), "request_id": request_id,
            "complete_token": complete_token, "hold_amount": hold_amount}


@app.post("/infer/{request_id}/complete")
def complete(request_id: str, body: CompleteBody):
    req = models.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"unknown request '{request_id}'")
    if req["status"] == "completed":
        raise HTTPException(status_code=409, detail="request already completed")
    # [P12] authenticate: only whoever received the token from /infer may complete this request.
    expected = req.get("complete_token")
    if expected and not secrets.compare_digest(str(body.complete_token or ""), str(expected)):
        raise HTTPException(status_code=401, detail="invalid or missing complete_token")
    # [P12] settle from the plan WE recorded at /infer, never the caller-reported node_ids — so a
    # completion can only ever pay the nodes the coordinator actually routed (incl. the chosen replica).
    plan = json.loads(req["plan_node_ids"]) if req.get("plan_node_ids") else list(body.node_ids)
    tokens = max(0, min(int(body.tokens_generated), int(req["max_tokens"] or body.tokens_generated)))
    # server-side recount clamp, same trust posture as the tokens_generated clamp above: the
    # coordinator can't tokenize (stays torch-free) but CAN bound a lying driver's report
    # against the prompt LENGTH it recorded at /infer (never the text itself, see SAFETY.md),
    # so a lie can only under-report, never over-charge past the hold.
    prompt_tokens = max(0, min(int(body.prompt_tokens), req.get("prompt_len") or 0))
    # complete_request's UPDATE is conditioned on WHERE status='pending' and reports whether IT
    # was the call that flipped the row — the true single-writer guard. The status check above
    # (line ~352) is only a fast pre-check; two concurrent /complete calls with the SAME valid
    # token both pass it before either commits, so settle() must gate on THIS return value,
    # not the pre-check, or the race pays every racer instead of paying once (post-audit fix).
    won = models.complete_request(request_id, tokens, body.duration_ms, plan)
    if not won:
        raise HTTPException(status_code=409, detail="request already completed")
    plan_nodes = [models.get_node(nid) for nid in plan]
    rewards = ledger.settle(request_id, req.get("wallet_id"), req.get("hold_amount") or 0.0,
                            prompt_tokens, tokens, plan_nodes)
    return {"status": "completed", "request_id": request_id, "rewards": rewards}


@app.get("/pricing")
def pricing():
    """Single source of truth for what a request costs -- kills the old duplicated constant
    (config.py vs api/openai_compat.py had the same NRN_PER_REQUEST defined twice)."""
    return {"price_per_1k_weighted_tokens": config.PRICE_PER_1K_WEIGHTED,
           "input_weight": config.INPUT_WEIGHT, "coordinator_fee": config.COORDINATOR_FEE,
           "reference_model": config.MODEL_ID, "reference_layers": config.TOTAL_LAYERS,
           "head_bonus_layer_equivalents": config.HEAD_BONUS_LE}


@app.get("/supply")
def supply():
    """Public per-bucket supply + the fixed-1e9 invariant check. Only genesis buckets +
    escrow are named -- never a per-wallet or per-node balance (same privacy posture as the
    public /dashboard, which also never shows individual balances)."""
    return models.supply_snapshot()


def require_link_secret(x_wallet_link_secret):
    """Shared gate for every endpoint that may only be called by a trusted driver or by the
    operator -- factored out of the copies in /wallet/oauth and /wallet/{id}/violation so a
    new privileged endpoint can't silently ship without it (which is exactly how
    /wallet/faucet ended up world-callable)."""
    if not (isinstance(x_wallet_link_secret, str)
           and secrets.compare_digest(x_wallet_link_secret, config.WALLET_LINK_SECRET)):
        raise HTTPException(status_code=401, detail="invalid or missing X-Wallet-Link-Secret")


@app.post("/wallet/oauth")
def wallet_oauth(body: WalletOAuthBody, x_wallet_link_secret: str = Header(default=None)):
    """Resolve (or create) the wallet for a (provider, external_id) OAuth identity. Gated by
    a shared secret -- the CALLER (a driver process) is trusted to have already verified this
    identity with the real OAuth provider; this endpoint itself does no verification of its
    own, so without the secret anyone could squat a wallet under an external_id they don't
    control. A brand-new wallet gets the faucet claimed in the SAME call (models.wallet_for_
    oauth), so login and spend-ability ship together."""
    if not (isinstance(x_wallet_link_secret, str)
           and secrets.compare_digest(x_wallet_link_secret, config.WALLET_LINK_SECRET)):
        raise HTTPException(status_code=401, detail="invalid or missing X-Wallet-Link-Secret")
    wallet_id, is_new = models.wallet_for_oauth(body.provider, body.external_id, body.email,
                                                email_verified=bool(body.email_verified))
    return {"wallet_id": wallet_id, "is_new": is_new}


@app.post("/wallet/{wallet_id}/violation")
def wallet_violation(wallet_id: str, body: ViolationBody,
                     x_wallet_link_secret: str = Header(default=None)):
    """Record that a driver's moderation gate (safety/moderation.py) blocked a request from
    this wallet's identity, escalating to a ban across MODERATION_BAN_THRESHOLD violations --
    a per-request block alone forgets who did it the moment the response is sent. Gated the
    same way as /wallet/oauth: only a driver that already judged this content (the only place
    plaintext ever exists in NEURON) may assert it happened. Deliberately accepts a category
    label only, never text -- the coordinator stays blind to plaintext even here."""
    if not (isinstance(x_wallet_link_secret, str)
           and secrets.compare_digest(x_wallet_link_secret, config.WALLET_LINK_SECRET)):
        raise HTTPException(status_code=401, detail="invalid or missing X-Wallet-Link-Secret")
    result = models.record_violation(wallet_id, body.direction, body.category)
    return {"wallet_id": wallet_id, **result}


@app.post("/wallet/faucet")
def wallet_faucet(body: WalletFaucetBody, x_wallet_link_secret: str = Header(default=None)):
    """One-time grant per wallet_id. Ships in the same release as the debit -- a wallet that
    can never receive anything can never spend anything either (TOKENOMICS.md §11.6).

    SECURITY: this endpoint used to be completely open, and models.claim_faucet CREATES the
    ledger row for whatever wallet_id it's handed. So anyone could POST an arbitrary string,
    get a funded wallet with a clean record, use it as an API bearer key, and mint a fresh one
    the moment it was banned -- no login, no cost, unlimited. That made the whole
    login/ban system decorative. Now gated like its sibling endpoints AND restricted to
    wallets that came from a real Google/GitHub login."""
    require_link_secret(x_wallet_link_secret)
    if not models.is_oauth_wallet(body.wallet_id):
        raise HTTPException(status_code=403,
                            detail="faucet is only available to wallets created by a real "
                                   "Google/GitHub login")
    if models.claim_faucet(body.wallet_id, config.FAUCET_AMOUNT_NRN):
        return {"wallet_id": body.wallet_id, "granted": config.FAUCET_AMOUNT_NRN}
    raise HTTPException(status_code=409, detail="faucet already claimed for this wallet")


# --------------------------------------------------------------------------- #
# Operator review + enforcement (see SAFETY.md)
# --------------------------------------------------------------------------- #
@app.post("/wallet/{wallet_id}/ban")
def wallet_ban(wallet_id: str, x_wallet_link_secret: str = Header(default=None)):
    """Ban an identity by hand. The automatic threshold only counts violations the DRIVER
    self-reports, and for a self-hosted install the driver is the user's own machine -- so a
    stripped client never reports itself and never trips it. This is the operator lever for
    everything the keyword filter misses (jailbreaks, paraphrase, abuse reports). Enforced at
    /infer, which is server-side, so it holds against a modified client."""
    require_link_secret(x_wallet_link_secret)
    if not models.set_ban(wallet_id, True):
        raise HTTPException(status_code=404, detail="unknown wallet")
    return {"wallet_id": wallet_id, "banned": True}


@app.post("/wallet/{wallet_id}/unban")
def wallet_unban(wallet_id: str, x_wallet_link_secret: str = Header(default=None)):
    """Reverse a ban (operator error, successful appeal, resolved false positive)."""
    require_link_secret(x_wallet_link_secret)
    if not models.set_ban(wallet_id, False):
        raise HTTPException(status_code=404, detail="unknown wallet")
    return {"wallet_id": wallet_id, "banned": False}


@app.get("/wallet/{wallet_id}/activity")
def wallet_activity(wallet_id: str, x_wallet_link_secret: str = Header(default=None)):
    """One identity's reviewable history -- who they are, their moderation events, and their
    recent requests -- for deciding whether to ban. Request rows carry prompt_len only, never
    prompt text (SAFETY.md), so this answers 'who did this, when, how much' and deliberately
    not 'what did they type'."""
    require_link_secret(x_wallet_link_secret)
    data = models.wallet_activity(wallet_id)
    if data["identity"] is None:
        raise HTTPException(status_code=404, detail="unknown wallet")
    return data


@app.get("/admin/identities")
def admin_identities(banned_only: bool = False, limit: int = 200,
                     x_wallet_link_secret: str = Header(default=None)):
    """Every identity that has ever logged in -- backs the admin review page."""
    require_link_secret(x_wallet_link_secret)
    return {"identities": models.list_identities(limit=limit, banned_only=banned_only)}


@app.get("/admin/sybil-flags")
def admin_sybil_flags(limit: int = 200, kind: str = None,
                      x_wallet_link_secret: str = Header(default=None)):
    """Sybil signals for operator review. Secret-gated and operator-only on purpose: these are
    unproven suspicions about specific people, false positives are expected (the hardware
    signature is only cores/RAM/OS), and publishing "this node looks fake" would be a public
    accusation the evidence cannot support. Nothing is blocked on any of it."""
    require_link_secret(x_wallet_link_secret)
    return {"flags": models.list_sybil_flags(limit=limit, kind=kind)}


@app.get("/wallet/{wallet_id}")
def wallet_balance(wallet_id: str):
    """wallet_id is an unguessable secret minted by wallet_for_oauth() (32 hex chars) -- same
    bearer-capability pattern this codebase already uses for complete_token/node tokens, so
    knowing it is the authorization. Not listable/enumerable anywhere."""
    row = models.get_ledger(wallet_id)
    if row is None or row.get("account_type") != "wallet":
        raise HTTPException(status_code=404, detail="unknown wallet")
    return {"wallet_id": wallet_id, "balance": row["balance"], "total_earned": row["total_earned"],
           "violation_count": row.get("violation_count", 0) or 0,
           "moderation_banned": bool(row.get("moderation_banned", 0))}


# --------------------------------------------------------------------------- #
# Auto-balance (Session 14) — assign layers by each node's measured speed
# --------------------------------------------------------------------------- #
def _balanced_plan():
    # only nodes cleared for live traffic (excludes probationary/flagged) are planned,
    # so the balancer never assigns layers to a node routing would skip (S12).
    nodes = [n for n in models.online_nodes() if n.get("ms_per_layer") and n.get("eligible")]
    # the driver (carries lm_head, head_ms > 0) goes first, then the rest
    nodes.sort(key=lambda n: (0 if (n.get("head_ms") or 0) > 0 else 1, n["layer_start"],
                              n["node_id"]))
    # Only PIPELINE_STAGES nodes become stages; the rest replicate one below. The balancer gives
    # every node it is handed its own contiguous slice, so passing it the whole roster produces a
    # chain of whatever length the roster happens to be -- and the driver routes exactly three
    # (config.PIPELINE_STAGES). Handing out a 4-stage chain is the same failure as the 1-stage
    # one that broke chat on 2026-08-07, just from the other direction.
    extra = nodes[config.PIPELINE_STAGES:]
    nodes = nodes[:config.PIPELINE_STAGES]
    bnodes = [{"node_id": n["node_id"], "ms_per_layer": n["ms_per_layer"],
               "head_ms": n.get("head_ms") or 0.0,
               # GPU is a tie-break in the balancer, never a speed multiplier — a node's speed
               # is the ms_per_layer it actually measured. See coordinator/balancer.py.
               "has_gpu": bool(n.get("has_gpu")), "gpu_vram_gb": n.get("gpu_vram_gb"),
               # Memory, so the balancer's hard cap is live here too. It has taken
               # `gb_per_layer` since Session 14 and no caller ever passed it, and no caller
               # passed RAM either — so the cap that exists to stop an OOM has never once been
               # applied to a real plan. /network/rebalance applies these ranges for real.
               "ram_gb": n.get("ram_gb")}
              for n in nodes]
    sm = serving_model()
    p = balancer.plan(bnodes, sm["layers"], model_tiers.gb_per_layer_for(sm["model_id"]))
    # Replicas go to the thinnest stage. A capped-out machine must still be given work: idle
    # earns its operator nothing and adds nothing, and replicas are how extra machines turn into
    # throughput rather than a deeper pipeline ([P16]).
    stages = [{"layer_start": a["layer_start"], "layer_end": a["layer_end"]}
              for a in p.get("assignment", [])]
    if stages and extra:
        depth = [1] * len(stages)
        for n in extra:
            i = min(range(len(stages)), key=lambda k: (depth[k], k))
            depth[i] += 1
            p["assignment"].append({"node_id": n["node_id"], "replica": True, **stages[i]})
    return p


@app.get("/network/plan")
def network_plan():
    """The layer split the balancer recommends from nodes' measured speeds (advisory)."""
    p = _balanced_plan()
    if not p.get("assignment"):
        return {"assignment": [], "note": "no online node has reported ms_per_layer yet"}
    return p


@app.post("/network/rebalance")
def network_rebalance(_=Depends(require_register_secret)):
    """Apply the balanced plan: update each node's stored layer range so /infer routes the
    optimal split. Call when nodes join/leave; a node reloads only if its range moved."""
    p = _balanced_plan()
    changed = []
    for a in p.get("assignment", []):
        models.update_layers(a["node_id"], a["layer_start"], a["layer_end"])
        changed.append({"node_id": a["node_id"], "layers": [a["layer_start"], a["layer_end"]]})
    extra = {k: p[k] for k in ("balanced_bottleneck_ms", "equal_split_bottleneck_ms",
                               "speedup_vs_equal") if k in p}
    return {"status": "rebalanced", "assignments": changed, **extra}


@app.post("/network/layers")
def network_set_layers(body: SetLayersBody, _=Depends(require_register_secret)):
    """Set an EXPLICIT layer range per node, overriding every automatic split.

    The balancer optimises for stage time and will happily produce a split no CLIENT can use:
    the driver holds a fixed shard (`neuron_driver.S1`, layers 0..S1-1) and rejects any chain
    whose first stage is not exactly that (node_a.py). So a speed-optimal 0-12 / 13-27 split is
    unroutable by a driver built for 0-9, and the only ways out were to re-download the driver's
    shard or to keep re-rolling the balancer until it happened to agree. Neither is a plan.

    Validated as a WHOLE before anything is written: the ranges must be contiguous, non
    overlapping, start at 0 and end at the serving model's last layer. A partial application
    would leave the network in a state no chain can be built from, which is the failure this
    endpoint exists to end rather than to cause.

    Nodes not named are left alone -- they keep whatever range they had, which is how a machine
    stays a replica of a stage while the stages themselves are pinned.
    """
    total = serving_model()["layers"]
    known = {n["node_id"] for n in models.list_nodes()}
    spans = []
    for node_id, span in (body.layers or {}).items():
        if node_id not in known:
            raise HTTPException(status_code=404, detail=f"no such node: {node_id}")
        if not isinstance(span, list) or len(span) != 2:
            raise HTTPException(status_code=400,
                                detail=f"{node_id}: expected [start, end], got {span!r}")
        lo, hi = int(span[0]), int(span[1])
        if not 0 <= lo <= hi < total:
            raise HTTPException(
                status_code=400,
                detail=f"{node_id}: [{lo}, {hi}] is not inside 0..{total - 1}")
        spans.append((lo, hi, node_id))

    # Contiguity is checked over the DISTINCT ranges, so two nodes may share one range (they are
    # replicas of that stage) without that reading as an overlap.
    stages = sorted({(lo, hi) for lo, hi, _ in spans})
    if stages:
        if stages[0][0] != 0 or stages[-1][1] != total - 1:
            raise HTTPException(
                status_code=400,
                detail=f"the ranges given cover {stages[0][0]}..{stages[-1][1]}, "
                       f"not the whole model 0..{total - 1}")
        for (a_lo, a_hi), (b_lo, b_hi) in zip(stages, stages[1:]):
            if b_lo != a_hi + 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"gap or overlap between {a_lo}-{a_hi} and {b_lo}-{b_hi}")

    for lo, hi, node_id in spans:
        models.update_layers(node_id, lo, hi)
    return {"status": "set", "stages": [list(s) for s in stages],
            "assignments": [{"node_id": n, "layers": [lo, hi]} for lo, hi, n in spans]}


# --------------------------------------------------------------------------- #
# Part 4 — Ledger
# --------------------------------------------------------------------------- #
def _require_own_token(node_id: str, token: str | None):
    """A node's earnings are private: only the holder of that node's own token may read
    them (the token is issued once at registration and never shown to anyone else)."""
    node = models.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node '{node_id}'")
    if not token or not secrets.compare_digest(str(token), str(node["node_token"])):
        raise HTTPException(status_code=401,
                            detail="this ledger is private to the node — X-Node-Token required")
    return node


@app.get("/ledger/{node_id}")
def get_ledger(node_id: str, x_node_token: str = Header(default=None)):
    _require_own_token(node_id, x_node_token)
    row = models.get_ledger(node_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no ledger for '{node_id}'")
    return {
        "node_id": node_id,
        "balance": round(row["balance"], 4),
        "total_earned": round(row["total_earned"], 4),
        "requests_served": row["requests_served"],
    }


# --------------------------------------------------------------------------- #
# Payout address binding (blockchain/MIGRATION_PLAN.md blocker 1)
# --------------------------------------------------------------------------- #
@app.get("/node/{node_id}/payout-challenge")
def payout_challenge(node_id: str, address: str = None,
                     x_node_token: str = Header(default=None)):
    """Issue the single nonce this node must sign to bind (or change) its payout address.

    Gated on the node's own token: a nonce is not a secret, but handing them out to anyone
    would let a stranger invalidate a node's in-flight challenge at will. Pass `?address=` to
    get back the exact message text — that is what a human pastes into a wallet's "sign
    message" box, and it must match byte for byte.
    """
    _require_own_token(node_id, x_node_token)
    nonce = models.issue_payout_challenge(node_id)
    out = {"node_id": node_id, "nonce": nonce,
           "expires_in_seconds": config.PAYOUT_CHALLENGE_TTL}
    if address:
        try:
            checksummed = payout.normalize_address(address)
        except payout.PayoutError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        out["address"] = checksummed
        out["message"] = payout.binding_message(node_id, checksummed, nonce)
    return out


@app.post("/node/{node_id}/payout-address")
def bind_payout_address(node_id: str, body: PayoutBindBody,
                        x_node_token: str = Header(default=None),
                        x_register_secret: str = Header(default=None)):
    """Bind the EVM address this node's NRN is paid to, proving control of it.

    Auth is deliberately two-layered. The node's own token says *this node* is asking; the
    signature says *the address owner* consents. Neither alone is enough, because neither
    alone is convincing: a token can be copied off a disk, and a signature says nothing about
    which node it was meant for unless the node_id is inside it (it is).

    Rebinding an already-bound address additionally needs `old_signature` from the currently
    bound key, so a stolen token cannot redirect earnings. The register secret overrides that
    — the recovery path for a genuinely lost key, and a deliberately human decision.
    """
    _require_own_token(node_id, x_node_token)
    operator = (isinstance(x_register_secret, str)
                and secrets.compare_digest(x_register_secret, config.REGISTRATION_SECRET))
    try:
        result = payout.bind(node_id, body.address, body.nonce, body.signature,
                             old_signature=body.old_signature, operator_override=operator)
    except payout.PayoutError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return result


@app.get("/node/{node_id}/payout-address")
def read_payout_address(node_id: str, x_node_token: str = Header(default=None)):
    """A node's own binding. Private, like its balance: a payout address is a persistent
    pseudonymous identifier, and publishing the map from node to address would tie every
    node's earnings together on-chain for anyone watching."""
    _require_own_token(node_id, x_node_token)
    bound = models.get_payout_address(node_id)
    return {"node_id": node_id, "payout_address": bound["payout_address"] if bound else None,
            "bound_at": bound["bound_at"] if bound else None}


# --------------------------------------------------------------------------- #
# Part 5 — Status + dashboard
# --------------------------------------------------------------------------- #
def _network_summary():
    nodes = models.list_nodes()
    online = [n for n in nodes if n["status"] == "online"]
    usable = [n for n in online if n.get("eligible")]        # cleared for live traffic
    # flagged = failed PoC (S16); probationary = open-join, not yet verified (S12)
    flagged = [n for n in online if n.get("flagged")]
    probationary = [n for n in online if n.get("standing") == "probationary"]
    sm = serving_model()
    sm_layers = sm["layers"]
    covered = set()
    for n in usable:
        covered.update(range(n["layer_start"], n["layer_end"] + 1))
    total_covered = len(covered & set(range(sm_layers)))
    # Reuses the roster already fetched above rather than calling build_chain(), which would hit
    # the DB a second time inside a function the dashboard calls on every page load.
    shape = router.chain_shape(nodes, sm_layers)
    return {
        "total_nodes": len(nodes),
        "online_nodes": len(online),
        "eligible_nodes": len(usable),
        "flagged_nodes": len(flagged),
        "probationary_nodes": len(probationary),
        "total_layers_covered": total_covered,
        "total_layers": sm_layers,
        # WHICH layers are missing, as [start, end] ranges. "21/28 covered" tells an operator
        # the chain is broken; this tells them where to put a node to fix it, and it is the
        # same fact the dashboard's coverage strip and the doctor's failure message use.
        "uncovered_layers": _ranges(sorted(set(range(sm_layers)) - covered)),
        # ROUTABILITY, which is not the same fact as coverage and was never reported.
        # `total_layers_covered == total_layers` asks "is every layer served somewhere"; a
        # driver asks "does this walk to 2 or 3 stages", and a roster can pass the first while
        # failing the second -- one node holding the whole model wins the chain walk from cursor
        # 0 and swallows the stages below it. Live for hours on 2026-08-09 with everything below
        # reading green. PROBLEMS.md [P32].
        "stages": shape["stages"],
        "chain_ranges": shape["ranges"],
        "routable": shape["routable"],
        # Reported separately because it is a DIFFERENT failure from a bad stage count, and the
        # remedy differs: this one means stage 1 is the wrong width for the driver's fixed shard.
        "stage1_ok": shape["stage1_ok"],
        "expected_stage1": shape["expected_stage1"],
        # Nodes serving a DIFFERENT model from the network. Never benign: such a node cannot
        # produce correct activations for the chain it sits in, and the user sees a dropped
        # socket rather than a wrong answer. [P33].
        "model_mismatch": [n["node_id"] for n in models.model_mismatches(sm["model_id"])],
        # network_healthy now means "a request can actually complete", which is what every
        # consumer already believed it meant: the dashboard dot, neuron_doctor's verdict, the
        # landing page and ui/app.py all treat it as "is the network working". Coverage alone
        # answered a narrower question while presenting as that one.
        "network_healthy": (total_covered == sm_layers and shape["routable"]
                            and not models.model_mismatches(sm["model_id"])),
    }, nodes


def _ago(ts):
    """"12s ago" / "4m ago" / "3h ago". A raw epoch on a dashboard is not information."""
    if not ts:
        return "never"
    d = max(0, int(time.time() - ts))
    if d < 60:
        return f"{d}s ago"
    if d < 3600:
        return f"{d // 60}m ago"
    if d < 86400:
        return f"{d // 3600}h ago"
    return f"{d // 86400}d ago"


def _ranges(values):
    """[0,1,2,7,8] -> [[0,2],[7,8]]. Contiguous runs read as one gap, not five."""
    out = []
    for v in values:
        if out and v == out[-1][1] + 1:
            out[-1][1] = v
        else:
            out.append([v, v])
    return out


@app.get("/status")
def status():
    # coordinator_version is what makes an unattended update verifiable. "The service came back
    # up" is not the same claim as "the new code is running" -- a half-extracted tarball, a
    # restart that raced the file swap, or a rollback that quietly succeeded all leave a
    # perfectly healthy process serving the OLD build. selfupdate.py gates on this.
    network, _ = _network_summary()
    return {"coordinator_version": config.COORDINATOR_VERSION,
            "network": network, "stats": models.network_stats()}


@app.post("/network/model")
def network_set_model(body: SetModelBody, _=Depends(require_register_secret)):
    """Pin the model the network serves — and REMOTELY move every node onto it.

    Two things at once, and both matter:

    1. Which model the network runs stops being a function of how many machines are awake. Four
       PCs coming online should not silently retier the product (2026-08-10, when exactly that
       started a 7B migration nobody asked for).
    2. It is the only way to move nodes onto a model **without touching them**. A volunteer's PC
       is 100 km away and behind a NAT; "restart the agent" is not an instruction this product
       can give. The migration handshake — prepare, download, report ready, cut over together —
       already does remote reloads (it is how the network moved to 7B this morning). Nothing
       could aim it deliberately until now.

    `model_id: null` clears the pin and hands the decision back to the capacity ladder.
    """
    if body.model_id is None:
        models.set_setting("pinned_model_id", "")
        return {"status": "cleared", "serving": serving_model()}
    tier = model_tiers.tier_for(body.model_id)
    if tier is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown model '{body.model_id}'. Known: "
                   f"{[t['model_id'] for t in model_tiers.TIERS]}")
    models.set_setting("pinned_model_id", tier["model_id"])
    return {"status": "pinned", "model_id": tier["model_id"], "layers": tier["layers"],
            "serving_now": serving_model(),
            "note": "nodes migrate on the next health sweep: they download the slice, report "
                    "ready, and cut over together. No node needs to be touched."}


@app.get("/network/slots")
def network_slots():
    """Where the network needs cover, and what that hour currently pays.

    Public on purpose. This is a recruiting signal, not operator detail: a volunteer deciding
    whether to leave a machine on overnight should be able to see that 03:00 UTC pays the cap
    because nobody is holding layers 10-18 then. It publishes replica DEPTH and multipliers,
    never node ids or addresses -- the same privacy line /node/list already draws.

    New capability rather than a restatement of /status: /status says whether the chain works
    NOW, this says where it is about to stop working."""
    sm = serving_model()
    return emission.coverage_report(models.list_nodes(), sm["layers"])


@app.get("/network/model")
def network_model():
    """The model the network is serving now, the capacity behind it, and what it takes to
    unlock the next tier (auto-model-tiering). Selection is capacity-driven with hysteresis;
    calling this also advances the selection using the current time. `serving` is what the
    network runs RIGHT NOW; the tier ladder is what its capacity QUALIFIES for."""
    with _tier_lock:
        snap = model_tiers.snapshot(models.list_nodes(), _tier_controller, now=time.time())
    snap["serving"] = serving_model()
    return snap


@app.get("/network/migration")
def network_migration():
    """Current model-migration status (Build 3): phase, target, per-node readiness."""
    with _migration_lock:
        return _migration.status()


@app.get("/network/gap-heal")
def network_gap_heal():
    """Current self-heal status: is a coverage gap being closed right now, with which idle
    node(s), and are they ready yet. Separate from /network/migration -- self-heal never
    changes the serving model, only reassigns already-idle capacity to close a gap."""
    with _migration_lock:
        return _migration.heal_status()


@app.get("/node/{node_id}/migration")
def node_migration(node_id: str):
    """The target slice a migrating node should prepare (download), or {migrating:false}.
    Read-only + rate-limited; a node polls this during a migration to fetch its new range."""
    with _migration_lock:
        asg = _migration.assignment_for(node_id)
    return asg or {"migrating": False}


@app.post("/node/{node_id}/migration-ready")
def node_migration_ready(node_id: str, _node=Depends(require_node_token)):
    """A node reports it downloaded the target slice and can serve its target range. Token-gated.
    Cutover (flip serving to the target) happens once every planned node has reported ready."""
    with _migration_lock:
        ok = _migration.mark_ready(node_id)
        st = _migration.status() if ok else None
    return {"node_id": node_id, "acknowledged": ok, "status": st}


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """Operator console for reviewing identities and banning abusers.

    The PAGE is public but carries no data -- every figure on it comes from the
    secret-gated /admin/identities and /wallet/{id}/activity endpoints, which the browser
    calls with the operator's key held in sessionStorage. So the key never appears in a URL,
    never lands in server logs or browser history, and closing the tab forgets it. Serving
    the empty shell unauthenticated is what makes that possible."""
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEURON — identity review</title>
<style>
 :root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2328;--muted:#6a737d;--line:#e3e6ea;--brand:#4f46e5;
   --danger:#c5221f;--ok:#137333}
 @media(prefers-color-scheme:dark){:root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;
   --muted:#8b949e;--line:#2a2f37;--brand:#8b8cf7}}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--ink);
   font:14px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
 .wrap{max-width:1150px;margin:0 auto;padding:1.5rem 1rem 4rem}
 h1{margin:0 0 .2rem;font-size:1.4rem} h1 span{color:var(--brand)}
 .sub{color:var(--muted);margin-bottom:1.2rem}
 .bar{display:flex;gap:.5rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap}
 input,button{font:inherit;padding:.45rem .7rem;border-radius:8px;border:1px solid var(--line)}
 input{background:var(--panel);color:var(--ink);min-width:260px}
 button{background:var(--brand);color:#fff;border:0;cursor:pointer}
 button.ghost{background:transparent;color:var(--ink);border:1px solid var(--line)}
 button.danger{background:var(--danger)} button.ok{background:var(--ok)}
 .tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);
   border-radius:10px}
 table{border-collapse:collapse;width:100%;min-width:860px}
 th,td{border-bottom:1px solid var(--line);padding:.5rem .7rem;text-align:left;
   white-space:nowrap}
 th{background:var(--line);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
 tr:last-child td{border-bottom:0}
 .pill{padding:2px 8px;border-radius:10px;font-size:12px;color:#fff;display:inline-block}
 .muted{color:var(--muted)} .mono{font-family:ui-monospace,Consolas,monospace;font-size:12px}
 .empty{padding:2rem;text-align:center;color:var(--muted)}
 dialog{border:1px solid var(--line);border-radius:12px;background:var(--panel);color:var(--ink);
   max-width:760px;width:92%}
 dialog::backdrop{background:rgba(0,0,0,.5)}
</style></head><body><div class="wrap">
<h1>NE<span>U</span>RON — identity review</h1>
<div class="sub">Every account that has signed in. Ban here and the block takes effect at
<code>/infer</code> — server-side, so it holds even against a modified client.</div>
<div class="bar">
  <input id="key" type="password" placeholder="operator key (X-Wallet-Link-Secret)">
  <button id="load">Load</button>
  <button id="toggle" class="ghost">Show banned only</button>
  <span id="msg" class="muted"></span>
</div>
<div class="tablewrap"><table>
<thead><tr><th>identity</th><th>provider</th><th>email verified</th><th>violations</th>
<th>requests</th><th>balance</th><th>last seen</th><th>status</th><th></th></tr></thead>
<tbody id="rows"><tr><td colspan="9" class="empty">Enter your operator key and press Load.</td></tr></tbody>
</table></div>
<h1 style="margin-top:2.5rem">Sybil signals</h1>
<div class="sub">Weak by design and <b>nothing is blocked on any of it</b>. The hardware
signature is only CPU count, RAM and OS, so two identical laptops collide and a VM can report
whatever it likes — expect false positives and treat these as "worth a look", not as proof.
Real Sybil resistance arrives when NRN is worth faking for.</div>
<div class="tablewrap"><table>
<thead><tr><th>when</th><th>kind</th><th>subject</th><th>node</th><th>detail</th></tr></thead>
<tbody id="flagrows"><tr><td colspan="5" class="empty">Load to see flags.</td></tr></tbody>
</table></div>
<dialog id="detail"><div style="padding:1.2rem"><h3 id="dtitle" style="margin:0 0 .6rem"></h3>
<div id="dbody" class="mono"></div>
<div style="margin-top:1rem;text-align:right"><button class="ghost" id="dclose">Close</button></div>
</div></dialog>
</div>
<script>
const $=s=>document.querySelector(s);
let bannedOnly=false;
const key=()=>$("#key").value.trim()||sessionStorage.getItem("neuronAdminKey")||"";
const hdr=()=>({"X-Wallet-Link-Secret":key()});
const when=t=>t?new Date(t*1000).toLocaleString():"—";
const esc=s=>String(s==null?"":s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",
  '"':"&quot;","'":"&#39;"}[c]));

async function load(){
  const k=key();
  if(!k){$("#msg").textContent="operator key required";return;}
  sessionStorage.setItem("neuronAdminKey",k);
  $("#msg").textContent="loading…";
  let r;
  try{ r=await fetch("/admin/identities?banned_only="+bannedOnly,{headers:hdr()}); }
  catch(e){ $("#msg").textContent="network error"; return; }
  if(r.status===401){$("#msg").textContent="wrong operator key";return;}
  if(!r.ok){$("#msg").textContent="error "+r.status;return;}
  const {identities}=await r.json();
  $("#msg").textContent=identities.length+" identit"+(identities.length===1?"y":"ies");
  $("#rows").innerHTML = identities.length? identities.map(i=>{
    const banned=!!i.moderation_banned;
    return `<tr>
      <td>${esc(i.email)||'<span class="muted">no email</span>'}
          <div class="mono muted">${esc(i.wallet_id)}</div></td>
      <td>${esc(i.provider)}</td>
      <td>${i.email_verified?'<span class="pill" style="background:var(--ok)">verified</span>'
                            :'<span class="pill" style="background:#9aa0a6">no</span>'}</td>
      <td>${i.violation_count}</td><td>${i.request_count}</td>
      <td>${Number(i.balance).toFixed(2)}</td><td>${when(i.last_seen)}</td>
      <td>${banned?'<span class="pill" style="background:var(--danger)">banned</span>'
                  :'<span class="pill" style="background:var(--ok)">active</span>'}</td>
      <td><button class="ghost act" data-w="${esc(i.wallet_id)}">View</button>
          <button class="${banned?'ok':'danger'} ban" data-w="${esc(i.wallet_id)}"
                  data-b="${banned?1:0}">${banned?'Unban':'Ban'}</button></td></tr>`;
  }).join("") : '<tr><td colspan="9" class="empty">No identities yet.</td></tr>';
  loadFlags();
}

async function loadFlags(){
  let r;
  try{ r=await fetch("/admin/sybil-flags",{headers:hdr()}); }catch(e){ return; }
  if(!r.ok) return;
  const {flags}=await r.json();
  $("#flagrows").innerHTML = flags.length? flags.map(f=>`<tr>
      <td>${when(f.created_at)}</td>
      <td><span class="pill" style="background:#f9ab00">${esc(f.kind)}</span></td>
      <td class="mono">${esc(f.subject)}</td>
      <td class="mono">${esc(f.node_id)||'—'}</td>
      <td class="muted">${esc(f.detail)}</td></tr>`).join("")
    : '<tr><td colspan="5" class="empty">No signals — nothing has looked duplicated yet.</td></tr>';
}

document.addEventListener("click",async e=>{
  const b=e.target.closest("button"); if(!b) return;
  if(b.id==="load"){ load(); return; }
  if(b.id==="toggle"){ bannedOnly=!bannedOnly;
    b.textContent=bannedOnly?"Show all":"Show banned only"; load(); return; }
  if(b.id==="dclose"){ $("#detail").close(); return; }
  const w=b.dataset.w; if(!w) return;
  if(b.classList.contains("ban")){
    const isBanned=b.dataset.b==="1";
    if(!confirm((isBanned?"Unban":"Ban")+" this identity?\\n\\n"+w)) return;
    const r=await fetch("/wallet/"+encodeURIComponent(w)+(isBanned?"/unban":"/ban"),
                        {method:"POST",headers:hdr()});
    $("#msg").textContent = r.ok ? (isBanned?"unbanned":"banned") : "failed ("+r.status+")";
    load(); return;
  }
  if(b.classList.contains("act")){
    const r=await fetch("/wallet/"+encodeURIComponent(w)+"/activity",{headers:hdr()});
    if(!r.ok){ $("#msg").textContent="activity failed ("+r.status+")"; return; }
    const d=await r.json();
    $("#dtitle").textContent=(d.identity.email||"(no email)")+" — "+d.identity.provider;
    const ev=d.moderation_events.length? d.moderation_events.map(e=>
        "· "+when(e.created_at)+"  ["+esc(e.direction)+"] "+esc(e.category)).join("<br>")
      : '<span class="muted">no moderation events</span>';
    const rq=d.requests.length? d.requests.slice(0,25).map(q=>
        "· "+when(q.created_at)+"  "+esc(q.status)+"  prompt_len="+q.prompt_len+
        "  tokens="+(q.tokens_generated==null?"—":q.tokens_generated)).join("<br>")
      : '<span class="muted">no requests</span>';
    $("#dbody").innerHTML="<b>wallet</b><br>"+esc(d.identity.wallet_id)+
      "<br><br><b>joined</b> "+when(d.identity.created_at)+
      " &nbsp; <b>last seen</b> "+when(d.identity.last_seen)+
      "<br><br><b>moderation events</b><br>"+ev+
      "<br><br><b>recent requests</b> <span class='muted'>(length only — never prompt text)</span><br>"+rq;
    $("#detail").showModal();
  }
});
$("#key").addEventListener("keydown",e=>{ if(e.key==="Enter") load(); });
if(sessionStorage.getItem("neuronAdminKey")){ $("#key").value=sessionStorage.getItem("neuronAdminKey"); load(); }
</script></body></html>"""


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    network, nodes = _network_summary()
    stats = models.network_stats()
    # Privacy: per-node balances are NOT shown here. The public dashboard is network
    # health only; each node sees its own earnings at /node/{id}/dashboard (token-gated).

    healthy = network["network_healthy"]
    # Two different failures, and saying "incomplete" for both sent an operator looking for a
    # missing node when every layer was present. The unroutable case is the one that reads as
    # fine everywhere else, so it is the one that has to name itself. PROBLEMS.md [P32].
    if healthy:
        banner_text = "HEALTHY — every layer has a node, and the chain routes"
    elif network["uncovered_layers"]:
        banner_text = "DEGRADED — chain incomplete, no request can complete"
    else:
        banner_text = (f"DEGRADED — every layer is covered but the chain walks to "
                       f"{network['stages']} stage(s); a driver needs "
                       f"{config.MIN_PIPELINE_STAGES}-{config.PIPELINE_STAGES}")

    rows = ""
    for n in nodes:
        st = n.get("standing", "trusted")
        gpu = "<span class='tick'>✓</span>" if n.get("has_gpu") else "<span class='dash'>—</span>"
        # Ms/layer is what the balancer solves the split from ([P7]); showing it makes an
        # unbalanced network legible instead of something only /network/plan knows about.
        # An EXPIRED figure is not a current one. Printing the raw column here is how 4146.6
        # was read as live evidence about a node routing had already stopped believing
        # ([P34]) -- the freshness rule lives in `router`, so this cannot drift from it again.
        # An expired figure reads as UNKNOWN here, not as itself-with-a-caveat. Routing already
        # scores this node at the default prior, so "—" is the more accurate public answer -- and
        # a stale `4146.6` printed beside peers at 8-20 still says "that volunteer's machine is
        # terrible" however it is labelled, which is the same public verdict the standing column
        # was removed for. The raw figure is kept for the operator's own page, where it is
        # evidence about a measurement rather than a judgement shown to everyone else.
        ms = router.ms_per_layer_fresh(n)
        speed = (f"{ms:.1f}" if ms is not None else
                 "<span class='dash' title='no current measurement — the node re-measures "
                 "hourly; until then it is scored at the default prior'>—</span>")
        # STANDING AND REPUTATION ARE NOT PUBLISHED PER NODE. They are the same class of fact as
        # the balances above -- personal to one operator -- and a harsher one: `flagged · 4%` is a
        # public verdict on an identifiable volunteer's machine, readable by everyone else on the
        # network. It is also the fact most likely to be WRONG, because a flag can be produced by
        # the coordinator's own bookkeeping rather than by the node ([P37]: three honest machines,
        # and 2026-08-11's `1/23` earned entirely on a range this coordinator had moved).
        #
        # Nothing is hidden that a visitor needs: the capacity line above already reads "across N
        # eligible node(s)", the coverage strip shows whether any layer is unbacked, and the
        # excluded count is stated in aggregate below. What goes is the pillory -- one named node
        # carrying a number the network cannot yet justify. The operator sees the full detail,
        # with the reason and the remedy, on their own token-gated page.
        rows += (
            f"<tr>"
            f"<td class='id'>{n['node_id']}</td>"
            f"<td class='nw'>{n['layer_start']}–{n['layer_end']}</td>"
            f"<td>{theme.pill(n['status'])}</td>"
            f"<td>{n.get('cores', '-')}</td>"
            f"<td>{n.get('ram_gb', '-')}</td>"
            f"<td>{gpu}</td>"
            f"<td>{speed}</td>"
            f"<td style='color:#6b7280'>{_ago(n.get('last_seen'))}</td>"
            f"</tr>"
        )

    # Coverage strip: which layers actually have an eligible node behind them. The number
    # alone ("21/28") never said where the hole was, so the one question it raises -- where do
    # I put a node -- could only be answered by reading /node/list by hand.
    gaps = {i for lo, hi in network["uncovered_layers"] for i in range(lo, hi + 1)}
    cells = "".join(f"<i class='gap' title='layer {i}: no node'>{i}</i>" if i in gaps
                    else f"<i title='layer {i}: covered'>{i}</i>"
                    for i in range(network["total_layers"]))
    if gaps:
        missing = ", ".join(f"{lo}–{hi}" if lo != hi else f"{lo}"
                            for lo, hi in network["uncovered_layers"])
        cov_key = (f"<span style='color:#b91c1c;font-weight:600'>Missing: layers {missing}</span>"
                   f" — a node placed here makes the network able to serve again. "
                   f"Joining nodes are auto-placed into the gap.")
    else:
        cov_key = ("Every layer of the serving model has at least one eligible node behind it. "
                   "Layers with more than one node are served by whichever replica is free.")

    # Aggregate capacity, from the nodes that can actually take traffic. Per-node hardware is
    # already in the table; the sum is what tells a visitor whether this is a demo or a network.
    usable = [n for n in nodes if n["status"] == "online" and n.get("eligible")]
    cap = (f"{sum(n.get('cores') or 0 for n in usable)} cores &middot; "
           f"{sum(n.get('ram_gb') or 0 for n in usable)} GB RAM &middot; "
           f"{sum(1 for n in usable if n.get('has_gpu'))} GPU(s) "
           f"across {len(usable)} eligible node(s)")
    waiting = network["probationary_nodes"]
    waiting_line = ""
    if waiting:
        # [P24]: a node can be online, healthy and excluded from every chain. That fact only
        # existed in the standing column of one table row; here it is stated.
        waiting_line = (
            f"<p class='callout'><b>{waiting} node(s) awaiting verification</b> — they are "
            f"online but serve no requests and earn no NRN until proof-of-compute confirms "
            f"them. Verification is automatic; a node stuck here for hours means the "
            f"network's verifiers are not running.</p>")

    # The excluded count, in AGGREGATE and without naming anyone. A visitor's real question is
    # "does this network work", and the honest answer is that it routes around a node that is not
    # serving -- which is the design doing its job, not a warning. Naming the machine answers a
    # question nobody asked and reads as an accusation against a volunteer who, on the evidence
    # of [P37], may well be innocent.
    # ROLLOUT VISIBILITY, in aggregate. Per-node versions stay off the public page for the same
    # reason standing does -- but "is this network patched" is a fair question for a visitor, and
    # until 0.20.2 nobody could answer it at all, including the operator. `unknown` is its own
    # bucket rather than being folded into "old": an agent too old to report its version is not
    # the same as one known to be behind, and merging them would invent certainty.
    online_nodes = [n for n in nodes if n["status"] == "online"]
    on_latest = sum(1 for n in online_nodes if n.get("agent_version") == config.AGENT_VERSION)
    unknown_ver = sum(1 for n in online_nodes if not n.get("agent_version"))
    version_line = ""
    if online_nodes:
        parts = [f"<b>{on_latest} of {len(online_nodes)}</b> online node(s) on the latest agent "
                 f"(v{config.AGENT_VERSION})"]
        if unknown_ver:
            parts.append(f"{unknown_ver} running a build too old to report its version")
        stuck = [n for n in online_nodes
                 if n.get("auto_update") == 0 and n.get("agent_version") != config.AGENT_VERSION]
        if stuck:
            parts.append(f"{len(stuck)} with auto-update switched off, so they will not move on "
                         f"their own")
        version_line = (f"<p class='callout'>{' · '.join(parts)}. Nodes check once a day and "
                        f"never mid-request, so a rollout takes up to 24 hours.</p>")

    excluded = network["flagged_nodes"]
    excluded_line = ""
    if excluded:
        excluded_line = (
            f"<p class='callout'><b>{excluded} node(s) excluded from routing</b> — the network "
            f"is serving around them and no request is sent their way. A node lands here after "
            f"repeated failed checks, which is usually a half-downloaded slice or a placement "
            f"the coordinator and the node disagree about. The operator sees the reason and the "
            f"fix on their own dashboard.</p>")

    # Model tier (auto-model-tiering): the biggest model this network can back, plus the
    # ladder and the "grow to unlock the next model" prompt. Read-only here (now=None) —
    # the health loop is what advances the hysteresis over time.
    tier = model_tiers.snapshot(nodes, _tier_controller)
    serv = serving_model()
    serv_name = next((t["name"] for t in tier["tiers"] if t["model_id"] == serv["model_id"]),
                     serv["model_id"])
    # serving = what nodes run now; ready = capacity qualifies but not yet migrated; locked = not
    # enough; won't fit = the network HAS the RAM but not on any single machine, so no per-node
    # split of that model exists and the migration will (correctly) never start. Calling that
    # "ready" was the dashboard promising an upgrade that could not happen.
    ladder = ""
    for t in tier["tiers"]:
        tstate = ("serving" if t["name"] == serv_name
                  else "won't fit" if t["feasible"] and not t["placeable"]
                  else "ready" if t["feasible"] else "locked")
        ladder += (
            f"<tr><td><b>{t['name']}</b></td><td class='id'>{t['model_id']}</td>"
            f"<td>{t['min_nodes']} nodes · {t['min_ram_gb']:.0f} GB</td>"
            f"<td>{theme.pill(tstate)}</td></tr>"
        )
    gap = tier["next_tier"]
    gap_line = ""
    if gap and (gap["need_nodes"] > 0 or gap["need_ram_gb"] > 0):
        need = f"+{gap['need_nodes']} node(s)"
        if gap["need_ram_gb"] > 0:
            need += f" and +{gap['need_ram_gb']:.0f} GB RAM"
        gap_line = (f"<p class='callout'>Grow the network by "
                    f"<b>{need}</b> and it auto-upgrades to the <b>{gap['name']}</b> model.</p>")

    # Migration status: was invisible before (only the raw /network/migration JSON showed it,
    # so a stuck/slow migration had no operator-facing signal at all — post-audit fix).
    with _migration_lock:
        mstatus = _migration.status()
    migration_line = ""
    if mstatus["phase"] == "preparing":
        migration_line = (
            f"<p class='callout'>Migrating to "
            f"<b>{mstatus['target']['model_id']}</b> — "
            f"{mstatus['ready_count']}/{mstatus['plan_size']} node(s) ready "
            f"(still serving <b>{serv_name}</b> until cutover).</p>")
    elif mstatus["blocked"]:
        # A qualified upgrade that is not happening needs a reason on the page, or the network
        # looks stuck. It isn't stuck — it is declining to OOM-kill the machines it runs on.
        b = mstatus["blocked"]
        migration_line = (
            f"<p class='callout'>Not upgrading to <b>{b['model_id']}</b>: the network has the "
            f"total RAM, but no single node can hold its share — {b['capacity_shortfall']} of "
            f"{b['layers']} layer(s) have nowhere to live. Bigger machines (not just more of "
            f"them) unlock it. Still serving <b>{serv_name}</b>.</p>")

    body = f"""
<h1>Live network</h1>
<div class="sub">Network of Existing Utilised Resources — open nodes ·
  serving <b>{serv['model_id']}</b> · <span title="the agent release this network offers for
  download. Nodes update on their own daily check, so some may still be on an older build —
  the coordinator is not told which version a node runs.">latest agent v{config.AGENT_VERSION}</span>
   · auto-refresh 5s</div>
{theme.banner(healthy, banner_text)}
<div class="stats">
  <div class="stat"><div class="n">{serv_name}</div><div class="l">serving now</div></div>
  <div class="stat"><div class="n">{network['eligible_nodes']}/{network['online_nodes']}</div>
    <div class="l">nodes serving / online</div></div>
  <div class="stat"><div class="n">{network['total_layers_covered']}/{network['total_layers']}</div>
    <div class="l">layers covered</div></div>
  <div class="stat"><div class="n">{stats['total_requests_served']}</div>
    <div class="l">requests served</div></div>
  <div class="stat"><div class="n">{round(stats['total_nrn_distributed'], 2)}</div>
    <div class="l">NRN distributed</div></div>
</div>

<h2>Layer coverage — every layer needs a node</h2>
<div class="panel" style="padding:1.1rem">
  <div class="cov">{cells}</div>
  <div class="cov-key">{cov_key}</div>
</div>

<h2>Model tier — scales with the network</h2>
<div class="panel"><div class="table-wrap"><table>
  <tr><th>tier</th><th>model</th><th>needs</th><th>state</th></tr>
  {ladder}
</table></div></div>
{gap_line}
{migration_line}

<h2>Nodes — {cap}</h2>
<div class="panel"><div class="table-wrap"><table>
  <tr><th>node</th><th>layers</th><th>status</th><th>cores</th>
      <th>RAM GB</th><th>GPU</th><th>ms/layer</th><th>last seen</th></tr>
  {rows}
</table></div></div>
{waiting_line}
{excluded_line}
{version_line}

<div class="note">
  <strong>What is not on this page.</strong> Earnings, node addresses, and each node's standing
  and proof-of-compute record are private — a machine's verification history is a judgement about
  one volunteer's computer, and it belongs to them, not to everyone else on the network. Each
  operator sees their own numbers in the NEURON app (tray &rarr; My Dashboard), authenticated
  with that node's own token. {stats['total_tokens_generated']:,} tokens have been generated
  across {stats['total_requests_served']} request(s). NRN has no cash value.
</div>"""
    return theme.page("NEURON — live network", body)


# --------------------------------------------------------------------------- #
# Per-node private dashboard — a node operator's own numbers, token-gated
# --------------------------------------------------------------------------- #
@app.get("/node/{node_id}/dashboard", response_class=HTMLResponse)
def node_dashboard(node_id: str, token: str = None,
                   x_node_token: str = Header(default=None)):
    """The node's OWN view: balance, total earned, spent, requests served, standing,
    reputation. Auth = that node's token (query `?token=` for the browser link the tray
    opens, or the X-Node-Token header). Nobody else's earnings are visible anywhere."""
    node = _require_own_token(node_id, token or x_node_token)
    led = models.get_ledger(node_id) or {"balance": 0, "total_earned": 0, "requests_served": 0}
    network, _ = _network_summary()
    spent = round(led["total_earned"] - led["balance"], 4)   # real once wallet debits land (§11)
    st = node.get("standing", "trusted")
    rep = node.get("reputation")
    bound = models.get_payout_address(node_id)
    payout_html = (
        f"<code>{bound['payout_address']}</code> "
        f"<span style='color:#6b7280'>(where your NRN goes if the ledger moves on-chain)</span>"
        if bound else
        "<span style='color:#6b7280'>not set — your NRN has nowhere to go on-chain. "
        "The agent binds one automatically; see INSTALL.md.</span>")
    gpu_row = ""
    if node.get("has_gpu"):
        vram = node.get("gpu_vram_gb")
        gpu_row = (f"<tr><td class='key'>GPU</td><td>{node.get('gpu_name') or 'detected'}"
                   f"{f' · {vram} GB VRAM' if vram else ''} "
                   f"<span style='color:#6b7280'>(reported as capacity; visible only to "
                   f"you)</span></td></tr>")
    # Deliberately blank rather than stale on the operator's OWN page: a figure taken while the
    # machine thrashed, kept past its TTL, tells an honest volunteer their PC is 500x slower
    # than its peers -- an accusation the coordinator itself has already stopped believing.
    ms = router.ms_per_layer_fresh(node)
    # THE OPERATOR'S OWN PAGE KEEPS THE DETAIL THE PUBLIC ONE AGGREGATES. This is the person who
    # can actually act on it: if their node is behind, they are the only one who can switch
    # auto-update back on or run the installer by hand.
    av = node.get("agent_version")
    latest = config.AGENT_VERSION
    if not av:
        ver_txt = ("<span class='dash'>not reported</span> <span style='color:#6b7280'>"
                   "(a build older than 0.20.2 does not say)</span>")
    elif av == latest:
        ver_txt = f"v{av} <span style='color:#6b7280'>(current)</span>"
    else:
        ver_txt = (f"v{av} <span style='color:#b45309'>— v{latest} is available</span>")
    bits = [ver_txt]
    if node.get("auto_update") == 0:
        bits.append("<span style='color:#b45309'>auto-update is OFF, so this node will not "
                    "update itself</span>")
    elif av and av != latest:
        bits.append("nodes check once a day and never mid-request")
    chk, chk_at = node.get("update_check"), node.get("update_checked_at")
    if chk and chk not in ("current",):
        bits.append(f"last check: <b>{chk}</b>{f' ({_ago(chk_at)})' if chk_at else ''}")
    version_row = f"<tr><td class=\"key\">agent</td><td>{' · '.join(bits)}</td></tr>"
    # What this node's standing actually MEANS for it, rather than a bare word. A probationary
    # operator's real question is "why is my balance not moving?" ([P24]).
    if st == "probationary":
        standing_note = ("<p class='callout'>Your node is <b>online but not yet serving</b>. "
                         "It answers verification challenges only, and earns nothing, until "
                         "proof-of-compute confirms it — normally within minutes. Nothing on "
                         "your machine needs changing.</p>")
    elif st == "flagged":
        standing_note = ("<p class='callout'>Your node has <b>failed proof-of-compute</b> often "
                         "enough to be excluded from routing. A wrong answer is usually a "
                         "half-downloaded slice: stop the agent, delete the slice directory, "
                         "and let it re-download.</p>")
    else:
        standing_note = ""

    body = f"""
<h1 style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:20px">{node_id}</h1>
<div class="sub">your node's private dashboard · auto-refresh 5s · {theme.pill(st)}</div>
<div class="stats">
  <div class="stat"><div class="n">{round(led['balance'], 3)}</div>
    <div class="l">NRN balance</div></div>
  <div class="stat"><div class="n">{round(led['total_earned'], 3)}</div>
    <div class="l">total earned</div></div>
  <div class="stat"><div class="n">{spent}</div><div class="l">spent (usage)</div></div>
  <div class="stat"><div class="n">{led['requests_served']}</div>
    <div class="l">requests served</div></div>
</div>
{standing_note}

<h2>This node</h2>
<div class="panel"><div class="table-wrap"><table>
  <tr><td class="key">status</td><td>{theme.pill(node['status'])}
      <span style="color:#6b7280">last seen {_ago(node.get('last_seen'))}</span></td></tr>
  <tr><td class="key">layers served</td><td>{node['layer_start']}–{node['layer_end']}
      of {network['total_layers']}</td></tr>
  <tr><td class="key">your endpoint</td><td><code>{node['tailscale_ip']}:{node['port']}</code>
      <span style="color:#6b7280">(how the network reaches you — private to this page)</span></td></tr>
  <tr><td class="key">hardware</td><td>{node.get('cores', '-')} cores ·
      {node.get('ram_gb', '-')} GB RAM{f" · {ms:.1f} ms/layer measured" if isinstance(ms, (int, float)) else ""}</td></tr>
  {gpu_row}
  {version_row}
  <tr><td class="key">proof-of-compute</td><td>{f'{rep:.0%}' if rep is not None else 'no challenges yet'}
      (passed {node.get('challenges_passed', 0)} / failed {node.get('challenges_failed', 0)})</td></tr>
  <tr><td class="key">payout address</td><td>{payout_html}</td></tr>
</table></div></div>

<h2>The network you are part of</h2>
<div class="panel"><div class="table-wrap"><table>
  <tr><td class="key">nodes</td><td>{network['eligible_nodes']} serving ·
      {network['online_nodes']} online · {network['total_nodes']} registered</td></tr>
  <tr><td class="key">coverage</td><td>{network['total_layers_covered']}/{network['total_layers']}
      layers{'' if network['network_healthy'] else (' — the chain is incomplete, so no request can complete right now' if network['uncovered_layers'] else f" — every layer is covered, but the chain walks to {network['stages']} stage(s) and a driver needs {config.MIN_PIPELINE_STAGES}-{config.PIPELINE_STAGES}, so no request can complete right now")}</td></tr>
  <tr><td class="key">chain</td><td>{network['stages']} stage(s) {network['chain_ranges']} ·
      {'routable' if network['routable'] else 'NOT routable'}</td></tr>
  <tr><td class="key">full picture</td><td><a href="/dashboard">the live network dashboard</a>
      (no earnings, no addresses)</td></tr>
</table></div></div>

<div class="note"><strong>Keep this URL private.</strong> It contains your node token, which is
what makes this page yours alone — anyone holding it can read these numbers. "Spent" becomes
live once wallet spending ships.</div>"""
    # nav_links=False + no_referrer: this URL carries the node's token, so no link on the page
    # may hand it to a third party through the Referer header.
    return theme.page(f"NEURON — {node_id}", body, nav_links=False, no_referrer=True)


# --------------------------------------------------------------------------- #
# Agent auto-update (Session 9)
# --------------------------------------------------------------------------- #
@app.get("/agent/version")
def agent_version():
    """What the auto-updater reads. `version` is kept as the first key for older agents that
    only look at that; the rest is what makes an unattended update safe to actually perform.

    sha256 empty means "do not install" — see config.AGENT_SHA256."""
    return {"version": config.AGENT_VERSION,
            "download_url": config.AGENT_DOWNLOAD_URL,
            "sha256": config.AGENT_SHA256,
            # `rollback` lets an operator move the fleet BACKWARDS onto an older build. Agents
            # before 0.20.1 ignore the field entirely, which is the correct behaviour for them:
            # they simply stay where they are rather than acting on something they cannot verify.
            "rollback": config.AGENT_ROLLBACK}


@app.get("/models")
def models_catalog():
    """Models the network can serve (Session 15). Today: the one default model."""
    return {"default": model_registry.DEFAULT_MODEL, "models": model_registry.list_models()}


@app.get("/")
def root():
    return {"service": "NEURON Coordinator", "docs": "/docs", "dashboard": "/dashboard"}
