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

from coordinator import (auth, balancer, config, genesis, ledger, migration,
                         model_registry, model_tiers, models, payout, router)
import relay_auth


# --------------------------------------------------------------------------- #
# Request bodies
# --------------------------------------------------------------------------- #
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
    gpu_vram_gb: float | None = None    # total VRAM; lets the balancer size a slice against
                                        # the GPU's memory instead of only system RAM.
    gpu_name: str | None = None         # e.g. "NVIDIA GeForce RTX 4070". Operator-only in
                                        # /node/list — a card model is fingerprinting detail,
                                        # like `platform`.


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
_serving = {"model_id": config.MODEL_ID, "layers": config.TOTAL_LAYERS}
_serving_lock = threading.Lock()


def serving_model():
    with _serving_lock:
        return dict(_serving)


def set_serving_model(model_id, layers):
    """Point the network at a different model (used by migration, Build 3)."""
    with _serving_lock:
        _serving["model_id"] = model_id
        _serving["layers"] = int(layers)


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


async def health_loop():
    prune_due_at = 0.0
    while True:
        await asyncio.sleep(config.HEALTH_CHECK_INTERVAL_S)
        try:
            for node_id in models.sweep():
                print(f"[health] node '{node_id}' went OFFLINE "
                      f"(no ping in {config.HEARTBEAT_TIMEOUT_S}s)")
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
            # advance any model migration toward the qualified target tier
            target = {"model_id": tier["model_id"], "layers": tier["layers"]}
            with _migration_lock:
                # self-heal first: closes a coverage gap in the CURRENTLY serving model using
                # idle surplus nodes, only while no real tier migration is in flight (it's a
                # no-op the instant update() below starts preparing one).
                healing_before = _migration.heal_status()["healing"]
                _migration.self_heal(models.list_nodes(), serving_model(), time.time(),
                                     apply_gap_heal)
                healing_after = _migration.heal_status()["healing"]
                if healing_after and not healing_before:
                    print(f"[gap-heal] coverage gap detected, reassigning idle node(s) "
                          f"(serving={serving_model()['model_id']})")
                elif healing_before and not healing_after:
                    print(f"[gap-heal] coverage restored (serving={serving_model()['model_id']})")

                before = _migration.phase
                _migration.update(models.list_nodes(), target, serving_model(),
                                  time.time(), apply_migration_cutover)
                if _migration.phase != before:
                    print(f"[migration] {before} -> {_migration.phase} "
                          f"(serving={serving_model()['model_id']})")
        except Exception as e:  # never let the loop die
            print(f"[health] sweep error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.init_db()
    if genesis.seed_genesis():
        print("[coordinator] genesis buckets seeded (fixed-supply ledger, Phase 0)")
    genesis.verify_invariant()   # fail startup loudly rather than serve on a broken supply
    print(f"[coordinator] up | db={config.DB_PATH} | layers={config.TOTAL_LAYERS} | "
          f"timeout={config.HEARTBEAT_TIMEOUT_S}s")
    task = asyncio.create_task(health_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="NEURON Coordinator", version="0.1", lifespan=lifespan)
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
    fingerprint = models.register_node(
        body.node_id, tailscale_ip, port, body.layer_start,
        body.layer_end, body.cores, body.ram_gb, token,
        ms_per_layer=body.ms_per_layer, head_ms=body.head_ms, trusted=trusted,
        platform=body.platform, has_gpu=body.has_gpu, gpu_vram_gb=body.gpu_vram_gb,
        gpu_name=body.gpu_name)
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
    resp = {
        "status": "registered",
        "standing": standing,
        "assigned_layers": [body.layer_start, body.layer_end],
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
def node_placement():
    """Advise a joining node which slice to serve (zero-config open join, S20). No auth — a
    node calls this before it has a token; it is read-only and rate-limited by the middleware.
    Includes the serving model_id so the node downloads the right model's slice."""
    sm = serving_model()
    return {"total_layers": sm["layers"], "model_id": sm["model_id"],
            **router.suggest_placement(total=sm["layers"])}


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
def ping(node_id: str, _node=Depends(require_node_token)):
    """Heartbeat. Carries `coordinator_url` because this is the one call every live node makes
    continuously — it is how a change of address reaches the whole network within a heartbeat
    instead of never."""
    models.touch_node(node_id)
    return {"status": "alive", "node_id": node_id, "last_seen": time.time(),
            "coordinator_url": config.PUBLIC_URL}


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
    n = models.get_node(node_id)
    return {"node_id": node_id, "passed": body.passed, "reputation": n["reputation"],
            "flagged": n["flagged"], "challenges_passed": n["challenges_passed"],
            "challenges_failed": n["challenges_failed"]}


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
    nodes.sort(key=lambda n: (0 if (n.get("head_ms") or 0) > 0 else 1, n["layer_start"]))
    bnodes = [{"node_id": n["node_id"], "ms_per_layer": n["ms_per_layer"],
               "head_ms": n.get("head_ms") or 0.0,
               # GPU is a tie-break in the balancer, never a speed multiplier — a node's speed
               # is the ms_per_layer it actually measured. See coordinator/balancer.py.
               "has_gpu": bool(n.get("has_gpu")), "gpu_vram_gb": n.get("gpu_vram_gb")}
              for n in nodes]
    return balancer.plan(bnodes, serving_model()["layers"])


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
    sm_layers = serving_model()["layers"]
    covered = set()
    for n in usable:
        covered.update(range(n["layer_start"], n["layer_end"] + 1))
    total_covered = len(covered & set(range(sm_layers)))
    return {
        "total_nodes": len(nodes),
        "online_nodes": len(online),
        "eligible_nodes": len(usable),
        "flagged_nodes": len(flagged),
        "probationary_nodes": len(probationary),
        "total_layers_covered": total_covered,
        "total_layers": sm_layers,
        "network_healthy": total_covered == sm_layers,
    }, nodes


@app.get("/status")
def status():
    network, _ = _network_summary()
    return {"network": network, "stats": models.network_stats()}


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
    banner_color = "#137333" if healthy else "#c5221f"
    banner_text = "HEALTHY" if healthy else "DEGRADED — chain incomplete"

    standing_colors = {"trusted": "#137333", "verified": "#1a73e8",
                       "probationary": "#f9ab00", "flagged": "#c5221f"}
    rows = ""
    for n in nodes:
        badge = "#137333" if n["status"] == "online" else "#c5221f"
        st = n.get("standing", "trusted")
        sbadge = standing_colors.get(st, "#5f6368")
        rows += (
            f"<tr>"
            f"<td>{n['node_id']}</td>"
            f"<td>{n['layer_start']}–{n['layer_end']}</td>"
            f"<td><span style='color:#fff;background:{badge};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{n['status']}</span></td>"
            f"<td><span style='color:#fff;background:{sbadge};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{st}</span></td>"
            f"<td>{n.get('cores','-')}</td>"
            f"<td>{n.get('ram_gb','-')}</td>"
            f"</tr>"
        )

    # Model tier (auto-model-tiering): the biggest model this network can back, plus the
    # ladder and the "grow to unlock the next model" prompt. Read-only here (now=None) —
    # the health loop is what advances the hysteresis over time.
    tier = model_tiers.snapshot(nodes, _tier_controller)
    serv = serving_model()
    serv_name = next((t["name"] for t in tier["tiers"] if t["model_id"] == serv["model_id"]),
                     serv["model_id"])
    # serving = what nodes run now; ready = capacity qualifies but not yet migrated; locked = not enough.
    tstate_colors = {"serving": "#137333", "ready": "#1a73e8", "locked": "#9aa0a6"}
    ladder = ""
    for t in tier["tiers"]:
        tstate = ("serving" if t["name"] == serv_name
                  else "ready" if t["feasible"] else "locked")
        tc = tstate_colors[tstate]
        ladder += (
            f"<tr><td>{t['name']}</td><td style='font-size:13px'>{t['model_id']}</td>"
            f"<td>{t['min_nodes']} nodes · {t['min_ram_gb']:.0f} GB</td>"
            f"<td><span style='color:#fff;background:{tc};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{tstate}</span></td></tr>"
        )
    gap = tier["next_tier"]
    gap_line = ""
    if gap and (gap["need_nodes"] > 0 or gap["need_ram_gb"] > 0):
        need = f"+{gap['need_nodes']} node(s)"
        if gap["need_ram_gb"] > 0:
            need += f" and +{gap['need_ram_gb']:.0f} GB RAM"
        gap_line = (f"<p style='color:#5f6368;font-size:14px'>Grow the network by "
                    f"<b>{need}</b> and it auto-upgrades to the <b>{gap['name']}</b> model.</p>")

    # Migration status: was invisible before (only the raw /network/migration JSON showed it,
    # so a stuck/slow migration had no operator-facing signal at all — post-audit fix).
    with _migration_lock:
        mstatus = _migration.status()
    migration_line = ""
    if mstatus["phase"] == "preparing":
        migration_line = (
            f"<p style='color:#1a73e8;font-size:14px'>Migrating to "
            f"<b>{mstatus['target']['model_id']}</b> — "
            f"{mstatus['ready_count']}/{mstatus['plan_size']} node(s) ready "
            f"(still serving <b>{serv_name}</b> until cutover).</p>")

    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>NEURON Coordinator</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#202124}}
 h1{{margin:0 0 .25rem}} .sub{{color:#5f6368;margin-bottom:1.5rem}}
 .banner{{color:#fff;background:{banner_color};padding:.6rem 1rem;border-radius:8px;
   font-weight:600;display:inline-block;margin-bottom:1.25rem}}
 table{{border-collapse:collapse;width:100%;max-width:920px}}
 th,td{{border:1px solid #dadce0;padding:.5rem .7rem;text-align:left;font-size:14px}}
 th{{background:#f1f3f4}}
 .cards{{display:flex;gap:1rem;margin:1.25rem 0}}
 .card{{border:1px solid #dadce0;border-radius:8px;padding:.8rem 1.2rem;min-width:150px}}
 .card .n{{font-size:1.7rem;font-weight:700}} .card .l{{color:#5f6368;font-size:13px}}
</style></head><body>
<h1>NEURON Coordinator</h1>
<div class="sub">Network of Existing Utilised Resources — Open Nodes · auto-refresh 5s</div>
<div class="banner">{banner_text}</div>
<div class="cards">
  <div class="card" style="border-color:#137333"><div class="n">{serv_name}</div>
    <div class="l">serving now</div></div>
  <div class="card"><div class="n">{network['online_nodes']}/{network['total_nodes']}</div>
    <div class="l">nodes online</div></div>
  <div class="card"><div class="n">{network['total_layers_covered']}/{network['total_layers']}</div>
    <div class="l">layers covered</div></div>
  <div class="card"><div class="n">{stats['total_requests_served']}</div>
    <div class="l">requests served</div></div>
  <div class="card"><div class="n">{round(stats['total_nrn_distributed'],2)}</div>
    <div class="l">NRN distributed</div></div>
</div>
<h2 style="font-size:1.05rem;margin:1.5rem 0 .5rem">Model tier — scales with the network</h2>
<table style="max-width:920px">
  <tr><th>tier</th><th>model</th><th>needs</th><th>state</th></tr>
  {ladder}
</table>
{gap_line}
{migration_line}
<h2 style="font-size:1.05rem;margin:1.5rem 0 .5rem">Nodes</h2>
<table>
  <tr><th>node</th><th>layers</th><th>status</th><th>standing</th><th>cores</th>
      <th>RAM GB</th></tr>
  {rows}
</table>
<p style="color:#5f6368;font-size:13px;margin-top:1rem">
  Earnings and node addresses are private: each node operator sees their own numbers in the
  NEURON app (tray &rarr; My Dashboard) — authenticated with that node's own token.</p>
</body></html>"""


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
    st_color = {"trusted": "#137333", "verified": "#1a73e8",
                "probationary": "#f9ab00", "flagged": "#c5221f"}.get(st, "#5f6368")
    rep = node.get("reputation")
    bound = models.get_payout_address(node_id)
    payout_html = (
        f"<code>{bound['payout_address']}</code> "
        f"<span style='color:#5f6368'>(where your NRN goes if the ledger moves on-chain)</span>"
        if bound else
        "<span style='color:#5f6368'>not set — your NRN has nowhere to go on-chain. "
        "The agent binds one automatically; see INSTALL.md.</span>")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>NEURON — {node_id}</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#202124}}
 h1{{margin:0 0 .25rem}} .sub{{color:#5f6368;margin-bottom:1.5rem}}
 .badge{{color:#fff;background:{st_color};padding:2px 10px;border-radius:10px;font-size:13px}}
 .cards{{display:flex;gap:1rem;margin:1.25rem 0;flex-wrap:wrap}}
 .card{{border:1px solid #dadce0;border-radius:8px;padding:.8rem 1.2rem;min-width:150px}}
 .card .n{{font-size:1.7rem;font-weight:700}} .card .l{{color:#5f6368;font-size:13px}}
 table{{border-collapse:collapse;max-width:560px}}
 th,td{{border:1px solid #dadce0;padding:.45rem .7rem;text-align:left;font-size:14px}}
 th{{background:#f1f3f4;width:190px}}
 .note{{color:#5f6368;font-size:13px;margin-top:1.25rem}}
</style></head><body>
<h1>{node_id}</h1>
<div class="sub">your node's private dashboard · auto-refresh 5s ·
  <span class="badge">{st}</span></div>
<div class="cards">
  <div class="card"><div class="n">{round(led['balance'], 3)}</div><div class="l">NRN balance</div></div>
  <div class="card"><div class="n">{round(led['total_earned'], 3)}</div><div class="l">total earned</div></div>
  <div class="card"><div class="n">{spent}</div><div class="l">spent (usage)</div></div>
  <div class="card"><div class="n">{led['requests_served']}</div><div class="l">requests served</div></div>
</div>
<table>
  <tr><th>status</th><td>{node['status']}</td></tr>
  <tr><th>layers served</th><td>{node['layer_start']}–{node['layer_end']}</td></tr>
  <tr><th>your endpoint</th><td>{node['tailscale_ip']}:{node['port']}
      <span style="color:#5f6368">(how the network reaches you — private to this page)</span></td></tr>
  <tr><th>reputation</th><td>{rep if rep is not None else 'no challenges yet'}
      (passed {node.get('challenges_passed', 0)} / failed {node.get('challenges_failed', 0)})</td></tr>
  <tr><th>network</th><td>{network['online_nodes']} nodes online ·
      {network['total_layers_covered']}/{network['total_layers']} layers covered</td></tr>
  <tr><th>payout address</th><td>{payout_html}</td></tr>
</table>
<p class="note">Keep this URL private — it contains your node token, which is what makes
this page yours alone. "Spent" becomes live once wallet spending ships.</p>
</body></html>"""


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
            "sha256": config.AGENT_SHA256}


@app.get("/models")
def models_catalog():
    """Models the network can serve (Session 15). Today: the one default model."""
    return {"default": model_registry.DEFAULT_MODEL, "models": model_registry.list_models()}


@app.get("/")
def root():
    return {"service": "NEURON Coordinator", "docs": "/docs", "dashboard": "/dashboard"}
