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

from coordinator import (balancer, config, ledger, migration, model_registry, model_tiers,
                         models, router)


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


class InferBody(BaseModel):
    prompt: str
    max_tokens: int = 200


class CompleteBody(BaseModel):
    tokens_generated: int
    duration_ms: int
    node_ids: list[str]
    complete_token: str | None = None   # [P12] token issued by /infer; required to settle


class AttestBody(BaseModel):
    passed: bool
    max_err: float | None = None


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


async def health_loop():
    while True:
        await asyncio.sleep(config.HEALTH_CHECK_INTERVAL_S)
        try:
            for node_id in models.sweep():
                print(f"[health] node '{node_id}' went OFFLINE "
                      f"(no ping in {config.HEARTBEAT_TIMEOUT_S}s)")
            with _tier_lock:
                prev = _tier_controller.active()["name"]
                tier = _tier_controller.update(models.list_nodes(), time.time())
                if tier["name"] != prev:
                    print(f"[tier] network now qualifies for {prev} -> {tier['name']}")
            # advance any model migration toward the qualified target tier
            target = {"model_id": tier["model_id"], "layers": tier["layers"]}
            with _migration_lock:
                before = _migration.phase
                _migration.update(models.list_nodes(), target, serving_model(),
                                  time.time(), set_serving_model)
                if _migration.phase != before:
                    print(f"[migration] {before} -> {_migration.phase} "
                          f"(serving={serving_model()['model_id']})")
        except Exception as e:  # never let the loop die
            print(f"[health] sweep error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models.init_db()
    print(f"[coordinator] up | db={config.DB_PATH} | layers={config.TOTAL_LAYERS} | "
          f"timeout={config.HEARTBEAT_TIMEOUT_S}s")
    task = asyncio.create_task(health_loop())
    try:
        yield
    finally:
        task.cancel()


app = FastAPI(title="NEURON Coordinator", version="0.1", lifespan=lifespan)


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
def register(body: RegisterBody, x_register_secret: str = Header(default=None)):
    trusted = classify_registration(x_register_secret)
    # open join: a secret-less registration must not hijack an existing TRUSTED node id.
    if not trusted:
        existing = models.get_node(body.node_id)
        if existing and existing["trusted"]:
            raise HTTPException(
                status_code=409,
                detail=f"node id '{body.node_id}' is reserved by a trusted node; "
                       f"registering it requires the secret")
    token = secrets.token_hex(config.TOKEN_BYTES)
    tailscale_ip, port, relay_block = body.tailscale_ip, body.port, None
    if body.behind_nat and config.RELAY_ENABLED:
        relay_port = _assign_relay_port(body.node_id)
        tailscale_ip, port = config.RELAY_HOST, relay_port    # peers reach it via the relay
        relay_block = {"host": config.RELAY_HOST, "control_port": config.RELAY_CONTROL_PORT,
                       "data_port": config.RELAY_DATA_PORT, "public_port": relay_port}
    models.register_node(body.node_id, tailscale_ip, port, body.layer_start,
                         body.layer_end, body.cores, body.ram_gb, token,
                         ms_per_layer=body.ms_per_layer, head_ms=body.head_ms, trusted=trusted)
    resp = {
        "status": "registered",
        "standing": "trusted" if trusted else "probationary",
        "assigned_layers": [body.layer_start, body.layer_end],
        "node_token": token,
    }
    if not trusted:
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
    (the proof-of-compute verifier is the legitimate consumer). node_token never leaves."""
    show_addr = x_register_secret == config.REGISTRATION_SECRET
    hidden = {"node_token"} if show_addr else {"node_token", "tailscale_ip", "port"}
    nodes = [{k: v for k, v in n.items() if k not in hidden} for n in models.list_nodes()]
    return {"nodes": nodes}


@app.delete("/node/{node_id}")
def unregister(node_id: str, _node=Depends(require_node_token)):
    models.delete_node(node_id)
    return {"status": "unregistered", "node_id": node_id}


# --------------------------------------------------------------------------- #
# Part 2 — Health check
# --------------------------------------------------------------------------- #
@app.get("/node/{node_id}/ping")
def ping(node_id: str, _node=Depends(require_node_token)):
    models.touch_node(node_id)
    return {"status": "alive", "node_id": node_id, "last_seen": time.time()}


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
    chain, missing = router.build_chain(total=serving_model()["layers"])
    if missing:
        raise HTTPException(
            status_code=503,
            detail=f"incomplete chain - missing layers {router.missing_str(missing)}",
        )
    request_id = str(uuid.uuid4())
    plan_node_ids = [n["node_id"] for n in chain]          # the chain WE chose (incl. replica)
    complete_token = secrets.token_hex(config.TOKEN_BYTES)  # only the caller who got this may complete
    models.create_request(request_id, body.prompt, body.max_tokens, plan_node_ids, complete_token)
    return {"chain": router.chain_public(chain), "request_id": request_id,
            "complete_token": complete_token}


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
    models.complete_request(request_id, tokens, body.duration_ms, plan)
    rewards = ledger.distribute(plan)
    return {"status": "completed", "request_id": request_id, "rewards": rewards}


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
               "head_ms": n.get("head_ms") or 0.0} for n in nodes]
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
</table>
<p class="note">Keep this URL private — it contains your node token, which is what makes
this page yours alone. "Spent" becomes live once wallet spending ships.</p>
</body></html>"""


# --------------------------------------------------------------------------- #
# Agent auto-update (Session 9)
# --------------------------------------------------------------------------- #
@app.get("/agent/version")
def agent_version():
    return {"version": config.AGENT_VERSION}


@app.get("/models")
def models_catalog():
    """Models the network can serve (Session 15). Today: the one default model."""
    return {"default": model_registry.DEFAULT_MODEL, "models": model_registry.list_models()}


@app.get("/")
def root():
    return {"service": "NEURON Coordinator", "docs": "/docs", "dashboard": "/dashboard"}
