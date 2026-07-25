"""NEURON coordinator — FastAPI app (the brain of the network).

Run:  uvicorn coordinator.main:app --reload --port 8000   (from C:\\Users\\optin\\neuron)

Registry + health + routing + ledger + dashboard. Node-management calls are
token-gated (Part 6): registration needs the shared X-Register-Secret; a node's
own ping/delete need its X-Node-Token.
"""
import asyncio
import json
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coordinator import balancer, config, ledger, model_registry, models, router


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
async def health_loop():
    while True:
        await asyncio.sleep(config.HEALTH_CHECK_INTERVAL_S)
        try:
            for node_id in models.sweep():
                print(f"[health] node '{node_id}' went OFFLINE "
                      f"(no ping in {config.HEARTBEAT_TIMEOUT_S}s)")
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
    node calls this before it has a token; it is read-only and rate-limited by the middleware."""
    return {"total_layers": config.TOTAL_LAYERS, **router.suggest_placement()}


@app.get("/node/list")
def node_list():
    nodes = [{k: v for k, v in n.items() if k != "node_token"} for n in models.list_nodes()]
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
        return sliceinfo.slice_info(config.MODEL_ID, node["layer_start"],
                                    node["layer_end"], config.TOTAL_LAYERS)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"could not read model header: {e}")


# --------------------------------------------------------------------------- #
# Part 3 — Request routing
# --------------------------------------------------------------------------- #
@app.post("/infer")
def infer(body: InferBody):
    chain, missing = router.build_chain()
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
    return balancer.plan(bnodes, config.TOTAL_LAYERS)


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
@app.get("/ledger/{node_id}")
def get_ledger(node_id: str):
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
    covered = set()
    for n in usable:
        covered.update(range(n["layer_start"], n["layer_end"] + 1))
    total_covered = len(covered & set(range(config.TOTAL_LAYERS)))
    return {
        "total_nodes": len(nodes),
        "online_nodes": len(online),
        "eligible_nodes": len(usable),
        "flagged_nodes": len(flagged),
        "probationary_nodes": len(probationary),
        "total_layers_covered": total_covered,
        "network_healthy": total_covered == config.TOTAL_LAYERS,
    }, nodes


@app.get("/status")
def status():
    network, _ = _network_summary()
    return {"network": network, "stats": models.network_stats()}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    network, nodes = _network_summary()
    stats = models.network_stats()
    ledgers = {l["node_id"]: l for l in models.node_ledgers()}

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
        led = ledgers.get(n["node_id"], {})
        rows += (
            f"<tr>"
            f"<td>{n['node_id']}</td>"
            f"<td>{n['layer_start']}–{n['layer_end']}</td>"
            f"<td><span style='color:#fff;background:{badge};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{n['status']}</span></td>"
            f"<td><span style='color:#fff;background:{sbadge};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{st}</span></td>"
            f"<td>{n['tailscale_ip']}:{n['port']}</td>"
            f"<td>{n.get('cores','-')}</td>"
            f"<td>{n.get('ram_gb','-')}</td>"
            f"<td>{round(led.get('balance',0),3)}</td>"
            f"<td>{led.get('requests_served',0)}</td>"
            f"</tr>"
        )

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
  <div class="card"><div class="n">{network['online_nodes']}/{network['total_nodes']}</div>
    <div class="l">nodes online</div></div>
  <div class="card"><div class="n">{network['total_layers_covered']}/{config.TOTAL_LAYERS}</div>
    <div class="l">layers covered</div></div>
  <div class="card"><div class="n">{stats['total_requests_served']}</div>
    <div class="l">requests served</div></div>
  <div class="card"><div class="n">{round(stats['total_nrn_distributed'],2)}</div>
    <div class="l">NRN distributed</div></div>
</div>
<table>
  <tr><th>node</th><th>layers</th><th>status</th><th>standing</th><th>address</th><th>cores</th>
      <th>RAM GB</th><th>NRN balance</th><th>served</th></tr>
  {rows}
</table>
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
