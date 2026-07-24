"""NEURON coordinator — FastAPI app (the brain of the network).

Run:  uvicorn coordinator.main:app --reload --port 8000   (from C:\\Users\\optin\\neuron)

Registry + health + routing + ledger + dashboard. Node-management calls are
token-gated (Part 6): registration needs the shared X-Register-Secret; a node's
own ping/delete need its X-Node-Token.
"""
import asyncio
import secrets
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from coordinator import config, ledger, models, router


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


class InferBody(BaseModel):
    prompt: str
    max_tokens: int = 200


class CompleteBody(BaseModel):
    tokens_generated: int
    duration_ms: int
    node_ids: list[str]


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def require_register_secret(x_register_secret: str = Header(default=None)):
    if x_register_secret != config.REGISTRATION_SECRET:
        raise HTTPException(status_code=401, detail="invalid or missing X-Register-Secret")


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


# --------------------------------------------------------------------------- #
# Part 1 — Node registry
# --------------------------------------------------------------------------- #
@app.post("/node/register")
def register(body: RegisterBody, _=Depends(require_register_secret)):
    token = secrets.token_hex(config.TOKEN_BYTES)
    models.register_node(body.node_id, body.tailscale_ip, body.port,
                         body.layer_start, body.layer_end, body.cores, body.ram_gb, token)
    return {
        "status": "registered",
        "assigned_layers": [body.layer_start, body.layer_end],
        "node_token": token,
    }


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
    models.create_request(request_id, body.prompt, body.max_tokens)
    return {"chain": router.chain_public(chain), "request_id": request_id}


@app.post("/infer/{request_id}/complete")
def complete(request_id: str, body: CompleteBody):
    req = models.get_request(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail=f"unknown request '{request_id}'")
    if req["status"] == "completed":
        raise HTTPException(status_code=409, detail="request already completed")
    models.complete_request(request_id, body.tokens_generated, body.duration_ms, body.node_ids)
    rewards = ledger.distribute(body.node_ids)
    return {"status": "completed", "request_id": request_id, "rewards": rewards}


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
    covered = set()
    for n in online:
        covered.update(range(n["layer_start"], n["layer_end"] + 1))
    total_covered = len(covered & set(range(config.TOTAL_LAYERS)))
    return {
        "total_nodes": len(nodes),
        "online_nodes": len(online),
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

    rows = ""
    for n in nodes:
        badge = "#137333" if n["status"] == "online" else "#c5221f"
        led = ledgers.get(n["node_id"], {})
        rows += (
            f"<tr>"
            f"<td>{n['node_id']}</td>"
            f"<td>{n['layer_start']}–{n['layer_end']}</td>"
            f"<td><span style='color:#fff;background:{badge};padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{n['status']}</span></td>"
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
  <tr><th>node</th><th>layers</th><th>status</th><th>address</th><th>cores</th>
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


@app.get("/")
def root():
    return {"service": "NEURON Coordinator", "docs": "/docs", "dashboard": "/dashboard"}
