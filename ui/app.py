"""
NEURON — Chat UI server  [Session 10, driver extracted in Session 11]

A web interface where anyone types a prompt and gets a response produced by the
NEURON node network — no local model on the user's side. This server IS node_a
(the driver): it holds the embed + layers 0..S1-1 + lm_head shard, asks the
coordinator for a live node chain, runs inference across the machines, and streams
each token back to the browser as it is produced.

The generation loop lives in `neuron_driver` (shared with the OpenAI-compatible API
in api/openai_compat.py). This server just wraps the driver's events as SSE for the
browser, serves the page, and exposes a /network status endpoint. It also MOUNTS the
OpenAI-compatible API at /v1/* (Session 11) so one process serves both.

Because it plays node_a's role it must run on the node_a machine (the Windows PC that
owns layers 0..S1-1). Reuses node_a.py / common.py unchanged.

Run (from C:\\Users\\optin\\neuron, node_a machine):
    .venv\\Scripts\\python.exe -m uvicorn ui.app:app --host 0.0.0.0 --port 8080
    then open http://localhost:8080  (API docs at http://localhost:8080/api-docs)

Env overrides:
    NEURON_COORDINATOR   coordinator base URL   (default http://150.230.22.250:8001)
    NEURON_S1            layers node_a owns = 0..S1-1  (default 10)
    NEURON_MAX_TOKENS    hard cap on tokens per response (default 512)
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from neuron_driver import DRIVER
from api.openai_compat import router as openai_router, docs_html
from rag import retriever as rag

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "http://150.230.22.250:8001").rstrip("/")
STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("neuron.ui")


@asynccontextmanager
async def lifespan(app: FastAPI):
    DRIVER.ensure_loaded()
    print(f"[ui] ready | coordinator={COORDINATOR} | chat at / , OpenAI API at /v1")
    yield


app = FastAPI(title="NEURON Chat", version="0.2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(openai_router)          # Session 11: /v1/* on the same server


class ChatBody(BaseModel):
    prompt: str
    max_tokens: int = 128
    use_rag: bool = False       # Session 15: retrieve current web context first


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "chat.html"))


@app.get("/api-docs", response_class=HTMLResponse)
def api_docs():
    return docs_html()


@app.get("/network")
def network():
    """Live node count + health, for the UI header. Talks to the coordinator
    server-side so the browser never needs to reach the coordinator directly."""
    try:
        st = requests.get(f"{COORDINATOR}/status", timeout=8).json()
        nodes = requests.get(f"{COORDINATOR}/node/list", timeout=8).json()["nodes"]
    except requests.RequestException as e:
        return {"reachable": False, "error": e.__class__.__name__, "coordinator": COORDINATOR}
    net = st["network"]
    return {
        "reachable": True,
        "coordinator": COORDINATOR,
        "online_nodes": net["online_nodes"],
        "total_nodes": net["total_nodes"],
        "layers_covered": net["total_layers_covered"],
        "healthy": net["network_healthy"],
        "requests_served": st["stats"]["total_requests_served"],
        "nrn_distributed": round(st["stats"]["total_nrn_distributed"], 3),
        "nodes": [
            {"node_id": n["node_id"], "layers": n["assigned_layers"],
             "status": n["status"], "cores": n.get("cores"), "ram_gb": n.get("ram_gb")}
            for n in nodes
        ],
    }


# --------------------------------------------------------------------------- #
# Chat — stream tokens produced by the node chain (SSE for the browser)
# --------------------------------------------------------------------------- #
def _drive(prompt: str, max_new: int, use_rag: bool = False):
    content = prompt
    if use_rag:
        content, sources = rag.retrieve_and_augment(prompt)
        yield sse("sources", {"sources": sources, "used": bool(sources)})
    input_ids = DRIVER.encode_chat([{"role": "user", "content": content}])
    for ev in DRIVER.stream(input_ids, max_new, COORDINATOR, prompt):
        if ev["type"] == "meta":
            yield sse("meta", {"request_id": ev["request_id"], "nodes": ev["nodes"],
                               "node_ids": ev["node_ids"], "cost_nrn": ev["cost_nrn"]})
        elif ev["type"] == "token":
            yield sse("token", {"text": ev["text"]})
        elif ev["type"] == "done":
            yield sse("done", {"tokens": ev["completion_tokens"],
                               "latency_ms": ev["latency_ms"], "tok_per_s": ev["tok_per_s"]})
        elif ev["type"] == "error":
            # a dropped/offline node mid-chain surfaces here — previously silent server-side,
            # so the founder would only learn about a real stranger's failed request if they
            # reported it themselves (post-audit fix: at least get it in the server log).
            log.warning("chat stream error for prompt %r: %s", prompt[:80], ev["detail"])
            yield sse("error", {"detail": ev["detail"]})


@app.post("/chat")
def chat(body: ChatBody):
    prompt = (body.prompt or "").strip()
    if not prompt:
        def _empty():
            yield sse("error", {"detail": "empty prompt"})
        return StreamingResponse(_empty(), media_type="text/event-stream")
    return StreamingResponse(
        _drive(prompt, body.max_tokens, body.use_rag),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )
