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
    NEURON_COORDINATOR   coordinator base URL   (default https://neuronnet.duckdns.org)
    NEURON_S1            layers node_a owns = 0..S1-1  (default 10)
    NEURON_MAX_TOKENS    hard cap on tokens per response (default 512)
"""
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

import common
from neuron_driver import DRIVER
from api.openai_compat import router as openai_router, docs_html
from engine import local_gguf
from rag import retriever as rag
from safety import moderation
from ui import conversations
from ui import oauth as oauth_module

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "https://neuronnet.duckdns.org").rstrip("/")
STATIC_DIR = Path(__file__).resolve().parent / "static"
log = logging.getLogger("neuron.ui")
SESSION_SECRET = os.environ.get("NEURON_SESSION_SECRET", "neuron-session-dev-secret")
# Force every request through the node chain instead of the local engine (see _drive).
FORCE_NETWORK = os.environ.get("NEURON_FORCE_NETWORK") == "1"
# Real multi-turn chat resends prior turns every request -- cap how many so a long-running
# conversation can't silently grow past the model's context length. Cheap safeguard, not full
# truncation logic (a future upgrade could summarize instead of just dropping the oldest).
MAX_HISTORY_MESSAGES = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline-driver shard ONLY if we actually need it. When this machine can run
    # the serving model itself, the driver role is never exercised, and forcing it here would
    # make a fresh install sit through a ~1.4 GB slice download before chat works -- on top of
    # the quantized weights it also needs. Exactly one of the two gets fetched. If the network
    # path is later required, _drive calls DRIVER.stream(), which loads on demand.
    if local_gguf.can_serve(common.MODEL_ID):
        log.info("local engine can serve %s — skipping the pipeline-driver shard load",
                 common.MODEL_ID)
    else:
        DRIVER.ensure_loaded()
    print(f"[ui] ready | coordinator={COORDINATOR} | chat at / , OpenAI API at /v1")
    yield


app = FastAPI(title="NEURON Chat", version="0.2", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(openai_router)          # Session 11: /v1/* on the same server
app.include_router(oauth_module.router)    # Workstream B: /auth/login|callback|me|logout


class ChatBody(BaseModel):
    prompt: str
    max_tokens: int = 128
    use_rag: bool = False       # Session 15: retrieve current web context first
    conversation_id: str | None = None   # None -> a new conversation is created server-side


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


@app.get("/wallet/balance")
def wallet_balance_proxy(request: Request):
    """Proxies the coordinator's GET /wallet/{id} for the logged-in session's wallet, so the
    browser never needs to reach the coordinator directly (same pattern as /network)."""
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return {"logged_in": False}
    try:
        r = requests.get(f"{COORDINATOR}/wallet/{wallet_id}", timeout=8)
        r.raise_for_status()
        data = r.json()
        return {"logged_in": True, "wallet_id": wallet_id, "email": request.session.get("email"),
               "balance": data["balance"], "total_earned": data["total_earned"]}
    except requests.RequestException as e:
        return {"logged_in": True, "wallet_id": wallet_id, "error": str(e)}


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
        # Whether this machine can answer on its own (engine/local_gguf.py). The page uses it
        # to decide if an incomplete chain is actually a problem for THIS user: if we can serve
        # locally, a short-staffed network is not a broken chat.
        "local_capable": local_gguf.available(common.MODEL_ID),
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
def _drive(prompt: str, max_new: int, wallet_id: str, use_rag: bool = False,
          conversation_id: str | None = None):
    # Input moderation gate (Workstream A) — checked on the RAW user prompt, before RAG
    # augmentation and before anything is dispatched to the node chain. This is the driver
    # (the only place plaintext exists in NEURON), so this is the correct — and only —
    # place a check belongs; compute nodes never see text at all. See safety/moderation.py.
    verdict = moderation.check_text(prompt)
    if verdict.blocked:
        pre_id = f"blocked-{uuid.uuid4().hex[:12]}"
        moderation.log_event("in", verdict.category, pre_id, snippet=prompt)
        moderation.report_violation(COORDINATOR, wallet_id, "in", verdict.category)
        yield sse("error", {"detail": "This request was blocked by NEURON's acceptable-use "
                                      "policy (see SAFETY.md).", "code": "content_policy_violation"})
        return

    # Real multi-turn memory (Workstream: Chat UI redesign) -- load prior turns from the
    # driver-side conversation store (ui/conversations.py) BEFORE augmenting/dispatching the
    # new one, so the model actually sees conversation history instead of treating every
    # message as independent. A missing/foreign conversation_id (bad id, wrong wallet) is
    # treated the same as "no conversation" -- starts a fresh one rather than erroring, since
    # the browser can't always know if its cached id is still valid.
    prior_messages = []
    if conversation_id:
        existing = conversations.get_conversation(conversation_id, wallet_id)
        if existing is not None:
            prior_messages = [{"role": m["role"], "content": m["content"]}
                              for m in existing["messages"][-MAX_HISTORY_MESSAGES:]]
        else:
            conversation_id = None
    if conversation_id is None:
        conversation_id = conversations.create_conversation(wallet_id, title=prompt[:40])

    content = prompt
    if use_rag:
        content, sources = rag.retrieve_and_augment(prompt)
        yield sse("sources", {"sources": sources, "used": bool(sources)})
    messages = prior_messages + [{"role": "user", "content": content}]

    # Tiered execution (see engine/local_gguf.py). If this machine can hold the serving model
    # itself, run it here: measured 36 ms/token quantized vs 240 ms/token fp32 across the node
    # pipeline, so a full answer takes ~10s instead of ~40 minutes -- and a 1.5B model split
    # across three PCs was only ever buying a network hop and a bottleneck stage. The pipeline
    # is for models this machine CANNOT hold, which is the case only it can serve. Falls back
    # automatically, so an incomplete chain no longer means "responses will fail".
    # NEURON_FORCE_NETWORK=1 sends the request over the node chain even when this machine
    # could answer locally. Without it the distributed path is UNTESTABLE from the UI on any
    # machine capable of local execution -- which is every dev machine -- so verifying that a
    # chain routes, serves and settles NRN meant hand-running node_a.py with a wallet id
    # copied out of a browser session. Off by default; local-first is still the right tiering.
    if local_gguf.available(common.MODEL_ID) and not FORCE_NETWORK:
        events = local_gguf.stream(messages, max_new, common.MODEL_ID,
                                   coordinator=COORDINATOR, wallet_id=wallet_id)
    else:
        # Load the driver shard on demand. lifespan() skips it whenever the local engine
        # *could* serve, and its comment claimed "_drive calls DRIVER.stream(), which loads on
        # demand" -- but nothing loaded anything: encode_chat() runs BEFORE stream() and dereferences
        # self.tok, so taking this branch on a machine that skipped the startup load died with
        # `AttributeError: 'NoneType' object has no attribute 'apply_chat_template'`, surfacing
        # in the browser as a bare "connection lost: network error". Reachable without
        # NEURON_FORCE_NETWORK too: startup skips on can_serve() but this branch is chosen on
        # available(), which is can_serve() AND the weights already being on disk -- so any
        # install between "capable" and "downloaded" hit it.
        DRIVER.ensure_loaded()
        events = DRIVER.stream(DRIVER.encode_chat(messages), max_new, COORDINATOR,
                               prompt, wallet_id)

    full_text = ""
    for ev in events:
        if ev["type"] == "meta":
            yield sse("meta", {"request_id": ev["request_id"], "nodes": ev["nodes"],
                               "node_ids": ev["node_ids"], "cost_nrn": ev["cost_nrn"],
                               "conversation_id": conversation_id,
                               "local": ev.get("local", False)})
        elif ev["type"] == "token":
            yield sse("token", {"text": ev["text"]})
        elif ev["type"] == "done":
            full_text = ev.get("text", "")
            # Persist the real exchange only on a genuine completion -- never a blocked or
            # errored turn, mirroring "never bill a blocked generation." The user's ORIGINAL
            # prompt is stored (not the RAG-augmented version) so history reflects what they
            # actually typed.
            conversations.add_message(conversation_id, wallet_id, "user", prompt)
            conversations.add_message(conversation_id, wallet_id, "assistant", full_text)
            yield sse("done", {"tokens": ev["completion_tokens"],
                               "latency_ms": ev["latency_ms"], "tok_per_s": ev["tok_per_s"],
                               "cost_nrn": ev.get("cost_nrn")})
        elif ev["type"] == "error":
            # a dropped/offline node mid-chain surfaces here — previously silent server-side,
            # so the founder would only learn about a real stranger's failed request if they
            # reported it themselves (post-audit fix: at least get it in the server log).
            log.warning("chat stream error for prompt %r: %s", prompt[:80], ev["detail"])
            yield sse("error", {"detail": ev["detail"], "code": ev.get("code")})


@app.post("/chat")
def chat(body: ChatBody, request: Request):
    prompt = (body.prompt or "").strip()
    if not prompt:
        def _empty():
            yield sse("error", {"detail": "empty prompt"})
        return StreamingResponse(_empty(), media_type="text/event-stream")
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        # NRN is now a real fixed-supply ledger (Workstream B) -- /infer requires a wallet_id,
        # so chatting requires being logged in. Fail here with a clear, actionable message
        # rather than letting an anonymous request die deep in the driver with a 422/402.
        def _login_required():
            # Wording matters here: this is the first thing a new user sees, and the old text
            # ("NEURON spends NRN from your wallet to pay the nodes that serve you") is now
            # wrong for the common case -- a machine that runs the model itself pays nobody.
            # The honest reason is accountability: every request is tied to a real account so
            # abuse can be acted on (SAFETY.md). Takes one click; nothing to install or set up.
            yield sse("error", {"detail": "Sign in to start chatting. It takes one click, and "
                                          "it's how NEURON keeps the network accountable — "
                                          "answers on this machine are free.",
                                "code": "login_required"})
        return StreamingResponse(_login_required(), media_type="text/event-stream")
    return StreamingResponse(
        _drive(prompt, body.max_tokens, wallet_id, body.use_rag, body.conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


# --------------------------------------------------------------------------- #
# Conversation history (driver-side, per-wallet) — the ChatGPT-style sidebar
# --------------------------------------------------------------------------- #
@app.get("/conversations")
def list_conversations_endpoint(request: Request):
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return {"conversations": []}
    return {"conversations": conversations.list_conversations(wallet_id)}


@app.get("/conversations/{conversation_id}")
def get_conversation_endpoint(conversation_id: str, request: Request):
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"detail": "not logged in"}, status_code=401)
    conv = conversations.get_conversation(conversation_id, wallet_id)
    if conv is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return conv


@app.delete("/conversations/{conversation_id}")
def delete_conversation_endpoint(conversation_id: str, request: Request):
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"detail": "not logged in"}, status_code=401)
    ok = conversations.delete_conversation(conversation_id, wallet_id)
    if not ok:
        return JSONResponse({"detail": "not found"}, status_code=404)
    return {"deleted": True}
