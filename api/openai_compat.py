"""
NEURON — OpenAI-compatible API  [Session 11]

Drop-in for the OpenAI REST API. Point any OpenAI SDK's `base_url` at this server
and existing code works unchanged — same request/response shapes, same streaming
(`stream=True` -> `data: {chunk}` ... `data: [DONE]`).

    from openai import OpenAI
    client = OpenAI(base_url="http://<node_a-host>:8081/v1", api_key="<your NRN wallet>")
    client.chat.completions.create(model="neuron", messages=[...])

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions      (stream + non-stream)
    POST /v1/completions           (legacy, stream + non-stream)
    GET  /docs                     (human usage docs)

The API key is your NRN wallet address; each request costs 1.0 NRN (reported in
`usage.nrn_cost` and the `X-NRN-Cost` header). Because it IS the node_a driver, this
server runs on the node_a machine. Generation is greedy/deterministic for now, so
`temperature`/`top_p` are accepted and ignored. Reuses neuron_driver (shared with
the Chat UI) — nothing in common.py / node_*.py is modified.

Run standalone:
    .venv\\Scripts\\python.exe -m uvicorn api.openai_compat:app --host 0.0.0.0 --port 8081
Or it is auto-mounted into the Chat UI server (ui.app) at the same /v1 paths.
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, Header
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

import common
from coordinator import model_registry
from neuron_driver import DRIVER
from safety import moderation

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "http://150.230.22.250:8001").rstrip("/")
MODEL_ID = common.MODEL_ID
MODEL_ALIASES = {MODEL_ID, "neuron", "neuron-1", "gpt-3.5-turbo"}  # accept common names
NRN_PER_REQUEST = 1.0

router = APIRouter()


# --------------------------------------------------------------------------- #
# Request bodies (extra fields ignored so real OpenAI payloads never 422)
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str = ""


class ChatBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    stream: bool = False
    stream_options: dict | None = None


class CompletionBody(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str | None = None
    prompt: str | list[str] = ""
    max_tokens: int | None = None
    stream: bool = False
    stream_options: dict | None = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _error_response(status: int, message: str, err_type: str, code: str):
    return JSONResponse(status_code=status,
                        content={"error": {"message": message, "type": err_type,
                                           "param": None, "code": code}})


def _auth(authorization: str | None):
    """Returns the wallet (str) or a JSONResponse error. The bearer key = NRN wallet."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return _error_response(
            401, "Missing bearer token. Use your NRN wallet address as the API key.",
            "invalid_request_error", "invalid_api_key")
    wallet = authorization.split(" ", 1)[1].strip()
    if not wallet:
        return _error_response(401, "Empty API key.", "invalid_request_error",
                               "invalid_api_key")
    return wallet


def _moderate_or_error(text: str):
    """Input moderation gate (Workstream A) — checked here, before DRIVER.ensure_loaded()
    and before anything is dispatched to the node chain. This API server IS the driver (the
    only place plaintext exists in NEURON), so this is the correct place for the check; see
    safety/moderation.py. Returns a JSONResponse to return directly if blocked, else None."""
    verdict = moderation.check_text(text)
    if verdict.blocked:
        moderation.log_event("in", verdict.category, "api-" + uuid.uuid4().hex[:12], snippet=text)
        return _error_response(
            400, "This request was blocked by NEURON's acceptable-use policy (see SAFETY.md).",
            "invalid_request_error", "content_policy_violation")
    return None


def _stream_headers(wallet: str):
    return {"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
            "Connection": "keep-alive", "X-NRN-Cost": str(NRN_PER_REQUEST),
            "X-NRN-Wallet": wallet}


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _usage(prompt_tokens: int, completion_tokens: int):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "nrn_cost": NRN_PER_REQUEST}


def _want_usage(body) -> bool:
    return bool(body.stream_options and body.stream_options.get("include_usage"))


# --------------------------------------------------------------------------- #
# GET /v1/models
# --------------------------------------------------------------------------- #
@router.get("/v1/models")
def list_models():
    now = int(time.time())
    data = [{"id": m["id"], "object": "model", "created": now, "owned_by": "neuron"}
            for m in model_registry.list_models()]
    data.append({"id": "neuron", "object": "model", "created": now, "owned_by": "neuron"})
    return {"object": "list", "data": data}


@router.get("/v1/models/{model_id:path}")
def get_model(model_id: str):
    return {"id": model_id, "object": "model", "created": int(time.time()),
            "owned_by": "neuron"}


# --------------------------------------------------------------------------- #
# POST /v1/chat/completions
# --------------------------------------------------------------------------- #
@router.post("/v1/chat/completions")
def chat_completions(body: ChatBody, authorization: str = Header(default=None)):
    wallet = _auth(authorization)
    if isinstance(wallet, JSONResponse):
        return wallet
    messages = [{"role": m.role, "content": m.content} for m in body.messages]
    blocked = _moderate_or_error("\n".join(m["content"] for m in messages))
    if blocked is not None:
        return blocked
    DRIVER.ensure_loaded()

    input_ids = DRIVER.encode_chat(messages)
    max_new = body.max_tokens or 256
    router_prompt = next((m["content"] for m in reversed(messages)
                          if m["role"] == "user"), "")
    cid = "chatcmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model_name = body.model or MODEL_ID

    def chunk(delta, finish):
        return {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    if body.stream:
        def gen():
            for ev in DRIVER.stream(input_ids, max_new, COORDINATOR, router_prompt):
                if ev["type"] == "error":
                    yield _sse({"error": {"message": ev["detail"], "type": "server_error",
                                          "code": "chain_unavailable"}})
                    yield "data: [DONE]\n\n"
                    return
                if ev["type"] == "meta":
                    yield _sse(chunk({"role": "assistant"}, None))
                elif ev["type"] == "token":
                    yield _sse(chunk({"content": ev["text"]}, None))
                elif ev["type"] == "done":
                    yield _sse(chunk({}, ev["finish_reason"]))
                    if _want_usage(body):
                        final = chunk({}, ev["finish_reason"])
                        final["choices"] = []
                        final["usage"] = _usage(ev["prompt_tokens"], ev["completion_tokens"])
                        yield _sse(final)
                    yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=_stream_headers(wallet))

    # non-streaming: consume the generator, build one response
    text, finish, pt, ct, err = "", "stop", 0, 0, None
    for ev in DRIVER.stream(input_ids, max_new, COORDINATOR, router_prompt):
        if ev["type"] == "error":
            err = ev["detail"]
            break
        if ev["type"] == "done":
            text, finish = ev["text"], ev["finish_reason"]
            pt, ct = ev["prompt_tokens"], ev["completion_tokens"]
    if err is not None:
        return _error_response(503, err, "server_error", "chain_unavailable")
    resp = {"id": cid, "object": "chat.completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": _usage(pt, ct)}
    return JSONResponse(resp, headers={"X-NRN-Cost": str(NRN_PER_REQUEST),
                                       "X-NRN-Wallet": wallet})


# --------------------------------------------------------------------------- #
# POST /v1/completions  (legacy text completions)
# --------------------------------------------------------------------------- #
@router.post("/v1/completions")
def completions(body: CompletionBody, authorization: str = Header(default=None)):
    wallet = _auth(authorization)
    if isinstance(wallet, JSONResponse):
        return wallet
    prompt = body.prompt[0] if isinstance(body.prompt, list) and body.prompt else body.prompt
    if not isinstance(prompt, str):
        prompt = ""
    blocked = _moderate_or_error(prompt)
    if blocked is not None:
        return blocked
    DRIVER.ensure_loaded()

    input_ids = DRIVER.encode_text(prompt)
    max_new = body.max_tokens or 128
    cid = "cmpl-" + uuid.uuid4().hex
    created = int(time.time())
    model_name = body.model or MODEL_ID

    def chunk(text, finish):
        return {"id": cid, "object": "text_completion", "created": created,
                "model": model_name,
                "choices": [{"index": 0, "text": text, "logprobs": None,
                             "finish_reason": finish}]}

    if body.stream:
        def gen():
            for ev in DRIVER.stream(input_ids, max_new, COORDINATOR, prompt):
                if ev["type"] == "error":
                    yield _sse({"error": {"message": ev["detail"], "type": "server_error",
                                          "code": "chain_unavailable"}})
                    yield "data: [DONE]\n\n"
                    return
                if ev["type"] == "token":
                    yield _sse(chunk(ev["text"], None))
                elif ev["type"] == "done":
                    yield _sse(chunk("", ev["finish_reason"]))
                    if _want_usage(body):
                        final = chunk("", ev["finish_reason"])
                        final["choices"] = []
                        final["usage"] = _usage(ev["prompt_tokens"], ev["completion_tokens"])
                        yield _sse(final)
                    yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers=_stream_headers(wallet))

    text, finish, pt, ct, err = "", "stop", 0, 0, None
    for ev in DRIVER.stream(input_ids, max_new, COORDINATOR, prompt):
        if ev["type"] == "error":
            err = ev["detail"]
            break
        if ev["type"] == "done":
            text, finish = ev["text"], ev["finish_reason"]
            pt, ct = ev["prompt_tokens"], ev["completion_tokens"]
    if err is not None:
        return _error_response(503, err, "server_error", "chain_unavailable")
    resp = {"id": cid, "object": "text_completion", "created": created,
            "model": model_name,
            "choices": [{"index": 0, "text": text, "logprobs": None,
                         "finish_reason": finish}],
            "usage": _usage(pt, ct)}
    return JSONResponse(resp, headers={"X-NRN-Cost": str(NRN_PER_REQUEST),
                                       "X-NRN-Wallet": wallet})


# --------------------------------------------------------------------------- #
# Human usage docs (self-contained; no external assets)
# --------------------------------------------------------------------------- #
def docs_html() -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEURON API — OpenAI-compatible</title>
<style>
 :root{{--bg:#f6f7f9;--panel:#fff;--ink:#1f2328;--muted:#6a737d;--line:#e3e6ea;
   --brand:#4f46e5;--code:#0f172a;--codeink:#e2e8f0}}
 @media(prefers-color-scheme:dark){{:root{{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;
   --muted:#8b949e;--line:#2a2f37;--brand:#8b8cf7;--code:#0b1020;--codeink:#e2e8f0}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--ink);
   font:15px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}}
 .wrap{{max-width:820px;margin:0 auto;padding:2rem 1.2rem 4rem}}
 h1{{margin:.2rem 0}} h1 span{{color:var(--brand)}}
 .sub{{color:var(--muted);margin-bottom:1.5rem}}
 h2{{margin-top:2rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}}
 code{{background:var(--line);padding:.1rem .35rem;border-radius:4px;font-size:.9em}}
 pre{{background:var(--code);color:var(--codeink);padding:1rem;border-radius:10px;
   overflow-x:auto;font-size:13px;line-height:1.5}}
 pre code{{background:none;padding:0}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0}}
 th,td{{border:1px solid var(--line);padding:.45rem .6rem;text-align:left;font-size:14px}}
 th{{background:var(--line)}}
 .note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--brand);
   padding:.7rem 1rem;border-radius:8px;margin:1rem 0;font-size:14px}}
</style></head><body><div class="wrap">
<h1>NE<span>U</span>RON API</h1>
<div class="sub">OpenAI-compatible inference over a network of volunteer machines.
Change one URL, keep your code.</div>

<div class="note"><b>Base URL:</b> <code>http://&lt;this-host&gt;/v1</code> &nbsp;·&nbsp;
<b>API key:</b> your NRN wallet address &nbsp;·&nbsp;
<b>Model:</b> <code>{MODEL_ID}</code> (alias <code>neuron</code>) &nbsp;·&nbsp;
<b>Cost:</b> {NRN_PER_REQUEST} NRN / request</div>

<h2>Python — OpenAI SDK (drop-in)</h2>
<pre><code>from openai import OpenAI

client = OpenAI(
    base_url="http://&lt;this-host&gt;/v1",   # &lt;-- the only change
    api_key="&lt;your NRN wallet address&gt;",
)

resp = client.chat.completions.create(
    model="neuron",
    messages=[{{"role": "user", "content": "Why is the sky blue?"}}],
    max_tokens=60,
)
print(resp.choices[0].message.content)

# streaming
for chunk in client.chat.completions.create(
        model="neuron",
        messages=[{{"role": "user", "content": "Count to five."}}],
        stream=True):
    print(chunk.choices[0].delta.content or "", end="")</code></pre>

<h2>curl</h2>
<pre><code>curl http://&lt;this-host&gt;/v1/chat/completions \\
  -H "Authorization: Bearer &lt;your NRN wallet&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"neuron","messages":[{{"role":"user","content":"Hello"}}],"max_tokens":40}}'</code></pre>

<h2>Endpoints</h2>
<table>
<tr><th>Method</th><th>Path</th><th>Notes</th></tr>
<tr><td>GET</td><td>/v1/models</td><td>list available models</td></tr>
<tr><td>POST</td><td>/v1/chat/completions</td><td>chat; <code>stream</code> supported</td></tr>
<tr><td>POST</td><td>/v1/completions</td><td>legacy text completion; <code>stream</code> supported</td></tr>
</table>

<div class="note"><b>Notes.</b> Generation is greedy/deterministic for now, so
<code>temperature</code>, <code>top_p</code>, <code>n</code> and similar sampling
params are accepted and ignored. Each response also returns <code>X-NRN-Cost</code>
and <code>X-NRN-Wallet</code> headers and a <code>usage.nrn_cost</code> field.
Persisting a per-wallet balance debit is coordinator-side economics (a later
session); today the wallet is recorded and the cost reported.</div>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Standalone app (also mountable into ui.app via `router`)
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    DRIVER.ensure_loaded()
    print(f"[api] OpenAI-compatible API up | coordinator={COORDINATOR} | model={MODEL_ID}")
    yield


def create_app() -> FastAPI:
    application = FastAPI(title="NEURON OpenAI-compatible API", version="0.1",
                         lifespan=lifespan, docs_url=None, redoc_url=None)
    application.include_router(router)

    @application.get("/docs", response_class=HTMLResponse)
    def docs():
        return docs_html()

    @application.get("/")
    def root():
        return {"service": "NEURON OpenAI-compatible API", "docs": "/docs",
                "base_url": "/v1", "model": MODEL_ID}

    return application


app = create_app()
