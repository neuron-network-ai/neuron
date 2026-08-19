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
import threading
import time
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
from agent import updater
from ui import app_assets
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
        # Pull a bigger model in the background if this machine can hold one. Deliberately a
        # thread: it is a multi-GB download and must never delay the server coming up, and
        # best_local_model() keeps answering with whatever is already cached until it lands.
        threading.Thread(target=local_gguf.prefetch_best, daemon=True).start()
    else:
        DRIVER.ensure_loaded()
    # [P46], the half a build-time guard cannot cover: is what was COPIED onto this machine
    # coherent? The build test proves the source tree is; nothing checked the install, and the
    # symptom is a blank page with every other signal green. Logged at startup because that is
    # the one moment somebody is watching, and loudly because the alternative is silence that
    # looks identical to health.
    _assets = app_assets.verify(str(APP_DIR))
    if _assets["built"] and not _assets["ok"]:
        log.error("INSTALL IS INCOMPLETE — /next would render a blank page. index.html asks "
                  "for %d file(s) that are not here: %s. %s",
                  len(_assets["missing"]), ", ".join(_assets["missing"]), app_assets.REMEDY)
    elif _assets["orphans"]:
        # Not broken, and worth saying: this is what a merged-rather-than-replaced copy looks
        # like, and it is the state that turns into the failure above on the next build.
        log.warning("%d orphaned bundle(s) from an earlier build are sitting in the app "
                    "directory (%s). Nothing is broken, but this is the fingerprint of an "
                    "install that was merged rather than replaced.",
                    len(_assets["orphans"]), ", ".join(_assets["orphans"][:4]))
    print(f"[ui] ready | coordinator={COORDINATOR} | chat at / , OpenAI API at /v1")
    yield


app = FastAPI(title="NEURON Chat", version="0.2", lifespan=lifespan)
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="lax")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# The built app references its chunks as ./assets/... relative to /next, so they have to
# be reachable at /assets/ as well. Mounted only when a build exists, so a source checkout
# that has never run npm still starts.
if (STATIC_DIR / "app" / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR / "app" / "assets")),
              name="app-assets")
app.include_router(openai_router)          # Session 11: /v1/* on the same server
app.include_router(oauth_module.router)    # Workstream B: /auth/login|callback|me|logout


class ChatBody(BaseModel):
    prompt: str
    max_tokens: int = 128
    use_rag: bool = False       # Session 15: retrieve current web context first
    # Send THIS request over the node chain even when this machine could answer it itself.
    # Per-request, unlike NEURON_FORCE_NETWORK, which is a process-wide env var the operator
    # has to restart the app to set — invisible from the product and therefore useless to the
    # person who actually wants to see what the network does. Asked for directly: "I want to
    # see the speed of network".
    use_network: bool = False
    conversation_id: str | None = None   # None -> a new conversation is created server-side


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "chat.html"))


# The React rewrite, served alongside the page it will eventually replace rather than instead
# of it. `/` keeps working exactly as before, so the 67 assertions in ui/test_chat_ui.py -- 59
# of which read chat.html's source text -- stay meaningful while this is built. Swapping the
# routes is a one-line change on the day it earns it.
#
# Built assets only: `cd ui/web && npm install && npm run build` writes them here. Node is a
# BUILD dependency; a volunteer never needs a JavaScript runtime, and the PyInstaller bundle
# ships these files exactly as it ships chat.html.
APP_DIR = STATIC_DIR / "app"


@app.get("/next")
def index_next():
    built = APP_DIR / "index.html"
    if not built.exists():
        return HTMLResponse(
            "<h1>Not built</h1><p>Run <code>npm install &amp;&amp; npm run build</code> in "
            "<code>ui/web/</code>. The existing chat is unaffected at <a href='/'>/</a>.</p>",
            status_code=503)
    # [P46]: a broken install must not be served as a blank page. Checked here as well as at
    # startup because the directory can be changed under a running app -- that is exactly how it
    # happened, a copy landing on top of an installation nobody stopped.
    state = app_assets.verify(str(APP_DIR))
    if not state["ok"]:
        log.error("serving /next with %d missing asset(s): %s",
                  len(state["missing"]), ", ".join(state["missing"]))
        return HTMLResponse(
            "<h1>This install is incomplete</h1>"
            "<p>The page asks for files that are not on this machine, so it would render as a "
            "blank screen:</p><ul>"
            + "".join(f"<li><code>{m}</code></li>" for m in state["missing"])
            + f"</ul><p>{app_assets.REMEDY}</p>"
            "<p>The chat at <a href='/'>/</a> is unaffected and still works.</p>",
            status_code=503)
    return FileResponse(str(built))


# --------------------------------------------------------------------------- #
# Is there a newer build? ([P48] item 1 taught the comparison; this is the surface)
# --------------------------------------------------------------------------- #
# Cached, because this is a network call and the page asks on every load. An hour is the right
# order: `updater.CHECK_SECONDS` is a day, so a tighter TTL here would poll far more often than
# the thing it reports on without ever learning anything new.
_UPDATE_TTL_S = 3600.0
_update_cache = {"at": 0.0, "data": None}
_update_lock = threading.Lock()


@app.get("/app/update")
def app_update():
    """What build is running, and is there a newer one?

    The operator's own dashboard could already answer this and the app could not — so the
    person who has to act on an update was the one person not told about it. `_version_gt`
    compares NUMERICALLY ([P48]): `"0.20.10" > "0.20.9"` is False as strings, so a lexical
    comparison breaks at the tenth patch release, and a node AHEAD of the network must never be
    told to downgrade.

    Never raises and never blocks the page: an unreachable coordinator returns
    `available: false` with the reason, because "we could not check" must not render as "you
    are up to date" ([P24], and the whole of 0.20.2's reasoning).
    """
    now = time.time()
    with _update_lock:
        cached = _update_cache["data"]
        if cached is not None and (now - _update_cache["at"]) < _UPDATE_TTL_S:
            return cached
    running = updater.LOCAL_VERSION
    try:
        info = updater.remote_info(COORDINATOR)
        latest = str(info.get("version") or "")
        data = {
            "running": running,
            "latest": latest or None,
            # `is_newer`, not `!=`: the founder's own machine ran a locally-built 0.20.4 while
            # the network advertised 0.20.3, and equality-as-currency told a node ahead of the
            # network to downgrade.
            "available": bool(latest) and updater.is_newer(latest, running),
            "rollback": bool(info.get("rollback")),
            "url": info.get("url") or None,
            "error": None,
        }
    except Exception as e:                      # noqa: BLE001 - a status call must not fail loudly
        data = {"running": running, "latest": None, "available": False, "rollback": False,
                "url": None, "error": str(e)}
    with _update_lock:
        _update_cache.update(at=now, data=data)
    return data


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


@app.get("/wallet/nodes")
def wallet_nodes_proxy(request: Request):
    """Every machine on the signed-in account, and what they add up to.

    One person contributing several computers is the ordinary case — a desktop, a laptop, an old
    machine in a cupboard — and the coordinator has always allowed it, because `owner_wallet_id`
    is just a column on `nodes` with no cap on how many rows carry the same value. What was
    missing was any way to SEE it: the only query was node -> owner, so you could claim every
    machine you owned and still only ever see the one you were sitting at.

    The wallet id comes from the SESSION and is never accepted from the page, same rule as every
    other wallet route here — otherwise this would be an endpoint for reading somebody else's
    machine list.
    """
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return {"logged_in": False, "nodes": []}
    try:
        r = requests.get(f"{COORDINATOR}/wallet/{wallet_id}/nodes", timeout=8)
        r.raise_for_status()
        data = r.json()
        # wallet_id is echoed by the coordinator; drop it rather than hand a spendable
        # credential back to the browser that already knows better than to need it.
        data.pop("wallet_id", None)
        return {"logged_in": True, **data}
    except requests.RequestException as e:
        log.warning("wallet nodes proxy failed: %s", e)
        return {"logged_in": True, "error": str(e), "nodes": []}


class PayoutBindBody(BaseModel):
    address: str
    nonce: str
    signature: str
    old_signature: str | None = None


@app.get("/wallet/payout")
def payout_address_read(request: Request):
    """The address this session's wallet currently pays out to, if any."""
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return {"logged_in": False}
    try:
        r = requests.get(f"{COORDINATOR}/wallet/{wallet_id}/payout-address", timeout=8)
        r.raise_for_status()
        return {"logged_in": True, **r.json()}
    except requests.RequestException as e:
        return {"logged_in": True, "error": str(e)}


@app.get("/wallet/payout/challenge")
def payout_challenge_proxy(address: str, request: Request):
    """Fetch the nonce and the exact text this session's wallet must sign.

    Proxied server-side for the same reason /wallet/balance is: the browser never talks to the
    coordinator, whose CORS is deliberately named-origins and GET-only. It also means the
    wallet id comes from the SESSION rather than from anything the page could be talked into
    sending — so this route cannot be used to start a binding against somebody else's wallet
    even by a page that has somehow learned their id.
    """
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"error": "sign in first"}, status_code=401)
    try:
        r = requests.get(f"{COORDINATOR}/wallet/{wallet_id}/payout-challenge",
                         params={"address": address}, timeout=8)
        if r.status_code == 400:
            return JSONResponse({"error": r.json().get("detail", "bad address")},
                                status_code=400)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("payout challenge failed: %s", e)
        return JSONResponse({"error": "could not reach the coordinator"}, status_code=502)


@app.post("/wallet/payout/bind")
def payout_bind_proxy(body: PayoutBindBody, request: Request):
    """Submit the signature. The coordinator does the real verification -- recovering the
    signer and requiring it to equal the address being claimed -- so this adds no trust of its
    own beyond binding the request to the logged-in session.

    A 400 from the coordinator is passed through verbatim: payout.py writes those messages to
    be read by the person who caused them ("sign with the key for the address you are
    claiming"), and replacing them with a generic failure would throw away the only thing that
    tells someone what they got wrong.
    """
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"error": "sign in first"}, status_code=401)
    try:
        r = requests.post(f"{COORDINATOR}/wallet/{wallet_id}/payout-address",
                          json=body.model_dump(), timeout=12)
        if r.status_code == 400:
            return JSONResponse({"error": r.json().get("detail", "binding refused")},
                                status_code=400)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("payout bind failed: %s", e)
        return JSONResponse({"error": "could not reach the coordinator"}, status_code=502)


# --------------------------------------------------------------------------- #
# This machine's own node: recording who owns its earnings  ([P39] phase 2)
#
# A node's balance is credentialed only by the node_token in config.json. Lose the file, lose
# the money, and migrate_ledger.py skips the account as `unmapped`. The fix is to record the
# wallet a person actually signs into -- and this is the one moment both facts are in the same
# process: the agent knows its node_id, and the session knows who just logged in.
#
# NODE_TOKEN is read from the environment the agent put it in and never leaves this process.
# The browser sees a node_id and an address; it never sees the token, and it never supplies the
# wallet id either -- that comes from the session, so this cannot be used to bind somebody
# else's wallet as the owner.
# --------------------------------------------------------------------------- #
# Resolved PER CALL, not once at import, and that distinction is a live bug rather than a
# nicety. `local_chat.start_local_chat` does `os.environ.setdefault("NEURON_NODE_TOKEN", …)`
# once at agent startup and this module read it once at import — two snapshots of a value the
# agent ROTATES. `register()` issues a fresh token on a relay-ticket refresh, on stale-token
# recovery, or on any re-registration (agent.py:1103), and from that moment every call here
# carried a dead credential.
#
# Observed 2026-08-18: /node/owner answering `401 Client Error` for the machine's own node
# while `node_server` on the same box answered challenges correctly. The claim panel reads
# that endpoint, so a token rotation silently switched off the one feature this UI exists to
# offer — the same blank panel as before, reached by a different road.
#
# The agent's config.json is the authority (it is what `_save()` writes on rotation); the
# environment stays a fallback so a dev shell override still works and a driver-only machine,
# which has neither, still reports `is_node: false`.
def _node_identity():
    """(node_id, node_token) as they are NOW. Never raises — a missing or half-written config
    is 'this machine serves no node', which is a state the UI already renders correctly."""
    env_id = os.environ.get("NEURON_NODE_ID") or None
    env_tok = os.environ.get("NEURON_NODE_TOKEN") or None
    try:
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"),
                                                              ".local", "share")
        for path in (Path(base) / "NEURON" / "config.json",
                     Path(__file__).resolve().parent.parent / "agent" / "config.json"):
            if path.exists():
                cfg = json.loads(path.read_text(encoding="utf-8"))
                nid, tok = cfg.get("node_id"), cfg.get("node_token")
                if nid and tok:
                    return nid, tok
    except (OSError, ValueError):
        pass
    return env_id, env_tok


class NodeBindBody(BaseModel):
    address: str
    nonce: str
    signature: str
    old_signature: str | None = None


@app.get("/node/owner")
def node_owner(request: Request):
    """Does this machine serve a node, and is its owner recorded yet?

    Drives whether the UI offers the prompt at all. A driver-only machine has no node_id and
    gets `is_node: false` — there is nothing to own, and asking would be noise.
    """
    wallet_id = request.session.get("wallet_id")
    NODE_ID, NODE_TOKEN = _node_identity()
    if not (NODE_ID and NODE_TOKEN):
        return {"is_node": False, "logged_in": bool(wallet_id)}
    try:
        r = requests.get(f"{COORDINATOR}/node/{NODE_ID}/payout-address",
                         headers={"X-Node-Token": NODE_TOKEN}, timeout=8)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return {"is_node": True, "node_id": NODE_ID, "logged_in": bool(wallet_id),
                "error": str(e)}
    owner = data.get("owner_wallet_id")
    return {"is_node": True, "node_id": NODE_ID, "logged_in": bool(wallet_id),
            "payout_address": data.get("payout_address"),
            "owner_wallet_id": owner,
            # The prompt is worth showing only when there is something to record AND somebody
            # to record it against.
            "needs_owner": bool(wallet_id) and not owner,
            # `unclaimed` is the FACT; `needs_owner` is only "can be recorded right now".
            # Conflating them hid the whole feature: the claim panel rendered on `needs_owner`,
            # which is false while nobody is signed in, so a machine that was earning with its
            # NRN tied to a file on disk displayed NOTHING -- and the prompt explaining why you
            # should sign in was itself gated on being signed in. Zero nodes on the live network
            # had an owner recorded, and this is why.
            "unclaimed": not owner}


@app.get("/node/payout/challenge")
def node_payout_challenge(address: str, request: Request):
    """The nonce and exact text this MACHINE's node must sign, for `address`."""
    NODE_ID, NODE_TOKEN = _node_identity()
    if not (NODE_ID and NODE_TOKEN):
        return JSONResponse({"error": "this machine does not serve a node"}, status_code=404)
    if not request.session.get("wallet_id"):
        return JSONResponse({"error": "sign in first"}, status_code=401)
    try:
        r = requests.get(f"{COORDINATOR}/node/{NODE_ID}/payout-challenge",
                         params={"address": address},
                         headers={"X-Node-Token": NODE_TOKEN}, timeout=8)
        if r.status_code == 400:
            return JSONResponse({"error": r.json().get("detail", "bad address")},
                                status_code=400)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("node payout challenge failed: %s", e)
        return JSONResponse({"error": "could not reach the coordinator"}, status_code=502)


@app.post("/node/payout/bind")
def node_payout_bind(body: NodeBindBody, request: Request):
    """Bind the address AND record the owner in one call.

    `owner_wallet_id` is taken from the SESSION, never from the request body — the page cannot
    nominate somebody else as the owner of this machine's earnings. The coordinator records it
    only if the signature verifies, which is what makes a copied node_token insufficient to
    move ownership later.
    """
    NODE_ID, NODE_TOKEN = _node_identity()
    if not (NODE_ID and NODE_TOKEN):
        return JSONResponse({"error": "this machine does not serve a node"}, status_code=404)
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"error": "sign in first"}, status_code=401)
    payload = {**body.model_dump(), "owner_wallet_id": wallet_id}
    try:
        r = requests.post(f"{COORDINATOR}/node/{NODE_ID}/payout-address", json=payload,
                          headers={"X-Node-Token": NODE_TOKEN}, timeout=12)
        if r.status_code == 400:
            return JSONResponse({"error": r.json().get("detail", "binding refused")},
                                status_code=400)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("node payout bind failed: %s", e)
        return JSONResponse({"error": "could not reach the coordinator"}, status_code=502)


def _local_payout_key(create=False):
    """(address, private_key) for the key THIS machine generated, or (None, None).

    `agent/payout_key.py` mints this on an early start and binds it, which is why an operator
    who has never touched a wallet still has an address on file ([P53]).

    `create` is off by default and that is deliberate. `load_or_create` does what it says, so
    reading the key to decide whether we CAN claim would mint one as a side effect — including
    in the case where the node is bound to somebody else's address and the answer is going to
    be "no". A key generated by a refusal is a key nobody knows exists.
    """
    try:
        from agent import payout_key
    except ImportError:                                     # pragma: no cover - packaging guard
        return None, None
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"),
                                                          ".local", "share")
    state = str(Path(base) / "NEURON")
    try:
        if not create and not os.path.exists(payout_key.key_path(state)):
            return None, None
        return payout_key.load_or_create(state)
    except (OSError, ValueError) as e:
        log.warning("could not read this machine's payout key: %s", e)
        return None, None


def _sign_binding(private_key, message):
    """Sign the challenge with this machine's own key. A seam, so the claim endpoint can be
    tested without a real secp256k1 key or the `eth_account` dependency."""
    from agent import payout_key
    return payout_key.sign_binding(private_key, message)


@app.post("/node/claim")
def node_claim(request: Request):
    """Record the signed-in account as the owner of this machine's node. No wallet needed.

    **Why this exists, and why the wallet path was not enough ([P53]).** Claiming used to mean
    `claimNodeEarnings`: connect a browser extension, sign a challenge with it, and bind THAT
    address. But `agent/payout_key.py` has already generated a key for this node and bound it
    automatically, long before anyone signs in. So the wallet's address is a DIFFERENT address,
    the bind is therefore a REBIND, and `require_rebind_authority` refuses it without
    `old_signature` from the incumbent key. Every self-hosted node reaches that state on its
    own, so the first genuine claim on the live network failed — and the error told the
    operator their key might be lost while the key sat in `payout_key.json` on the same disk.

    **What this does instead.** Re-bind the address that is ALREADY bound, signed here by the
    key that bound it, carrying `owner_wallet_id` from the session. `require_rebind_authority`
    exempts a bind to the same address — `if not current_address or current_address.lower() ==
    new_address.lower(): return` — so no `old_signature` is needed and no register secret. The
    payout address does not move. Only the owner is recorded, which is the whole point: the
    node belongs to a Google/GitHub identity rather than to a file on one disk.

    **What it deliberately cannot do.** If the bound address is one this machine has no key
    for, this refuses and says so. That case is a genuine address CHANGE and it must keep
    needing the incumbent key — that requirement is what stops a copied `node_token` from
    redirecting somebody's earnings, and making the account-claim able to override it would
    hand an attacker the exact bypass the control exists to prevent.
    """
    NODE_ID, NODE_TOKEN = _node_identity()
    if not (NODE_ID and NODE_TOKEN):
        return JSONResponse({"error": "this machine does not serve a node"}, status_code=404)
    wallet_id = request.session.get("wallet_id")
    if not wallet_id:
        return JSONResponse({"error": "sign in first"}, status_code=401)

    headers = {"X-Node-Token": NODE_TOKEN}
    try:
        # The BINDING is read first, because it decides everything after it: with nothing bound
        # this machine may mint a key and bind it, and with something bound it may only re-bind
        # the very same address. Reading the key first would have minted one even on the path
        # that goes on to refuse.
        cur = requests.get(f"{COORDINATOR}/node/{NODE_ID}/payout-address",
                           headers=headers, timeout=8)
        cur.raise_for_status()
        bound = cur.json().get("payout_address")

        address, private_key = _local_payout_key(create=not bound)
        if not address:
            return JSONResponse(
                {"error": (f"this node pays out to {bound}, and this machine holds no key at "
                           f"all — nothing here can sign for it. Use the browser-wallet claim, "
                           f"or ask the operator to rebind with the register secret."
                           if bound else
                           "this machine could not create a payout key, so it has nothing to "
                           "sign with. Use the browser-wallet claim instead.")},
                status_code=409)
        if bound and bound.lower() != address.lower():
            return JSONResponse(
                {"error": f"this node pays out to {bound}, which is not an address this "
                          f"machine holds a key for. Changing it needs a signature from that "
                          f"address — use the browser-wallet claim."}, status_code=409)

        ch = requests.get(f"{COORDINATOR}/node/{NODE_ID}/payout-challenge",
                          params={"address": address}, headers=headers, timeout=8)
        if ch.status_code == 400:
            return JSONResponse({"error": ch.json().get("detail", "bad address")},
                                status_code=400)
        ch.raise_for_status()
        ch = ch.json()

        body = {"address": address, "nonce": ch["nonce"],
                "signature": _sign_binding(private_key, ch["message"]),
                # From the SESSION, never the request body — same rule as /node/payout/bind:
                # the page must not be able to nominate somebody else as owner.
                "owner_wallet_id": wallet_id}
        r = requests.post(f"{COORDINATOR}/node/{NODE_ID}/payout-address", json=body,
                          headers=headers, timeout=12)
        if r.status_code == 400:
            return JSONResponse({"error": r.json().get("detail", "claim refused")},
                                status_code=400)
        r.raise_for_status()
        out = r.json()
        log.info("node %s claimed by the signed-in account (address unchanged: %s)",
                 NODE_ID, address)
        return {"node_id": NODE_ID, "owner_wallet_id": out.get("owner_wallet_id", wallet_id),
                "payout_address": out.get("payout_address", address), "rebound": False}
    except requests.RequestException as e:
        log.warning("account claim failed: %s", e)
        return JSONResponse({"error": "could not reach the coordinator"}, status_code=502)


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
        # The page hardcoded "/28" in its degraded banner, which silently becomes a lie the
        # moment the network migrates to a bigger tier. It also had no way to say WHICH layers
        # are missing, which is the only part of that message anyone can act on.
        "total_layers": net["total_layers"],
        "uncovered_layers": net.get("uncovered_layers", []),
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
           use_network: bool = False,
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
    if not verdict.screened:
        # Passed the gate only because the gate has nothing to say about this script. That is
        # NOT the same as "clean", and reporting the two identically made the share of traffic
        # going through unexamined unmeasurable. Recorded locally (never sent to the
        # coordinator, like every other moderation line) so the coverage gap has evidence
        # behind it instead of being an assumption. No snippet: this is not a violation.
        moderation.log_event("in", f"unscreened_script:{verdict.script}",
                             f"unscreened-{uuid.uuid4().hex[:12]}")

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
    # Serve the biggest model THIS machine can hold, which is usually larger than the one the
    # network serves (that is capped by its weakest member). Falls back to the network's model,
    # then to the chain.
    # `use_network` is the per-request form of the same override. Either one sends this
    # request down the chain; neither changes the default, because local-first is still the
    # right tiering for a model that fits.
    local_model = (None if (FORCE_NETWORK or use_network)
                   else local_gguf.best_local_model(common.MODEL_ID))
    if local_model:
        events = local_gguf.stream(messages, max_new, local_model,
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
        elif ev["type"] == "reroute":
            # A node died and the driver recovered onto a fresh chain. The answer is unchanged
            # (token-identical, per test_node_death.py); only the timing is. Forwarded so the
            # page can say why the stream paused instead of appearing to hang.
            yield sse("reroute", {"at_token": ev["at_token"], "nodes": ev["nodes"]})
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
                               "cost_nrn": ev.get("cost_nrn"),
                               # The driver records these deliberately ("a recovered answer is
                               # still a degraded one"). They were dropped here, so a request
                               # that survived two node deaths looked identical to one that
                               # sailed through -- the opposite of RESILIENCE.md's rule that a
                               # fallback must be visible in the response metadata.
                               "reroutes": len(ev.get("reroutes") or [])})
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
        _drive(prompt, body.max_tokens, wallet_id, body.use_rag, body.use_network,
               body.conversation_id),
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
