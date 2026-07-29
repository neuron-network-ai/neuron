"""coordinator/auth.py — Google/GitHub login, run ONCE for the whole network.

WHY THIS MOVED OFF THE AGENT. Login used to live in ui/oauth.py, on the driver -- which was
fine while the driver was one machine the founder ran. It stopped being fine the moment every
installed agent started serving its own Chat UI (agent/local_chat.py): an OAuth *client secret*
would then have to exist on every stranger's PC. That leaves two impossible options --

  * ship the secret inside the installer: it is extractable from any distributable binary, and
    this repo is public, so anyone could impersonate the app to Google; or
  * ask each user to create their own Google Cloud project: nobody will, and it is absurd to
    ask someone to do OAuth admin to send a chat message.

So the secret lives HERE, on the coordinator -- an actual server, already trusted to route
requests and hold the ledger, and already the only thing that can mint a wallet
(models.wallet_for_oauth). The founder configures one OAuth app once; every install just
clicks "Sign in".

THE FLOW (agent never sees a client secret, coordinator never sees the user's session):

  1. agent  -> GET  /auth/login/{provider}?port=8080     (port = its own loopback Chat UI)
  2. here   -> redirect to Google/GitHub
  3. user approves; provider -> GET /auth/callback/{provider}?code&state
  4. here   -> exchange code with the provider, resolve the identity, mint/find the wallet,
               then redirect to http://127.0.0.1:<port>/auth/adopt?code=<one-time>
  5. agent  -> POST /auth/exchange {code}  ->  {wallet_id, email}, single use, 120s TTL

The hand-back code is deliberately NOT the wallet_id: a wallet_id is a bearer capability that
spends real NRN, and it would otherwise land in a URL, browser history and any proxy log. The
one-time code is worthless once redeemed.

Uses plain `requests` rather than authlib: the coordinator deliberately carries no heavy deps
(no torch, and now no OAuth client either), and the authorization-code flow is a few HTTP calls.
"""
import logging
import secrets
import time
import urllib.parse

import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse

from coordinator import config, models

log = logging.getLogger("neuron.coordinator.auth")
router = APIRouter()

PROVIDERS = {
    "google": {
        "authorize": "https://accounts.google.com/o/oauth2/v2/auth",
        "token": "https://oauth2.googleapis.com/token",
        "userinfo": "https://openidconnect.googleapis.com/v1/userinfo",
        "scope": "openid email profile",
    },
    "github": {
        "authorize": "https://github.com/login/oauth/authorize",
        "token": "https://github.com/login/oauth/access_token",
        "userinfo": "https://api.github.com/user",
        "scope": "read:user user:email",
    },
}

# state -> (provider, loopback_port, created_at); pending logins, and one-time hand-back codes.
# In-process dicts are correct here: both live for seconds, and a coordinator restart during a
# login just means the user clicks again.
_pending = {}
_handback = {}
STATE_TTL_S = 600
CODE_TTL_S = 120


def _cfg(provider):
    return (getattr(config, f"{provider.upper()}_CLIENT_ID", None),
            getattr(config, f"{provider.upper()}_CLIENT_SECRET", None))


def configured():
    return sorted(p for p in PROVIDERS if all(_cfg(p)))


def _sweep(store, ttl):
    now = time.time()
    for k in [k for k, v in store.items() if now - v[-1] > ttl]:
        store.pop(k, None)


def _redirect_uri(provider):
    return f"{config.PUBLIC_BASE_URL.rstrip('/')}/auth/callback/{provider}"


@router.get("/auth/providers")
def auth_providers():
    """What the agent's Chat UI asks before drawing login buttons."""
    return {"providers": configured(), "login_base": config.PUBLIC_BASE_URL.rstrip("/")}


@router.get("/auth/login/{provider}")
def auth_login(provider: str, port: int = Query(..., ge=1024, le=65535)):
    """Start a login. `port` is the caller's own loopback Chat UI port, where the result is
    handed back. Only a port is accepted -- never a full URL -- so this cannot be turned into
    an open redirect that mails a live login code to somebody else's server."""
    if provider not in PROVIDERS:
        raise HTTPException(status_code=404, detail=f"unknown provider '{provider}'")
    cid, _ = _cfg(provider)
    if not cid:
        raise HTTPException(
            status_code=501,
            detail=f"{provider} login is not configured on this coordinator "
                   f"(set NEURON_{provider.upper()}_CLIENT_ID / _CLIENT_SECRET)")
    _sweep(_pending, STATE_TTL_S)
    state = secrets.token_urlsafe(24)
    _pending[state] = (provider, port, time.time())
    q = urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": _redirect_uri(provider),
        "response_type": "code", "scope": PROVIDERS[provider]["scope"], "state": state,
    })
    return RedirectResponse(f"{PROVIDERS[provider]['authorize']}?{q}")


def _identity(provider, access_token):
    """(external_id, email, email_verified) from the provider."""
    p = PROVIDERS[provider]
    r = requests.get(p["userinfo"], timeout=15,
                     headers={"Authorization": f"Bearer {access_token}",
                              "Accept": "application/json"})
    r.raise_for_status()
    me = r.json()
    if provider == "google":
        return me["sub"], me.get("email"), bool(me.get("email_verified"))
    email, verified = me.get("email"), False
    if not email:                      # GitHub hides private addresses from /user
        er = requests.get("https://api.github.com/user/emails", timeout=15,
                          headers={"Authorization": f"Bearer {access_token}",
                                   "Accept": "application/json"})
        if er.ok and er.json():
            chosen = next((e for e in er.json() if e.get("primary")), er.json()[0])
            email, verified = chosen.get("email"), bool(chosen.get("verified"))
    return str(me["id"]), email, verified


@router.get("/auth/callback/{provider}", response_class=HTMLResponse)
def auth_callback(provider: str, code: str = "", state: str = ""):
    _sweep(_pending, STATE_TTL_S)
    entry = _pending.pop(state, None)
    if provider not in PROVIDERS or entry is None or entry[0] != provider:
        raise HTTPException(status_code=400, detail="invalid or expired login state")
    _, port, _ts = entry
    cid, csecret = _cfg(provider)
    try:
        tr = requests.post(PROVIDERS[provider]["token"], timeout=15,
                           headers={"Accept": "application/json"},
                           data={"client_id": cid, "client_secret": csecret, "code": code,
                                 "redirect_uri": _redirect_uri(provider),
                                 "grant_type": "authorization_code"})
        tr.raise_for_status()
        access_token = tr.json().get("access_token")
        if not access_token:
            raise ValueError("provider returned no access_token")
        external_id, email, verified = _identity(provider, access_token)
    except Exception as e:
        log.warning("%s login failed: %s", provider, e)
        raise HTTPException(status_code=502, detail=f"{provider} login failed")

    wallet_id, is_new = models.wallet_for_oauth(provider, external_id, email,
                                                email_verified=verified)
    _sweep(_handback, CODE_TTL_S)
    hb = secrets.token_urlsafe(24)
    _handback[hb] = (wallet_id, email, time.time())
    log.info("login ok: %s %s (new=%s)", provider, email or external_id, is_new)
    # Loopback only, and only the port the caller supplied -- see auth_login.
    return RedirectResponse(f"http://127.0.0.1:{port}/auth/adopt?code={hb}")


@router.post("/auth/exchange")
def auth_exchange(body: dict):
    """Redeem the one-time code for the wallet. Single use: pop, not read."""
    _sweep(_handback, CODE_TTL_S)
    entry = _handback.pop((body or {}).get("code"), None)
    if entry is None:
        raise HTTPException(status_code=400, detail="invalid or expired code")
    wallet_id, email, _ts = entry
    return {"wallet_id": wallet_id, "email": email}
