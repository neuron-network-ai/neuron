"""ui/oauth.py — sign-in for the local Chat UI, delegated to the coordinator.

This file used to run the OAuth dance itself, which required a Google/GitHub **client secret**
in this process. That was workable while the driver was one machine the founder ran. It became
unshippable once every installed agent started serving its own Chat UI (agent/local_chat.py):
the secret would have to sit on every stranger's PC, where anyone holding the installer can
extract it from the binary. The alternative, asking each user to create their own Google
Cloud project before they can send a message, is not a product.

So the secret now lives on the coordinator (coordinator/auth.py), which is an actual server,
is already trusted to route requests and hold the ledger, and is already the only thing that
can mint a wallet. The founder configures ONE OAuth app for the whole network; every install
just gets a working "Sign in with Google" button and needs no configuration at all.

What is left here is the loopback half of the handshake:

    GET  /auth/login/{provider}  -> bounce the browser to the coordinator, telling it which
                                    local port to hand the result back to
    GET  /auth/adopt?code=...    <- coordinator sends the browser back here; we redeem the
                                    one-time code server-to-server and store the wallet in
                                    this browser's session

The code in that redirect is deliberately NOT the wallet_id: a wallet_id is a bearer capability
that spends real NRN, and it must never appear in a URL, browser history or proxy log. The
one-time code is single-use and worthless once redeemed.
"""
import logging
import os

import requests
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "https://neuronnet.duckdns.org").rstrip("/")
# The port this Chat UI is actually reachable on, so the coordinator knows where to send the
# browser back. Set by agent/local_chat.py; the default matches its DEFAULT_PORT.
LOCAL_PORT = int(os.environ.get("NEURON_LOCAL_CHAT_PORT", "8080"))

log = logging.getLogger("neuron.ui.oauth")
router = APIRouter()


def providers_configured():
    """Ask the coordinator which logins it can offer. Returns [] if it is unreachable or has
    no OAuth app configured, which is what makes the UI say 'login not configured'."""
    try:
        r = requests.get(f"{COORDINATOR}/auth/providers", timeout=8)
        r.raise_for_status()
        return r.json().get("providers", [])
    except requests.RequestException:
        return []


@router.get("/auth/login/{provider}")
def login(provider: str):
    if provider not in ("google", "github"):
        raise HTTPException(status_code=404, detail=f"unknown provider '{provider}'")
    if provider not in providers_configured():
        raise HTTPException(
            status_code=501,
            detail=f"{provider} login is not configured on the coordinator "
                   f"({COORDINATOR}). See coordinator/auth.py's setup notes.")
    return RedirectResponse(f"{COORDINATOR}/auth/login/{provider}?port={LOCAL_PORT}")


@router.get("/auth/adopt")
def adopt(code: str, request: Request):
    """Redeem the coordinator's one-time code for this browser's session."""
    try:
        r = requests.post(f"{COORDINATOR}/auth/exchange", json={"code": code}, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        log.warning("could not redeem login code: %s", e)
        raise HTTPException(status_code=502, detail="could not complete sign-in")
    request.session["wallet_id"] = data["wallet_id"]
    request.session["email"] = data.get("email")
    return RedirectResponse(url="/")


@router.get("/auth/me")
def me(request: Request):
    return {"wallet_id": request.session.get("wallet_id"),
            "email": request.session.get("email"),
            "providers_configured": providers_configured()}


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "logged out"}
