"""ui/oauth.py — Google/GitHub login for chat wallet identity (Workstream B).

WHY OAuth here and not in the coordinator: the coordinator stays torch-free and (per its own
design) never needs to be trusted with an OAuth client secret -- login lives on the DRIVER
(this process), which already handles plaintext and already talks to the coordinator over
HTTP for everything wallet-related. After a successful provider login, this calls the
coordinator's POST /wallet/oauth (shared-secret gated) to resolve/create the wallet_id, then
stores it in a signed session cookie. The coordinator endpoint trusts THIS process's claim
because we hold NEURON_WALLET_LINK_SECRET -- it does no OAuth verification of its own.

SETUP — you need real app credentials from each provider before login works:
  Google:  https://console.cloud.google.com/apis/credentials
           -> Create OAuth client ID (Web application)
           -> Authorized redirect URI: http://<this-host>/auth/callback/google
           -> set NEURON_GOOGLE_CLIENT_ID / NEURON_GOOGLE_CLIENT_SECRET
  GitHub:  https://github.com/settings/developers -> New OAuth App
           -> Authorization callback URL: http://<this-host>/auth/callback/github
           -> set NEURON_GITHUB_CLIENT_ID / NEURON_GITHUB_CLIENT_SECRET
  Both:    set NEURON_SESSION_SECRET (random string) to sign the session cookie, and
           NEURON_WALLET_LINK_SECRET (must match the coordinator's env of the same name).

Without a provider's credentials set, /auth/login/<that provider> returns a clear 501 instead
of crashing -- the rest of the chat UI keeps working unauthenticated (no wallet, no spend).
"""
import os

import requests
from authlib.integrations.starlette_client import OAuth
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

COORDINATOR = os.environ.get("NEURON_COORDINATOR", "http://150.230.22.250:8001").rstrip("/")
WALLET_LINK_SECRET = os.environ.get("NEURON_WALLET_LINK_SECRET", "neuron-wallet-link-dev-secret")

router = APIRouter()
oauth = OAuth()

_PROVIDERS_CONFIGURED = set()

_google_id = os.environ.get("NEURON_GOOGLE_CLIENT_ID")
_google_secret = os.environ.get("NEURON_GOOGLE_CLIENT_SECRET")
if _google_id and _google_secret:
    oauth.register(
        name="google", client_id=_google_id, client_secret=_google_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )
    _PROVIDERS_CONFIGURED.add("google")

_github_id = os.environ.get("NEURON_GITHUB_CLIENT_ID")
_github_secret = os.environ.get("NEURON_GITHUB_CLIENT_SECRET")
if _github_id and _github_secret:
    oauth.register(
        name="github", client_id=_github_id, client_secret=_github_secret,
        access_token_url="https://github.com/login/oauth/access_token",
        authorize_url="https://github.com/login/oauth/authorize",
        api_base_url="https://api.github.com/",
        client_kwargs={"scope": "read:user user:email"},
    )
    _PROVIDERS_CONFIGURED.add("github")


def _link_wallet(provider, external_id, email=None, email_verified=False):
    """Call the coordinator's shared-secret-gated endpoint to resolve/create the wallet.
    email_verified is the PROVIDER's own assertion (Google's OIDC claim / GitHub's verified
    flag) -- recorded so an operator reviewing an abusive identity can tell a real, verified
    account from a throwaway."""
    r = requests.post(f"{COORDINATOR}/wallet/oauth",
                      json={"provider": provider, "external_id": str(external_id), "email": email,
                            "email_verified": bool(email_verified)},
                      headers={"X-Wallet-Link-Secret": WALLET_LINK_SECRET}, timeout=15)
    r.raise_for_status()
    return r.json()   # {"wallet_id": ..., "is_new": ...}


@router.get("/auth/login/{provider}")
async def login(provider: str, request: Request):
    if provider not in ("google", "github"):
        raise HTTPException(status_code=404, detail=f"unknown provider '{provider}'")
    if provider not in _PROVIDERS_CONFIGURED:
        raise HTTPException(
            status_code=501,
            detail=f"{provider} login is not configured on this server -- missing "
                   f"NEURON_{provider.upper()}_CLIENT_ID/SECRET. See ui/oauth.py's setup docs.")
    redirect_uri = str(request.url_for("callback", provider=provider))
    client = oauth.create_client(provider)
    return await client.authorize_redirect(request, redirect_uri)


@router.get("/auth/callback/{provider}", name="callback")
async def callback(provider: str, request: Request):
    if provider not in _PROVIDERS_CONFIGURED:
        raise HTTPException(status_code=501, detail=f"{provider} login is not configured")
    client = oauth.create_client(provider)
    token = await client.authorize_access_token(request)

    if provider == "google":
        userinfo = token.get("userinfo") or await client.userinfo(token=token)
        external_id, email = userinfo["sub"], userinfo.get("email")
        email_verified = bool(userinfo.get("email_verified"))
    else:   # github -- no OIDC userinfo; a separate API call, and email can be private
        profile = (await client.get("user", token=token)).json()
        external_id = profile["id"]
        email = profile.get("email")
        email_verified = False
        if not email:
            emails = (await client.get("user/emails", token=token)).json()
            primary = next((e for e in emails if e.get("primary")), None)
            chosen = primary or (emails[0] if emails else None)
            if chosen:
                email, email_verified = chosen["email"], bool(chosen.get("verified"))

    link = _link_wallet(provider, external_id, email, email_verified)
    request.session["wallet_id"] = link["wallet_id"]
    request.session["email"] = email
    return RedirectResponse(url="/")


@router.get("/auth/me")
def me(request: Request):
    return {"wallet_id": request.session.get("wallet_id"),
           "email": request.session.get("email"),
           "providers_configured": sorted(_PROVIDERS_CONFIGURED)}


@router.post("/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"status": "logged out"}
