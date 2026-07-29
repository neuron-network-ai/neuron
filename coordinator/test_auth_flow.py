"""coordinator/test_auth_flow.py — run: python -m coordinator.test_auth_flow

Login is configured ONCE on the coordinator instead of on every installed agent. That is not a
convenience: an OAuth client secret cannot live on a stranger's PC (it is extractable from any
shipped binary, and this repo is public), and asking each user to create a Google Cloud project
before they can send a message is not a product.

What must hold:

  * a login can only be started for a provider the coordinator actually has credentials for;
  * the hand-back to the agent is LOOPBACK-ONLY and derived from a port number, never a
    caller-supplied URL -- otherwise this endpoint would mail a live login code to whatever
    server an attacker named;
  * the code handed back is NOT the wallet_id (a bearer capability that spends real NRN, so it
    must never sit in a URL, browser history or proxy log), and it is single-use and expiring;
  * a successful login mints exactly the same wallet models.wallet_for_oauth would, carrying the
    provider's email_verified claim through for abuse review.

The provider is mocked; no network, isolated temp DB.
"""
import os
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_authflow_"), "a.db")
os.environ["NEURON_PUBLIC_BASE_URL"] = "http://coordinator.test:8001"

from coordinator import auth, config, models   # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.ok = payload, status, status < 400

    def json(self):
        return self._p

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


def main():
    models.init_db()
    config.GOOGLE_CLIENT_ID = "test-client-id"
    config.GOOGLE_CLIENT_SECRET = "test-client-secret"
    config.GITHUB_CLIENT_ID = config.GITHUB_CLIENT_SECRET = None

    # ---- only configured providers are offered ---- #
    check("configured() lists only providers with real credentials",
          auth.configured() == ["google"])
    try:
        auth.auth_login("github", port=8080)
        check("unconfigured provider refuses to start a login", False)
    except Exception as e:
        check("unconfigured provider refuses to start a login",
              getattr(e, "status_code", None) == 501)
    try:
        auth.auth_login("myspace", port=8080)
        check("unknown provider -> 404", False)
    except Exception as e:
        check("unknown provider -> 404", getattr(e, "status_code", None) == 404)

    # ---- starting a login redirects to the provider with our registered redirect_uri ---- #
    resp = auth.auth_login("google", port=8080)
    loc = resp.headers["location"]
    check("redirects to Google", loc.startswith(auth.PROVIDERS["google"]["authorize"]))
    check("sends our client_id", "client_id=test-client-id" in loc)
    check("redirect_uri is the coordinator's PUBLIC url, not a guess from the request",
          "coordinator.test%3A8001%2Fauth%2Fcallback%2Fgoogle" in loc.replace("%2F", "%2F"))
    check("client SECRET is never in a browser-visible redirect", "test-client-secret" not in loc)
    state = [k for k in auth._pending][0]

    # ---- the provider comes back -> wallet minted, one-time code handed to loopback ---- #
    real_get, real_post = auth.requests.get, auth.requests.post
    auth.requests.post = lambda *a, **k: FakeResp({"access_token": "tok-abc"})
    auth.requests.get = lambda *a, **k: FakeResp(
        {"sub": "google-user-1", "email": "user@example.com", "email_verified": True})
    try:
        cb = auth.auth_callback("google", code="prov-code", state=state)
    finally:
        auth.requests.get, auth.requests.post = real_get, real_post

    target = cb.headers["location"]
    check("hands back to LOOPBACK only", target.startswith("http://127.0.0.1:8080/auth/adopt?"))
    handback = target.split("code=")[1]

    wallet, _ = models.wallet_for_oauth("google", "google-user-1")
    check("the wallet_id itself never appears in the redirect URL", wallet not in target)
    check("hand-back code is not the wallet_id", handback != wallet)
    check("login minted a real OAuth-backed wallet", models.is_oauth_wallet(wallet))
    ident = models.list_identities()[0]
    check("provider's email_verified claim is recorded for abuse review",
          ident["email"] == "user@example.com" and ident["email_verified"] == 1)

    # ---- redeeming the code ---- #
    got = auth.auth_exchange({"code": handback})
    check("exchange returns the wallet", got["wallet_id"] == wallet)
    check("exchange returns the email", got["email"] == "user@example.com")
    try:
        auth.auth_exchange({"code": handback})
        check("a hand-back code is SINGLE USE", False)
    except Exception as e:
        check("a hand-back code is SINGLE USE", getattr(e, "status_code", None) == 400)
    try:
        auth.auth_exchange({"code": "never-issued"})
        check("an unknown code is rejected", False)
    except Exception as e:
        check("an unknown code is rejected", getattr(e, "status_code", None) == 400)

    # ---- a stale/forged state cannot complete a login ---- #
    try:
        auth.auth_callback("google", code="x", state="forged-state")
        check("forged state is rejected", False)
    except Exception as e:
        check("forged state is rejected", getattr(e, "status_code", None) == 400)
    check("state is consumed, so a callback cannot be replayed", state not in auth._pending)

    # ---- the agent cannot name an arbitrary hand-back destination ---- #
    import inspect
    sig = inspect.signature(auth.auth_login)
    check("login takes a PORT, never a redirect URL (no open-redirect surface)",
          "port" in sig.parameters and not any(
              n in sig.parameters for n in ("redirect_uri", "return_url", "next")))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
