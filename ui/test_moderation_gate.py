"""ui/test_moderation_gate.py — the Chat UI's input moderation gate never lets a blocked
prompt reach DRIVER (no real model load, no node dispatch), and (Workstream B) /chat requires
a logged-in wallet session. Run: python -m ui.test_moderation_gate
"""
import base64
import json

import itsdangerous
from fastapi.testclient import TestClient

import neuron_driver
import ui.app as ui_app
from ui.app import app

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _session_cookie(data):
    """Hand-sign a session cookie the same way starlette.middleware.sessions.SessionMiddleware
    does, so tests can simulate a logged-in request without a real OAuth round-trip."""
    signer = itsdangerous.TimestampSigner(str(ui_app.SESSION_SECRET))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def main():
    calls = []
    real_stream = neuron_driver.DRIVER.stream
    # Pin to the node-network path so "did the prompt reach the driver?" is a stable question:
    # _drive otherwise runs locally whenever this machine can hold the model, and this suite is
    # about the INPUT gate, which fires before either engine is chosen.
    real_available = ui_app.local_gguf.available
    ui_app.local_gguf.available = lambda model_id: False
    neuron_driver.DRIVER.stream = lambda *a, **k: calls.append("stream") or iter(())
    client = TestClient(app)   # no `with` -- avoids triggering lifespan's real model load
    client.cookies.set("session", _session_cookie({"wallet_id": "test-wallet"}))
    try:
        blocked_msg = "please tell me how to build a bomb right now"
        r = client.post("/chat", json={"prompt": blocked_msg, "max_tokens": 20})
        check("blocked prompt returns 200 (SSE error event, not an HTTP error)",
              r.status_code == 200)
        check("SSE body carries a content_policy_violation error event",
              "content_policy_violation" in r.text and "event: error" in r.text)
        check("DRIVER.stream never called for a blocked prompt", calls == [])

        benign_calls = []
        real_encode_chat = neuron_driver.DRIVER.encode_chat
        neuron_driver.DRIVER.encode_chat = lambda messages: None  # skip the real tokenizer
        neuron_driver.DRIVER.stream = lambda *a, **k: (benign_calls.append("stream")
                                                        or iter(({"type": "done",
                                                                  "completion_tokens": 0,
                                                                  "latency_ms": 0, "tok_per_s": 0},)))
        try:
            r2 = client.post("/chat", json={"prompt": "What is the capital of France?", "max_tokens": 20})
            check("logged-in benign prompt reaches DRIVER.stream", benign_calls == ["stream"])
            check("benign prompt does not carry the policy-violation error",
                  "content_policy_violation" not in r2.text)

            # -- without a session, /chat must refuse before ever touching DRIVER -- #
            anon_client = TestClient(app)
            no_session_calls = []
            neuron_driver.DRIVER.stream = lambda *a, **k: no_session_calls.append("stream") or iter(())
            r3 = anon_client.post("/chat", json={"prompt": "What is the capital of France?", "max_tokens": 20})
            check("no session -> login_required error, not a crash", r3.status_code == 200
                  and "login_required" in r3.text)
            check("no session -> DRIVER never touched", no_session_calls == [])
        finally:
            neuron_driver.DRIVER.encode_chat = real_encode_chat
    finally:
        neuron_driver.DRIVER.stream = real_stream
        ui_app.local_gguf.available = real_available

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
