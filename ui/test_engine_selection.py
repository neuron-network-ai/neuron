"""ui/test_engine_selection.py — run: python -m ui.test_engine_selection

Tiered execution: _drive runs the model on THIS machine when it can hold it, and falls back to
the node pipeline when it cannot.

Why the fallback direction matters as much as the fast path: before this, an incomplete chain
meant the Chat UI showed "Responses will fail" and nothing worked. Now a short-staffed network
is only a problem for a machine that cannot serve itself. Conversely, a machine that CANNOT
hold the model must still reach the network -- that is the case only the network can serve, and
it is the whole reason NEURON exists.

Both engines are mocked; no model load, no coordinator.
"""
import base64
import json
import os
import tempfile

os.environ.setdefault("NEURON_SESSION_SECRET", "test-session-secret")
os.environ.setdefault("NEURON_CONVERSATIONS_DB",
                      os.path.join(tempfile.mkdtemp(prefix="neuron_engine_"), "c.db"))

import itsdangerous                       # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import neuron_driver                       # noqa: E402
import ui.app as ui_app                    # noqa: E402
from ui.app import app                     # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _session_cookie(data):
    s = itsdangerous.TimestampSigner(os.environ["NEURON_SESSION_SECRET"])
    return s.sign(base64.b64encode(json.dumps(data).encode())).decode()


def _events(text, local):
    meta = {"type": "meta", "request_id": "r1", "node_ids": [] if local else ["node_a"],
            "nodes": 0 if local else 1, "cost_nrn": 0.0 if local else 0.5}
    if local:
        meta["local"] = True
    return iter((meta, {"type": "token", "text": text},
                 {"type": "done", "text": text, "completion_tokens": 2, "prompt_tokens": 1,
                  "finish_reason": "stop", "latency_ms": 10, "tok_per_s": 5,
                  "cost_nrn": 0.0 if local else 0.5}))


def main():
    real_avail, real_local_stream = ui_app.local_gguf.available, ui_app.local_gguf.stream
    real_stream, real_encode = neuron_driver.DRIVER.stream, neuron_driver.DRIVER.encode_chat
    used = []
    ui_app.local_gguf.stream = lambda *a, **k: used.append("local") or _events("local answer", True)
    neuron_driver.DRIVER.stream = lambda *a, **k: used.append("network") or _events("net answer", False)
    neuron_driver.DRIVER.encode_chat = lambda messages: None

    client = TestClient(app)
    client.cookies.set("session", _session_cookie({"wallet_id": "w_test"}))
    try:
        # ---- machine CAN hold the model -> run here, bill nothing, touch no node ---- #
        used.clear()
        ui_app.local_gguf.available = lambda model_id: True
        r = client.post("/chat", json={"prompt": "hello", "max_tokens": 20})
        check("local-capable machine uses the local engine", used == ["local"])
        check("local answer streams to the browser", "local answer" in r.text)
        check("local run reports no nodes involved", '"nodes": 0' in r.text)
        check("local run costs 0 NRN", '"cost_nrn": 0.0' in r.text)
        check("local run is flagged so the UI can say so", '"local": true' in r.text)

        # ---- machine CANNOT hold it -> the network, which is the point of NEURON ---- #
        used.clear()
        ui_app.local_gguf.available = lambda model_id: False
        r = client.post("/chat", json={"prompt": "hello", "max_tokens": 20})
        check("machine that cannot hold the model falls back to the node network",
              used == ["network"])
        check("network answer streams to the browser", "net answer" in r.text)
        check("network run reports the serving nodes", '"nodes": 1' in r.text)
        check("network run is billed", '"cost_nrn": 0.5' in r.text)

        # ---- /network tells the page which case it is (drives the warning banner) ---- #
        ui_app.local_gguf.available = lambda model_id: True
        body = client.get("/network").json()
        check("/network reports local_capable so the UI can stop crying wolf",
              body.get("local_capable") is True)
        ui_app.local_gguf.available = lambda model_id: False
        check("/network reports local_capable False when only the network can serve",
              client.get("/network").json().get("local_capable") is False)

        # ---- login is still required on BOTH paths (abuse accountability, SAFETY.md) ---- #
        used.clear()
        ui_app.local_gguf.available = lambda model_id: True
        anon = TestClient(app)
        r = anon.post("/chat", json={"prompt": "hello", "max_tokens": 20})
        check("local execution still requires a login", "login_required" in r.text)
        check("...and never reaches any engine", used == [])
    finally:
        ui_app.local_gguf.available, ui_app.local_gguf.stream = real_avail, real_local_stream
        neuron_driver.DRIVER.stream, neuron_driver.DRIVER.encode_chat = real_stream, real_encode

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
