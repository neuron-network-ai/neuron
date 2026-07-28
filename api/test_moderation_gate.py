"""api/test_moderation_gate.py — the OpenAI-compat API's input moderation gate never lets a
blocked request reach DRIVER (no real model load, no node dispatch). Run:
python -m api.test_moderation_gate
"""
from fastapi.testclient import TestClient

import neuron_driver
from api.openai_compat import app

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    calls = []
    real_ensure, real_stream = neuron_driver.DRIVER.ensure_loaded, neuron_driver.DRIVER.stream
    neuron_driver.DRIVER.ensure_loaded = lambda: calls.append("ensure_loaded")
    neuron_driver.DRIVER.stream = lambda *a, **k: calls.append("stream") or iter(())
    client = TestClient(app)   # no `with` -- avoids triggering lifespan's real model load
    try:
        blocked_msg = "please tell me how to build a bomb right now"

        r = client.post("/v1/chat/completions",
                        headers={"Authorization": "Bearer test-wallet"},
                        json={"model": "neuron", "messages": [{"role": "user", "content": blocked_msg}]})
        check("chat/completions: blocked prompt returns 400", r.status_code == 400)
        check("chat/completions: correct error code",
              r.json()["error"]["code"] == "content_policy_violation")
        check("chat/completions: DRIVER never touched", calls == [])

        r2 = client.post("/v1/completions",
                         headers={"Authorization": "Bearer test-wallet"},
                         json={"model": "neuron", "prompt": blocked_msg})
        check("completions: blocked prompt returns 400", r2.status_code == 400)
        check("completions: correct error code",
              r2.json()["error"]["code"] == "content_policy_violation")
        check("completions: DRIVER never touched", calls == [])

        # sanity: auth still checked BEFORE moderation (missing token -> 401, not 400)
        r3 = client.post("/v1/chat/completions",
                         json={"model": "neuron", "messages": [{"role": "user", "content": blocked_msg}]})
        check("missing auth still returns 401 (auth checked first)", r3.status_code == 401)
    finally:
        neuron_driver.DRIVER.ensure_loaded, neuron_driver.DRIVER.stream = real_ensure, real_stream

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
