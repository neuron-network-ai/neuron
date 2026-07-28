"""ui/test_moderation_gate.py — the Chat UI's input moderation gate never lets a blocked
prompt reach DRIVER (no real model load, no node dispatch). Run: python -m ui.test_moderation_gate
"""
from fastapi.testclient import TestClient

import neuron_driver
from ui.app import app

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    calls = []
    real_stream = neuron_driver.DRIVER.stream
    neuron_driver.DRIVER.stream = lambda *a, **k: calls.append("stream") or iter(())
    client = TestClient(app)   # no `with` -- avoids triggering lifespan's real model load
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
            check("benign prompt reaches DRIVER.stream", benign_calls == ["stream"])
            check("benign prompt does not carry the policy-violation error",
                  "content_policy_violation" not in r2.text)
        finally:
            neuron_driver.DRIVER.encode_chat = real_encode_chat
    finally:
        neuron_driver.DRIVER.stream = real_stream

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
