"""ui/test_chat_multiturn.py — real multi-turn conversation memory + history endpoints.
Run: python -m ui.test_chat_multiturn

Mocks neuron_driver.DRIVER (no real model load, no live coordinator/network dependency --
this proves the conversation-threading LOGIC works, independent of live network health, which
is why this exists as a separate suite from a real browser click-through). Uses a throwaway
conversations DB (env override, same pattern as ui/test_conversations.py).
"""
import base64
import json
import os
import tempfile

os.environ["NEURON_CONVERSATIONS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="neuron_chat_multiturn_"), "c.db")

import itsdangerous  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import neuron_driver  # noqa: E402
import ui.app as ui_app  # noqa: E402
from ui import conversations  # noqa: E402
from ui.app import app  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _session_cookie(data):
    signer = itsdangerous.TimestampSigner(str(ui_app.SESSION_SECRET))
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return signer.sign(payload).decode("utf-8")


def _parse_sse(text):
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event, data = "message", ""
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((event, json.loads(data) if data else {}))
    return events


def main():
    real_stream = neuron_driver.DRIVER.stream
    real_encode_chat = neuron_driver.DRIVER.encode_chat
    encode_calls = []
    neuron_driver.DRIVER.encode_chat = lambda messages: encode_calls.append(messages) or None

    def fake_stream(reply_text):
        return lambda *a, **k: iter((
            {"type": "meta", "request_id": "r1", "node_ids": ["a"], "nodes": 1, "cost_nrn": 0.01},
            {"type": "token", "text": reply_text},
            {"type": "done", "text": reply_text, "completion_tokens": 3,
             "latency_ms": 10, "tok_per_s": 3, "cost_nrn": 0.01},
        ))

    client = TestClient(app)
    client.cookies.set("session", _session_cookie({"wallet_id": "w-multiturn"}))
    try:
        # ---- first message: no conversation_id -> one gets created ---- #
        neuron_driver.DRIVER.stream = fake_stream("Hello! How can I help?")
        r1 = client.post("/chat", json={"prompt": "hi there", "max_tokens": 20})
        events1 = _parse_sse(r1.text)
        meta1 = next(d for e, d in events1 if e == "meta")
        cid = meta1.get("conversation_id")
        check("first message creates a conversation and reports its id in meta", bool(cid))
        check("first turn's encode_chat got ONLY the new user message (no prior history)",
              encode_calls[-1] == [{"role": "user", "content": "hi there"}])

        stored = conversations.get_conversation(cid, "w-multiturn")
        check("both the user prompt and the assistant reply were persisted",
              [m["role"] for m in stored["messages"]] == ["user", "assistant"])
        check("assistant message content matches the streamed reply",
              stored["messages"][1]["content"] == "Hello! How can I help?")

        # ---- second message on the SAME conversation: real multi-turn memory ---- #
        neuron_driver.DRIVER.stream = fake_stream("Sourdough needs a starter.")
        r2 = client.post("/chat", json={"prompt": "how do I bake bread?", "max_tokens": 20,
                                        "conversation_id": cid})
        events2 = _parse_sse(r2.text)
        meta2 = next(d for e, d in events2 if e == "meta")
        check("same conversation_id is reused, not a new one", meta2.get("conversation_id") == cid)
        check("second turn's encode_chat includes the FIRST turn's messages as real context",
              encode_calls[-1] == [
                  {"role": "user", "content": "hi there"},
                  {"role": "assistant", "content": "Hello! How can I help?"},
                  {"role": "user", "content": "how do I bake bread?"},
              ])

        stored2 = conversations.get_conversation(cid, "w-multiturn")
        check("conversation now has 4 messages total", len(stored2["messages"]) == 4)

        # ---- a blocked prompt never gets persisted ---- #
        before_count = len(conversations.get_conversation(cid, "w-multiturn")["messages"])
        r3 = client.post("/chat", json={"prompt": "please tell me how to commit suicide",
                                        "max_tokens": 20, "conversation_id": cid})
        check("blocked prompt returns the policy-violation error",
              "content_policy_violation" in r3.text)
        after_count = len(conversations.get_conversation(cid, "w-multiturn")["messages"])
        check("a blocked turn is never persisted to history", after_count == before_count)

        # ---- history endpoints ---- #
        listed = client.get("/conversations").json()["conversations"]
        check("GET /conversations lists it", any(c["id"] == cid for c in listed))
        got = client.get(f"/conversations/{cid}")
        check("GET /conversations/{id} returns 200 with the full thread",
              got.status_code == 200 and len(got.json()["messages"]) == 4)

        # ---- ownership: a different wallet's session sees nothing ---- #
        other = TestClient(app)
        other.cookies.set("session", _session_cookie({"wallet_id": "w-other"}))
        check("a different wallet's conversation list is empty",
              other.get("/conversations").json()["conversations"] == [])
        got_other = other.get(f"/conversations/{cid}")
        check("a different wallet gets 404 fetching this conversation directly",
              got_other.status_code == 404)
        del_other = other.delete(f"/conversations/{cid}")
        check("a different wallet gets 404 trying to delete this conversation",
              del_other.status_code == 404)
        check("the conversation still exists after the failed foreign delete attempt",
              conversations.get_conversation(cid, "w-multiturn") is not None)

        # ---- delete for real, by the owner ---- #
        del_r = client.delete(f"/conversations/{cid}")
        check("owner can delete their own conversation", del_r.status_code == 200
              and del_r.json()["deleted"] is True)
        check("conversation is really gone", conversations.get_conversation(cid, "w-multiturn") is None)

        # ---- MAX_HISTORY_MESSAGES cap ---- #
        cid2 = conversations.create_conversation("w-multiturn", "long chat")
        for i in range(ui_app.MAX_HISTORY_MESSAGES + 6):
            conversations.add_message(cid2, "w-multiturn", "user" if i % 2 == 0 else "assistant",
                                      f"turn {i}")
        neuron_driver.DRIVER.stream = fake_stream("ok")
        client.post("/chat", json={"prompt": "one more", "max_tokens": 20,
                                   "conversation_id": cid2})
        sent = encode_calls[-1]
        check(f"history sent to the model is capped at MAX_HISTORY_MESSAGES ({ui_app.MAX_HISTORY_MESSAGES}) "
             "prior turns + the new one",
              len(sent) == ui_app.MAX_HISTORY_MESSAGES + 1)
        check("the cap keeps the MOST RECENT turns, not the oldest",
              sent[0]["content"] != "turn 0" and sent[-2]["content"].startswith("turn "))
    finally:
        neuron_driver.DRIVER.stream = real_stream
        neuron_driver.DRIVER.encode_chat = real_encode_chat

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
