"""ui/test_conversations.py — driver-side conversation history. Run: python -m ui.test_conversations

Uses a throwaway DB (env override set before import, same pattern as coordinator's DB-backed
tests) -- no real network, no live coordinator. Covers create/list/get/delete and, critically,
ownership isolation between two different wallets (this is per-wallet private history sitting
on the driver, so a wrong/foreign wallet_id must never see or touch another wallet's data).
"""
import os
import tempfile

os.environ["NEURON_CONVERSATIONS_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="neuron_conversations_"), "c.db")

from ui import conversations  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _fake_clock():
    """A strictly-increasing fake clock for ordering-sensitive assertions. Windows' time.time()
    can have coarse (~15ms) resolution, so two calls a few lines apart in a tight test loop can
    legitimately return the SAME value -- that's real, observed flakiness in this suite, not a
    bug in list_conversations' ORDER BY updated_at DESC (ties are effectively unreachable in
    real usage, where a human takes much longer than 15ms between actions)."""
    state = {"t": 1_700_000_000.0}
    def tick():
        state["t"] += 1.0
        return state["t"]
    return tick


def main():
    real_time = conversations.time.time
    conversations.time.time = _fake_clock()
    try:
        _run_checks()
    finally:
        conversations.time.time = real_time

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


def _run_checks():
    # ---- create + get ---- #
    cid = conversations.create_conversation("w1", "My first chat about sourdough")
    check("create_conversation returns an id", bool(cid))
    conv = conversations.get_conversation(cid, "w1")
    check("get_conversation returns the conversation", conv is not None)
    check("title is truncated/stored", conv["title"] == "My first chat about sourdough")
    check("new conversation starts with no messages", conv["messages"] == [])

    # ---- add_message + ordering ---- #
    check("add_message succeeds for the owning wallet",
          conversations.add_message(cid, "w1", "user", "how do I bake sourdough?"))
    check("add_message succeeds for the assistant turn",
          conversations.add_message(cid, "w1", "assistant", "First, make a starter..."))
    conv2 = conversations.get_conversation(cid, "w1")
    check("messages come back in insertion order",
          [m["role"] for m in conv2["messages"]] == ["user", "assistant"])
    check("message content round-trips exactly",
          conv2["messages"][0]["content"] == "how do I bake sourdough?")

    # ---- list_conversations ---- #
    cid2 = conversations.create_conversation("w1", "second chat")
    listed = conversations.list_conversations("w1")
    check("list_conversations returns both of this wallet's conversations",
          {c["id"] for c in listed} == {cid, cid2})
    check("newly-created cid2 sorts first (most-recently-updated)", listed[0]["id"] == cid2)
    conversations.add_message(cid, "w1", "user", "one more message to touch cid again")
    listed2 = conversations.list_conversations("w1")
    check("touching cid again moves it back to the front", listed2[0]["id"] == cid)

    # ---- ownership isolation: wallet w2 must see NOTHING of w1's data ---- #
    check("a foreign wallet cannot read w1's conversation",
          conversations.get_conversation(cid, "w2") is None)
    check("a foreign wallet's add_message on w1's conversation fails, nothing inserted",
          not conversations.add_message(cid, "w2", "user", "injected"))
    check("the injection attempt did not actually add a message",
          len(conversations.get_conversation(cid, "w1")["messages"]) == 3)
    check("list_conversations for a wallet with nothing returns empty, not an error",
          conversations.list_conversations("w2") == [])

    # ---- delete ---- #
    check("a foreign wallet cannot delete w1's conversation",
          not conversations.delete_conversation(cid, "w2"))
    check("w1's conversation still exists after the failed foreign delete",
          conversations.get_conversation(cid, "w1") is not None)
    check("the owning wallet can delete its own conversation",
          conversations.delete_conversation(cid, "w1"))
    check("deleted conversation is really gone", conversations.get_conversation(cid, "w1") is None)
    check("deleting again is a clean no-op, not an error",
          not conversations.delete_conversation(cid, "w1"))

    # ---- unknown conversation id ---- #
    check("a totally unknown conversation id returns None, not an error",
          conversations.get_conversation("does-not-exist", "w1") is None)


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
