"""Feedback typed in the app reaches a human — run:
    python -m coordinator.test_feedback_reaches_someone

The product had no way to tell anybody anything. Each reply carried a thumbs up/down, and
`handleFeedbackMessage` wrote it to the sender's own browser storage and stopped there — so
every opinion anyone has ever formed about this product exists only on the machine that formed
it. A rating nobody can read is a control that pretends to listen.

This relays a typed report to the project's Discord. Three things are pinned here, and the
first two are about what must NOT happen:

  * a webhook is a WRITE CREDENTIAL for the project's chat server, so it lives on the
    coordinator and never in the installed app, where anyone could read it out of the bundle
    and post as NEURON;
  * anything credential-shaped is redacted before it leaves. People paste whatever is on
    screen into a feedback box, and the wallet id is a SPENDING key the UI has to show them —
    posting one into a chat room nobody can un-see is a harm the sender never intended;
  * an unconfigured relay says so (503) rather than swallowing the message, because the UI
    turns that into the Discord invite and the report still gets somewhere;
  * IT IS NOT ANONYMOUS AND IT IS NOT UNLIMITED. A public write pipe into a chat room humans
    read is owned by the first person who finds it. Feedback requires a real Google/GitHub
    login and each account gets a small hourly quota — the global per-IP limiter is a DDoS
    guard at 120 requests a minute, which would wave through 7,200 messages an hour.

Posted as an EMBED rather than a line of text: feedback is read in a hurry, on a phone, mixed
into a conversation, so it needs a coloured bar to find and labelled fields instead of a
run-on. Purely a readability change — the previous plain-text format was checked in the real
channel and rendered correctly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import HTTPException  # noqa: E402

from coordinator import config, main as co  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeResp:
    status_code = 204

    def raise_for_status(self):
        pass


W = None            # a real OAuth-linked wallet, minted below


def main():
    global W
    real_post, real_hook = co.requests.post, config.DISCORD_WEBHOOK_URL
    real_quota = config.FEEDBACK_PER_HOUR
    co.models.init_db()
    W, _ = co.models.wallet_for_oauth("test", "feedback-user", "feedback@example.com")
    config.FEEDBACK_PER_HOUR = 1000        # the quota gets its own test at the end
    co._feedback_hits.clear()
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return FakeResp()

    try:
        # -- the relay being off is an answer, not a black hole ------------------------ #
        config.DISCORD_WEBHOOK_URL = ""
        try:
            co.feedback(co.FeedbackBody(text="the sidebar is too narrow", wallet_id=W))
            check("an unconfigured relay refuses rather than silently dropping it", False)
        except HTTPException as e:
            check("an unconfigured relay refuses rather than silently dropping it",
                  e.status_code == 503 and "not configured" in str(e.detail))

        config.DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/test/hook"
        co.requests.post = fake_post

        # -- empty is refused before anything is posted -------------------------------- #
        sent.clear()
        try:
            co.feedback(co.FeedbackBody(text="   ", wallet_id=W))
            check("empty feedback is refused", False)
        except HTTPException as e:
            check("empty feedback is refused", e.status_code == 400)
        check("...and nothing was posted for it", not sent)

        # -- an ordinary report gets through ------------------------------------------- #
        sent.clear()
        out = co.feedback(co.FeedbackBody(text="the wallet panel is much better now",
                                          category="UI", wallet_id=W))
        check("a report is delivered", out.get("delivered") is True)
        check("...to the configured webhook", sent["url"] == config.DISCORD_WEBHOOK_URL)
        emb = sent["body"]["embeds"][0]
        check("...carrying the text", "much better now" in emb["description"])
        check("...and the category as the title, so it can be triaged", emb["title"] == "UI")

        # -- THE ONE THAT MATTERS: a pasted credential never leaves --------------------- #
        wallet = "wallet_9f3c7a21b8e64d5fa0c1e7b2d4839af6"
        sha = "2c7820d063d16ed53039ec0a42e0ddfbce5beb36d7c7360654beb322bfdd2721"
        sent.clear()
        co.feedback(co.FeedbackBody(text=f"my id is {wallet} and the hash was {sha}", wallet_id=W))
        body = sent["body"]["embeds"][0]["description"]
        check("a pasted wallet id is redacted", wallet not in body)
        check("a pasted 64-hex secret is redacted", sha not in body)
        check("...and the rest of the sentence survives, so the report is still readable",
              "my id is" in body and "the hash was" in body)

        # -- context is whitelisted, not forwarded wholesale --------------------------- #
        sent.clear()
        co.feedback(co.FeedbackBody(text="slow today", wallet_id=W, context={
            "version": "0.20.23", "platform": "win32", "nodes_online": 3, "is_node": True,
            "wallet_id": wallet, "prompt": "something private"}))
        emb = sent["body"]["embeds"][0]
        fields = {f["name"]: f["value"] for f in emb.get("fields", [])}
        check("declared context fields are included", fields.get("Version") == "0.20.23")
        check("...and undeclared ones are dropped, whatever the client sent",
              not any("wallet" in n.lower() for n in fields)
              and "something private" not in repr(fields))
        # Keys are relabelled for a human reader rather than dumped as identifiers.
        check("context keys are given readable names", "Runs a node" in fields)

        # -- Discord's 2000-char limit is handled here, not by a failed post ------------ #
        sent.clear()
        out = co.feedback(co.FeedbackBody(text="x" * (config.FEEDBACK_MAX_CHARS + 500), wallet_id=W))
        check("an over-long report is truncated rather than lost", out.get("truncated") is True)
        check("...and says so in the message",
              "(truncated)" in sent["body"]["embeds"][0]["description"])

        # -- the invite is served, so it can be rotated without a release -------------- #
        check("the community endpoint hands out the invite",
              str(co.community().get("discord", "")).startswith("https://discord.gg/"))
        # -- ANONYMOUS IS REFUSED. Without this the endpoint is a public write pipe into
        #    the project's chat room, and one person can send a thousand messages. ---------- #
        for who, label in ((None, "no wallet at all"), ("", "an empty wallet"),
                           ("wallet_invented_by_the_caller", "a wallet nobody logged into")):
            try:
                co.feedback(co.FeedbackBody(text="spam", wallet_id=who))
                check(f"{label} is refused", False)
            except HTTPException as e:
                check(f"{label} is refused", e.status_code == 403)

        # -- and a signed-in account still cannot flood it ----------------------------- #
        config.FEEDBACK_PER_HOUR = 3
        co._feedback_hits.clear()
        sent.clear()
        codes = []
        for i in range(5):
            try:
                co.feedback(co.FeedbackBody(text=f"report {i}", wallet_id=W))
                codes.append(200)
            except HTTPException as e:
                codes.append(e.status_code)
        check("the first reports go through", codes[:3] == [200, 200, 200])
        check("...and the rest are refused with 429, not silently dropped",
              codes[3:] == [429, 429])

        # -- the sender is identifiable in the channel WITHOUT exposing them ----------- #
        config.FEEDBACK_PER_HOUR = 1000
        co._feedback_hits.clear()
        sent.clear()
        co.feedback(co.FeedbackBody(text="who sent this?", wallet_id=W))
        foot = sent["body"]["embeds"][0]["footer"]["text"]
        check("the embed says which sender it came from", foot.startswith("from "))
        check("...as a short hash, never the wallet id itself", W not in foot and len(foot) < 20)
    finally:
        co.requests.post, config.DISCORD_WEBHOOK_URL = real_post, real_hook
        config.FEEDBACK_PER_HOUR = real_quota

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
