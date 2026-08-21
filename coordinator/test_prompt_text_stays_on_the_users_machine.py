"""The prompt text never reaches the coordinator — run:
    python -m coordinator.test_prompt_text_stays_on_the_users_machine

PRIVACY.md tells anyone who reads it that the coordinator cannot read their prompt. That was
true of STORAGE — `models.create_request` keeps `prompt_len`, a character count, and stopped
writing the text — and false of TRANSMISSION: `node_a.coord_get_chain` put the whole prompt in
the body of every /infer, so the user's words arrived at this process on every single network
request. Nothing read them. Both uses were `len(body.prompt)`.

"Nothing reads it" is not the same promise as "it is not sent". A field that arrives is in this
process's memory, in any core dump taken of it, and in any proxy in front of it that logs
bodies — and none of that is visible to the person who read the promise.

So the length is computed on the user's own machine and the text stays there. Two directions
matter and both are asserted here:

  * a request carrying `prompt_chars` and NO text is priced and recorded exactly as before;
  * a request from a driver already installed, which still sends text, keeps working — the
    coordinator can therefore be deployed BEFORE the release that stops sending it, which is
    the only safe order.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import node_a  # noqa: E402
from coordinator import config, ledger, models  # noqa: E402
from coordinator.main import InferBody  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeResp:
    status_code = 200

    def json(self):
        return {"chain": [], "request_id": "r-1", "complete_token": "t", "hold_amount": 1.0}


def main():
    # -- the driver: what actually leaves the user's machine ---------------------------- #
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, body=json)
        return FakeResp()

    real_post = node_a.requests.post
    node_a.requests.post = fake_post
    try:
        try:
            node_a.coord_get_chain("http://co", "the secret question I typed", 50,
                                   expected_s1=None, wallet_id="w-1",
                                   prompt_tokens_estimate=7)
        except Exception:                                            # noqa: BLE001
            pass          # the empty chain fails downstream; the POST body is what is on trial
    finally:
        node_a.requests.post = real_post

    body = sent.get("body") or {}
    check("the driver posts no prompt field at all", "prompt" not in body)
    check("no value in the body contains the prompt text",
          not any(isinstance(v, str) and "secret question" in v for v in body.values()))
    check("it sends the LENGTH instead, computed on this machine",
          body.get("prompt_chars") == len("the secret question I typed"))
    check("and still sends the real tokenizer count for the cost hold",
          body.get("prompt_tokens_estimate") == 7)

    # -- the coordinator: same pricing, from a length instead of a string ---------------- #
    prompt = "how do I make a sourdough starter?"
    new = InferBody(prompt_chars=len(prompt), max_tokens=64, wallet_id="w-x")
    old = InferBody(prompt=prompt, max_tokens=64, wallet_id="w-x")
    check("a body with no prompt text is accepted at all", new.prompt is None)

    def chars_of(b):
        return b.prompt_chars if b.prompt_chars is not None else len(b.prompt or "")

    check("prompt_chars and the old text agree on the character count",
          chars_of(new) == chars_of(old) == len(prompt))
    check("...so the hold quoted from each is identical",
          ledger.quote(new.max_tokens, max(1, chars_of(new) // 3))
          == ledger.quote(old.max_tokens, max(1, chars_of(old) // 3)))

    # A driver that knows its real token count overrides the char/3 estimate either way.
    est = InferBody(prompt_chars=len(prompt), max_tokens=64, wallet_id="w-x",
                    prompt_tokens_estimate=9)
    check("a driver's real tokenizer count still wins over the character estimate",
          est.prompt_tokens_estimate == 9)

    # -- what is recorded is a NUMBER, and there is nowhere for text to land ------------- #
    models.init_db()
    rid = f"r-privacy-{uuid.uuid4().hex[:8]}"     # unique: this runs against a real dev DB
    models.create_request(rid, len(prompt), 64, ["n-1"], "tok",
                          wallet_id="w-x", hold_amount=0.0)
    row = None
    with models._db() as c:                                          # noqa: SLF001
        cur = c.execute("SELECT prompt, prompt_len FROM requests WHERE request_id = ?", (rid,))
        row = cur.fetchone()
    check("the stored row holds the character count", row is not None and row[1] == len(prompt))
    check("...and no prompt text, even when the column still exists for old rows",
          row is not None and (row[0] is None or row[0] == ""))

    # -- the guard that keeps this from quietly coming back ------------------------------ #
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
               encoding="utf-8").read()
    infer_src = src[src.index("def infer(body: InferBody):"):]
    infer_src = infer_src[:infer_src.index("\n@app.")]
    uses = [ln.strip() for ln in infer_src.splitlines()
            if "body.prompt" in ln and "prompt_chars" not in ln and "prompt_tokens" not in ln]
    check("/infer reads body.prompt in exactly one place — the compatibility fallback",
          len(uses) == 1 and "len(body.prompt or" in uses[0])

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
