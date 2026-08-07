"""coordinator/test_nodelogs.py — run: python -m coordinator.test_nodelogs

The load-bearing tests here are the redaction ones. This feature takes a file off a volunteer's
personal computer and puts it on a server and then on an operator's screen; if it ever carries a
live node token or somebody's name out with it, the feature is a liability rather than a
diagnostic. Everything else is bookkeeping.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_logs_"), "n.db")

from coordinator import models, nodelogs  # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def main():
    models.init_db()
    nodelogs.init()

    # ---- redaction: nothing secret survives ---------------------------------- #
    secret = "8d10177dec7380c12298aecf00608bdcbbfe8830ab07ffe1"          # a real-shaped token
    cases = [
        (f'registering with "node_token": "{secret}"', secret),
        (f"X-Node-Token: {secret}", secret),
        (f"Authorization: Bearer {secret}", secret),
        (f"X-Register-Secret: {secret}", secret),
        (f"relay ticket={secret} accepted", secret),
        (f'{{"api_key": "{secret}"}}', secret),
        (f"payout key 0x{'ab' * 32}", "ab" * 32),
    ]
    for text, leaked in cases:
        out = nodelogs.redact(text)
        check(f"redacts: {text[:38]}...", leaked not in out)

    check("a windows home directory does not name the person",
          "optin" not in nodelogs.redact(r"loading C:\Users\optin\neuron\agent\config.json"))
    check("...nor a linux one",
          "ubuntu" not in nodelogs.redact("reading /home/ubuntu/neuron/coordinator/neuron.db"))
    check("but the surrounding text still reads",
          "config.json" in nodelogs.redact(r"loading C:\Users\optin\neuron\agent\config.json"))
    check("ordinary log lines are untouched",
          nodelogs.redact("heartbeat ok - active, layers 0-13") ==
          "heartbeat ok - active, layers 0-13")
    check("empty input is safe", nodelogs.redact(None) == "" and nodelogs.redact("") == "")

    # ---- tail: bounded, and cut at a line boundary ---------------------------- #
    big = "".join(f"line {i}\n" for i in range(20000))
    t = nodelogs.tail_bytes(big, limit=1000)
    check("tail is capped", len(t.encode()) <= 1000)
    check("tail keeps the END of the file, which is where the failure is",
          t.rstrip().endswith("line 19999"))
    check("tail starts at a line boundary, not mid-line", t.split("\n")[0].startswith("line "))
    check("a short file is returned whole", nodelogs.tail_bytes("abc\n", limit=1000) == "abc\n")

    # ---- request / clear ------------------------------------------------------ #
    check("nothing is wanted initially", nodelogs.wanted("n1") is False)
    nodelogs.request(["n1", "n2"])
    check("a request is visible to the node", nodelogs.wanted("n1") and nodelogs.wanted("n2"))
    check("pending lists them", nodelogs.pending() == ["n1", "n2"])
    check("requesting twice does not duplicate", nodelogs.request(["n1"]) == ["n1"]
          and nodelogs.pending() == ["n1", "n2"])
    nodelogs.clear("n2")
    check("clear removes just that one", nodelogs.pending() == ["n1"])

    # ---- store: redacts server-side too, and stops the upload loop ------------- #
    got = nodelogs.store("n1", f"starting up\nX-Node-Token: {secret}\ndone\n")
    check("storing clears the request, so the node uploads once and stops",
          nodelogs.wanted("n1") is False)
    body = nodelogs.get("n1")["body"]
    check("the secret is not in the database even though the node sent it",
          secret not in body)
    check("...and the rest of the log survived", "starting up" in body and "done" in body)
    check("store reports what it kept", got["lines"] == 3 and got["bytes"] > 0)

    over = nodelogs.store("n1", "".join(f"x{i}\n" for i in range(200000)))
    check("an oversized upload is capped server-side, not trusted to have been capped",
          over["bytes"] <= nodelogs.MAX_BYTES)

    # ---- reading -------------------------------------------------------------- #
    check("get returns None for a node that never uploaded", nodelogs.get("nope") is None)
    nodelogs.store("n2", "second node\n")
    s = nodelogs.summary()
    check("summary lists every node holding a log", {r["node_id"] for r in s} == {"n1", "n2"})
    check("summary carries no bodies", all("body" not in r for r in s))

    # ---- one row per node: this is a diagnostic, not an archive ---------------- #
    nodelogs.store("n2", "third upload\n")
    check("a new upload replaces the old one", nodelogs.get("n2")["body"] == "third upload\n")
    check("...and does not add a row", len(nodelogs.summary()) == 2)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
