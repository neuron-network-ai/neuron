"""agent/test_uninstall_deregister.py — run: python -m agent.test_uninstall_deregister

Uninstall fired a DELETE at the coordinator and ignored everything that came back: exceptions
swallowed by a bare `except: pass`, and every non-2xx counted as success because the status was
never read. Then it printed "NEURON removed. Thank you for contributing..." either way.

That is not cosmetic, because `new_node_id()` mints a fresh random suffix on every install (on
purpose -- it is what stopped hostname collisions locking people out). So a reinstall never
reclaims the old registration: a silently-failed deregistration orphans a node id on the network
permanently, holding a layer range nobody serves. Seen live on the founder's own machine, listed
twice as `agent-<host>` and `agent-<host>-67e4eb` with the chain stuck DEGRADED behind the dead
one.

So the tests are about what the user is TOLD. A failure has to be visible and has to name the
one action that fixes it, because the obvious instinct -- reinstall -- provably does not.
"""
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import uninstall                    # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class FakeResponse:
    def __init__(self, status):
        self.status_code = status


def patched_delete(status=None, raises=None):
    calls = []

    def _delete(url, headers=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        if raises is not None:
            raise raises
        return FakeResponse(status)
    return _delete, calls


CFG = {"coordinator": "https://c.example", "node_id": "agent-host-67e4eb",
       "node_token": "tok"}


def main():
    real_delete = uninstall.requests.delete
    try:
        print("\n-- what counts as success")
        for status, expect_ok in ((200, True), (204, True), (404, True),
                                  (401, False), (403, False), (500, False)):
            uninstall.requests.delete, calls = patched_delete(status)
            got_ok, detail = uninstall._deregister(CFG)
            check(f"HTTP {status} -> {'ok' if expect_ok else 'failure'}", got_ok is expect_ok,
                  f"got {got_ok}: {detail}")
        check("the DELETE goes to the right node with its token",
              calls[0]["url"] == "https://c.example/node/agent-host-67e4eb"
              and calls[0]["headers"]["X-Node-Token"] == "tok", str(calls))

        print("\n-- 404 is success, not an error")
        uninstall.requests.delete, _ = patched_delete(404)
        got_ok, detail = uninstall._deregister(CFG)
        check("a node already gone counts as removed", got_ok and "already gone" in detail)

        print("\n-- 401 names the real cause")
        uninstall.requests.delete, _ = patched_delete(401)
        _, detail = uninstall._deregister(CFG)
        check("it says another copy replaced the token",
              "replaced" in detail and "token" in detail, detail)

        print("\n-- a network failure is a failure, not a shrug")
        uninstall.requests.delete, _ = patched_delete(
            raises=uninstall.requests.RequestException("boom"))
        got_ok, detail = uninstall._deregister(CFG)
        check("an unreachable coordinator returns False", got_ok is False)
        check("and says so", "could not reach" in detail, detail)

        print("\n-- what the user is told")
        d = tempfile.mkdtemp(prefix="neuron-uninstall-")
        cfg_path = os.path.join(d, "config.json")

        def run_uninstall(status):
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(dict(CFG, slice_dir="./model_slice/"), f)
            uninstall.requests.delete, _ = patched_delete(status)
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = uninstall.main(["--config", cfg_path])
            return rc, buf.getvalue()

        rc, out = run_uninstall(200)
        check("a clean uninstall exits 0", rc == 0)
        check("and says it deregistered", "deregistered" in out, out)
        check("with no scary warning", "WARNING" not in out, out)
        check("the config is deleted either way", not os.path.exists(cfg_path))

        rc, out = run_uninstall(401)
        check("a failed deregistration exits non-zero", rc == 1)
        check("it is not reported as a clean removal",
              "COULD NOT DEREGISTER" in out, out)
        check("the warning names the node id that is still listed",
              "agent-host-67e4eb" in out.split("WARNING")[-1], out)
        check("it says reinstalling will NOT fix it -- the instinct that made this worse",
              "Reinstalling will not clear it" in out, out)
        check("and it names the one action that does",
              "X-Register-Secret" in out and "DELETE" in out, out)
        check("local cleanup still happens on a failed deregistration",
              not os.path.exists(cfg_path))
    finally:
        uninstall.requests.delete = real_delete

    # ---------------------------------------------------------------- [P53]
    # What the uninstaller says about MONEY. It deleted config.json — the only record of which
    # node id this machine was — and printed lifetime earnings as a thank-you in the same
    # breath. On 2026-08-18 that ran on the founder's own machine and 33.49 unclaimed NRN
    # became addressable only by an id nothing had written down. A reinstall does not get it
    # back: new_node_id() mints a fresh suffix on purpose.
    real_get = uninstall.requests.get
    real_owner = uninstall._owner_and_balance
    try:
        d = tempfile.mkdtemp()
        cfg_path = os.path.join(d, "config.json")

        def run(owner, balance):
            # Clear the note first, so "no note exists afterwards" means THIS run declined to
            # write one rather than that no run ever has.
            leftover = os.path.join(d, "unclaimed-earnings.json")
            if os.path.exists(leftover):
                os.remove(leftover)
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(dict(CFG, slice_dir="./model_slice/"), f)
            uninstall.requests.delete, _ = patched_delete(200)
            uninstall._owner_and_balance = lambda cfg: (owner, balance)
            buf = io.StringIO()
            with redirect_stdout(buf):
                uninstall.main(["--config", cfg_path])
            return buf.getvalue()

        print("\n-- [P53] unclaimed NRN is named, not thanked away")
        out = run(None, 33.4858)
        check("an unclaimed balance is warned about", "WARNING" in out and "33.49" in out, out)
        check("...it names the node id the money is under",
              "agent-host-67e4eb" in out.split("WARNING")[-1], out)
        check("...it says reinstalling will NOT recover it, which is the instinct",
              "NOT get it back" in out, out)
        check("...and it names the action that would have prevented it",
              "Claim with my account" in out, out)
        note = os.path.join(d, "unclaimed-earnings.json")
        check("...and the node id survives on disk for recovery", os.path.exists(note))
        if os.path.exists(note):
            saved = json.load(open(note, encoding="utf-8"))
            check("......with the id a sweep needs", saved["node_id"] == "agent-host-67e4eb")
            check("......and the balance at the moment it was lost",
                  abs(saved["balance_at_removal"] - 33.4858) < 1e-6)
            check("......and NO token, because deregistration already killed it",
                  "token" not in json.dumps(saved).lower(), json.dumps(saved))

        print("\n-- a CLAIMED node says the opposite, because the opposite is true")
        out = run("w_alice", 33.4858)
        check("a claimed balance raises no warning", "WARNING" not in out, out)
        check("...and says plainly that the money is unaffected",
              "NOT affected by this removal" in out, out)
        check("...and writes no recovery note, because nothing is stranded",
              not os.path.exists(os.path.join(d, "unclaimed-earnings.json")))

        print("\n-- a node that earned nothing is not given a scare")
        out = run(None, 0.0)
        check("a zero balance warns about nothing", "WARNING" not in out, out)
    finally:
        uninstall.requests.get = real_get
        uninstall._owner_and_balance = real_owner

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
