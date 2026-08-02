"""agent/test_updater.py — run: python -m agent.test_updater

The updater is the only path by which a fix ever reaches a stranger, and it is also the only
code that downloads an executable and runs it on their machine unattended. Both halves are
tested here, but the refusals matter more than the happy path: a bug in "should I install this"
ships a binary to every volunteer automatically.

Ordered by how badly each ends if broken:
  1. an unverified or mismatched download is never executed;
  2. an update never lands while the node is serving;
  3. nothing the updater does can take the node down.
"""
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import updater                       # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class FakeResp:
    def __init__(self, payload=None, status=200, body=b"", raises=None):
        self._payload, self.status_code, self._body = payload, status, body
        self._raises = raises

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise updater.requests.HTTPError(str(self.status_code))

    def iter_content(self, n):
        yield self._body

    def __enter__(self):
        if self._raises:
            raise self._raises
        return self

    def __exit__(self, *a):
        return False


def fake_get(payload=None, status=200, body=b"", raises=None):
    """One stub serving two very different calls: the JSON version check and the binary
    download. Keying on the URL keeps a single check_once() run coherent -- an earlier version
    handed the empty JSON body to the downloader and the happy path 'failed' for the wrong
    reason entirely."""
    def _get(url, timeout=None, stream=False, **kw):
        if raises:
            raise raises
        if url.endswith("/agent/version"):
            return FakeResp(payload, status, b"")
        return FakeResp(None, status, body)
    return _get


def main():
    real_get, real_frozen = updater.requests.get, updater.is_frozen
    tmp = tempfile.mkdtemp(prefix="neuron-upd-")

    print("\n-- version comparison")
    check("0.17.0 is newer than 0.16.5", updater.is_newer("0.17.0", "0.16.5"))
    check("0.16.5 is not newer than 0.17.0", not updater.is_newer("0.16.5", "0.17.0"))
    check("equal is not newer", not updater.is_newer("0.17.0", "0.17.0"))
    check("garbage is never newer", not updater.is_newer("banana", "0.17.0"))
    check("an empty remote version is never newer", not updater.is_newer("", "0.17.0"))

    print("\n-- rule 3: a broken coordinator never decides anything")
    try:
        updater.requests.get = fake_get(raises=updater.requests.ConnectionError("down"))
        check("unreachable -> None, not 'up to date'", updater.remote_info("http://c") is None)
        check("check_once reports it and stops",
              updater.check_once("http://c") == "unreachable")
        updater.requests.get = fake_get(payload=None)
        check("non-JSON -> None", updater.remote_info("http://c") is None)
        updater.requests.get = fake_get(payload={"nope": 1})
        check("missing version field -> None", updater.remote_info("http://c") is None)
        updater.requests.get = fake_get(payload={"version": "0.18.0"})
        info = updater.remote_info("http://c")
        check("a version-only response still parses (older coordinators)",
              info["version"] == "0.18.0" and info["sha256"] == "")
    finally:
        updater.requests.get = real_get

    print("\n-- rule 1: nothing unverified is ever executed")
    body = b"pretend installer bytes"
    good = hashlib.sha256(body).hexdigest()
    try:
        updater.requests.get = fake_get(body=body)
        check("no URL -> refused", updater.download_and_verify("", good, tmp) is None)
        check("no published hash -> refused",
              updater.download_and_verify("http://x/s.exe", "", tmp) is None)
        bad = "0" * 64
        got = updater.download_and_verify("http://x/setup.exe", bad, tmp)
        check("a hash mismatch -> refused", got is None)
        check("and the rejected file is deleted, not left lying around",
              not os.path.exists(os.path.join(tmp, "setup.exe")))
        got = updater.download_and_verify("http://x/setup.exe", good, tmp)
        check("a matching hash -> accepted", got is not None and os.path.exists(got))
        check("and the bytes on disk are what was hashed",
              open(got, "rb").read() == body)
        updater.requests.get = fake_get(raises=updater.requests.ConnectionError("cut"))
        check("a failed download -> refused, not a partial install",
              updater.download_and_verify("http://x/s2.exe", good, tmp) is None)
    finally:
        updater.requests.get = real_get

    print("\n-- rule 2: never mid-request")
    try:
        updater.requests.get = fake_get(payload={
            "version": "9.9.9", "download_url": "http://x/setup.exe", "sha256": good},
            body=body)
        updater.is_frozen = lambda: True
        check("busy -> deferred, nothing downloaded",
              updater.check_once("http://c", busy=lambda: True) == "deferred-busy")

        def exploding_busy():
            raise RuntimeError("cannot tell")
        check("a busy-check that raises is treated as BUSY, not idle",
              updater.check_once("http://c", busy=exploding_busy) == "deferred-busy")

        print("\n-- the happy path, with the install stubbed")
        applied = {}
        real_apply = updater.apply_update
        updater.apply_update = lambda p, exit_after=True: applied.setdefault("path", p) or True
        try:
            verdict = updater.check_once("http://c", busy=lambda: False, dest_dir=tmp,
                                         exit_after=False)
        finally:
            updater.apply_update = real_apply
        check("idle + newer + verified -> installs", verdict == "installing", verdict)
        check("and it installed the file it verified",
              applied.get("path", "").endswith("setup.exe"), str(applied))

        print("\n-- a source checkout is never overwritten")
        updater.is_frozen = lambda: False
        check("source mode refuses to self-modify",
              updater.check_once("http://c", busy=lambda: False) == "source-manual")

        print("\n-- already current")
        updater.is_frozen = lambda: True
        updater.requests.get = fake_get(payload={"version": updater.LOCAL_VERSION})
        check("same version -> nothing happens",
              updater.check_once("http://c", busy=lambda: False) == "current")
    finally:
        updater.requests.get = real_get
        updater.is_frozen = real_frozen

    print("\n-- the busy signal itself")
    from agent import node_server
    node_server._serving_conns = 0
    node_server._last_activity = 0.0
    check("idle node is not busy", node_server.is_busy() is False)
    node_server._serving_enter()
    check("an open connection is busy", node_server.is_busy() is True)
    node_server._serving_exit()
    check("still busy right after it closes (grace period covers token round trips)",
          node_server.is_busy() is True)
    check("and idle again once the grace period has passed",
          node_server.is_busy(idle_seconds=0) is False)
    node_server._serving_enter(); node_server._serving_enter(); node_server._serving_exit()
    check("nested connections are counted, not toggled", node_server.is_busy() is True)
    node_server._serving_exit()
    node_server._serving_conns = 0

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
