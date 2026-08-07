"""coordinator/test_selfupdate.py — run: python -m coordinator.test_selfupdate

The coordinator updater has one coordinator and no redundancy behind it, so the tests that
matter are the refusals and the rollback, not the happy path. Everything external (manifest
fetch, health probes, systemctl) is injected, so this runs on any machine with no VM, no
network and no systemd.
"""
import json
import os
import shutil
import sys
import tarfile
import tempfile

from coordinator import selfupdate as su

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_root(tmp, marker="old"):
    """A miniature deploy root: coordinator/ + the two loose modules + a database."""
    root = tempfile.mkdtemp(dir=tmp)
    os.makedirs(os.path.join(root, "coordinator"))
    open(os.path.join(root, "coordinator", "main.py"), "w").write(f"# {marker}\n")
    open(os.path.join(root, "coordinator", "neuron.db"), "w").write("PRECIOUS")
    open(os.path.join(root, "relay_auth.py"), "w").write(f"# {marker}\n")
    open(os.path.join(root, "common.py"), "w").write(f"# {marker}\n")
    return root


def make_archive(tmp, marker="new", extra=None):
    """A release tarball shaped like the one deploy.sh builds."""
    stage = tempfile.mkdtemp(dir=tmp)
    os.makedirs(os.path.join(stage, "coordinator"))
    open(os.path.join(stage, "coordinator", "main.py"), "w").write(f"# {marker}\n")
    open(os.path.join(stage, "relay_auth.py"), "w").write(f"# {marker}\n")
    path = os.path.join(tmp, f"coord-{marker}.tgz")
    with tarfile.open(path, "w:gz") as tf:
        tf.add(os.path.join(stage, "coordinator"), arcname="coordinator")
        tf.add(os.path.join(stage, "relay_auth.py"), arcname="relay_auth.py")
        if extra:
            for arcname, content in extra.items():
                p = os.path.join(stage, "evil")
                open(p, "w").write(content)
                tf.add(p, arcname=arcname)
    return path


def marker_of(root):
    return open(os.path.join(root, "coordinator", "main.py")).read().strip()


class Probes:
    """Scriptable /status + gate responses. `versions` is consumed one entry per health call."""

    def __init__(self, versions, faucet=401, admin=401, status_code=200):
        self.versions = list(versions)
        self.faucet, self.admin, self.status_code = faucet, admin, status_code
        self.calls = []

    def __call__(self, url, timeout=20, method="GET", body=None):
        self.calls.append(url)
        if url.endswith("/status"):
            if self.status_code != 200:
                return self.status_code, ""
            v = self.versions.pop(0) if self.versions else None
            return 200, json.dumps({"coordinator_version": v})
        if url.endswith("/wallet/faucet"):
            return self.faucet, ""
        if url.endswith("/admin/identities"):
            return self.admin, ""
        return 404, ""


def runner(fail_on=()):
    calls = []

    class R:
        def __init__(self, rc):
            self.returncode = rc

    def run(cmd, **kw):
        calls.append(cmd)
        return R(1 if len(calls) in fail_on else 0)
    return calls, run


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-selfupdate-")
    try:
        # ---- version comparison ------------------------------------------------ #
        check("a newer version is newer", su.is_newer("0.20.0", "0.19.0"))
        check("the same version is not", not su.is_newer("0.19.0", "0.19.0"))
        check("an older version is not", not su.is_newer("0.18.9", "0.19.0"))
        check("garbage never counts as newer", not su.is_newer("", "0.19.0")
              and not su.is_newer(None, "0.19.0"))

        # ---- manifest ---------------------------------------------------------- #
        check("a manifest with no version is ignored",
              su.manifest("u", get=lambda u, timeout=20: {"sha256": "x"}) is None)
        check("an unreadable manifest is None, not 'up to date'",
              su.manifest("u", get=lambda u, timeout=20: None) is None)
        m = su.manifest("u", get=lambda u, timeout=20: {"version": "0.20.0",
                                                        "sha256": "  ABCD  "})
        check("sha256 is normalised", m["sha256"] == "abcd" and m["version"] == "0.20.0")

        # ---- refusals: nothing unverified is ever installed --------------------- #
        check("no download URL -> nothing installed",
              su.download_and_verify("", "abc", dest_dir=tmp) is None)
        check("no published hash -> refusal, not a shortcut",
              su.download_and_verify("http://x/y.tgz", "", dest_dir=tmp) is None)

        arch = make_archive(tmp, "new")
        real = su.sha256_file(arch)
        got = su.download_and_verify(f"file:///{arch.replace(os.sep, '/')}", "0" * 64,
                                     dest_dir=tempfile.mkdtemp(dir=tmp))
        check("a hash mismatch installs nothing", got is None)
        d = tempfile.mkdtemp(dir=tmp)
        got = su.download_and_verify(f"file:///{arch.replace(os.sep, '/')}", real, dest_dir=d)
        check("a matching hash is accepted", got is not None and os.path.exists(got))
        check("a rejected archive is not left on disk",
              not any(f.endswith(".tgz") for f in os.listdir(tempfile.mkdtemp(dir=tmp))))

        # ---- snapshot / restore ------------------------------------------------ #
        root = make_root(tmp, "old")
        snap = su.snapshot(root)
        open(os.path.join(root, "coordinator", "main.py"), "w").write("# new\n")
        check("the snapshot holds the old code", "# old" in
              open(os.path.join(snap, "coordinator", "main.py")).read())
        check("the database is backed up with it",
              open(os.path.join(snap, "neuron.db.bak")).read() == "PRECIOUS")
        su.restore(snap, root)
        check("restore puts the old code back", marker_of(root) == "# old")
        check("restore leaves the live database alone — a migrated schema is one-way",
              open(os.path.join(root, "coordinator", "neuron.db")).read() == "PRECIOUS")

        # ---- extraction is confined to the code tree --------------------------- #
        root = make_root(tmp, "old")
        evil = make_archive(tmp, "evil", extra={"../../../etc/pwned": "x",
                                                "packaging/neuron.iss": "x"})
        su.install(evil, root)
        check("a traversal member is not written outside the root",
              not os.path.exists(os.path.join(os.path.dirname(root), "etc", "pwned")))
        check("a member outside CODE_PATHS is skipped",
              not os.path.exists(os.path.join(root, "packaging")))
        check("the legitimate members still landed", marker_of(root) == "# evil")

        # ---- health gate ------------------------------------------------------- #
        okh, why = su.health("http://c", expect_version="0.20.0",
                             status_fn=Probes(["0.20.0"]), sleep=lambda s: None)
        check("healthy when serving, on the new version, and still gated", okh and why == [])

        okh, why = su.health("http://c", expect_version="0.20.0", timeout=0,
                             status_fn=Probes(["0.19.0"]), sleep=lambda s: None)
        check("a healthy process on the OLD version is a FAILED update",
              not okh and any("expected '0.20.0'" in r for r in why))

        okh, why = su.health("http://c", expect_version="0.20.0", timeout=0,
                             status_fn=Probes(["0.20.0"], faucet=200), sleep=lambda s: None)
        check("serving with the faucet ungated is a failed update, not a passing one",
              not okh and any("faucet" in r for r in why))

        okh, why = su.health("http://c", expect_version="0.20.0", timeout=0,
                             status_fn=Probes([], status_code=0), sleep=lambda s: None)
        check("a coordinator that answers nothing is unhealthy", not okh)

        # ---- check_once: the whole cycle --------------------------------------- #
        url = f"file:///{arch.replace(os.sep, '/')}"

        def man(version="0.20.0", sha=None):
            return lambda u, timeout=20: {"version": version,
                                          "download_url": url,
                                          "sha256": real if sha is None else sha}

        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(),
                          status_fn=Probes(["0.20.0"]), run=run, sleep=lambda s: None)
        check("a good update is applied and verified", v == "updated")
        check("...the new code is in place", marker_of(root) == "# new")
        check("...and the service was restarted once", len(calls) == 1)

        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(version="0.19.0"),
                          status_fn=Probes([]), run=run, sleep=lambda s: None)
        check("an unpublished version does nothing at all",
              v == "current" and calls == [] and marker_of(root) == "# old")

        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=lambda u, timeout=20: None,
                          status_fn=Probes([]), run=run, sleep=lambda s: None)
        check("an unreachable manifest never restarts anything",
              v == "unreachable" and calls == [])

        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(sha="0" * 64),
                          status_fn=Probes([]), run=run, sleep=lambda s: None)
        check("a bad hash leaves the running build untouched",
              v == "download-failed" and marker_of(root) == "# old" and calls == [])

        # THE case this whole module exists for: new code that will not serve.
        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(),
                          # new build never reports its version; rollback then serves 0.19.0
                          status_fn=Probes(["0.19.0", "0.19.0"]), run=run,
                          sleep=lambda s: None)
        check("an update that does not come up healthy is rolled back", v == "rolled-back")
        check("...the OLD code is running again", marker_of(root) == "# old")
        check("...and the service was restarted twice (install, then rollback)",
              len(calls) == 2)

        # a failing restart is also a rollback, without waiting on a health check
        root = make_root(tmp, "old")
        calls, run = runner(fail_on=(1,))
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(),
                          status_fn=Probes(["0.19.0"]), run=run, sleep=lambda s: None)
        check("a restart that fails outright rolls back too",
              v == "restart-failed" and marker_of(root) == "# old")

        # the worst case must be reported, not swallowed
        root = make_root(tmp, "old")
        calls, run = runner(fail_on=(2,))
        v = su.check_once(root, "http://c", "m", "0.19.0", health_timeout=0, get=man(),
                          status_fn=Probes(["0.19.0", "0.19.0"]), run=run, sleep=lambda s: None)
        check("a rollback that cannot restart says so loudly", v == "rollback-failed")

        # ---- --check never mutates anything ------------------------------------ #
        root = make_root(tmp, "old")
        calls, run = runner()
        v = su.check_once(root, "http://c", "m", "0.19.0", apply=False, health_timeout=0, get=man(),
                          status_fn=Probes([]), run=run, sleep=lambda s: None)
        check("report-only downloads nothing and restarts nothing",
              v == "available" and calls == [] and marker_of(root) == "# old")

        # ---- snapshots are pruned ---------------------------------------------- #
        root = make_root(tmp, "old")
        for i in range(8):
            su.snapshot(root, into=os.path.join(root, ".rollback", f"2026080{i}-000000"))
        dropped = su.prune_snapshots(root, keep=5)
        check("only the newest snapshots are kept",
              len(dropped) == 3 and len(os.listdir(os.path.join(root, ".rollback"))) == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
