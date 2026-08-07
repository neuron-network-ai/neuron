"""
coordinator/selfupdate.py — bring the coordinator up to a PUBLISHED version, or put it back.

Every coordinator fix so far has reached production only when the founder was at their machine
to run `deploy.sh`. The self-heal work in [R6]/[P25] is the example that made this urgent: the
network sat DEGRADED for days with a fix written and tested, because shipping it required a
human. A network that can heal itself but cannot receive the code that heals it is only half
autonomous.

The agent has had this since Session 9 (`agent/updater.py`) and the three rules there apply
unchanged — never install an unverified binary, never update at a bad moment, never let the
updater take the thing down. But the blast radius here is categorically different. A bad agent
update costs one volunteer's machine out of several. A bad coordinator update costs the entire
network, and there is exactly one coordinator, so nothing is left to notice or repair it. That
difference is what the rest of this file is about.

**Runs OUTSIDE the coordinator process.** A systemd timer, not a thread in `main.py`. An updater
living inside the thing it updates dies with it: the one scenario the rollback exists for --
new code that will not serve -- is precisely the scenario where an in-process updater is no
longer running to perform the rollback.

**The manifest is not served by the coordinator.** `agent/updater.py` reads `/agent/version`
from the coordinator, which is right for a node; it would be circular here. A broken coordinator
cannot advertise its own replacement, and a bad deploy able to rewrite the manifest could erase
the only record of what the version was supposed to be. So it comes from a static file in the
repo (`config.COORDINATOR_MANIFEST_URL`) and publishing is a git push.

**Nothing moves until you publish.** The version in the manifest is compared against the running
build. No manifest, no SHA-256, or a hash mismatch means nothing is installed -- an empty hash
is a refusal, never a shortcut.

**Health is proved, not assumed, and failure is undone.** After the restart the updater checks
that /status answers, that it reports the version we just installed, and that the auth gates
still reject anonymous callers -- the same gates `deploy.sh` verifies, because a coordinator
serving happily with `/wallet/faucet` wide open is a worse outcome than one that is down. Any
failure restores the snapshot taken before the swap and restarts again. The database is copied
aside first and never written to by this module.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

# Directories this updater is allowed to replace, relative to the deploy root. Same set
# deploy.sh ships, and deliberately explicit: an update must never be able to write outside
# the code tree, and never touch neuron.db.
CODE_PATHS = ("coordinator", "relay_auth.py", "common.py")
DB_RELPATH = os.path.join("coordinator", "neuron.db")
SERVICE = "neuron-coordinator"
HEALTH_TIMEOUT_S = 90
HEALTH_POLL_S = 3
KEEP_SNAPSHOTS = 5


def log(msg):
    """One stream, timestamped. This runs under systemd, so stdout IS the journal."""
    print(f"[selfupdate {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# version comparison — same semantics as the agent updater
# --------------------------------------------------------------------------- #
def _parse(v):
    return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def is_newer(remote, local):
    try:
        r, l = _parse(remote), _parse(local)
        return bool(r) and r > l
    except Exception:                                           # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def http_json(url, timeout=20):
    """GET a JSON document, or None. None means "could not ask", never "up to date"."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:      # noqa: S310
            return json.loads(r.read().decode())
    except Exception as e:                                      # noqa: BLE001
        log(f"manifest unreadable ({e.__class__.__name__}: {e}) — staying put")
        return None


def http_status(url, timeout=20, method="GET", body=None):
    """Return (status_code, body_text). A refused connection is status 0, not an exception:
    every caller here treats 'no answer' as a health failure, not a crash."""
    req = urllib.request.Request(url, method=method)              # noqa: S310
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as r:   # noqa: S310
            return r.status, r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except Exception:                                           # noqa: BLE001
        return 0, ""


def manifest(url, get=http_json):
    """{'version', 'download_url', 'sha256'} or None. A manifest missing a version is treated
    as unreadable rather than as a downgrade instruction."""
    data = get(url)
    if not isinstance(data, dict) or not data.get("version"):
        if data is not None:
            log("manifest carried no version — ignoring it")
        return None
    return {"version": str(data["version"]),
            "download_url": data.get("download_url") or "",
            "sha256": (data.get("sha256") or "").strip().lower()}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download_and_verify(url, expected_sha256, dest_dir=None, timeout=600):
    """Fetch `url`; return its path only if it hashes to `expected_sha256`.

    A missing expected hash is a refusal (see the module docstring). A mismatch deletes the file
    rather than leaving a rejected archive on disk for something else to unpack.
    """
    if not url:
        log("a version is published but no download URL — nothing to install")
        return None
    if not expected_sha256:
        log("a version is published but no SHA-256 — refusing to install an unverified build")
        return None
    dest_dir = dest_dir or tempfile.mkdtemp(prefix="neuron-coord-update-")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]) or "coordinator.tgz")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(path, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
    except Exception as e:                                      # noqa: BLE001
        log(f"download failed ({e.__class__.__name__}) — staying on the current build")
        _quiet_remove(path)
        return None
    actual = sha256_file(path)
    if actual != expected_sha256:
        log(f"REJECTED: archive hashes to {actual[:16]}… but the manifest published "
            f"{expected_sha256[:16]}…. Nothing was installed.")
        _quiet_remove(path)
        return None
    log(f"archive verified (sha256 {actual[:16]}…)")
    return path


def _quiet_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# snapshot / install / rollback
# --------------------------------------------------------------------------- #
def snapshot(root, into=None, paths=CODE_PATHS):
    """Copy the current code aside and return the snapshot directory.

    This is the whole rollback story, so it happens BEFORE anything is extracted. The database
    is copied too -- not because an update should ever touch it (nothing here writes to it, and
    `models.init_db()` migrates the schema forward on startup), but because a schema migration
    is one-way and the backup is worthless if taken after the new code has already run.
    """
    into = into or os.path.join(root, ".rollback", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(into, exist_ok=True)
    for rel in paths:
        src = os.path.join(root, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(into, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(src, dst)
    db = os.path.join(root, DB_RELPATH)
    if os.path.exists(db):
        shutil.copy2(db, os.path.join(into, "neuron.db.bak"))
    log(f"snapshot taken: {into}")
    return into


def prune_snapshots(root, keep=KEEP_SNAPSHOTS):
    """A VM with 1 GB of RAM has a small disk too. Keep the last few, drop the rest."""
    base = os.path.join(root, ".rollback")
    if not os.path.isdir(base):
        return []
    kept = sorted(os.listdir(base))
    dropped = kept[:-keep] if len(kept) > keep else []
    for d in dropped:
        shutil.rmtree(os.path.join(base, d), ignore_errors=True)
    return dropped


def _safe_members(tf, root):
    """Yield only members that land inside `root` and inside CODE_PATHS.

    An archive can name `../../etc/systemd/...` or an absolute path, and tarfile will happily
    write it. The archive is SHA-256 verified against the manifest before we get here, so this
    is defence in depth rather than the primary control -- but the primary control is a hash in
    a file in a git repo, and this costs four lines.
    """
    allowed = tuple(p.rstrip("/") for p in CODE_PATHS)
    for m in tf.getmembers():
        name = m.name.lstrip("./")
        if m.issym() or m.islnk():
            continue
        if os.path.isabs(m.name) or ".." in name.split("/"):
            log(f"skipped suspicious archive member: {m.name}")
            continue
        top = name.split("/")[0]
        if top not in allowed:
            continue
        target = os.path.realpath(os.path.join(root, name))
        if not target.startswith(os.path.realpath(root) + os.sep):
            log(f"skipped out-of-tree archive member: {m.name}")
            continue
        yield m


def install(archive, root):
    """Extract the verified archive over the code tree. Never touches neuron.db."""
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(root, members=_safe_members(tf, root))     # noqa: S202
    log(f"unpacked into {root}")


def restore(snap, root, paths=CODE_PATHS):
    """Put the snapshot back. The database is deliberately NOT restored: rolling code back is
    safe, silently rolling a migrated schema back is not, and nothing here writes to it."""
    for rel in paths:
        src = os.path.join(snap, rel)
        if not os.path.exists(src):
            continue
        dst = os.path.join(root, rel)
        if os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    log(f"restored code from {snap}")


def restart(service=SERVICE, run=subprocess.run):
    r = run(["sudo", "systemctl", "restart", service], capture_output=True, text=True)
    ok = getattr(r, "returncode", 1) == 0
    log(f"systemctl restart {service}: {'ok' if ok else 'FAILED'}")
    return ok


# --------------------------------------------------------------------------- #
# health gate
# --------------------------------------------------------------------------- #
def health(base, expect_version=None, timeout=HEALTH_TIMEOUT_S, poll=HEALTH_POLL_S,
           status_fn=http_status, sleep=time.sleep):
    """Is the coordinator actually serving, running what we installed, and still gated?

    Returns (ok, [reasons]). The version check is the one that catches the failures a plain
    200 does not: a restart that raced the file swap, or a rollback that already happened,
    both leave a perfectly healthy process serving the old build.
    """
    base = base.rstrip("/")
    deadline = time.time() + timeout
    reasons = []
    while True:
        reasons = []
        code, body = status_fn(f"{base}/status")
        if code != 200:
            reasons.append(f"/status answered {code or 'nothing'}")
        elif expect_version:
            try:
                got = (json.loads(body) or {}).get("coordinator_version")
            except ValueError:
                got = None
            if got != expect_version:
                reasons.append(f"/status reports version {got!r}, expected {expect_version!r}")
        # The gates deploy.sh verifies. A coordinator that serves but has its auth switched off
        # is a failed update, not a successful one.
        code, _ = status_fn(f"{base}/wallet/faucet", method="POST",
                            body={"wallet_id": "selfupdate-probe"})
        if code != 401:
            reasons.append(f"/wallet/faucet answered {code}, expected 401")
        code, _ = status_fn(f"{base}/admin/identities")
        if code != 401:
            reasons.append(f"/admin/identities answered {code}, expected 401")
        if not reasons:
            return True, []
        if time.time() >= deadline:
            return False, reasons
        sleep(poll)


# --------------------------------------------------------------------------- #
# one cycle
# --------------------------------------------------------------------------- #
def check_once(root, base, manifest_url, local_version, apply=True,
               health_timeout=HEALTH_TIMEOUT_S,
               get=http_json, status_fn=http_status, run=subprocess.run, sleep=time.sleep):
    """One update cycle. Returns a short verdict string — what the tests assert on.

    Verdicts: unreachable · current · available · download-failed · restart-failed ·
              rolled-back · rollback-failed · updated
    """
    info = manifest(manifest_url, get=get)
    if info is None:
        return "unreachable"
    if not is_newer(info["version"], local_version):
        return "current"

    log(f"coordinator {info['version']} is published (running {local_version})")
    if not apply:
        return "available"

    archive = download_and_verify(info["download_url"], info["sha256"])
    if archive is None:
        return "download-failed"

    snap = snapshot(root)
    try:
        install(archive, root)
    except Exception as e:                                      # noqa: BLE001
        log(f"extract failed ({e.__class__.__name__}: {e}) — restoring")
        restore(snap, root)
        restart(run=run)
        return "rolled-back"
    finally:
        _quiet_remove(archive)

    if not restart(run=run):
        restore(snap, root)
        restart(run=run)
        return "restart-failed"

    ok, reasons = health(base, expect_version=info["version"], timeout=health_timeout,
                         status_fn=status_fn, sleep=sleep)
    if ok:
        prune_snapshots(root)
        log(f"updated to {info['version']} and verified live")
        return "updated"

    for r in reasons:
        log(f"HEALTH FAILED: {r}")
    log("rolling back to the previous build")
    restore(snap, root)
    if not restart(run=run):
        log("ROLLBACK RESTART FAILED — the coordinator is down and needs a human. "
            f"Snapshot kept at {snap}")
        return "rollback-failed"
    ok2, reasons2 = health(base, expect_version=local_version, timeout=health_timeout,
                           status_fn=status_fn, sleep=sleep)
    if ok2:
        log(f"rolled back to {local_version}; it is serving again. The published build "
            f"{info['version']} was NOT installed — fix it and publish again.")
        return "rolled-back"
    for r in reasons2:
        log(f"POST-ROLLBACK STILL UNHEALTHY: {r}")
    log(f"the coordinator is not serving even on the previous build. Snapshot kept at {snap}")
    return "rollback-failed"


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Update the NEURON coordinator to the published "
                                            "version, verifying it and rolling back on failure.")
    p.add_argument("--root", default=os.environ.get("NEURON_DEPLOY_DIR", "/home/ubuntu/neuron"),
                   help="deploy root containing coordinator/ (default: %(default)s)")
    p.add_argument("--base", default=None,
                   help="URL to health-check after restart (default: config.PUBLIC_URL)")
    p.add_argument("--manifest", default=None,
                   help="manifest URL (default: config.COORDINATOR_MANIFEST_URL)")
    p.add_argument("--check", action="store_true",
                   help="report only — never download, install or restart")
    args = p.parse_args(argv)

    sys.path.insert(0, args.root)
    from coordinator import config                              # noqa: PLC0415

    verdict = check_once(
        root=args.root,
        base=args.base or config.PUBLIC_URL,
        manifest_url=args.manifest or config.COORDINATOR_MANIFEST_URL,
        local_version=config.COORDINATOR_VERSION,
        apply=not args.check)
    print(verdict)
    # Only a state needing a human is a non-zero exit: systemd should not mail the founder
    # because the coordinator was already up to date.
    return 1 if verdict in ("rollback-failed", "restart-failed") else 0


if __name__ == "__main__":
    sys.exit(main())
