"""
agent/updater.py — keep the agent current without asking anyone to reinstall.

This matters more than it looks. Every fix so far has reached exactly one machine — the
founder's — because the only delivery mechanism was "download the installer again". A stranger
will not do that. They will run a build with a dead Chat UI, or an uninstaller that lies about
deregistering, forever.

The previous version of this file downloaded `agent.py` and swapped it in. That could never
work for the shipped product: the app is a frozen PyInstaller bundle, the code lives inside
`neuron-agent.exe`, and there is no `agent.py` on disk to replace. It was also never called
from anywhere. So it is rewritten around what the app actually is.

**Frozen build** (what strangers run): download the published installer, verify its SHA-256
against what the coordinator advertises, then run it silently and exit so it can replace the
files underneath us. Inno Setup does an in-place upgrade; config.json, the payout key and the
model slice live in %LOCALAPPDATA%\\NEURON and are untouched.

**Source checkout** (what the founder runs): refuse to touch anything and say so. A working
tree is many modules, possibly with local edits — silently overwriting it would be hostile,
and `git pull` is one command.

Three rules, ordered by how badly they end if broken:

1. **Never install an unverified binary.** No hash from the coordinator, or a mismatch, means
   nothing runs. Pushing an unchecked executable to every volunteer's machine is the worst
   thing this project could do, and it would be doing it automatically.
2. **Never update mid-request.** A node that vanishes during inference surfaces to the driver
   as "socket closed mid-message" and kills the answer for everyone on that chain.
3. **Never let the updater take the node down.** Unreachable coordinator, truncated download,
   malformed response — all logged and skipped. Serving continues.
"""
import hashlib
import logging
import os
import subprocess
import sys
import tempfile
import time

import requests

LOCAL_VERSION = "0.18.0"          # bump together with packaging/neuron.iss
CHECK_SECONDS = 24 * 3600
DOWNLOAD_TIMEOUT = 600
HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("neuron.agent.updater")


def _parse(v):
    return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def is_newer(remote, local):
    try:
        r, l = _parse(remote), _parse(local)
        return bool(r) and r > l
    except Exception:                                           # noqa: BLE001
        return False


def is_frozen():
    return bool(getattr(sys, "frozen", False))


def remote_info(base, timeout=15):
    """{'version', 'download_url', 'sha256'} from the coordinator, or None.

    None means "could not ask" -- never "up to date". A network blip must not be read as a
    decision by the caller.
    """
    try:
        r = requests.get(f"{base.rstrip('/')}/agent/version", timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        log.warning("update check skipped: coordinator unreachable (%s)", e.__class__.__name__)
        return None
    except ValueError:
        log.warning("update check skipped: coordinator sent a non-JSON version response")
        return None
    if not isinstance(data, dict) or not data.get("version"):
        log.warning("update check skipped: version response carried no version")
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


def _quiet_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def download_and_verify(url, expected_sha256, dest_dir=None, timeout=DOWNLOAD_TIMEOUT):
    """Fetch `url`, return its path only if it hashes to `expected_sha256`.

    A missing expected hash is a refusal, not a shortcut (rule 1). A mismatch deletes the file
    rather than leaving a rejected executable on disk for something else to find.
    """
    if not url:
        log.warning("update available but the coordinator published no download URL")
        return None
    if not expected_sha256:
        log.warning("update available but the coordinator published no SHA-256 — refusing to "
                    "install an unverified build")
        return None

    dest_dir = dest_dir or tempfile.mkdtemp(prefix="neuron-update-")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, os.path.basename(url.split("?")[0]) or "neuron-update.exe")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        log.warning("update download failed (%s) — staying on %s",
                    e.__class__.__name__, LOCAL_VERSION)
        _quiet_remove(path)
        return None

    actual = sha256_file(path)
    if actual != expected_sha256:
        log.error("update REJECTED: %s hashes to %s… but the coordinator published %s…. "
                  "Nothing was installed.", os.path.basename(path), actual[:16],
                  expected_sha256[:16])
        _quiet_remove(path)
        return None
    log.info("update %s verified (sha256 %s…)", os.path.basename(path), actual[:16])
    return path


def apply_update(installer_path, exit_after=True):
    """Run the verified installer and get out of its way.

    Inno Setup cannot replace files this process is executing, so: start it detached, then
    exit. `/VERYSILENT` because nobody is watching a tray app. An upgrade over the top keeps
    config.json, the payout key and the model slice — they live in %LOCALAPPDATA%\\NEURON,
    not in the program directory.
    """
    if not installer_path or not os.path.exists(installer_path):
        return False
    cmd = [installer_path, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
           "/CLOSEAPPLICATIONS", "/RESTARTAPPLICATIONS"]
    flags = 0
    if os.name == "nt":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    try:
        subprocess.Popen(cmd, creationflags=flags, close_fds=True)
    except OSError as e:
        log.error("could not launch the update installer: %s", e)
        return False
    log.info("update installer launched — exiting so it can replace the app")
    if exit_after:
        # os._exit, not sys.exit: daemon threads (node server, tunnel, tray) would keep the
        # process alive holding the very files the installer needs to overwrite.
        time.sleep(1.0)
        os._exit(0)
    return True


def check_once(base, busy=None, dest_dir=None, exit_after=True):
    """One update cycle. Returns a short verdict string — what the tests assert on.

    `busy` is a zero-argument callable answering "is this node serving right now", passed in
    rather than imported so the updater carries no dependency on the node server.
    """
    info = remote_info(base)
    if info is None:
        return "unreachable"
    if not is_newer(info["version"], LOCAL_VERSION):
        return "current"

    log.info("agent %s is available (running %s)", info["version"], LOCAL_VERSION)
    if not is_frozen():
        log.info("running from a source checkout — not touching it. Update with: git pull")
        return "source-manual"
    if busy is not None:
        try:
            if busy():
                log.info("update deferred: this node is serving a request right now")
                return "deferred-busy"
        except Exception:                                       # noqa: BLE001
            # A broken busy-check must never be read as "idle". Defer; try again next cycle.
            log.warning("update deferred: could not determine whether this node is busy")
            return "deferred-busy"

    path = download_and_verify(info["download_url"], info["sha256"], dest_dir=dest_dir)
    if path is None:
        return "download-failed"
    return "installing" if apply_update(path, exit_after=exit_after) else "install-failed"


def update_loop(base, stop=None, busy=None, interval=CHECK_SECONDS, initial_delay=60,
                enabled=True):
    """Check shortly after startup, then every `interval` seconds.

    The initial delay lets registration, the slice download and the node server settle; an
    update racing startup would be the least useful possible moment to restart.

    `enabled` is the operator's `auto_update` setting, passed in rather than read from disk for
    the same reason `busy` is: this module stays free of any dependency on where config lives,
    which differs between a frozen install (%LOCALAPPDATA%) and a source checkout.

    Checked before the initial delay, so a node with auto-update off never waits and never
    contacts the coordinator about versions at all. It is read once, at startup — changing the
    setting takes effect the next time the agent starts, which INSTALL.md says plainly.

    Deliberately NOT applied to `check_once()` or the CLI below: an operator who explicitly runs
    `python -m agent.updater --apply` is asking for an update by hand, and a background setting
    should not silently refuse a foreground request.
    """
    if not enabled:
        log.info("auto-update disabled (auto_update=false) — this node will not check for or "
                 "install new versions; update it yourself by installing a newer build")
        return
    if stop is not None:
        if stop.wait(initial_delay):
            return
    else:
        time.sleep(initial_delay)
    while True:
        try:
            check_once(base, busy=busy)
        except Exception as e:                                  # noqa: BLE001
            log.warning("update check failed (%s: %s)", e.__class__.__name__, e)
        if stop is not None:
            if stop.wait(interval):
                return
        else:
            time.sleep(interval)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Check for a newer NEURON agent.")
    p.add_argument("--coordinator", required=True)
    p.add_argument("--apply", action="store_true",
                   help="actually download and install (default: report only)")
    args = p.parse_args(argv)

    info = remote_info(args.coordinator)
    if info is None:
        print("could not reach the coordinator")
        return 2
    print(f"local  : {LOCAL_VERSION}")
    print(f"remote : {info['version']}")
    if not is_newer(info["version"], LOCAL_VERSION):
        print("up to date")
        return 0
    if not args.apply:
        print(f"update available at {info['download_url'] or '(no URL published)'} — "
              f"re-run with --apply to install")
        return 0
    print(check_once(args.coordinator, exit_after=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
