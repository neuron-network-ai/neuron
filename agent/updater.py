"""
agent/updater.py — keep the agent current.

On startup and every 6 hours, ask the coordinator for the published agent version
(GET /agent/version). If it is newer than this build, download the new agent.py
and restart. ARM-compatible (pure Python + requests).
"""
import argparse
import logging
import os
import sys
import time

import requests

LOCAL_VERSION = "0.3.0"          # bump when publishing a new agent
CHECK_SECONDS = 6 * 3600
HERE = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("neuron.agent.updater")


def _parse(v):
    return tuple(int(x) for x in str(v).split(".") if x.isdigit())


def is_newer(remote, local):
    try:
        return _parse(remote) > _parse(local)
    except Exception:
        return False


def remote_version(base):
    r = requests.get(f"{base.rstrip('/')}/agent/version", timeout=10)
    r.raise_for_status()
    return r.json()["version"]


def download_and_replace(base, update_url=None):
    """Fetch the new agent.py. Returns True on success. Non-fatal if unavailable."""
    url = update_url or f"{base.rstrip('/')}/agent/download"
    try:
        r = requests.get(url, timeout=30)
    except requests.RequestException as e:
        log.warning("update download failed: %s", e)
        return False
    if r.status_code != 200 or not r.content:
        log.warning("update available but no download endpoint (%s) — manual update", r.status_code)
        return False
    tmp = os.path.join(HERE, "agent.py.new")
    with open(tmp, "wb") as f:
        f.write(r.content)
    os.replace(tmp, os.path.join(HERE, "agent.py"))
    log.info("downloaded new agent.py (%d bytes)", len(r.content))
    return True


def restart():
    log.info("restarting agent to apply update ...")
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "agent.py")])


def run_loop(base, stop=None, update_url=None, on_update=restart):
    """Background loop: poll for updates until stopped."""
    while stop is None or not stop.is_set():
        try:
            remote = remote_version(base)
            if is_newer(remote, LOCAL_VERSION):
                log.info("new agent version %s available (have %s)", remote, LOCAL_VERSION)
                if download_and_replace(base, update_url):
                    on_update()
                    return
            else:
                log.info("agent up to date (%s)", LOCAL_VERSION)
        except requests.RequestException as e:
            log.warning("version check failed: %s", e)
        if stop is not None:
            stop.wait(CHECK_SECONDS)
        else:
            time.sleep(CHECK_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", required=True)
    args = ap.parse_args()
    remote = remote_version(args.coordinator)
    print(f"local={LOCAL_VERSION}  remote={remote}  newer_available={is_newer(remote, LOCAL_VERSION)}")


if __name__ == "__main__":
    main()
