"""
agent/uninstall.py — clean removal.

Deregisters the node, stops the agent, removes auto-start, and deletes the model
slice + config.json. Leaves the agent code in place so reinstalling is one command.
Prints the lifetime NRN contribution. ARM-compatible.
"""
import json
import os
import platform
import shutil
import subprocess

import sys

import requests

# Match the agent's writable-state location (installed app: %LOCALAPPDATA%\NEURON).
if getattr(sys, "frozen", False):
    _base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    HERE = os.path.join(_base, "NEURON")
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
LAUNCHD_LABEL = "com.neuron.agent"
LAUNCHD_PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
KEEPALIVE_PATH = os.path.join(HERE, "neuron-keepalive.sh")
CRON_TAG = "# NEURON-AGENT"


def _total_earned(cfg):
    try:
        r = requests.get(f"{cfg['coordinator'].rstrip('/')}/ledger/{cfg['node_id']}", timeout=8,
                         headers={"X-Node-Token": cfg.get("node_token", "")})
        if r.status_code == 200:
            return r.json().get("total_earned", 0.0)
    except Exception:
        pass
    return 0.0


def _owner_and_balance(cfg):
    """(owner_wallet_id, balance) for this node, or (None, 0.0) if it cannot be read.

    Both are needed before anything is deleted, because together they answer the only
    question that matters at uninstall time: **is this machine's NRN reachable by a person
    after the config is gone?** An owner means yes — it is recorded against a Google/GitHub
    account that outlives the disk. No owner means the balance is addressable only by a
    node_id that is about to be deleted and never reissued.
    """
    base = cfg.get("coordinator", "").rstrip("/")
    nid, tok = cfg.get("node_id"), cfg.get("node_token", "")
    if not (base and nid):
        return None, 0.0
    owner, balance = None, 0.0
    try:
        r = requests.get(f"{base}/node/{nid}/payout-address",
                         headers={"X-Node-Token": tok}, timeout=8)
        if r.status_code == 200:
            owner = r.json().get("owner_wallet_id")
    except requests.RequestException:
        pass
    try:
        r = requests.get(f"{base}/ledger/{nid}", headers={"X-Node-Token": tok}, timeout=8)
        if r.status_code == 200:
            balance = float(r.json().get("balance") or 0.0)
    except (requests.RequestException, TypeError, ValueError):
        pass
    return owner, balance


def _write_recovery_note(cfg, balance, state_dir=None):
    """Leave the node id behind so unclaimed NRN can still be found afterwards.

    This file is the difference between "33.49 NRN is stranded" and "33.49 NRN is stranded on
    agent-optinovate-6ff49d". `coordinator/claim_node_earnings.py` sweeps by node_id, so the id
    is the whole of what recovery needs — and it is the one thing deleting config.json destroys.
    Deliberately holds NO token: the coordinator kills it on deregistration, so keeping it would
    be a dead secret on disk pretending to be a key.
    """
    # Beside the config being removed, not beside HERE: --config is what decides
    # which install this is, and a test (or a second install) must not write into
    # the real state directory.
    state_dir = state_dir or HERE
    path = os.path.join(state_dir, "unclaimed-earnings.json")
    try:
        os.makedirs(state_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"node_id": cfg.get("node_id"),
                       "coordinator": cfg.get("coordinator"),
                       "balance_at_removal": round(balance, 4),
                       "removed_at": int(__import__("time").time()),
                       "note": "This NRN was never claimed to an account. It is still on the "
                               "coordinator's ledger under node_id. To recover it, sign in to "
                               "NEURON and ask the operator to run claim_node_earnings.py for "
                               "this node_id."}, f, indent=2)
        return path
    except OSError:
        return None


def _deregister(cfg):
    """Remove this node from the coordinator. Returns (ok, human-readable detail).

    This used to fire the DELETE and ignore everything that came back -- exceptions swallowed
    by a bare `except: pass`, and every non-2xx treated as success because the status was never
    read. A failed deregistration is not cosmetic: `new_node_id()` mints a FRESH random suffix
    on every install (deliberately -- it is what stopped hostname collisions locking people
    out), so a reinstall never reclaims the old registration. The orphan then sits on the
    network forever, holding a layer range nobody serves, while the uninstaller cheerfully
    reports success. Seen live: one machine listed twice as `agent-<host>` and
    `agent-<host>-67e4eb`, with the chain stuck DEGRADED behind the dead one.

    The 401 case is the likely one and worth its own message: the coordinator re-mints a
    node's token on every registration, so any copy that re-registered after this config was
    written left the token on disk dead.
    """
    url = f"{cfg['coordinator'].rstrip('/')}/node/{cfg['node_id']}"
    try:
        r = requests.delete(url, headers={"X-Node-Token": cfg.get("node_token", "")}, timeout=8)
    except requests.RequestException as exc:
        return False, f"could not reach the coordinator ({type(exc).__name__})"
    if r.status_code in (200, 204):
        return True, "deregistered"
    if r.status_code == 404:
        return True, "already gone from the coordinator"
    if r.status_code == 401:
        return False, ("the coordinator rejected this node's token — another copy of the agent "
                       "registered after this one and replaced it")
    return False, f"the coordinator answered HTTP {r.status_code}"


def _stop_agent():
    if IS_WINDOWS:
        try:
            import psutil
            mine = os.getpid()
            for p in psutil.process_iter(["pid", "cmdline"]):
                if p.info["pid"] == mine:
                    continue
                cl = " ".join(p.info.get("cmdline") or [])
                if "agent.agent" in cl or cl.endswith("agent.py") \
                        or os.path.join("agent", "agent.py") in cl:
                    p.kill()
        except Exception:
            pass
    elif IS_MACOS:
        # unload both stops the running process (KeepAlive-managed) and de-registers it —
        # the plist FILE itself is removed separately in _remove_startup, mirroring Linux.
        subprocess.run(["launchctl", "unload", "-w", LAUNCHD_PLIST_PATH], check=False)
    else:
        subprocess.run(["systemctl", "--user", "stop", "neuron-agent"], check=False)
        subprocess.run(["systemctl", "--user", "disable", "neuron-agent"], check=False)


def _remove_startup():
    if IS_WINDOWS:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                r"Software\Microsoft\Windows\CurrentVersion\Run",
                                0, winreg.KEY_SET_VALUE) as key:
                winreg.DeleteValue(key, "NEURONAgent")
            print("  removed Windows auto-start entry")
        except FileNotFoundError:
            pass
        except OSError:
            pass
    elif IS_MACOS:
        if os.path.exists(LAUNCHD_PLIST_PATH):
            os.remove(LAUNCHD_PLIST_PATH)
            print("  removed LaunchAgent")
    else:
        dst = os.path.expanduser("~/.config/systemd/user/neuron-agent.service")
        if os.path.exists(dst):
            os.remove(dst)
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            print("  removed systemd user service")
        _remove_cron_fallback()


def _remove_cron_fallback():
    """Drop the crontab lines + keepalive script install.py adds on Linux when linger is
    unavailable. Missing this would leave a cron job resurrecting an uninstalled agent
    every two minutes — the uninstall would appear to work and silently not."""
    out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    if out.returncode == 0 and CRON_TAG in out.stdout:
        kept = [ln for ln in out.stdout.splitlines() if CRON_TAG not in ln]
        subprocess.run(["crontab", "-"], input="\n".join(kept).strip() + "\n",
                       capture_output=True, text=True, check=False)
        print("  removed cron auto-start entries")
    if os.path.exists(KEEPALIVE_PATH):
        os.remove(KEEPALIVE_PATH)
        print("  removed keepalive script")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH)
    # ignore extra flags (the installer invokes this as `<app>.exe --deregister`)
    args, _ = ap.parse_known_args(argv)
    config_path = args.config

    cfg = json.load(open(config_path)) if os.path.exists(config_path) else {}
    earned = _total_earned(cfg) if cfg.get("node_id") else 0.0
    # Read BEFORE deregistering: the coordinator kills this node's token on DELETE, so after
    # that point neither the owner nor the balance can be read at all, and the uninstaller
    # would have to guess about the very thing it is about to make unreachable.
    owner, balance = _owner_and_balance(cfg) if cfg.get("node_id") else (None, 0.0)

    dereg_ok, dereg_detail = True, None
    if cfg.get("node_id") and cfg.get("node_token"):
        dereg_ok, dereg_detail = _deregister(cfg)
        print(f"  {'deregistered' if dereg_ok else 'COULD NOT DEREGISTER'}: {dereg_detail}")
    _stop_agent()
    _remove_startup()

    slice_dir = os.path.join(HERE, os.path.normpath(cfg.get("slice_dir", "./model_slice/")))
    if os.path.isdir(slice_dir):
        shutil.rmtree(slice_dir, ignore_errors=True)
        print(f"  deleted slice dir {slice_dir}")
    if os.path.exists(config_path):
        os.remove(config_path)
        print(f"  deleted {os.path.basename(config_path)}")

    print(f"\nNEURON removed. Thank you for contributing {earned:.2f} NRN total.")

    # THE LINE THIS FILE WAS MISSING, and it cost 33.49 NRN on 2026-08-18 ([P53]).
    # A node's balance lives against its node_id. `new_node_id()` mints a fresh random suffix
    # on every install, deliberately, so a reinstall NEVER comes back as the same node — and
    # config.json, the only record of which id this machine was, has just been deleted three
    # lines above. Printing lifetime earnings as a thank-you at that exact moment reads as a
    # receipt for money that has in fact just become unaddressable.
    #
    # Claimed earnings are fine and should be said so plainly: they are recorded against a
    # Google/GitHub account and survive the disk, which is the entire point of the claim.
    if owner:
        print(f"  Your earnings are recorded to your account and are NOT affected by this "
              f"removal.")
    elif balance > 0:
        note = _write_recovery_note(cfg, balance, os.path.dirname(config_path))
        print(f"\n  WARNING: {balance:.2f} NRN on this machine was never claimed to an account.")
        print(f"  It is held under the node id '{cfg.get('node_id')}', and reinstalling will "
              f"NOT get it back —\n  a new install registers under a NEW id, on purpose.")
        print(f"  To claim next time: open NEURON, sign in with Google or GitHub, and press "
              f"'Claim with my account'\n  BEFORE uninstalling.")
        if note:
            print(f"  The node id has been saved to {note} so it can still be recovered.")

    if not dereg_ok:
        # Say it plainly rather than let a silent orphan degrade the network. Reinstalling
        # will NOT fix this -- the new install takes a new node id on purpose.
        print(f"\n  WARNING: this machine is still listed on the network as "
              f"'{cfg['node_id']}'.\n"
              f"  Reason: {dereg_detail}.\n"
              f"  Reinstalling will not clear it — a new install registers under a new id, so\n"
              f"  the old one would linger and hold a layer range nobody serves.\n"
              f"  Ask the network operator to remove it:\n"
              f"      DELETE {cfg['coordinator'].rstrip('/')}/node/{cfg['node_id']}\n"
              f"      header: X-Register-Secret: <operator secret>")
    return 0 if dereg_ok else 1


if __name__ == "__main__":
    sys.exit(main())
