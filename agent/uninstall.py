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


def _total_earned(cfg):
    try:
        r = requests.get(f"{cfg['coordinator'].rstrip('/')}/ledger/{cfg['node_id']}", timeout=8,
                         headers={"X-Node-Token": cfg.get("node_token", "")})
        if r.status_code == 200:
            return r.json().get("total_earned", 0.0)
    except Exception:
        pass
    return 0.0


def _deregister(cfg):
    try:
        requests.delete(f"{cfg['coordinator'].rstrip('/')}/node/{cfg['node_id']}",
                        headers={"X-Node-Token": cfg.get("node_token", "")}, timeout=8)
    except Exception:
        pass


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH)
    # ignore extra flags (the installer invokes this as `<app>.exe --deregister`)
    args, _ = ap.parse_known_args()
    config_path = args.config

    cfg = json.load(open(config_path)) if os.path.exists(config_path) else {}
    earned = _total_earned(cfg) if cfg.get("node_id") else 0.0

    if cfg.get("node_id") and cfg.get("node_token"):
        _deregister(cfg)
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


if __name__ == "__main__":
    main()
