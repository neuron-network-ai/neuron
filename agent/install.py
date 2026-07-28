"""
agent/install.py — one-command setup on a fresh machine.

  python install.py --coordinator http://150.230.22.250:8001

Creates config.json, (optionally) registers auto-start, and launches the agent in
the background. Auto-start uses the Windows HKCU Run key, a macOS LaunchAgent, or a
Linux systemd --user service — all fully removed by uninstall.py. ARM-compatible.

Flags: --layer-start/--layer-end (which layers this node claims; auto-assignment is
a later session), --no-startup (skip auto-start), --no-run (just write config).
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
LAUNCHD_LABEL = "com.neuron.agent"
LAUNCHD_PLIST_PATH = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")

DEFAULT_CONFIG = {
    "coordinator": "http://150.230.22.250:8001",
    "node_id": None, "node_token": None,
    "layer_start": 10, "layer_end": 18,
    "slice_dir": "./model_slice/",
    "max_cpu_pct": 2, "idle_threshold_seconds": 60, "log_level": "INFO",
}


def write_config(coordinator, layer_start, layer_end):
    cfg = dict(DEFAULT_CONFIG)
    cfg["coordinator"] = coordinator
    if layer_start is not None:
        cfg["layer_start"] = layer_start
    if layer_end is not None:
        cfg["layer_end"] = layer_end
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  wrote {CONFIG_PATH}")


def add_to_startup_windows():
    import winreg
    py = shutil.which("pythonw") or sys.executable
    cmd = f'"{py}" "{os.path.join(HERE, "agent.py")}"'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Windows\CurrentVersion\Run",
                        0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "NEURONAgent", 0, winreg.REG_SZ, cmd)
    print("  added Windows auto-start (HKCU\\...\\Run : NEURONAgent)")


def add_to_startup_macos():
    """A per-user LaunchAgent (~/Library/LaunchAgents) — the macOS equivalent of the
    Windows Run key / Linux systemd --user service above. RunAtLoad+KeepAlive means
    `launchctl load -w` both registers it for next login AND starts it right now, so
    (unlike the Linux path) no separate start_background() call is needed afterward."""
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>{LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{sys.executable}</string>
        <string>{os.path.join(HERE, "agent.py")}</string>
    </array>
    <key>WorkingDirectory</key><string>{HERE}</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardOutPath</key><string>{os.path.join(HERE, "agent.log")}</string>
    <key>StandardErrorPath</key><string>{os.path.join(HERE, "agent.log")}</string>
</dict>
</plist>
"""
    os.makedirs(os.path.dirname(LAUNCHD_PLIST_PATH), exist_ok=True)
    with open(LAUNCHD_PLIST_PATH, "w") as f:
        f.write(plist)
    subprocess.run(["launchctl", "unload", "-w", LAUNCHD_PLIST_PATH], check=False)  # in case of a stale prior load
    subprocess.run(["launchctl", "load", "-w", LAUNCHD_PLIST_PATH], check=False)
    print(f"  created + loaded LaunchAgent: {LAUNCHD_PLIST_PATH}")


def add_to_startup_linux():
    unit = (
        "[Unit]\nDescription=NEURON agent\nAfter=network-online.target\n\n"
        "[Service]\n"
        f"ExecStart={sys.executable} {os.path.join(HERE, 'agent.py')}\n"
        f"WorkingDirectory={HERE}\nRestart=on-failure\n\n"
        "[Install]\nWantedBy=default.target\n"
    )
    dst = os.path.expanduser("~/.config/systemd/user/neuron-agent.service")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "neuron-agent"], check=False)
    print(f"  created systemd user service: {dst}")


def start_background():
    if IS_WINDOWS:
        py = shutil.which("pythonw") or sys.executable
        subprocess.Popen([py, os.path.join(HERE, "agent.py")], cwd=HERE,
                         creationflags=0x00000008)   # DETACHED_PROCESS
    elif IS_MACOS:
        # add_to_startup_macos()'s RunAtLoad already started it if that ran; this covers
        # --no-startup (no LaunchAgent) by just spawning it directly, same as Windows.
        subprocess.Popen([sys.executable, os.path.join(HERE, "agent.py")], cwd=HERE,
                         start_new_session=True)
    else:
        subprocess.run(["systemctl", "--user", "start", "neuron-agent"], check=False)
    print("  agent launched in background")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", default=None)
    ap.add_argument("--layer-start", type=int, default=None)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--no-startup", action="store_true", help="do not add to auto-start")
    ap.add_argument("--no-run", action="store_true", help="only write config, do not launch")
    args = ap.parse_args()

    coordinator = args.coordinator
    if not coordinator:
        coordinator = input("Coordinator URL [http://150.230.22.250:8001]: ").strip() \
            or "http://150.230.22.250:8001"

    print(f"Installing NEURON agent (coordinator = {coordinator})")
    write_config(coordinator, args.layer_start, args.layer_end)
    if not args.no_startup:
        if IS_WINDOWS:
            add_to_startup_windows()
        elif IS_MACOS:
            add_to_startup_macos()
        else:
            add_to_startup_linux()
    if not args.no_run:
        start_background()
    print("NEURON agent installed. It will register, download its slice, and start earning.")


if __name__ == "__main__":
    main()
