"""
agent/install.py — one-command setup on a fresh machine.

  python install.py --coordinator https://neuronnet.duckdns.org
  python install.py --startup          # existing node: just make it survive reboots

Creates config.json, registers auto-start, and launches the agent in the background.
Auto-start uses the Windows HKCU Run key, a macOS LaunchAgent, or a Linux systemd
--user service — all fully removed by uninstall.py. ARM-compatible.

Flags: --startup (install/repair auto-start, keeping any existing config exactly as
it is), --layer-start/--layer-end (which layers this node claims), --no-startup (skip
auto-start), --no-run (just write config).

[P21] — WHY THE LINUX PATH IS MORE THAN A UNIT FILE. A `systemd --user` service does
NOT start at boot unless the user has *lingering* enabled; without it the unit only
runs while somebody is logged in, which is precisely the case a reboot destroys. So
the install enables linger (directly, then via sudo), and when neither is permitted —
a machine with no sudo, which is exactly a stranger's laptop — it falls back to cron,
which needs no privileges at all: an `@reboot` line plus a two-minute keepalive that
restarts the agent if it is not running. Restart=always covers a crash; linger or
cron covers a reboot; you need both or the node leaves the network on its first
power cycle and never comes back.
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
KEEPALIVE_PATH = os.path.join(HERE, "neuron-keepalive.sh")
CRON_TAG = "# NEURON-AGENT"
# The operator's verifier keepalive ([P24]) — a scheduled task on Windows, a cron line
# elsewhere. Tagged separately from the agent's own entries so removing one never removes both.
VERIFIER_TASK = "NEURONVerifierKeepalive"
VERIFIER_CRON_TAG = "# NEURON-VERIFIER"

# MUST stay in step with agent/agent.py's DEFAULT_CONFIG (kept as a copy rather than an
# import because agent.py pulls in torch via node_server, and the installer has to run before
# any of that is guaranteed to be importable). The two had drifted: this one still carried the
# Session-9 `max_cpu_pct: 2`, which agent.py reads as an explicit ceiling override that BEATS
# donation_mode -- so every fresh install was silently pinned to a 2% ceiling and spent its
# life paused, and a hardcoded 10-18 layer range collided with whoever already served it.
DEFAULT_CONFIG = {
    "coordinator": "https://neuronnet.duckdns.org",
    "node_id": None, "node_token": None, "model_id": None,
    # null = ask the coordinator where this machine fits (GET /node/placement, S20).
    # A stranger must never be asked to pick layer numbers.
    "layer_start": None, "layer_end": None,
    "slice_dir": "./model_slice/",
    # See agent.py's DEFAULT_CONFIG: "balanced" donates while the owner works instead of only
    # when the machine is idle. `device` is overwritten below with what this machine actually
    # has; "auto" is only the value a config carries if detection could not run.
    "donation_mode": "balanced", "idle_threshold_seconds": 60,
    "device": "auto",
    # Every node reaches its peers through the public relay by default ([P10]): a home
    # machine cannot accept inbound TCP, and a Tailscale address is meaningless to
    # anyone outside the founder's tailnet.
    "behind_nat": True, "log_level": "INFO",
    "local_chat": True, "local_chat_port": 8080,
}


def detect_device():
    """"gpu" if this machine has a card the build can genuinely use, else "cpu".

    Both halves have to be true, and conflating them is the mistake that shipped in v0.18.0: a
    card being present says nothing about whether this binary can address it. `torch.cuda`
    answers the question that matters, because it is the thing that would run a layer;
    `agent.gpu.detect_gpu()` is a hardware report and would say yes on a machine that cannot
    use it. Today the shipped torch is a `+cpu` build, so this returns "cpu" everywhere -- and
    it will start returning "gpu" on its own the day that changes, with no edit here.

    Deliberately concrete rather than "auto": a value written at install is one a person can
    read in their config file and change. Never raises -- detection failing means "cpu", not a
    failed install.
    """
    try:
        import torch
        return "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:                                               # noqa: BLE001
        return "cpu"


def write_config(coordinator, layer_start, layer_end):
    """MERGE into any existing config — never replace it.

    This used to write DEFAULT_CONFIG over whatever was there, which silently threw away
    `node_id`, `node_token`, the layer range this machine was actually serving and its
    register_secret, and re-pinned every node to layers 10-18. Running the documented
    auto-start command on a live node would have de-identified it.
    """
    cfg = None
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  ! existing {CONFIG_PATH} unreadable ({e}) — writing a fresh one")
    # Defaults are for a FRESH install only. Layering them under an existing config would
    # inject keys that node's operator deliberately left out, and some of them are not inert:
    # `max_cpu_pct` is read as an explicit ceiling override, so adding it silently demoted a
    # `donation_mode: max` machine to a 2% ceiling and it stopped advertising availability.
    if cfg is None:
        cfg = dict(DEFAULT_CONFIG)
        cfg["device"] = detect_device()
        print(f"  compute device: {cfg['device']}")
    if coordinator:
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


def add_verifier_to_startup(python_exe=None):
    """Auto-start verify_service.py — the OPERATOR's node verifier.

    Deliberately NOT part of the default --startup. Every stranger runs that path, and the
    verifier needs two things no stranger has or should have: the operator's
    NEURON_REGISTER_SECRET (node addresses are private and /attest is secret-gated) and
    PyTorch, because it recomputes each node's layers locally to check the answer. Bundling it
    into the normal install would put a permanently-failing service on every donor's machine
    and imply they ought to hold the network's registration secret.
    """
    repo = os.path.dirname(HERE)
    script = os.path.join(repo, "verify_service.py")
    if not os.path.exists(script):
        print(f"  ! {script} not found — verifier auto-start skipped")
        return False
    py = python_exe or sys.executable
    if IS_WINDOWS:
        import winreg
        # pythonw NEXT TO the chosen interpreter, never whatever is first on PATH: this repo's
        # working interpreter is a venv, while PATH here resolves to a bare Python 3.14 with no
        # PyTorch — so a PATH lookup writes a Run key that dies on import at every boot.
        cand = os.path.join(os.path.dirname(py), "pythonw.exe")
        pyw = cand if os.path.exists(cand) else py
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion\Run",
                            0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "NEURONVerifier", 0, winreg.REG_SZ,
                              f'"{pyw}" "{script}"')
        print("  added Windows auto-start for the verifier (HKCU\\...\\Run : NEURONVerifier)")
    else:
        unit = ("[Unit]\nDescription=NEURON node verifier\nAfter=network-online.target\n\n"
                "[Service]\n"
                f"ExecStart={py} {script}\nWorkingDirectory={repo}\n"
                "Restart=always\nRestartSec=30\n\n"
                "[Install]\nWantedBy=default.target\n")
        dst = os.path.expanduser("~/.config/systemd/user/neuron-verifier.service")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as f:
            f.write(unit)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "--user", "enable", "neuron-verifier"], check=False)
        print(f"  created systemd user service: {dst}")
    add_verifier_keepalive(py)
    # Auto-start launches with a bare environment, so an env var set in this shell would not
    # reach it; verify_service.load_secret() falls back to .env.coordinator for exactly that.
    if not os.environ.get("NEURON_REGISTER_SECRET") and \
            not os.path.exists(os.path.join(repo, ".env.coordinator")):
        print("  ! no NEURON_REGISTER_SECRET and no .env.coordinator — the verifier will start "
              "and exit immediately. Put the secret in one of them.")
    return True


def add_verifier_keepalive(python_exe=None):
    """Restart the verifier if it dies — the half auto-start does not cover.

    [P24]: the verifier was dead for two days on a machine where its auto-start was correctly
    installed. That is not a contradiction — the Windows Run key fires once at login and never
    again, so it survives a reboot and not a crash, and the Linux systemd unit written above
    covers the crash but leaves nothing checking it either. Whatever killed it, the network
    then had no way to admit new nodes, and every stranger who joined stayed probationary.

    So: a scheduled task (Windows) or a cron line (elsewhere) runs agent/verifier_keepalive.py
    every 5 minutes. It does nothing while the verifier is alive. Same shape as the agent's own
    cron keepalive, for the same reason — an outage should be bounded by a timer, not by
    somebody happening to look.
    """
    py = python_exe or sys.executable
    script = os.path.join(HERE, "verifier_keepalive.py")
    if IS_WINDOWS:
        cand = os.path.join(os.path.dirname(py), "pythonw.exe")
        pyw = cand if os.path.exists(cand) else py
        out = subprocess.run(
            ["schtasks", "/create", "/f", "/tn", VERIFIER_TASK, "/sc", "minute", "/mo", "5",
             "/tr", f'"{pyw}" "{script}"'],
            capture_output=True, text=True)
        if out.returncode == 0:
            print(f"  added a 5-minute keepalive task ({VERIFIER_TASK})")
            return True
        print(f"  ! could not create the keepalive task: {out.stdout.strip()} "
              f"{out.stderr.strip()}")
        return False
    line = f"*/5 * * * * cd {os.path.dirname(HERE)} && {py} {script} >/dev/null 2>&1 "\
           f"{VERIFIER_CRON_TAG}"
    text = _read_crontab()
    if VERIFIER_CRON_TAG in text:
        return True
    _write_crontab((text.rstrip() + "\n" + line + "\n").lstrip())
    print("  added a 5-minute cron keepalive for the verifier")
    return True


def remove_verifier_keepalive():
    """Undo add_verifier_keepalive. A keepalive nobody remembers installing would otherwise
    resurrect a deliberately stopped verifier every five minutes."""
    if IS_WINDOWS:
        subprocess.run(["schtasks", "/delete", "/f", "/tn", VERIFIER_TASK],
                       capture_output=True, text=True)
        return
    text = _read_crontab()
    if VERIFIER_CRON_TAG in text:
        kept = [ln for ln in text.splitlines() if VERIFIER_CRON_TAG not in ln]
        _write_crontab("\n".join(kept).strip() + "\n")


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


def _linger_enabled(user):
    out = subprocess.run(["loginctl", "show-user", user, "-p", "Linger"],
                         capture_output=True, text=True, check=False)
    return "Linger=yes" in out.stdout


def enable_linger():
    """Make this user's systemd instance start at BOOT, not just at login.

    Without lingering a `--user` unit is bound to a login session, so `WantedBy=
    default.target` fires when somebody logs in and never after an unattended reboot —
    the unit looks correctly installed and enabled the whole time. Try unprivileged
    first (polkit allows self-linger from an active seat), then sudo -n. Returns
    True/False; False is not fatal, it just means cron has to carry the boot half.
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    if not user:
        return False
    if _linger_enabled(user):
        print("  linger already enabled (user services start at boot)")
        return True
    for cmd in (["loginctl", "enable-linger", user],
                ["sudo", "-n", "loginctl", "enable-linger", user]):
        subprocess.run(cmd, capture_output=True, text=True, check=False)
        if _linger_enabled(user):
            print(f"  enabled linger for {user} ({' '.join(cmd[:2])}) — "
                  "the agent now starts at boot with no login")
            return True
    print(f"  ! could not enable linger for {user} (needs root) — "
          "installing the cron fallback so reboots are still covered")
    return False


def _write_keepalive():
    """A tiny guard script cron can call. It lives in its own FILE rather than inline in
    the crontab for a specific reason: the guard's own command line must NOT contain the
    string it greps for, or it matches itself and concludes the agent is already running
    (the `[a]gent.py` bracket trick that sessions.md documents for pkill, one level up)."""
    script = f"""#!/bin/sh
# NEURON agent keepalive — start the agent if it is not already running.
# Installed by agent/install.py; removed by agent/uninstall.py. Safe to run repeatedly:
# it exits immediately when the agent is up, so it never starts a second copy (two agents
# on one machine fight over the node_id and invalidate each other's token).
pgrep -f "[a]gent[.]py" >/dev/null 2>&1 && exit 0
cd "{HERE}" || exit 1
nohup setsid "{sys.executable}" "{os.path.join(HERE, 'agent.py')}" >> "{os.path.join(HERE, 'agent.log')}" 2>&1 &
exit 0
"""
    with open(KEEPALIVE_PATH, "w", newline="\n") as f:
        f.write(script)
    os.chmod(KEEPALIVE_PATH, 0o755)
    return KEEPALIVE_PATH


def _read_crontab():
    out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    return out.stdout if out.returncode == 0 else ""


def _write_crontab(text):
    p = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True, check=False)
    return p.returncode == 0


def install_cron_fallback():
    """@reboot + a 2-minute keepalive, via the user's own crontab (no root needed).

    @reboot alone would cover the reboot case; the recurring line is what makes the node
    self-heal after a crash, an OOM kill or a closed SSH session, and bounds "offline"
    at two minutes rather than "until somebody notices". Existing crontab lines are
    preserved — these machines run other jobs.
    """
    if not shutil.which("crontab"):
        print("  ! no crontab available — this node will NOT come back after a reboot")
        return False
    _write_keepalive()
    keep = [ln for ln in _read_crontab().splitlines() if CRON_TAG not in ln]
    keep += [f"@reboot {KEEPALIVE_PATH} {CRON_TAG}",
             f"*/2 * * * * {KEEPALIVE_PATH} {CRON_TAG}"]
    if not _write_crontab("\n".join(keep).strip() + "\n"):
        print("  ! could not write crontab — this node will NOT come back after a reboot")
        return False
    print(f"  installed cron auto-start + 2-minute keepalive ({KEEPALIVE_PATH})")
    return True


def add_to_startup_linux():
    unit = (
        "[Unit]\nDescription=NEURON agent\nAfter=network-online.target\n\n"
        "[Service]\n"
        f"ExecStart={sys.executable} {os.path.join(HERE, 'agent.py')}\n"
        f"WorkingDirectory={HERE}\n"
        # Restart=on-failure leaves a node dead after any clean-looking exit; on a
        # volunteer network the machine going away IS the normal case, so always.
        "Restart=always\nRestartSec=10\n\n"
        "[Install]\nWantedBy=default.target\n"
    )
    dst = os.path.expanduser("~/.config/systemd/user/neuron-agent.service")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        f.write(unit)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "neuron-agent"], check=False)
    print(f"  created systemd user service: {dst} (Restart=always)")
    # Belt and braces, deliberately: the keepalive is a no-op whenever systemd already has
    # the agent running, and it is the only thing standing between a no-sudo machine and
    # permanent disappearance on the first reboot.
    if not enable_linger():
        install_cron_fallback()


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
    ap.add_argument("--startup", action="store_true",
                    help="install/repair auto-start on an already-configured node, without "
                         "touching config.json or prompting for anything")
    ap.add_argument("--with-verifier", action="store_true",
                    help="OPERATOR ONLY: also auto-start verify_service.py, which promotes "
                         "joining nodes. Needs NEURON_REGISTER_SECRET and PyTorch, so it is "
                         "not part of the normal install.")
    ap.add_argument("--verifier-keepalive", action="store_true",
                    help="OPERATOR ONLY: install just the 5-minute keepalive that restarts "
                         "verify_service.py if it dies, and do nothing else. The repair path "
                         "for [P24] on a machine whose config must not be touched.")
    ap.add_argument("--no-startup", action="store_true", help="do not add to auto-start")
    ap.add_argument("--no-run", action="store_true", help="only write config, do not launch")
    args = ap.parse_args()

    # Deliberately before anything that writes config or launches a process: this flag is a
    # repair applied to a working machine, and it must not have side effects.
    if args.verifier_keepalive:
        print("Installing the verifier keepalive only.")
        add_verifier_keepalive()
        print("Done. It checks every 5 minutes and starts the verifier if it is not running.")
        return

    coordinator = args.coordinator
    # --startup is the repair path for a machine that is already a working node, so it must
    # never block on a prompt (it is run over SSH) and never rewrite a live config.
    if not coordinator and not (args.startup and os.path.exists(CONFIG_PATH)):
        coordinator = input("Coordinator URL [https://neuronnet.duckdns.org]: ").strip() \
            or "https://neuronnet.duckdns.org"

    print(f"Installing NEURON agent (coordinator = {coordinator or 'unchanged'})")
    write_config(coordinator, args.layer_start, args.layer_end)
    if not args.no_startup:
        if IS_WINDOWS:
            add_to_startup_windows()
        elif IS_MACOS:
            add_to_startup_macos()
        else:
            add_to_startup_linux()
    if args.with_verifier:
        add_verifier_to_startup()
    if not args.no_run:
        start_background()
    print("NEURON agent installed. It will register, download its slice, and start earning.")


if __name__ == "__main__":
    main()
