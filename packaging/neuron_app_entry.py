"""Frozen-app entry for the NEURON desktop app.

Dispatch by argument so one built exe covers every role:
  (no args)     -> system-tray app: runs the agent AND shows the tray icon with live
                   NRN balance / status / Pause / Dashboard / Quit (the desktop experience)
  --headless    -> run the agent with no tray (servers / no GUI)
  --deregister  -> deregister this node and delete its slice + config (the uninstaller calls this)

Kept separate from agent/agent.py so the top-level frozen script isn't named `agent` (which would
shadow the agent package and cause a circular import). The `--deregister` path imports only the
light uninstall module (no torch), so it runs fast.

**Nothing here may fail silently.** Tray mode hides the console before it imports anything, so a
failure in that import chain — the heaviest one in the product, torch included — produced no
console output, no log file and no window: a double-clicked exe that did nothing at all. That is
[P24]'s "0.18 produced no log". So this file records its own death to agent.log with the stdlib
alone, before it can depend on any of the code that might be broken.
"""
import os
import sys
import time
import traceback


def _log_path():
    """Where agent.log lives — computed WITHOUT importing agent.agent.

    Must mirror agent/agent.py's own rule: the frozen app keeps writable state in
    %LOCALAPPDATA%\\NEURON because it is installed into read-only Program Files. Duplicated
    deliberately: the import that fails is the one being recorded, so this cannot depend on it.
    """
    if getattr(sys, "frozen", False):
        base = (os.environ.get("LOCALAPPDATA")
                or os.path.join(os.path.expanduser("~"), ".local", "share"))
        return os.path.join(base, "NEURON", "agent.log")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "agent", "agent.log")


def _record_crash(what):
    """Append the current traceback to agent.log. Best effort, never raises."""
    path = _log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} [CRASH] {what}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass
    try:
        # ASCII only: the console codepage here may not be UTF-8, unlike the file.
        print(f"NEURON: {what} - details in {path}", file=sys.stderr)
    except Exception:
        pass
    return path


def _tell_the_user(path):
    """Windowed mode has no console to print to, so put the log's location on screen.

    Without this a tray-mode crash is invisible to the person it happened to: they
    double-click, nothing appears, and there is nothing to report. One message box turns that
    into "here is the file to send".
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            None,
            f"NEURON could not start.\n\nThe details were written to:\n{path}\n\n"
            f"Sending that file is enough to diagnose it.",
            "NEURON", 0x10)          # MB_ICONERROR
    except Exception:
        pass


def _hide_console():
    """Tray mode shows a GUI (the tray icon), so hide the console window PyInstaller
    allocates — that's what makes it a windowed app. --headless keeps its console for logs.
    Windows only; no-op elsewhere / if anything goes wrong."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)   # SW_HIDE
    except Exception:
        pass


def _main():
    args = sys.argv[1:]
    if "--deregister" in args:
        from agent.uninstall import main
        main()
    elif "--headless" in args:
        # Strip the dispatch flag before handing over: agent.main() parses sys.argv with
        # argparse, which does not know --headless and exits 2 with
        # "unrecognized arguments: --headless". The whole headless mode was unusable in the
        # packaged app -- `neuron-agent.exe --headless` printed a usage error and quit. Every
        # other flag (--config, --donation-mode, --port, ...) must still reach it, so filter
        # only this one rather than clearing argv.
        sys.argv = [sys.argv[0]] + [a for a in args if a != "--headless"]
        from agent.agent import main
        main()
    else:
        _hide_console()          # windowed tray: no lingering console window
        from agent.tray import main
        main()


if __name__ == "__main__":
    windowed = not ({"--headless", "--deregister"} & set(sys.argv[1:]))
    try:
        _main()
    except SystemExit:
        raise                    # argparse and clean exits are not crashes
    except BaseException:
        path = _record_crash("the NEURON app failed to start")
        if windowed:
            _tell_the_user(path)
        raise
