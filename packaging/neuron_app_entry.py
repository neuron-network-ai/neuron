"""Frozen-app entry for the NEURON desktop app.

Dispatch by argument so one built exe covers every role:
  (no args)     -> system-tray app: runs the agent AND shows the tray icon with live
                   NRN balance / status / Pause / Dashboard / Quit (the desktop experience)
  --headless    -> run the agent with no tray (servers / no GUI)
  --deregister  -> deregister this node and delete its slice + config (the uninstaller calls this)

Kept separate from agent/agent.py so the top-level frozen script isn't named `agent` (which would
shadow the agent package and cause a circular import). The `--deregister` path imports only the
light uninstall module (no torch), so it runs fast.
"""
import sys


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
    _main()
