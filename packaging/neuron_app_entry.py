"""Frozen-app entry for the NEURON desktop app.

Dispatch by argument so one built exe covers every role:
  (no args)     -> system-tray app: runs the agent AND shows the tray icon with live
                   NRN balance / status / Pause / Dashboard / Quit (the desktop experience)
  --startup     -> launched by the sign-in shortcut: identical, but opens no browser
  --headless    -> run the agent with no tray (servers / no GUI)
                   (headless deliberately takes NO single-instance lock: a server operator
                   running two configs on one box is a real case, and the ports they would
                   collide on already say so themselves)
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


# The port this app holds purely to mean "an instance is running". Nothing is served on it.
#
# A LOCK, NOT A FILE. A pid file survives a power cut and then lies about a process that no
# longer exists, and cleaning that up correctly is more code than this. A bound socket dies with
# the process it belongs to, which is the property actually wanted.
SINGLE_INSTANCE_PORT = 50998


def _claim_single_instance():
    """The lock socket if this is the only instance, or None if one is already running.

    Returns the socket so the CALLER holds it: letting it fall out of scope would close it and
    release the lock immediately, which is the classic way this pattern silently does nothing.

    Fails OPEN. If the socket cannot be created at all -- no loopback, a locked-down host, an
    antivirus intercepting it -- this returns the socket-less "go ahead" rather than refusing to
    start. A duplicate process is recoverable by quitting one; an app that will not launch is
    a support ticket from somebody who has already uninstalled it.
    """
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except OSError:
        return "unknown"                     # cannot tell -> start normally
    try:
        # No SO_REUSEADDR, deliberately: it would let a second instance bind the same port and
        # the lock would never fire.
        s.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
        s.listen(1)
        return s
    except OSError:
        s.close()
        return None                          # somebody already holds it


def _chat_url():
    """Where this machine's Chat UI lives.

    Reads config.json DIRECTLY rather than importing agent.agent for its CONFIG_PATH. That
    import is the heaviest chain in the product -- torch included, per this file's own opening
    note -- and this runs on the path a person takes when they double-click the icon to see the
    app. Making them wait through a model framework's import to open a browser tab would be
    absurd, and if that chain is broken this is the one path that must still work.

    Same rule as `_log_path` above, and duplicated for the same reason.
    """
    port = 8080
    try:
        import json
        with open(os.path.join(os.path.dirname(_log_path()), "config.json"),
                  encoding="utf-8") as f:
            port = json.load(f).get("local_chat_port") or 8080
    except Exception:                                # noqa: BLE001 - a default is fine here
        pass
    return f"http://127.0.0.1:{port}"


def _main():
    args = sys.argv[1:]
    if "--startup" in args:
        # Launched by the sign-in shortcut, not by a person. The agent reads this and stays
        # quiet; everything else about the run is identical. An env var rather than a threaded
        # parameter because it has to survive the dispatch into agent.main() and the tray.
        os.environ["NEURON_NO_BROWSER"] = "1"
        args = [a for a in args if a != "--startup"]
        sys.argv = [sys.argv[0]] + args
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
        # A SECOND LAUNCH OPENS THE PRODUCT INSTEAD OF FIGHTING THE FIRST.
        #
        # Every shortcut the installer creates -- Start Menu, desktop, startup -- points at this
        # exe, and a person who wants to "open NEURON" double-clicks one of them. Until now that
        # started a second agent, which then lost a fight over port 50999 and the Chat UI's own
        # port and retried forever: two processes, one of them useless, and no window either way.
        #
        # The tray was the ONLY route to the Chat UI, behind an icon Windows hides by default.
        # So the same double-click that used to do harm now does the obvious thing.
        lock = _claim_single_instance()
        if lock is None:
            import webbrowser
            webbrowser.open(_chat_url())
            return
        _hide_console()          # windowed tray: no lingering console window
        from agent.tray import main
        # Held for the life of the process. Assigned to a module global rather than left as a
        # local, so it cannot be garbage-collected out from under the lock it represents.
        globals()["_INSTANCE_LOCK"] = lock
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
