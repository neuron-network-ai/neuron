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


def _main():
    args = sys.argv[1:]
    if "--deregister" in args:
        from agent.uninstall import main
        main()
    elif "--headless" in args:
        from agent.agent import main
        main()
    else:
        from agent.tray import main
        main()


if __name__ == "__main__":
    _main()
