"""packaging/test_app_entry.py — run: python packaging/test_app_entry.py

(a script, not `-m`: packaging/ is deliberately not a Python package — it holds build inputs.)

Covers neuron_app_entry's argument dispatch. This is the ONLY entry point of the shipped
Windows app, and nothing tested it: `neuron-agent.exe --headless` exited 2 with
"unrecognized arguments: --headless", because the dispatch flag was passed straight through to
agent.main()'s argparse. Headless is the documented mode for servers and anything without a
GUI, and it was dead in every build.
"""
import sys
import types

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def _entry():
    """Import the entry module without PyInstaller around it."""
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_app_entry.py")
    spec = importlib.util.spec_from_file_location("neuron_app_entry", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run(argv, entry):
    """Dispatch with a fake argv, capturing which mode ran and what argv it saw."""
    seen = {}
    fake_agent = types.ModuleType("agent.agent")
    fake_agent.main = lambda: seen.update(mode="headless", argv=list(sys.argv))
    fake_tray = types.ModuleType("agent.tray")
    fake_tray.main = lambda: seen.update(mode="tray", argv=list(sys.argv))
    fake_uninst = types.ModuleType("agent.uninstall")
    fake_uninst.main = lambda: seen.update(mode="deregister", argv=list(sys.argv))
    saved = {k: sys.modules.get(k) for k in ("agent.agent", "agent.tray", "agent.uninstall")}
    sys.modules.update({"agent.agent": fake_agent, "agent.tray": fake_tray,
                        "agent.uninstall": fake_uninst})
    real_argv, real_hide = sys.argv, entry._hide_console
    sys.argv = ["neuron-agent.exe"] + argv
    entry._hide_console = lambda: None
    try:
        entry._main()
    finally:
        sys.argv, entry._hide_console = real_argv, real_hide
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return seen


def main():
    e = _entry()

    check("no args -> tray mode", run([], e).get("mode") == "tray")
    check("--deregister -> uninstaller", run(["--deregister"], e).get("mode") == "deregister")

    r = run(["--headless"], e)
    check("--headless -> agent mode", r.get("mode") == "headless")
    # the regression: agent.main() runs argparse over sys.argv, so the dispatch flag must be
    # gone by the time it is called or it exits 2 before doing anything.
    check("--headless is STRIPPED before agent.main() parses argv",
          "--headless" not in r.get("argv", []))

    r2 = run(["--headless", "--donation-mode", "max", "--port", "50999"], e)
    check("other flags still reach the agent",
          "--donation-mode" in r2.get("argv", []) and "max" in r2.get("argv", [])
          and "--port" in r2.get("argv", []))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
