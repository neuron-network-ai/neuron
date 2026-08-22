"""Clicking NEURON shows you NEURON — run: python -m agent.test_one_door

The app had no door. Every shortcut the installer creates points at the exe, the exe starts a
background process whose only surface is a tray icon Windows hides behind a chevron, and
nothing ever opened a browser. A person installed it, ticked "Start NEURON now", and watched
nothing happen. The Chat UI existed and was unreachable unless you already knew it was there.

Worse, the same double-click did two different things depending on state nobody can see: with
nothing running it started an agent silently; with one running it started a SECOND agent, which
lost a fight over port 50999 and the Chat UI's port and retried forever.

One rule replaces both: **a deliberate launch always ends at the Chat UI.** The distinction that
keeps a tab from ambushing somebody at sign-in is INTENT, not a counter — the startup shortcut
passes `--startup`, a human passes nothing. An earlier attempt used a marker file to open the
page exactly once per install, which is worse than it sounds: on day two the person is back to
hunting the hidden icon, which is the problem being solved.
"""
import os
import socket
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util as _u  # noqa: E402

import agent.agent as agentmod  # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _entry():
    """The frozen app's entry module, loaded by path (it is not importable as a package)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = _u.spec_from_file_location("neuron_entry",
                                      os.path.join(root, "packaging", "neuron_app_entry.py"))
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class FakeAgent:
    def __init__(self, port):
        self.cfg = {"local_chat_port": port}
        self._stop = threading.Event()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    entry = _entry()
    import webbrowser
    real_open = webbrowser.open
    opened = []
    webbrowser.open = lambda url: opened.append(url) or True
    prev_env = os.environ.get("NEURON_NO_BROWSER")
    os.environ.pop("NEURON_NO_BROWSER", None)
    try:
        # -- ONE INSTANCE. A second launch must not start a second agent ---------------- #
        first = entry._claim_single_instance()
        check("the first launch gets the lock", hasattr(first, "bind"))
        second = entry._claim_single_instance()
        check("a second launch is refused, so no rival agent starts", second is None)
        first.close()
        third = entry._claim_single_instance()
        check("...and the lock is released on quit, so it can start again",
              hasattr(third, "bind"))
        third.close()

        # -- the door leads somewhere: a refused launch opens the Chat UI --------------- #
        url = entry._chat_url()
        check("the entry knows where the Chat UI is, without importing the torch chain",
              url.startswith("http://127.0.0.1:"))
        # It must NOT reach into agent.agent for this: that import is the heaviest in the
        # product, and this runs when somebody double-clicks expecting a browser tab.
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "packaging", "neuron_app_entry.py"), encoding="utf-8").read()
        chat_fn = src[src.index("def _chat_url"):src.index("def _main")]
        check("...and _chat_url imports nothing from agent.agent",
              "from agent" not in chat_fn and "import agent" not in chat_fn)

        # -- A DELIBERATE LAUNCH OPENS THE PAGE, EVERY TIME ----------------------------- #
        port = _free_port()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", port))
        srv.listen(5)
        def _accept_quietly():
            # Swallows the OSError the close() below raises in here. A passing test that also
            # prints a stack trace makes somebody stop and read it — the ambiguity is the cost,
            # not the exception.
            while True:
                try:
                    srv.accept()[0].close()
                except OSError:
                    return

        threading.Thread(target=_accept_quietly, daemon=True).start()

        opened.clear()
        agentmod.Agent._open_chat_when_ready(FakeAgent(port))
        time.sleep(2.5)
        check("a launch opens the Chat UI once the port answers",
              opened == [f"http://127.0.0.1:{port}"])

        # THE REGRESSION THAT MATTERS: this used to be once per install.
        opened.clear()
        agentmod.Agent._open_chat_when_ready(FakeAgent(port))
        time.sleep(2.5)
        check("...and again on the NEXT launch, because a door people can only use once "
              "is not a door", opened == [f"http://127.0.0.1:{port}"])

        # -- but the machine starting itself is not a person clicking ------------------- #
        os.environ["NEURON_NO_BROWSER"] = "1"
        opened.clear()
        agentmod.Agent._open_chat_when_ready(FakeAgent(port))
        time.sleep(1.5)
        check("started at sign-in, nothing is opened", opened == [])
        os.environ.pop("NEURON_NO_BROWSER", None)

        # -- and it waits rather than greeting anyone with a connection error ----------- #
        srv.close()
        dead = _free_port()
        opened.clear()
        agentmod.Agent._open_chat_when_ready(FakeAgent(dead))
        time.sleep(2.5)
        check("with nothing listening yet, no browser is opened", opened == [])

        # -- the wiring the installer depends on ---------------------------------------- #
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        iss = open(os.path.join(root, "packaging", "neuron.iss"), encoding="utf-8").read()
        startup_line = next((l for l in iss.splitlines()
                             if "{userstartup}" in l and "Filename:" in l), "")
        check("the sign-in shortcut passes --startup", '"--startup"' in startup_line)
        desktop_line = next((l for l in iss.splitlines()
                             if "{autodesktop}" in l and "Filename:" in l), "")
        check("...and the desktop shortcut does NOT, so clicking it opens the page",
              desktop_line and "--startup" not in desktop_line)
        check("the entry strips --startup before argparse ever sees it",
              '[a for a in args if a != "--startup"]' in src)
    finally:
        webbrowser.open = real_open
        if prev_env is None:
            os.environ.pop("NEURON_NO_BROWSER", None)
        else:
            os.environ["NEURON_NO_BROWSER"] = prev_env

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
