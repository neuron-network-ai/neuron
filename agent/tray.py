"""
agent/tray.py — system tray icon + menu for the NEURON agent.

Runs the agent in a background thread and shows a tray icon whose colour reflects
state, with a menu showing NRN balance and Pause/Resume/Dashboard/Quit. Cross-
platform via pystray + Pillow (both ARM-compatible). Refreshes every 30 s from
GET /ledger/{node_id}. On desktops this is the agent's face; headless servers run
agent.py directly instead.

  python tray.py
"""
import json
import logging
import os
import sys
import threading
import time
import webbrowser

import requests
from PIL import Image, ImageDraw
import pystray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.agent import Agent, _setup_logging   # noqa: E402
from agent import resource_guard                 # noqa: E402

log = logging.getLogger("neuron.agent")   # same file as the agent's own log

# donation levels shown in the tray dial (value -> menu label)
DONATION_LABELS = [
    ("idle", "Idle — only spare compute"),
    ("balanced", "Balanced — while I work"),
    ("generous", "Generous — donate more"),
    ("max", "Max — always on"),
]

COLORS = {
    "active": (46, 160, 67),        # green — earning
    "idle": (150, 150, 150),        # grey — user active / paused
    "downloading": (240, 200, 20),  # yellow — fetching slice
    "error": (200, 40, 40),         # red — coordinator unreachable
    "starting": (150, 150, 150),
}


def icon_image(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color)
    return img


class Tray:
    def __init__(self, config_path=None):
        self.agent = Agent(config_path) if config_path else Agent()
        self.ledger = {"balance": 0.0, "total_earned": 0.0}
        self._ledger_error = None            # last non-200 from the ledger poll, if any
        self._last_logged_ledger_error = None   # so one bad status isn't logged every 30s
        self.icon = pystray.Icon("neuron", icon_image(COLORS["starting"]), "NEURON",
                                 menu=self._menu())

    def _effective_status(self):
        return "idle" if self.agent.user_paused.is_set() else self.agent.state.get("status", "starting")

    def _menu(self):
        def title(_):
            return f"NEURON — {self.ledger.get('balance', 0):.2f} NRN balance"

        def status(_):
            s = "Paused" if self.agent.user_paused.is_set() else \
                self.agent.state.get("status", "starting").capitalize()
            return f"Status: {s}"

        def earned(_):
            return f"Total earned: {self.ledger.get('total_earned', 0):.2f} NRN"

        def pause_label(_):
            return "Resume" if self.agent.user_paused.is_set() else "Pause"

        def chat_label(_):
            # "starting…" was shown for every non-ready state, including the one where it had
            # already failed and was never coming back. A permanently greyed "starting…" tells
            # the owner nothing and looks like the app is broken rather than one optional part.
            state = getattr(self.agent, "local_chat_state", "pending")
            if self._chat_ready():
                return "Open Chat UI"
            return {"failed": "Chat UI unavailable — see agent.log",
                    "disabled": "Chat UI disabled in config",
                    }.get(state, "Chat UI (starting…)")

        def balance_note(_):
            # Distinguish "you have earned nothing" from "we could not read your balance".
            return f"⚠ balance unavailable (HTTP {self._ledger_error})"

        donation = pystray.Menu(*[self._mode_item(m, label) for m, label in DONATION_LABELS])
        return pystray.Menu(
            pystray.MenuItem(title, None, enabled=False),
            pystray.MenuItem(status, None, enabled=False),
            pystray.MenuItem(earned, None, enabled=False),
            pystray.MenuItem(balance_note, None, enabled=False,
                             visible=lambda item: self._ledger_error is not None),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(chat_label, self._open_chat, enabled=lambda item: self._chat_ready()),
            pystray.MenuItem("API Docs", self._open_api_docs, enabled=lambda item: self._chat_ready()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_label, self._toggle_pause),
            pystray.MenuItem("Donation level", donation),
            pystray.MenuItem("My Dashboard", self._open_my_dashboard),
            pystray.MenuItem("Network Dashboard", self._open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

    # -- personal Chat UI / API (agent/local_chat.py, api/openai_compat.py) -- #
    def _chat_ready(self):
        return self.agent.local_chat_server is not None

    def _chat_base_url(self):
        port = self.agent.cfg.get("local_chat_port", 8080)
        return f"http://127.0.0.1:{port}"

    def _open_chat(self, icon, item):
        webbrowser.open(self._chat_base_url())

    def _open_api_docs(self, icon, item):
        webbrowser.open(f"{self._chat_base_url()}/api-docs")

    # -- donation dial ------------------------------------------------------- #
    def _mode_item(self, mode, label):
        return pystray.MenuItem(
            label, self._set_mode(mode),
            checked=lambda item, m=mode: self.agent.cfg.get("donation_mode", "idle") == m,
            radio=True,
        )

    def _set_mode(self, mode):
        def handler(icon, item):
            self.agent.cfg["donation_mode"] = mode
            self.agent.cfg.pop("max_cpu_pct", None)         # the dial supersedes the old override
            # rebuild the guard live so the change takes effect immediately (no restart)
            self.agent.guard = resource_guard.ResourceGuard(
                mode, self.agent.cfg.get("idle_threshold_seconds", 60))
            self.agent._save()
            self.icon.update_menu()
        return handler

    def _toggle_pause(self, icon, item):
        if self.agent.user_paused.is_set():
            self.agent.user_paused.clear()
        else:
            self.agent.user_paused.set()
        self.icon.update_menu()

    def _open_dashboard(self, icon, item):
        webbrowser.open(f"{self.agent.base}/dashboard")

    def _creds(self):
        """This node's CURRENT id and token, re-read from disk each time.

        The coordinator mints a fresh node_token on every registration, so the copy this
        process loaded at startup can go stale -- a re-registration (relay ticket refresh, a
        second copy of the agent, a restart that needs one) invalidates it. A stale token makes
        the earnings page 401 and, worse, makes the balance poll below silently return nothing,
        so the tray sits at "0.00 NRN" forever and the owner concludes they are earning zero.
        Whoever registered last wrote the good token to config.json; read it from there.
        """
        cfg = self.agent.cfg
        try:
            with open(self.agent.config_path) as f:
                disk = json.load(f)
            if disk.get("node_token"):
                cfg = disk
        except (OSError, ValueError):
            pass
        return cfg.get("node_id"), cfg.get("node_token")

    def _open_my_dashboard(self, icon, item):
        """The node's own token-gated page (balance/earned/served). Falls back to the
        public network dashboard if this machine hasn't registered yet."""
        nid, tok = self._creds()
        if nid and tok:
            webbrowser.open(f"{self.agent.base}/node/{nid}/dashboard?token={tok}")
        else:
            webbrowser.open(f"{self.agent.base}/dashboard")

    def _quit(self, icon, item):
        self.agent.stop()
        icon.stop()

    def _poll(self):
        while True:
            nid, tok = self._creds()
            if nid and tok:
                try:
                    # the ledger is private to this node -> authenticate with our own token
                    r = requests.get(f"{self.agent.base}/ledger/{nid}", timeout=8,
                                     headers={"X-Node-Token": tok})
                    if r.status_code == 200:
                        self.ledger = r.json()
                        self._ledger_error = None
                    else:
                        # Never swallow this. A 401 here means the displayed balance is not
                        # "you have earned nothing", it is "we could not ask" -- and those look
                        # identical in the menu. Unreported, it reads as the network not paying.
                        self._ledger_error = r.status_code
                        if r.status_code != self._last_logged_ledger_error:
                            log.warning("cannot read this node's balance (HTTP %d) — the "
                                        "displayed earnings are stale, not zero%s",
                                        r.status_code,
                                        "; this node's token has been superseded, most likely "
                                        "by another copy of the agent registering the same "
                                        "node id" if r.status_code == 401 else "")
                            self._last_logged_ledger_error = r.status_code
                except requests.RequestException:
                    pass
            self.icon.icon = icon_image(COLORS.get(self._effective_status(), COLORS["idle"]))
            self.icon.update_menu()
            time.sleep(30)

    def run(self):
        threading.Thread(target=self.agent.run, daemon=True).start()
        threading.Thread(target=self._poll, daemon=True).start()
        self.icon.run()


def main():
    _setup_logging("INFO")   # the tray hides the console, so keep a record in agent.log
    Tray().run()


if __name__ == "__main__":
    main()
