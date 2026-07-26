"""
agent/tray.py — system tray icon + menu for the NEURON agent.

Runs the agent in a background thread and shows a tray icon whose colour reflects
state, with a menu showing NRN balance and Pause/Resume/Dashboard/Quit. Cross-
platform via pystray + Pillow (both ARM-compatible). Refreshes every 30 s from
GET /ledger/{node_id}. On desktops this is the agent's face; headless servers run
agent.py directly instead.

  python tray.py
"""
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

        donation = pystray.Menu(*[self._mode_item(m, label) for m, label in DONATION_LABELS])
        return pystray.Menu(
            pystray.MenuItem(title, None, enabled=False),
            pystray.MenuItem(status, None, enabled=False),
            pystray.MenuItem(earned, None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(pause_label, self._toggle_pause),
            pystray.MenuItem("Donation level", donation),
            pystray.MenuItem("Open Dashboard", self._open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self._quit),
        )

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

    def _quit(self, icon, item):
        self.agent.stop()
        icon.stop()

    def _poll(self):
        while True:
            nid = self.agent.state.get("node_id")
            if nid:
                try:
                    r = requests.get(f"{self.agent.base}/ledger/{nid}", timeout=8)
                    if r.status_code == 200:
                        self.ledger = r.json()
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
