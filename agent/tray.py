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
from agent.agent import Agent, _setup_logging, crash_log   # noqa: E402
from agent import resource_guard                 # noqa: E402
from agent import updater                        # noqa: E402  (LOCAL_VERSION, shown in the tray)

log = logging.getLogger("neuron.agent")   # same file as the agent's own log

# donation levels shown in the tray dial (value -> menu label)
DONATION_LABELS = [
    ("idle", "Idle — only spare compute"),
    ("balanced", "Balanced — while I work"),
    ("generous", "Generous — donate more"),
    ("max", "Max — always on"),
]

# compute device shown in the tray (value -> menu label). Mirrors DONATION_LABELS above.
# "Automatic" is the honest default: it uses a GPU when one is genuinely usable and the CPU
# otherwise, which on this build is always the CPU — see _device_note().
DEVICE_LABELS = [
    ("auto", "Automatic — use a GPU if one works"),
    ("cpu", "CPU only"),
    ("gpu", "GPU"),
]

def _gpu_is_usable():
    """Can this build actually compute on a GPU? Not "is a card present" — that is a different
    question, and answering it instead is what produced a release announcing GPU support that
    could never run. Never raises: the tray must open on any machine."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:                                               # noqa: BLE001
        return False


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
        # What the device was when this process started. `common.DEVICE` is resolved at import
        # and never revisited, so choosing a different one in the menu cannot take effect until
        # a restart — comparing against this is how the menu knows to say so.
        self._device_at_start = self.agent.cfg.get("device", "auto")
        self.ledger = {"balance": 0.0, "total_earned": 0.0}
        self._ledger_error = None            # last non-200 from the ledger poll, if any
        self._last_logged_ledger_error = None   # so one bad status isn't logged every 30s
        # The hover tooltip is the one place a version number is free to read and costs
        # nothing to show. "which version are you running?" was previously unanswerable
        # without opening a log file, on the surface a non-technical volunteer actually uses.
        self.icon = pystray.Icon("neuron", icon_image(COLORS["starting"]),
                                 f"NEURON v{updater.LOCAL_VERSION}", menu=self._menu())
        # What the last notification was about, so a state that persists is announced once
        # rather than every refresh. Notification spam is worse than no notifications: the
        # first thing a user does with an app that cries wolf is silence it permanently.
        self._notified = None

    def _effective_status(self):
        return "idle" if self.agent.user_paused.is_set() else self.agent.state.get("status", "starting")

    def _menu(self):
        def title(_):
            return f"NEURON v{updater.LOCAL_VERSION} — {self.ledger.get('balance', 0):.2f} NRN"

        def status(_):
            # The agent has always computed a `detail` for every state -- "earning", "awaiting
            # verification — not yet earning", "coordinator unreachable: …", the list of
            # reasons it paused -- and this menu threw all of it away and showed one word.
            # "Status: Active" and "Status: Idle" are indistinguishable from a stub, which is
            # what makes the app feel like it is not doing anything. It always knew; it just
            # did not say.
            if self.agent.user_paused.is_set():
                return "Status: Paused by you"
            s = self.agent.state.get("status", "starting").capitalize()
            detail = (self.agent.state.get("detail") or "").strip()
            if not detail:
                return f"Status: {s}"
            # Keep the menu narrow: a long reason list is a tooltip, not a menu row.
            if len(detail) > 46:
                detail = detail[:45].rstrip(" ;,—-") + "…"
            return f"Status: {s} — {detail}"

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
        device_menu = pystray.Menu(*[self._device_item(d, label) for d, label in DEVICE_LABELS])
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
            pystray.MenuItem("Compute device", device_menu),
            # Callable text, like `title`/`status` above: the menu object is built once and
            # re-rendered by update_menu(), so a plain string would freeze at its first value
            # and the note would still claim "applies on restart" after a restart.
            pystray.MenuItem(lambda item: self._device_note(), None, enabled=False,
                             visible=lambda item: bool(self._device_note())),
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

    def _device_item(self, device, label):
        return pystray.MenuItem(
            label, self._set_device(device),
            checked=lambda item, d=device: self.agent.cfg.get("device", "auto") == d,
            radio=True,
        )

    def _set_device(self, device):
        def handler(icon, item):
            self.agent.cfg["device"] = device
            self.agent._save()
            self.icon.update_menu()
        return handler

    def _device_note(self):
        """The line under the device menu. Empty when there is nothing worth saying.

        Two honest statements, never a silent no-op. Picking GPU on this build changes the
        config and changes nothing else, because the shipped torch is CPU-only — a menu that
        appears to work and does not is worse than one that says so. And because
        `common.DEVICE` is resolved at import, ANY change here needs a restart; the tray is the
        only place a user finds that out.
        """
        want = self.agent.cfg.get("device", "auto")
        if want == "gpu" and not _gpu_is_usable():
            return "GPU selected — this build computes on CPU"
        if want != self._device_at_start:
            return "Device change applies on restart"
        return ""

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

    # A beat every PING_SECONDS (~30s); three missed rounds is stopped, not slow. Deliberately
    # generous — a laptop waking from sleep, or a home connection dropping for a minute, must
    # not paint the tray red for something that fixes itself on the next beat.
    BEAT_STALE_S = 5 * 60

    def _watch_heartbeat(self):
        """Is the agent still REACHING the coordinator? ([P51])

        `_supervise_agent` catches a loop that dies. This catches the other half: a loop that is
        still running and no longer getting through — the state that cost 81 minutes on
        2026-08-18, where the process was alive, the port was open, probes were answered
        correctly, and the coordinator had the node down as offline.

        Keyed on the last SUCCESSFUL beat rather than on the loop being alive, because those are
        exactly the two things that came apart. Logged once per transition, not every 30s: an
        alarm that repeats forever is one people silence.
        """
        last = self.agent.state.get("last_beat_at")
        if not last:
            return                      # never beaten yet — startup, not a stall
        stale = (time.time() - last) > self.BEAT_STALE_S
        if stale and not getattr(self, "_beat_warned", False):
            mins = int((time.time() - last) // 60)
            log.error("no successful heartbeat for %d minutes — the coordinator has almost "
                      "certainly dropped this node from routing, and it is earning nothing "
                      "even though this process is still running. See PROBLEMS.md [P51].",
                      mins)
            crash_log(f"no successful heartbeat for {mins} minutes")
            self._beat_warned = True
        elif not stale and getattr(self, "_beat_warned", False):
            log.info("heartbeats are getting through again")
            self._beat_warned = False

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
            self._watch_heartbeat()
            self.icon.icon = icon_image(COLORS.get(self._effective_status(), COLORS["idle"]))
            self.icon.update_menu()
            self._maybe_notify()
            time.sleep(30)

    # -- desktop notifications ------------------------------------------------ #
    # Restrained on purpose. Only events the owner CANNOT otherwise discover, and only on the
    # transition into them -- `_notified` holds the last key, so a state that persists for a
    # day is announced once. An agent that notifies on every poll gets muted by the OS within
    # an hour, and then the one notification that mattered never arrives either.
    #
    # Deliberately NOT notified: pause and resume (the user just did it), and the ordinary
    # active heartbeat (nothing to act on).
    def _notify_key(self):
        """(key, title, message) for a state worth interrupting someone about, or None."""
        st = self.agent.state
        status = st.get("status")
        detail = (st.get("detail") or "").strip()

        if self._ledger_error == 401:
            return ("token", "NEURON — this node's token was superseded",
                    "Another copy of the agent has registered the same node id. Earnings "
                    "shown here are stale until that is resolved.")
        if status == "error":
            # The two that strand a node silently: it looks online and serves nothing.
            return ("error:" + detail[:40], "NEURON — your node is not serving",
                    detail or "The node hit an error; see the log.")
        if self.agent.standing == "probationary":
            return ("probationary", "NEURON — waiting to be verified",
                    "Your node is online and healthy, but a verifier must confirm it before "
                    "it receives requests or earns NRN.")
        if self.agent.standing in ("verified", "trusted") and status == "active":
            return ("serving", "NEURON — your node is verified",
                    "It is now serving requests and earning NRN.")
        return None

    def _maybe_notify(self):
        try:
            entry = self._notify_key()
            if entry is None:
                self._notified = None      # recovered: allow the next occurrence to announce
                return
            key, title, message = entry
            if key == self._notified:
                return
            self._notified = key
            notify = getattr(self.icon, "notify", None)
            if notify:
                notify(message, title)
        except Exception:                                          # noqa: BLE001
            # A notification backend that is missing, disabled by policy, or simply refuses
            # must never take down the tray -- this is decoration on top of a node that is
            # doing its actual job.
            pass

    def _supervise_agent(self):
        """Run the agent loop and make sure its death can never be silent ([P51]).

        This was `threading.Thread(target=self.agent.run)` with no wrapper. An exception in
        that thread goes to `threading.excepthook`, which writes to `sys.stderr` — and tray
        mode is a FROZEN WINDOWED app whose console `_hide_console()` has already hidden, so
        stderr goes nowhere at all. The agent loop could therefore stop dead while the tray
        icon, the poll thread, the Chat UI and `node_server`'s listener all carried on: a
        process that is alive, holding its port, answering probes correctly, and not
        registered — with nothing written anywhere to say so.

        Observed live 2026-08-18: 81 minutes in exactly that state, the coordinator reading
        the node as offline and auto-repair collapsing the network onto the other machine.
        `agent.log` had no line from that process and no `[CRASH]` marker has ever been
        written, which is what a swallowed thread exception looks like from the outside.

        `crash_log` as well as `log.exception`, deliberately: this is the failure class where
        the logging config itself is a suspect, and crash_log writes with the stdlib alone.
        The tray goes red with a reason, because the person watching the icon is the only one
        who can act.
        """
        try:
            self.agent.run()
        except BaseException as e:                     # noqa: BLE001 - nothing may escape here
            log.exception("the agent loop STOPPED: %s", e)
            crash_log(f"the agent loop stopped: {e.__class__.__name__}: {e}")
            self.agent.state.update(
                status="error",
                detail=f"agent stopped ({e.__class__.__name__}) — restart NEURON")
            return
        # Returning is also a stop. `run()` is an endless loop, so reaching here at all means
        # something broke out of it -- and a silent return would be indistinguishable from a
        # healthy agent, which is the whole defect this method exists to close.
        log.error("the agent loop RETURNED without raising — it is no longer heartbeating, "
                  "so this node has stopped earning. Restart NEURON.")
        crash_log("the agent loop returned without raising")
        self.agent.state.update(status="error",
                                detail="agent stopped — restart NEURON")

    def run(self):
        threading.Thread(target=self._supervise_agent, daemon=True).start()
        threading.Thread(target=self._poll, daemon=True).start()
        self.icon.run()


def main():
    _setup_logging("INFO")   # the tray hides the console, so keep a record in agent.log
    try:
        Tray().run()
    except BaseException:
        # Constructing the Tray reads the config and builds an Agent, either of which can
        # throw on an in-place upgrade ([P24]). In windowed mode that traceback has no console
        # to land in, so it must reach the file explicitly.
        log.exception("the tray app failed to start")
        crash_log("the tray app failed to start")
        raise


if __name__ == "__main__":
    main()
