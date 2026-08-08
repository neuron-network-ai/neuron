"""agent/test_tray.py — run: python -m agent.test_tray

The tray IS the product for a stranger on Windows: it is the only thing they see, and the only
place they learn what they have earned. It had no tests, and three separate bugs shipped in it:

  1. the earnings link and the balance poll both used the node_token this process loaded at
     STARTUP. The coordinator re-mints that token on every registration, so after any
     re-registration the link 401s and the poll silently returns nothing -- leaving the menu
     reading "0.00 NRN" forever, which looks exactly like a network that does not pay.
  2. a non-200 from the ledger was swallowed entirely, so there was no way to tell
     "you earned nothing" from "we could not ask".
  3. a Chat UI that had already FAILED showed "Chat UI (starting…)", greyed, permanently.

pystray/PIL are stubbed so this runs headless.
"""
import json
import os
import sys
import tempfile
import types

# --- stub the GUI deps before importing the tray -------------------------------------- #
_ps = types.ModuleType("pystray")


class _MenuItem:
    def __init__(self, text, action=None, **kw):
        self.text, self.action, self.kw = text, action, kw


class _Menu:
    SEPARATOR = object()

    def __init__(self, *items):
        self.items = items


class _Icon:
    def __init__(self, name=None, icon=None, title=None, **k):
        self.icon = icon
        # Recorded because the tray shows the version here. A stub that silently drops an
        # attribute turns every assertion about it into a vacuous pass.
        self.title = title
        self.notified = []

    def notify(self, message, title=None):
        self.notified.append((title, message))

    def update_menu(self):
        pass

    def run(self):
        pass


_ps.MenuItem, _ps.Menu, _ps.Icon = _MenuItem, _Menu, _Icon
sys.modules.setdefault("pystray", _ps)

_pil = types.ModuleType("PIL")


class _Img:
    @staticmethod
    def new(*a, **k):
        return object()


class _Draw:
    def __init__(self, *a, **k):
        pass

    def ellipse(self, *a, **k):
        pass


_pil.Image, _pil.ImageDraw = _Img, types.SimpleNamespace(Draw=_Draw)
sys.modules.setdefault("PIL", _pil)

from agent import tray as traymod          # noqa: E402
from agent import agent as agentmod        # noqa: E402

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def _tray(tmpdir, token="tok-original"):
    path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="n1", node_token=token, layer_start=0, layer_end=9, local_chat=True)
    json.dump(cfg, open(path, "w"))
    return traymod.Tray(config_path=path), path


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron-tray-")

    # 1) a token rotated on disk by a re-registration must be picked up
    t, path = _tray(tmpdir)
    check("reads the startup token before anything changes", t._creds() == ("n1", "tok-original"))
    cfg = json.load(open(path))
    cfg["node_token"] = "tok-rotated"
    json.dump(cfg, open(path, "w"))
    check("picks up a token rotated on disk (the 401 / stuck-at-0.00 bug)",
          t._creds() == ("n1", "tok-rotated"))

    # the dashboard URL must carry the CURRENT token, not the one loaded at startup
    opened = []
    real_open, traymod.webbrowser.open = traymod.webbrowser.open, opened.append
    try:
        t._open_my_dashboard(None, None)
    finally:
        traymod.webbrowser.open = real_open
    check("My Dashboard link uses the current token",
          opened and "tok-rotated" in opened[0] and "/node/n1/dashboard" in opened[0])

    # 2) a non-200 from the ledger is surfaced, not swallowed
    t2, _ = _tray(tmpdir)
    calls = []

    class _Resp:
        status_code = 401

        def json(self):
            return {}

    real_get, real_sleep = traymod.requests.get, traymod.time.sleep

    def fake_get(*a, **k):
        calls.append(1)
        return _Resp()

    def stop_after_one(_s):
        raise KeyboardInterrupt

    traymod.requests.get, traymod.time.sleep = fake_get, stop_after_one
    try:
        t2._poll()
    except KeyboardInterrupt:
        pass
    finally:
        traymod.requests.get, traymod.time.sleep = real_get, real_sleep
    check("a 401 on the ledger is recorded, not swallowed", t2._ledger_error == 401)
    check("...and the balance is NOT overwritten with a fake zero",
          t2.ledger.get("total_earned", 0) == 0.0 and t2._ledger_error is not None)

    # 3) chat label distinguishes failed / disabled / starting
    t3, _ = _tray(tmpdir)
    labels = {}
    for state in ("pending", "failed", "disabled"):
        t3.agent.local_chat_state = state
        t3.agent.local_chat_server = None
        menu = t3._menu()
        labels[state] = next(i.text(None) for i in menu.items
                             if isinstance(i, traymod.pystray.MenuItem) and callable(i.text)
                             and "Chat UI" in str(i.text(None)))
    check("a FAILED chat says so instead of 'starting…' forever",
          "unavailable" in labels["failed"])
    check("a disabled chat says disabled", "disabled" in labels["disabled"].lower())
    check("a genuinely starting chat still says starting", "starting" in labels["pending"])

    # 4) the compute-device dial (Phase 3). The menu must not merely LOOK like it worked:
    #    picking GPU on a build with a CPU-only torch has to say so, because that is exactly
    #    the class of silent no-op v0.18.0 shipped ([P31]).
    t4, path4 = _tray(tmpdir)

    def device_menu(tray):
        for i in tray._menu().items:
            if isinstance(i, traymod.pystray.MenuItem) and i.text == "Compute device":
                return i.action
        return None

    def note(tray):
        for i in tray._menu().items:
            if (isinstance(i, traymod.pystray.MenuItem) and callable(i.text)
                    and i.action is None and i.kw.get("enabled") is False):
                txt = i.text(None)
                if txt and ("device" in txt.lower() or "GPU" in txt):
                    return txt
        return ""

    dm = device_menu(t4)
    check("the menu offers a Compute device submenu", dm is not None)
    check("...with automatic, CPU and GPU",
          [d for d, _ in traymod.DEVICE_LABELS] == ["auto", "cpu", "gpu"])
    check("...as radio items that tick the configured device",
          all(i.kw.get("radio") for i in dm.items)
          and dm.items[0].kw["checked"](None) is True)      # DEFAULT_CONFIG device is "auto"

    # selecting a device persists it, so it survives the restart it requires
    dm.items[1].action(None, None)                          # "CPU only"
    check("choosing a device saves it to the config file",
          json.load(open(path4))["device"] == "cpu")
    check("...and the menu now ticks it",
          device_menu(t4).items[1].kw["checked"](None) is True)
    check("...and says the change needs a restart, since common.DEVICE is import-time",
          "restart" in note(t4).lower())

    dm2 = device_menu(t4)
    dm2.items[2].action(None, None)                         # "GPU"
    if not traymod._gpu_is_usable():
        check("choosing GPU on a CPU-only build says so instead of silently doing nothing",
              "computes on CPU" in note(t4))

    dm3 = device_menu(t4)
    dm3.items[0].action(None, None)                         # back to "Automatic"
    check("returning to the starting device clears the restart note", note(t4) == "")

    # 5) the tray says what the app is and what it is doing
    #    "Status: Active" with no version and no detail is indistinguishable from a stub, and
    #    the agent had computed a `detail` for every state all along -- the menu discarded it.
    t5, _ = _tray(tmpdir)

    def row(tray, prefix):
        for i in tray._menu().items:
            if isinstance(i, traymod.pystray.MenuItem) and callable(i.text):
                txt = i.text(None)
                if txt and txt.startswith(prefix):
                    return txt
        return ""

    from agent import updater as _upd
    check("the tray names the version it is running",
          _upd.LOCAL_VERSION in row(t5, "NEURON v"))
    check("...and so does the hover tooltip, where it costs nothing to read",
          bool(t5.icon.title) and _upd.LOCAL_VERSION in t5.icon.title)

    t5.agent.state.update(status="active", detail="earning")
    check("the status row shows the detail the agent already computed",
          row(t5, "Status:") == "Status: Active — earning")
    t5.agent.state.update(status="error", detail="coordinator unreachable: timed out")
    check("an error state says what went wrong, not just 'Error'",
          "coordinator unreachable" in row(t5, "Status:"))
    t5.agent.state.update(status="idle", detail="x" * 90)
    check("a very long detail is truncated rather than stretching the menu",
          len(row(t5, "Status:")) < 70)
    t5.agent.user_paused.set()
    check("a user-initiated pause says so plainly",
          row(t5, "Status:") == "Status: Paused by you")
    t5.agent.user_paused.clear()

    # 6) notifications: only on transitions, and never fatal
    t6, _ = _tray(tmpdir)
    sent = []
    t6.icon.notify = lambda msg, title=None: sent.append((title, msg))

    t6.agent.standing = "probationary"
    t6.agent.state.update(status="active", detail="awaiting verification")
    t6._maybe_notify()
    check("a probationary node tells its owner it is not yet earning", len(sent) == 1)
    t6._maybe_notify(); t6._maybe_notify()
    check("...once, not on every poll — an app that cries wolf gets muted", len(sent) == 1)

    t6.agent.standing = "verified"
    t6._maybe_notify()
    check("promotion to verified is announced", len(sent) == 2 and "verified" in sent[-1][0])

    t6.agent.state.update(status="error", detail="node server not listening")
    t6._maybe_notify()
    check("a node that has stopped serving is announced",
          len(sent) == 3 and "not serving" in sent[-1][0])
    t6._maybe_notify()
    check("...and not repeated while it persists", len(sent) == 3)

    # Pause is user-initiated: they just clicked it, so telling them is noise.
    before = len(sent)
    t6.agent.state.update(status="idle", detail="paused by user")
    t6.agent.standing = "verified"
    t6._maybe_notify()
    check("an ordinary idle state is not notified", len(sent) == before)

    # A backend that refuses must never take down the tray.
    def _boom(msg, title=None):
        raise RuntimeError("no notification service")
    t6.icon.notify = _boom
    t6._notified = None
    t6.agent.standing = "probationary"
    t6.agent.state.update(status="active", detail="awaiting verification")
    t6._maybe_notify()
    check("a notification backend that raises does not crash the tray", True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
