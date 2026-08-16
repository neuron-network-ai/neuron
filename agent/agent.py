"""
agent/agent.py — the NEURON agent main loop.

Turns this machine into a NEURON node automatically:
  1. read agent/config.json
  2. register with the coordinator if we have no node_id (sends CPU/RAM/Tailscale IP)
  3. ask the coordinator which layers we own  (GET /node/{id}/slice-info)
  4. download ONLY that slice              (slice_downloader; skipped if present)
  5. start the generalized node server     (node_server; any layer range)
  6. heartbeat every 30 s — but only while the resource guard says the machine is
     idle; when the owner is using the machine we stop advertising availability so
     the coordinator routes elsewhere (in-flight requests still finish)
  7. log everything to agent/agent.log

Zero personal data leaves the machine: only node_id, layer range, core/RAM counts,
and the Tailscale IP. ARM-compatible (pure Python + psutil + requests).
"""
import argparse
import json
import logging
import os
import platform as _platform
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from urllib.parse import urlparse

import psutil
import requests

# Where writable state (config, log, model slice) lives. When frozen into an installed .exe
# the program sits in read-only Program Files, so state goes to %LOCALAPPDATA%\NEURON (Windows)
# / ~/.local/share/NEURON (elsewhere). As a normal script it's the agent/ dir, and we add the
# repo root to sys.path so `common` / `slice_downloader` resolve.
if getattr(sys, "frozen", False):
    _base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "share")
    HERE = os.path.join(_base, "NEURON")
    os.makedirs(HERE, exist_ok=True)
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(HERE))             # repo root

CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_PATH = os.path.join(HERE, "agent.log")


def crash_log(what):
    """Append `what` + the current traceback to agent.log using nothing but the stdlib.

    Deliberately NOT a logging handler. This has to work in the three situations where the
    logging config cannot help: before `_setup_logging()` has run, when the import that
    logging config itself depends on is the thing that just failed, and inside the frozen
    tray app, where the console is hidden so stderr goes nowhere a person can see.

    [P24]: v0.18 produced NO log file at all on a stranger's machine. A run that leaves no
    trace is indistinguishable from a run that never happened, and there was nothing the
    owner could send. Every path that can die before logging exists now calls this first.
    """
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} [CRASH] {what}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass                       # best effort by definition — there is nothing further to try
    try:
        # ASCII only: this goes to a console whose codepage may not be UTF-8, unlike the file.
        print(f"NEURON: {what} - details in {LOG_PATH}", file=sys.stderr)
    except Exception:
        pass


def _config_path_from_argv(argv=None):
    """The --config value if one was passed, else CONFIG_PATH. Needed at IMPORT time.

    argparse has not run yet and cannot: see _apply_device_preference for why this has to
    happen before main(). Deliberately forgiving -- a malformed --config is argparse's error to
    report properly a moment later, not a reason to fail here.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    return CONFIG_PATH


def _apply_device_preference(argv=None, environ=None):
    """Turn the config's `device` setting into NEURON_DEVICE. Returns a note for the log.

    **This must run before `common` is imported anywhere in this process**, and that is why it
    sits at module level instead of in main(). `common.DEVICE` is resolved exactly once, at
    import (`common.py:113`), and `from agent.node_server import NodeServer` below imports
    `common` — so by the time main() reads the config the device has already been chosen. A
    setting applied in main() would be read, saved, displayed in the tray, and do nothing.

    Rules, in order:
      * An explicit NEURON_DEVICE in the environment always wins. An operator who set it on the
        command line is not overruled by a file.
      * "auto" (or absent) sets nothing and lets `common._resolve_device()` decide, which is
        exactly the behaviour before this setting existed.
      * "cpu" pins the CPU.
      * "gpu" is honoured ONLY if torch reports a usable CUDA device. It is not enough that a
        card exists: `torch.device("cuda:0")` is accepted by torch without checking anything,
        so pinning it on a `+cpu` build would hand `common` a device that fails on first use.
        The shipped build IS a `+cpu` build, so this path returns the "detected but unusable"
        note -- said out loud rather than silently ignored, which is the whole point.

    Never raises: a device preference must not be the reason a node fails to start.
    """
    environ = os.environ if environ is None else environ
    try:
        if environ.get("NEURON_DEVICE", "").strip():
            return f"device: NEURON_DEVICE={environ['NEURON_DEVICE']} (environment wins)"
        try:
            with open(_config_path_from_argv(argv)) as f:
                pref = (json.load(f).get("device") or "auto").strip().lower()
        except (OSError, ValueError, AttributeError):
            return None                     # no config yet (first run) -- auto is correct
        if pref in ("", "auto"):
            return None
        if pref == "cpu":
            environ["NEURON_DEVICE"] = "cpu"
            return "device: cpu (from config)"
        if pref != "gpu":
            return f"device: {pref!r} is not one of auto/cpu/gpu — ignoring it, using auto"
        try:
            import torch
            usable = bool(torch.cuda.is_available())
        except Exception:                                           # noqa: BLE001
            usable = False
        if usable:
            environ["NEURON_DEVICE"] = "cuda:0"
            return "device: cuda:0 (from config)"
        return ("device: GPU requested, but THIS BUILD COMPUTES ON CPU — it ships a CPU-only "
                "torch, so no card can be used. Running on CPU.")
    except Exception:                                               # noqa: BLE001
        return None


# Read before the heavy imports below, because one of them (`node_server` -> `common`) resolves
# the execution device at import and never reconsiders it. Logged in main(), once logging is up.
_DEVICE_NOTE = _apply_device_preference()

# [P41] — and it runs BEFORE the heavy imports below, which is the whole point. `cpu_check` is
# stdlib-only and cheap; torch is neither, and it is torch's bundled MKL that has been reported
# executing an AVX-512 kernel on a CPU that has none. Probing after that import would mean the
# check lives downstream of the thing it is meant to protect against.
#
# `crash_log` rather than `log`, for the same reason it exists at all: this is before
# `_setup_logging()`, and in the frozen tray app stderr goes nowhere a person can see. A refusal
# nobody can read is the silent crash again, with extra steps.
#
# It refuses only a POSITIVE determination of x86-without-AVX2 — `refusal()` returns None for
# anything undetermined, and names an override in its own text, because the risk is documented
# elsewhere and has never been reproduced here.
from agent import cpu_check                               # noqa: E402

_CPU = cpu_check.probe()
_CPU_REFUSAL = cpu_check.refusal(_CPU)
if _CPU_REFUSAL:
    crash_log(_CPU_REFUSAL)
    raise SystemExit(2)

# The imports below are the heavy ones (node_server and local_chat pull in torch), and they
# run at MODULE level — before main(), before any config is read, before logging exists. A
# failure here used to be completely silent. It is the single most likely place for a version
# that "does nothing" to be dying, so it records itself before re-raising.
try:
    from agent import gpu                                 # noqa: E402
    from agent import local_chat                          # noqa: E402
    from agent import resource_guard                      # noqa: E402
    from agent.node_server import NodeServer              # noqa: E402
    import slice_downloader                               # noqa: E402
except BaseException:
    crash_log("the agent could not import its own modules — it never got as far as starting")
    raise

RETRY_SECONDS = 60
PING_SECONDS = 30
# How often the node re-times its own segment. Comfortably inside the coordinator's
# MS_PER_LAYER_TTL_S (6 h) so a healthy node's figure never ages out and falls back to the
# default prior — the TTL is there to release a node from a BAD reading, not to forget a good
# one. Long enough that the benchmark itself is not a meaningful load on a volunteer's machine.
REMEASURE_INTERVAL_S = 3600
MIGRATION_POLL_SECONDS = 20
# How long to wait for the node server to actually bind before treating setup as failed.
# A bind either works immediately or fails immediately; the only slow case is a previous
# copy of the agent still holding the port on its way out.
BIND_TIMEOUT_S = 20
# How often to prove this node is reachable at its PUBLIC relay endpoint, and how long to wait
# for that proof. Every fourth heartbeat (~2 min) rather than every one: it opens a real
# connection through the relay and back into our own server, so it is not free.
# One INFO line per unchanged state every N beats (~15 min at PING_SECONDS=30). Silence
# still has to mean dead -- see verify_service.py, which learned this the hard way: a
# healthy service that logs nothing looks identical to one that died on Monday.
HEARTBEAT_ALIVE_EVERY = 30
RELAY_PROBE_EVERY = 4
RELAY_PROBE_TIMEOUT_S = 20
# How often a verified node looks for a newcomer to vouch for. Slow on purpose: verifying is
# a favour to the network, not this node's job, and a newcomer waiting an extra minute costs
# nothing next to needing a human to be awake.
PEER_VERIFY_POLL_SECONDS = 60
# How many heartbeats a node may sit PROBATIONARY before it says so, and keeps saying so.
# 60 x 30 s = 30 minutes, comfortably longer than a healthy promotion takes (a peer verifier
# polls every 60 s) and far shorter than the three days [P24]'s stranger waited in silence.
PROBATION_WARN_BEATS = 60

# Written on first run if no config exists (so a freshly-installed app just works): open join,
# auto-placement, green idle donation, relay on. Matches agent/config.json.
DEFAULT_CONFIG = {
    "coordinator": "https://neuronnet.duckdns.org",
    "node_id": None, "node_token": None, "model_id": None,
    # layers_pinned: set only by --layers. Left false, a probationary node may be re-placed by
    # the coordinator when its slice turns out to duplicate someone else's.
    "layer_start": None, "layer_end": None, "layers_pinned": False,
    # The UTC hours this machine is usually left running, e.g. "0,1,2,3,4,5,6". A DECLARATION,
    # never a promise: the coordinator plans coverage against it and pays for hours actually
    # attended, and nothing is penalised for missing one. Penalising it would be aimed squarely
    # at phones, whose availability follows a charger rather than a decision -- and the fastest
    # way to have an app uninstalled is to fine someone for their charging habits.
    # None means "not stated"; the node still earns for whatever it does attend.
    "declared_slots": None,
    "slice_dir": "./model_slice/",
    # "balanced" donates while you work (yielding above 50% CPU, AC only) rather than only
    # when the machine is idle. A node that only ever runs when nobody is at the keyboard
    # contributes very little on a personal PC, and the network is small enough that the
    # difference matters. Existing configs are untouched -- this is the FRESH-install default.
    "donation_mode": "balanced", "idle_threshold_seconds": 60,
    # "auto" | "cpu" | "gpu". Which device this node computes on. Set at install time from
    # what the machine actually has, changeable from the tray. See _apply_device_preference():
    # it must be applied before common.py is imported, so it is read at module import, not in
    # main(). NOTE: this build cannot use a GPU whatever this says -- see agent/gpu.py.
    "device": "auto",
    "behind_nat": True, "log_level": "INFO",
    # Check daily for a newer build, verify its published SHA-256, and install it. On by
    # default because a stranger will not reinstall to pick up a fix, so a node that never
    # updates keeps its shipped bugs for good. Set false and this node never even asks the
    # coordinator about versions; read at startup, so it takes effect on the next start.
    "auto_update": True,
    # Where this node's NRN goes if the ledger moves on-chain. Leave null and the agent
    # generates its own key (agent/payout_key.py) and binds it for you. Set it to your own
    # wallet address instead and the agent will NOT generate one -- bind it yourself with
    # `python -m agent.bind_payout`, which needs a signature only your wallet can make.
    "payout_address": None,
    # every installed agent also runs its own personal Chat UI (agent/local_chat.py) --
    # your own front door to the network, on your own machine, off by default to the
    # internet (127.0.0.1 only). Independent of donation_mode: pausing compute-sharing
    # when you're active shouldn't also take away your own ability to use the network.
    "local_chat": True, "local_chat_port": 8080,
    # Google/GitHub OAuth for wallet login (ui/oauth.py) -- None until set. This is the only
    # way to hand credentials to a packaged, console-less desktop install: ui/oauth.py reads
    # plain os.environ, which nobody can set for a double-clicked tray app, so
    # start_local_chat() copies these into the process environment before importing ui.app.
    "oauth": {
        "google_client_id": None, "google_client_secret": None,
        "github_client_id": None, "github_client_secret": None,
        "session_secret": None, "wallet_link_secret": None,
    },
}


def ensure_config(path=CONFIG_PATH):
    """Create a default config on first run so an installed app needs no manual setup."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    return path


def load_config(path):
    """Read the config, falling back to the last known-good copy if it is unreadable.

    A truncated config.json is not a hypothetical: `_save()` used to truncate-then-write, so
    any interrupted save left one, and the agent then died before logging on every start
    ([P24]). Recovery matters more than it looks — the file holds `node_id` and `node_token`,
    which ARE the node's identity and its claim on everything it has earned. Regenerating them
    silently would orphan the owner's balance and register a stranger's machine as a brand-new
    node, so a broken config is repaired from `.prev` where possible and reported loudly where
    not. Never silently replaced with defaults.
    """
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as first:
        prev = path + ".prev"
        try:
            with open(prev) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            raise first             # nothing to fall back to; main() crash_logs the real reason
        log.error("%s is unreadable (%s) — recovered the previous copy from %s. The node keeps "
                  "its identity and earnings; anything changed since that copy was written is "
                  "lost.", path, first, prev)
        try:
            shutil.copy2(path, path + ".corrupt")     # keep the evidence, do not just bin it
        except OSError:
            pass
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)
        return cfg


log = logging.getLogger("neuron.agent")


def _setup_logging(level="INFO"):
    """Attach handlers to the `neuron` PARENT logger, not to `neuron.agent`.

    The tray says "Chat UI unavailable — see agent.log" when local_chat fails. The two things
    that actually fail there are the weight fetch (`neuron.engine.local_gguf`) and the model
    load (`neuron.driver`) -- and neither is a child of `neuron.agent`, so their records
    propagated to a root logger with no handlers and were discarded. In windowed tray mode
    there is no console either, so the reason a user was told to look up had nowhere to appear.
    Pointing someone at an empty log is worse than saying nothing.

    encoding="utf-8" because the existing file is full of mojibake where "·" was written
    through the console's ANSI codepage.
    """
    parent = logging.getLogger("neuron")
    parent.setLevel(getattr(logging, level, logging.INFO))
    log.setLevel(getattr(logging, level, logging.INFO))
    if parent.handlers:          # idempotent: tray mode and main() can both call this
        return
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        parent.addHandler(h)


def _version():
    """This build's version string, or "unknown". Cheap, lazy and never fatal.

    The first line of the log has to say which build wrote it. [P24] is a regression between
    two versions, and the log from the machine could not answer "which one is this?" — the
    answer had to be inferred from which lines were present.
    """
    try:
        from agent import updater
        return updater.LOCAL_VERSION
    except Exception:
        return "unknown"


# The dtypes `common.WEIGHT_DTYPE` can be. Duplicated from there ON PURPOSE: this process is
# the lightweight, ARM-compatible half and must not import torch, which `common` does at module
# scope. `agent/test_weight_dtype_report.py` asserts the two agree for every value and fails if
# either side changes alone -- the only honest way to hold a duplicated constant together.
WEIGHT_DTYPES = ("fp32", "fp16", "bf16")
DEFAULT_WEIGHT_DTYPE = "fp32"


def weight_dtype():
    """What this node will STORE weights at, as `common.py` will resolve it.

    Storage, not compute: `cast_linears` keeps every GEMM in fp32 whatever this says, because
    these CPUs have no half-precision GEMM ([P2]). So this halves a node's resident bytes
    without halving its arithmetic — which is what decides whether a model too big for one
    machine fits across two.

    An unrecognised value resolves to the default here rather than raising, because `common`
    would raise on it in the node_server process and the coordinator must still be told
    something true about THIS process. Reporting the default matches what the operator will
    actually get once they fix the typo, and reporting nothing would size the node identically.
    """
    v = os.environ.get("NEURON_WEIGHT_DTYPE", DEFAULT_WEIGHT_DTYPE).strip().lower()
    return v if v in WEIGHT_DTYPES else DEFAULT_WEIGHT_DTYPE


def detect_tailscale_ip():
    """Best-effort Tailscale IPv4 (100.64.0.0/10). Falls back to a 100.x interface addr."""
    for cmd in (["tailscale", "ip", "-4"],
                [r"C:\Program Files\Tailscale\tailscale.exe", "ip", "-4"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            if a.family == socket.AF_INET and a.address.startswith("100."):
                return a.address
    return socket.gethostbyname(socket.gethostname())


class Agent:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        ensure_config(config_path)
        self.cfg = load_config(config_path)
        # An in-place upgrade reads the PREVIOUS version's config ([P24]: 0.18 was installed
        # over 0.17 and kept %LOCALAPPDATA%\NEURON). A key this build expects and the older one
        # never wrote must not be a KeyError in the constructor — that fires before anything
        # is logged and takes the whole agent down with no explanation. Missing keys fall back
        # to DEFAULT_CONFIG here; they are NOT written into the user's file, because injecting
        # defaults an operator deliberately left out has its own failure mode (see
        # install.py's write_config).
        self.base = (self.cfg.get("coordinator")
                     or DEFAULT_CONFIG["coordinator"]).rstrip("/")
        # donation level (how much spare capacity to give); max_cpu_pct kept as a
        # back-compat explicit ceiling override for older configs.
        overrides = None
        if self.cfg.get("max_cpu_pct") is not None:
            overrides = {"cpu_ceiling": float(self.cfg["max_cpu_pct"])}
        self.guard = resource_guard.ResourceGuard(
            donation_mode=self.cfg.get("donation_mode", resource_guard.DEFAULT_MODE),
            idle_threshold_s=self.cfg.get("idle_threshold_seconds", 60),
            overrides=overrides)
        # live state the tray reads: status in {starting, downloading, active, idle, error}
        self.state = {"status": "starting", "node_id": self.cfg.get("node_id"),
                      "layers": None, "coordinator": self.base, "detail": ""}
        self._stop = threading.Event()
        # Last auto-update verdict, reported on the next registration. None until the first
        # check runs, and None means "not yet checked" -- never "up to date".
        self._last_update_check = None
        self.user_paused = threading.Event()   # set by the tray's Pause button
        self.relay = self.cfg.get("relay")     # relay params if this node is behind NAT
        self.server = None                     # the running NodeServer (migration reload target)
        self._tunnel_stop = None               # per-tunnel stop flag, so it can be restarted alone
        self.local_chat_server = None          # set once start_local_chat() finishes (tray readiness check)
        # "pending" until start_local_chat() has run, then "running" / "failed" / "disabled".
        # Without this a failed Chat UI is indistinguishable from a slow one -- the tray showed
        # "Chat UI (starting…)", disabled, forever, with no hint that it had already given up.
        self.local_chat_state = "pending"
        self.local_chat_error = None           # the reason, when state becomes "failed"
        # Standing as the COORDINATOR sees it, refreshed by every heartbeat. None until the
        # first ping answers. `_probation_beats` counts heartbeats spent probationary so the
        # agent can eventually say out loud that it is serving nothing (see note_standing).
        self.standing = None
        self._probation_beats = 0

    # -- config persistence -------------------------------------------------- #
    def _save(self):
        """Write the config so it can never be observed half-written.

        It used to be `json.dump(cfg, open(path, "w"))`: the open TRUNCATES immediately, the
        handle was never explicitly closed, and this runs from eight places including
        registration and migration cutover. Anything that stopped the process mid-write — an
        OS shutdown, a task kill, installing a new version over a running agent — left a
        truncated config.json. On the next start `json.load` raises, and in v0.18 that happened
        *before* logging existed, so the agent died silently with no file, on every start,
        forever. That is [P24]'s exact signature, and this is the most plausible mechanism found
        for it (unproven without the machine, but a real defect either way).

        Write to a temp file in the SAME directory, flush to disk, then os.replace() — which is
        atomic on both Windows and POSIX. The previous good copy is kept as .prev, because the
        thing at risk is node_id and node_token: that is the node's identity and its earnings,
        and regenerating it would silently orphan the owner's balance.
        """
        tmp = self.config_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.cfg, f, indent=2)
            f.flush()
            os.fsync(f.fileno())        # the rename is atomic; the CONTENT still has to be there
        if os.path.exists(self.config_path):
            try:
                shutil.copy2(self.config_path, self.config_path + ".prev")
            except OSError:
                pass                    # a missing backup must never block saving the real file
        os.replace(tmp, self.config_path)

    # -- coordinator calls --------------------------------------------------- #
    def ensure_placement(self):
        """Zero-config open join (S20): if config has no layer range, ask the coordinator
        where we fit (fills a gap, else replicates the weakest segment) and persist it."""
        if self.cfg.get("layer_start") is not None and self.cfg.get("layer_end") is not None:
            return
        p = self.ask_placement()
        self.cfg["layer_start"], self.cfg["layer_end"] = p["layer_start"], p["layer_end"]
        self._save()
        log.info("auto-placed on layers %d-%d (%s: %s)", p["layer_start"], p["layer_end"],
                 p.get("role"), p.get("reason"))

    def ask_placement(self, exclude_self=False):
        """GET /node/placement. `exclude_self` asks the coordinator to leave this node out of
        the roster it reasons over — i.e. "where would I go if I weren't already here?"."""
        params = {}
        if exclude_self and self.cfg.get("node_id"):
            params["exclude"] = self.cfg["node_id"]
        r = requests.get(f"{self.base}/node/placement", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def reconsider_placement(self):
        """Re-ask where this node belongs, and move if the answer changed. Returns True if the
        configured range was replaced (the caller must then re-register).

        Only ever called while PROBATIONARY, and that restriction is the whole safety argument:
        a probationary node receives no live requests, so moving it costs the network nothing
        and costs this machine one download. The range in config was decided at the instant we
        first asked, from whatever the coordinator could see then — and an unverified node used
        to be invisible to that view, so several machines joining around the same time were all
        told to take the identical gap and the layers nobody took stayed uncovered for good
        (three nodes on 0-13, 21-27 empty, live on 2026-08-07). Asking again with ourselves
        excluded is self-stabilising: if this node's range is genuinely needed, the coordinator
        sees the hole we would leave and hands the same range straight back, so a node that is
        where it should be never moves and there is nothing to oscillate.
        """
        if self.cfg.get("layers_pinned"):
            return False                      # operator ran --layers; never second-guess that
        # Advice is a nicety; the registration that just succeeded is the real thing. Anything
        # wrong with the answer -- unreachable coordinator, an older build that has no `exclude`
        # and no range in the body, a garbled payload -- means stay put, not fail.
        try:
            p = self.ask_placement(exclude_self=True)
            new = (int(p["layer_start"]), int(p["layer_end"]))
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            log.debug("placement re-check unusable, keeping layers %s-%s: %s",
                      self.cfg.get("layer_start"), self.cfg.get("layer_end"), e)
            return False
        cur = (self.cfg.get("layer_start"), self.cfg.get("layer_end"))
        if new == cur or new[0] > new[1]:
            return False
        log.warning("layers %d-%d are redundant — this node is moving to %d-%d (%s: %s). "
                    "The slice for the new range downloads now; nothing was being served from "
                    "the old one, because a probationary node receives no live requests.",
                    cur[0], cur[1], new[0], new[1], p.get("role"), p.get("reason"))
        self.cfg["layer_start"], self.cfg["layer_end"] = new
        self._save()
        return True

    def measure_ms_per_layer(self, layer_start, layer_end, iters=6):
        """Time one decode-shaped forward pass through THIS node's own layers, in ms/layer.

        Without this the coordinator's auto-balancer ([P7], Session 14) can never fire:
        `balancer.solve` needs each node's real speed, `models.register` has carried an
        `ms_per_layer` column since S14, and nothing ever filled it -- so /network/plan had
        no data and the whole subsystem sat dead while a slow laptop and a fast desktop were
        handed identical layer counts. On this trio that mis-split costs ~1.8x.

        Deliberately measures the same shape the pipeline actually runs (batch 1, one token),
        after the slice is loaded and before serving real traffic, so it reflects this
        machine on this model rather than a synthetic score.
        """
        import torch

        import common
        try:
            model = self.server.model
            h = model.config.hidden_size
            n_layers = max(layer_end - layer_start + 1, 1)
            layers = list(model.model.layers)[layer_start:layer_end + 1]
            cache = common.new_cache()
            x = torch.zeros(1, 1, h, dtype=common.DTYPE)

            def _sync():
                """CUDA kernels are queued, not run, so `perf_counter` around them measures
                how fast this machine can ENQUEUE work — a number that has nothing to do with
                how fast it computes. Without this a GPU node reports an absurdly low
                ms_per_layer, `balancer.solve` reads that as the fastest machine in the
                network and hands it nearly every layer, and the node is over-assigned into
                the same OOM [P31] is about — by a second route, through the speed field
                instead of the memory one. No-op on CPU."""
                if common.DEVICE.type == "cuda":
                    torch.cuda.synchronize()

            common._run_layers(model, layers, x, cache, 0)      # warm: first pass allocates
            _sync()                                             # ...and finish warming before t0
            t0 = time.perf_counter()
            for i in range(iters):
                common._run_layers(model, layers, x, cache, 1 + i)
            _sync()
            per_pass_ms = (time.perf_counter() - t0) / iters * 1000
            return round(per_pass_ms / n_layers, 4)
        except Exception as e:
            log.warning("could not measure ms_per_layer: %s", e)
            return None

    def report_speed(self, info):
        """Send this node's measured speed up so the balancer can use it. Re-registers,
        because `models.register`'s upsert COALESCEs ms_per_layer -- registration is the
        existing path for this field and needs no new endpoint."""
        ms = self.measure_ms_per_layer(info["layer_start"], info["layer_end"])
        if ms is None:
            return
        prev = self.cfg.get("ms_per_layer")
        if prev and ms > 0 and (max(prev, ms) / min(prev, ms)) > 2.0:
            # Worth saying out loud. A figure that moves by more than 2x between measurements is
            # either a machine whose load changed a lot or a reading taken under contention --
            # and the coordinator sizes stages and picks replicas from it, so a silent swing is
            # a silent routing change.
            log.info("ms/layer changed materially: %.3f -> %.3f", prev, ms)
        self.cfg["ms_per_layer"] = ms
        self._save()
        log.info("measured %.3f ms/layer over layers %d-%d", ms,
                 info["layer_start"], info["layer_end"])
        try:
            self.register()
        except Exception as e:
            log.warning("could not report ms_per_layer: %s", e)

    def use_relay(self):
        """Should peers reach this node through the public relay rather than directly?

        Defaults to TRUE ([P10]). A direct address only works between machines on the same
        tailnet or LAN — which is every developer's setup and no stranger's. A home machine
        behind NAT cannot accept an inbound connection at all, and a Tailscale 100.x address
        handed to somebody outside the tailnet is not merely slow, it is unroutable. The
        relay costs one extra hop and makes the address universally valid, so it is the
        right default and `behind_nat: false` is the exception a LAN cluster opts into.
        """
        return bool(self.cfg.get("behind_nat", True))

    def new_node_id(self):
        """A node id that will not collide with somebody else's machine.

        This used to be exactly `agent-{hostname}`, which collides deterministically: Windows
        ships defaults like DESKTOP-8F3K2P1, plenty of people run "laptop", and one person
        reinstalling produces the same id as before. That matters because the coordinator
        REFUSES a secret-less registration of an id that is already `trusted` or `verified`
        (the hijack guard, and rightly so) -- so the second machine to use a given hostname, or
        the same machine after losing its config, gets a 409 forever and can never join. Seen
        live as an endless "this node_id is registered with a different token" retry loop.
        The random suffix is generated once and persisted, so the id is stable for this install
        but unique across installs.
        """
        return f"agent-{socket.gethostname().lower()}-{uuid.uuid4().hex[:6]}"

    def register(self, _replaced=False):
        self.ensure_placement()
        ip = detect_tailscale_ip()
        if not self.cfg.get("node_id"):
            self.cfg["node_id"] = self.new_node_id()
            self._save()
        body = {
            "node_id": self.cfg["node_id"],
            "tailscale_ip": ip,
            "port": self.cfg.get("port", 50999),
            "layer_start": self.cfg["layer_start"],
            "layer_end": self.cfg["layer_end"],
            # The model this node is ACTUALLY serving, so a coordinator/node divergence is
            # detectable at all. Without it, both 28-layer Qwen2.5 tiers validate identically:
            # on 2026-08-10 the coordinator believed 1.5B while nodes ran 7B, every range and
            # coverage check passed, and the only symptom was the driver seeing a socket close
            # mid-message. Reported, never obeyed -- placement stays the coordinator's ([P33]).
            "model_id": self.cfg.get("model_id"),
            # Availability declaration (TOKENOMICS.md §11.4 emission). Sent only when the owner
            # has stated one -- an unset value must stay unset rather than becoming a guess the
            # coordinator then plans coverage around. On Android this should be DERIVED from
            # observed charge/plug history rather than asked for, since the phone already knows
            # when it is usually on a charger and its owner does not think in UTC hours.
            "declared_slots": self.cfg.get("declared_slots"),
            "cores": os.cpu_count(),
            "ram_gb": int(psutil.virtual_memory().total // 10**9),
            # What this node will STORE weights at, which halves or doubles every footprint the
            # coordinator sizes it from ([P43]). `balancer.weight_bytes_for` has read this field
            # since the dtype correction shipped and nothing has ever sent it, so every node has
            # been sized at the pessimistic 4 bytes/param regardless of what it runs.
            #
            # Read from the environment rather than from `common.WEIGHT_DTYPE`, because THIS
            # process must not import torch -- agent.py is the lightweight, ARM-compatible half
            # and `common` imports torch at module scope. That duplicates the default, which is
            # a drift risk, so `test_weight_dtype_report.py` asserts the two agree for every
            # supported value and fails if either side changes alone.
            "weight_dtype": weight_dtype(),
            # With cores/ram_gb this is the coarse hardware signature the coordinator groups
            # on to spot one machine registering many node ids. It is a signal an operator
            # reviews, never a block, and it says nothing a `User-Agent` header would not.
            "platform": _platform.platform(),
            # When true the coordinator ignores the address above and stores this node at a
            # relay endpoint instead, so every peer that asks for a chain is handed a public
            # host:port. That is the whole of "a stranger can be in the pipeline".
            "behind_nat": self.use_relay(),
        }
        # GPU capability. Reported for the operator's roster and for the day the pipeline can
        # use a card. It does NOT size this node's slice: `balancer.GPU_EXECUTION` is off,
        # because the shipped build ships a CPU-only torch and a loader that never moves
        # weights to a device, so a card cannot be used regardless of what is detected here.
        # Nothing in this block is a speed claim or a capacity claim.
        try:
            info = gpu.detect_gpu()
            body["has_gpu"] = info["has_gpu"]
            if info["has_gpu"]:
                body["gpu_vram_gb"] = info["gpu_vram_gb"]
                body["gpu_name"] = info["gpu_name"]
        except Exception as e:              # detection must never block registration
            log.debug("GPU detection failed, registering as CPU-only: %s", e)
            body["has_gpu"] = False
        # Feeds coordinator/balancer.py. None on the first registration (the slice is not
        # loaded yet, so there is nothing to time); report_speed() re-registers with the real
        # figure once the node server is up, and the upsert COALESCEs so a later None never
        # erases it.
        if self.cfg.get("ms_per_layer") is not None:
            body["ms_per_layer"] = self.cfg["ms_per_layer"]
        # WHY THIS NODE IS NOT ON THE VERSION THE COORDINATOR PUBLISHED. Nothing reported the
        # running build before 0.20.2: the version existed only in the first line of this
        # machine's own log, which nobody can read on a PC behind a NAT in somebody's house.
        # So a rollout could not be watched and -- the sharp one -- a ROLLBACK could not be
        # confirmed. The remedy shipped the same day fired blind.
        #
        # Three fields answer one question, and it takes all three: a node on an old build is
        # either yet to make its daily check, running with auto_update off, or failing the
        # download. Only the first fixes itself.
        body["agent_version"] = _version()
        body["auto_update"] = bool(self.cfg.get("auto_update", True))
        if self._last_update_check:
            body["update_check"] = self._last_update_check
        # Open join (Session 12): register with NO secret by default — anyone can join.
        # Only send the header if the operator explicitly set one (that path marks the
        # node TRUSTED and is for the founder's own dev nodes, not strangers).
        headers = {}
        secret = self.cfg.get("register_secret")
        if secret:
            headers["X-Register-Secret"] = secret
        # Self-recovery: a re-registration of an already-trusted/verified node_id with no
        # secret is rejected as a possible hijack (coordinator/main.py's guard) UNLESS it
        # proves ownership via the node's own CURRENT token. Never exercised before this
        # session -- setup() only ever called register() for a brand-new node; the new
        # stale-relay-ticket refresh (see setup()) is the first path that re-registers an
        # ALREADY-credentialed node, and without this header that legitimately hits a 409.
        existing_token = self.cfg.get("node_token")
        if existing_token:
            headers["X-Node-Token"] = existing_token
        r = requests.post(f"{self.base}/node/register", json=body, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        self.cfg["node_id"] = body["node_id"]
        self.cfg["node_token"] = data["node_token"]
        self.relay = data.get("relay")          # set when behind_nat -> auto-tunnel
        if self.relay:
            self.cfg["relay"] = self.relay
        elif body["behind_nat"]:
            # We asked to be relayed and were not given an endpoint, which means the
            # coordinator has RELAY_ENABLED off or its port pool is exhausted. Peers now hold
            # whatever local address we sent, so anyone off this LAN/tailnet silently cannot
            # reach us -- exactly the failure [P10] exists to prevent. Say so.
            log.warning("requested a relay endpoint but the coordinator returned none — this "
                        "node is only reachable at %s:%d, so peers outside this network "
                        "will NOT be able to connect", ip, body["port"])
        self._save()
        # Registration carries the coordinator's address too, so a node that re-registers
        # (ticket refresh, restart, recovery) picks up a move without waiting for a beat.
        self.adopt_coordinator_url(data)
        standing = data.get("standing", "trusted")
        gpu_note = (f", GPU {body['gpu_name']} {body['gpu_vram_gb']} GB"
                    if body.get("has_gpu") else "")
        log.info("registered as %s [%s], assigned layers %s (%d cores, %d GB%s, %s)",
                 body["node_id"], standing, data["assigned_layers"],
                 body["cores"], body["ram_gb"], gpu_note, ip)
        self.standing = standing
        # A probationary node's placement is the one that can still be wrong AND still be fixed
        # for free: wrong because it was chosen before the coordinator counted unverified nodes,
        # free because nothing routes to us yet. Re-ask once, and if the answer moved, register
        # again on the new range. `_replaced` bounds this to a single extra round trip -- the
        # second registration never re-checks, so there is no way to loop.
        if standing == "probationary" and not _replaced and self.reconsider_placement():
            return self.register(_replaced=True)
        if standing == "probationary":
            log.info("PROBATIONARY: serving challenges only — a verifier must confirm this "
                     "node (proof-of-compute) before it receives live requests or earns NRN")
        return data["assigned_layers"]

    def slice_info(self):
        r = requests.get(f"{self.base}/node/{self.cfg['node_id']}/slice-info", timeout=30)
        r.raise_for_status()
        return r.json()

    def ping(self):
        r = requests.get(f"{self.base}/node/{self.cfg['node_id']}/ping",
                         headers={"X-Node-Token": self.cfg["node_token"]}, timeout=10)
        r.raise_for_status()
        try:
            data = r.json()
        except ValueError:
            return                  # a ping that isn't JSON is still a successful heartbeat
        self.adopt_coordinator_url(data)
        self.note_standing(data.get("standing"))
        if data.get("want_logs"):
            self.upload_log()

    def upload_log(self):
        """Send a redacted tail of agent.log to the coordinator, because it asked.

        This log is the only account of what happened on this machine, and until now it existed
        solely on this machine -- so diagnosing a stranger's node meant asking a human to open a
        file and paste it ([P24]: three days of `heartbeat ok` while the node was doing nothing).
        The coordinator cannot come and get it: this node is behind NAT, which is what the relay
        exists to work around. So the coordinator raises a flag, the heartbeat carries it, and
        the node pushes once and stops.

        Redacted HERE as well as on arrival. The coordinator scrubs what it receives, but this
        file belongs to the person running this machine, and "the server will clean it up" is not
        a promise this end can verify. Whatever leaves does so already clean.

        Never raises: an operator wanting to read a log must not be able to take a serving node
        down, and a failed upload is worth strictly less than the node staying up.
        """
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                body = f.read()
        except OSError as e:
            body = f"(this node could not read its own log at {LOG_PATH}: {e})"
        try:
            import logtail                              # repo root; bundled in the frozen app
            body = logtail.clean_tail(body)
            requests.post(f"{self.base}/node/{self.cfg['node_id']}/logs",
                          headers={"X-Node-Token": self.cfg["node_token"]},
                          json={"body": body}, timeout=30)
            log.info("uploaded a %d-byte log tail at the coordinator's request",
                     len(body.encode("utf-8")))
        except requests.RequestException as e:
            log.debug("log upload failed (%s) — will retry on the next heartbeat", e)

    def note_standing(self, standing):
        """Track what the coordinator says this node's standing is, and SAY when it is stuck.

        [P24]: a stranger's node registered, downloaded its slice, bound its port, opened its
        relay tunnel and then logged `heartbeat ok — active` for three days. It was
        probationary the whole time — serving no requests, earning no NRN — because the
        operator's verifier had died. Every signal the owner could see said the machine was
        fine, and the one fact that mattered was never mentioned again after the registration
        line scrolled away. An agent that cannot say "I am online and useless" is why that
        went unnoticed for three days.
        """
        if not standing:
            return                  # older coordinator: it does not report standing at all
        if standing != "probationary":
            if self.standing == "probationary":
                log.info("VERIFIED — this node now receives live requests and earns NRN")
            self.standing, self._probation_beats = standing, 0
            return
        self.standing = "probationary"
        self._probation_beats += 1
        # Repeats rather than firing once: a log that scrolls has to keep saying it, and the
        # owner may only ever look at the tail.
        if self._probation_beats % PROBATION_WARN_BEATS == 0:
            log.warning(
                "still PROBATIONARY after %d minutes — this machine is healthy and reachable, "
                "but it serves no requests and earns no NRN until a verifier confirms it. "
                "Nothing here needs fixing; the network operator's verifier may be down. "
                "See PROBLEMS.md [P24].",
                round(self._probation_beats * PING_SECONDS / 60))

    @staticmethod
    def _normalize_url(url):
        """A usable absolute http(s) URL, or None.

        Strict on purpose. This value redirects every future call this node makes, so a
        malformed one does not deserve the benefit of the doubt -- garbage, a relative path or
        a non-http scheme is dropped rather than adopted and then failed on forever.
        """
        if not isinstance(url, str):
            return None
        url = url.strip().rstrip("/")
        if not url:
            return None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return None
        return url

    def adopt_coordinator_url(self, payload):
        """Follow the coordinator if it says it now lives somewhere else.

        The address is otherwise frozen in config.json at install time and nothing can revise
        it, so moving the public hostname would strand every existing node permanently -- a new
        installer could not help, because ensure_config only writes defaults when there is no
        config.json. The old host telling nodes where the new one is, while it is still up, is
        the only mechanism that works.

        The new address is PROBED before it is kept. Adopting blindly would replace one way to
        strand the network with a faster one: a single typo in the coordinator's PUBLIC_URL
        would be obeyed by every node at once, with nothing left able to correct them. If the
        probe fails we stay put and are simply told again on the next heartbeat.

        This is not a new trust boundary. A coordinator already tells nodes which layers to
        serve and which peers to talk to; telling them its own address is strictly less.
        """
        if not isinstance(payload, dict):
            return False
        offered = self._normalize_url(payload.get("coordinator_url"))
        if not offered or offered == self._normalize_url(self.cfg.get("coordinator")):
            return False

        current = self.cfg.get("coordinator")
        log.info("coordinator says it now lives at %s (we have %s) — checking before moving",
                 offered, current)
        try:
            probe = requests.get(f"{offered}/node/{self.cfg['node_id']}/ping",
                                 headers={"X-Node-Token": self.cfg["node_token"]}, timeout=15)
            probe.raise_for_status()
        except requests.RequestException as e:
            log.warning("NOT moving to %s — it did not answer (%s). Staying on %s.",
                        offered, e.__class__.__name__, current)
            return False

        self.cfg["coordinator"] = offered
        self.cfg["coordinator_previous"] = current
        self._save()
        self.base = offered
        log.info("coordinator address updated: %s -> %s (saved; all further calls use it)",
                 current, offered)
        return True

    # -- slice download ------------------------------------------------------ #
    @staticmethod
    def slice_tensors_on_disk(weights):
        """Every tensor name inside a downloaded slice, or None if the file is unreadable.

        Read from the safetensors header (an 8-byte little-endian length then JSON), so it
        reflects the bytes on disk rather than what some config claims about them.
        """
        try:
            with open(weights, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if not 0 < n < 100 * 1024 * 1024:
                    return None
                return set(json.loads(f.read(n).decode()))
        except (OSError, ValueError):
            return None

    @staticmethod
    def slice_layers_on_disk(weights):
        """Which decoder layers a downloaded slice actually contains, as (lo, hi) or None."""
        keys = Agent.slice_tensors_on_disk(weights)
        if not keys:
            return None
        idx = set()
        for k in keys:
            parts = k.split(".")
            if k.startswith("model.layers.") and len(parts) > 2 and parts[2].isdigit():
                idx.add(int(parts[2]))
        return (min(idx), max(idx)) if idx else None

    @staticmethod
    def _slice_covers(weights, have, want, info):
        """Does the slice on disk hold everything this node's NEW assignment needs?

        The layer range containing `want` is necessary but not sufficient: a first node also
        needs the embedding (and the tokenizer, which lives in separate files), and a last node
        needs the final norm. A slice downloaded as a MIDDLE has neither, so a middle promoted
        to first or last must still re-download however many layers it already holds.
        """
        if not (have[0] <= want[0] and have[1] >= want[1]):
            return False
        keys = Agent.slice_tensors_on_disk(weights)
        if not keys:
            return False
        if info.get("is_last_node") and "model.norm.weight" not in keys:
            return False
        if info.get("is_first_node"):
            # lm_head is deliberately NOT required separately: with tie_word_embeddings (Qwen's
            # default) it IS embed_tokens and never appears as its own tensor.
            if "model.embed_tokens.weight" not in keys:
                return False
            d = os.path.dirname(weights)
            if not any(os.path.exists(os.path.join(d, f))
                       for f in ("tokenizer.json", "tokenizer_config.json")):
                return False
        return True

    def ensure_slice(self, info):
        slice_dir = os.path.join(HERE, os.path.normpath(self.cfg["slice_dir"]))
        weights = os.path.join(slice_dir, "model.safetensors")
        if os.path.exists(weights):
            # Existence was the ONLY check, so a slice downloaded for one layer range was
            # happily reused when the node was later placed on a different one -- serving
            # another segment's weights while claiming this segment. Nothing detects that
            # locally: the node answers confidently with wrong activations, fails
            # proof-of-compute, and eventually gets flagged, with no clue why. Reachable
            # whenever placement changes: delete config.json and re-register, and the
            # coordinator hands you whichever gap needs filling, not the range you had.
            have = self.slice_layers_on_disk(weights)
            want = (info["layer_start"], info["layer_end"])
            # WHICH MODEL is this slice for? Layer numbers do not answer that. Two tiers with
            # the same layer count (Qwen2.5 1.5B and 7B both have 28) produce slices that are
            # indistinguishable by range -- so when a migration reassigned a node to the SAME
            # range on a different model, this said "already present, skipping download" and the
            # node served the old model's weights as the new one. Live 2026-08-10: three of four
            # nodes came out of a 7B -> 1.5B migration holding 7B weights, answered every request
            # with token soup, and were BILLED for it. Nothing downstream could tell, because
            # every check downstream was about layer numbers too.
            #
            # An unrecorded slice (downloaded before the marker existed) is treated as a
            # mismatch: provenance we cannot establish is exactly the case that produced this,
            # and one re-download is cheap against serving a corrupted stage.
            on_disk_model = slice_downloader.slice_provenance(slice_dir)
            wrong_model = on_disk_model != info["model_id"]
            if wrong_model:
                log.warning("cached slice is for %s but this node serves %s — discarding it. "
                            "Layer ranges cannot tell two models apart, so this is checked "
                            "explicitly.", on_disk_model or "an unrecorded model",
                            info["model_id"])
                shutil.rmtree(slice_dir, ignore_errors=True)
                have = None          # nothing on disk now; go straight to the download below
            elif have == want:
                log.info("slice already present (%s) — skipping download", slice_dir)
                return slice_dir
            if not wrong_model and have and self._slice_covers(weights, have, want, info):
                # A slice that CONTAINS the assigned range is as good as an exact one, and
                # re-downloading it is pure waste. `load_slice_model` builds a full model
                # skeleton and fills in whatever the file holds (strict=False), and this node
                # then runs only layers[lo:hi] -- the extra layers cost resident RAM and
                # nothing else. Worth the check because re-splitting is now routine: a node
                # that has been through a couple of re-splits has usually already downloaded
                # a superset of whatever it is asked for next, and throwing that away is how
                # a stranger's PC ends up downloading the same weights four times in an
                # afternoon (observed live 2026-08-07, ~20 GB wasted on one 8 GB machine).
                log.info("cached slice holds layers %d-%d, which covers this node's %d-%d — "
                         "reusing it, no download needed", have[0], have[1], want[0], want[1])
                return slice_dir
            if not wrong_model:
                log.warning("cached slice holds layers %s but this node serves %d-%d — "
                            "discarding it and downloading the right one",
                            f"{have[0]}-{have[1]}" if have else "an unreadable range",
                            want[0], want[1])
            shutil.rmtree(slice_dir, ignore_errors=True)
        self.state["status"] = "downloading"
        log.info("downloading slice: layers %d-%d (~%.2f GB) ...",
                 info["layer_start"], info["layer_end"], info["estimated_download_gb"])
        slice_downloader.download_slice(
            info["model_id"], info["layer_start"], info["layer_end"], slice_dir,
            is_first_node=info["is_first_node"], is_last_node=info["is_last_node"])
        return slice_dir

    # -- run ----------------------------------------------------------------- #
    def setup(self):
        """Register + slice-info + download + start the server. Retries on failure."""
        while not self._stop.is_set():
            try:
                # Also re-register (safe/idempotent -- the coordinator's hijack-guard allows
                # self-recovery with the current node_token) if this is a NAT'd node whose
                # cached relay config predates the relay-auth ticket system, or otherwise
                # never got a ticket. Without this, an already-registered node restarting
                # would carry that gap forever: register() is normally skipped once
                # credentials exist, so a missing ticket could never self-heal, and the
                # tunnel would churn forever against the relay's "bad/missing ticket" check.
                # Also covers a node that was registered with a DIRECT address and has since
                # been switched to relay mode (behind_nat flipped on): without a re-register
                # the coordinator would keep handing peers the old Tailscale/LAN address
                # forever, since register() is normally skipped once credentials exist.
                needs_relay = self.use_relay() and (
                    not self.cfg.get("relay") or not self.cfg["relay"].get("ticket"))
                if not self.cfg.get("node_id") or not self.cfg.get("node_token") or needs_relay:
                    self.register()
                info = self.slice_info()
                self.state.update(node_id=self.cfg["node_id"],
                                  layers=[info["layer_start"], info["layer_end"]])
                self.cfg["model_id"] = info["model_id"]      # what we're actually serving now
                # ...and the RANGE we are actually serving, for exactly the same reason.
                #
                # The coordinator owns placement (PROBLEMS.md [P32]). This node already SERVES
                # whatever slice-info returns; what it did NOT do was remember it, so config.json
                # kept a stale range and re-asserted it on the next registration -- silently
                # undoing `neuron fix` and collapsing the chain to one stage. Persisting it here
                # makes config a CACHE of the coordinator's answer rather than a rival opinion.
                assigned = (info["layer_start"], info["layer_end"])
                current = (self.cfg.get("layer_start"), self.cfg.get("layer_end"))
                if current != assigned and current != (None, None):
                    if self.cfg.get("layers_pinned"):
                        # An override that silently does nothing is the same bug with a manual
                        # trigger, and worse -- someone chose this deliberately and would read a
                        # quiet startup as confirmation. [P31] is exactly that mistake.
                        log.warning(
                            "--layers asked for %s-%s, but the coordinator assigns this node "
                            "%d-%d and placement is the coordinator's to decide. Serving %d-%d. "
                            "To change it for real, move it coordinator-side "
                            "(./coordinator/pin_layers.sh), not with --layers here.",
                            current[0], current[1], assigned[0], assigned[1],
                            assigned[0], assigned[1])
                    else:
                        log.info("layer range updated by the coordinator: %s-%s -> %d-%d",
                                 current[0], current[1], assigned[0], assigned[1])
                self.cfg["layer_start"], self.cfg["layer_end"] = assigned
                self._save()
                slice_dir = self.ensure_slice(info)
                port = self.cfg.get("port", 50999)
                if self.server is None:
                    # The SAME Event the tray's Pause toggles. Without it NodeServer built its
                    # own, `self.paused` was never read by anything, and pausing only skipped
                    # the heartbeat -- so a paused node kept serving live requests for up to
                    # HEARTBEAT_TIMEOUT_S (~90 s) while its owner believed it had stopped.
                    self.server = NodeServer(slice_dir, info["layer_start"], info["layer_end"],
                                             info.get("total_layers", 28),
                                             paused_flag=self.user_paused)
                threading.Thread(target=self.server.run, args=("0.0.0.0", port),
                                 daemon=True).start()
                # Do not proceed until the listener is genuinely accepting. run() reports a
                # failed bind rather than raising (it is a daemon thread, where an exception
                # vanishes), and an agent that heartbeats without checking advertises a node
                # that refuses every connection -- [P21]. Retrying the whole setup is the
                # right response: the usual cause is a previous copy still holding the port,
                # which resolves on its own within a minute.
                deadline = time.time() + BIND_TIMEOUT_S
                while (time.time() < deadline and not self.server.listening.is_set()
                       and self.server.bind_error is None):
                    time.sleep(0.2)
                if not self.server.listening.is_set():
                    detail = ("node server is not accepting connections on port "
                              f"{port}: {self.server.bind_error or 'timed out'}")
                    self.state.update(status="error", detail=detail)
                    log.error("%s — retrying in %ds", detail, RETRY_SECONDS)
                    self._stop.wait(RETRY_SECONDS)
                    continue
                log.info("node server listening on port %d", port)
                self.report_speed(info)
                # Session 12: if we're behind NAT the coordinator handed us relay params
                # at registration — start the outbound tunnel so peers can reach us with
                # NO inbound port. One-click NAT traversal; nothing for the user to do.
                relay = self.relay or self.cfg.get("relay")
                if relay:
                    self.start_tunnel(relay)
                return
            except requests.RequestException as e:
                # A 409 from /node/register is NOT unreachability -- it means this node_id is
                # already registered and our node_token no longer matches, because the
                # coordinator mints a fresh one on every registration. It happens whenever a
                # second copy of the agent registers the same node_id (a manual run alongside
                # the installed tray app is the usual way), which silently invalidates the
                # token the first copy is still holding in memory. Reporting that as
                # "coordinator unreachable" sends everyone hunting a network fault that does
                # not exist, so name it and say what actually fixes it.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 409 and not self.cfg.get("node_token"):
                    # We hold no token for this id, so we cannot prove we own it and never
                    # will: the coordinator's hijack guard will refuse this registration on
                    # every future attempt too. Retrying is an infinite loop. This is the
                    # "somebody else already has my hostname" / "I lost my config" case, and
                    # the honest answer is that we are a NEW node -- so take a new identity
                    # rather than sitting there forever claiming one we cannot open.
                    old = self.cfg.get("node_id")
                    self.cfg["node_id"] = self.new_node_id()
                    self._save()
                    log.warning("node id '%s' is already claimed by another machine and we "
                                "hold no token for it — joining as '%s' instead",
                                old, self.cfg["node_id"])
                    continue
                if status == 409:
                    detail = ("this node_id is registered with a different token — another "
                              "copy of the agent probably re-registered it. Restart this agent "
                              "to pick up the current token from config.json.")
                elif status is not None:
                    detail = f"coordinator refused the request (HTTP {status}): {e}"
                else:
                    detail = f"coordinator unreachable: {e}"
                self.state.update(status="error", detail=detail)
                log.warning("%s — retrying in %ds", detail, RETRY_SECONDS)
                # Re-read config from disk before retrying: if another copy of the agent
                # re-registered, it wrote the CURRENT token there, and this retry can recover
                # on its own instead of looping on a stale in-memory value forever.
                if status == 409:
                    try:
                        fresh = json.load(open(self.config_path))
                        if fresh.get("node_token") and fresh["node_token"] != self.cfg.get("node_token"):
                            self.cfg["node_token"] = fresh["node_token"]
                            self.cfg["relay"] = fresh.get("relay", self.cfg.get("relay"))
                            log.info("picked up a newer node_token from config.json — retrying")
                    except (OSError, ValueError):
                        pass
                self._stop.wait(RETRY_SECONDS)

    # -- relay tunnel: start, prove, restart --------------------------------- #
    def start_tunnel(self, relay):
        """(Re)start the outbound relay tunnel. Its own stop flag, separate from the agent's,
        so a dead tunnel can be replaced without taking the whole agent down."""
        import tunnel_client
        if self._tunnel_stop is not None:
            self._tunnel_stop.set()            # tell the old one to stop looping
        self._tunnel_stop = threading.Event()
        threading.Thread(
            target=tunnel_client.run_tunnel,
            kwargs=dict(node_id=self.cfg["node_id"], public_port=relay["public_port"],
                        relay_host=relay["host"], control_port=relay["control_port"],
                        data_port=relay["data_port"], local_host="127.0.0.1",
                        local_port=self.cfg.get("port", 50999), stop=self._tunnel_stop,
                        ticket=relay.get("ticket")),
            daemon=True).start()
        log.info("relay tunnel started — reachable via %s:%d (NAT-friendly)",
                 relay["host"], relay["public_port"])

    def relay_reachable(self):
        """Dial our OWN public relay endpoint and complete a real handshake.

        A plain TCP connect proves nothing: the relay accepts on the public port whether or not
        it can still reach this node, so a dead tunnel looks identical to a healthy one from
        outside. Only bytes coming back through relay -> tunnel -> our node server prove the
        whole path. This is the same reasoning as the listener check in setup(): a node that
        cannot be reached must not advertise itself, or routing feeds it real requests and they
        vanish.
        """
        relay = self.relay or self.cfg.get("relay")
        if not relay or self.server is None:
            return True                        # not relayed / not serving yet: nothing to prove
        import common
        lo, hi, n = self.server.lo, self.server.hi, self.server.n
        # Same config shapes the real pipeline and proof-of-compute use, so we exercise the
        # node exactly as a peer would rather than through some test-only path.
        msg = ({"type": "config", "s2": lo, "n": n} if hi == n - 1
               else {"type": "config", "s1": lo, "s2": hi + 1})
        s = None
        try:
            s = socket.create_connection((relay["host"], relay["public_port"]),
                                         timeout=RELAY_PROBE_TIMEOUT_S)
            s.settimeout(RELAY_PROBE_TIMEOUT_S)
            common.send_msg(s, msg)
            ack = common.recv_msg(s)
            common.send_msg(s, {"type": "bye"})
            return bool(ack.get("ok"))
        except Exception as e:
            log.warning("relay endpoint %s:%d did not answer (%s: %s)",
                        relay["host"], relay["public_port"], e.__class__.__name__, e)
            return False
        finally:
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass

    # -- peer verification: the network verifies itself ---------------------- #
    def peer_verify_loop(self):
        """Once verified, help verify newcomers.

        This is what makes joining independent of any one person. Before it, a stranger stayed
        probationary — reachable, but earning nothing and serving nothing — until the operator
        personally ran security/proof_of_compute against them. So the network could only grow
        while one particular laptop was switched on, and that laptop's owner held a secret
        nobody else could be given. Here every verified node does the same check against a
        newcomer and reports a verdict signed with its own node token; PEER_VERIFY_QUORUM
        distinct agreements promote them.

        Deliberately reuses the SAME proof-of-compute a human verifier ran — a peer is not
        trusted more cheaply, it is just not a person.
        """
        while not self._stop.is_set():
            self._stop.wait(PEER_VERIFY_POLL_SECONDS)
            if self._stop.is_set() or self.server is None:
                continue
            try:
                r = requests.get(f"{self.base}/node/verify-assignment",
                                 headers={"X-Node-Token": self.cfg.get("node_token", "")},
                                 timeout=15)
                if r.status_code in (401, 403):
                    continue          # not verified yet ourselves — nothing to do
                r.raise_for_status()
                job = r.json()
            except requests.RequestException:
                continue
            if not job.get("node_id"):
                continue
            try:
                verdict = self.challenge_peer(job)
            except Exception as e:
                # Could not reach them / they are mid-restart. Report NOTHING: a failed
                # attestation is permanent and unreachability is not evidence of cheating.
                log.info("peer verify: could not challenge %s (%s) — leaving it for another "
                         "verifier", job["node_id"], e.__class__.__name__)
                continue
            try:
                out = requests.post(f"{self.base}/node/{job['node_id']}/peer-attest",
                                    json=verdict,
                                    headers={"X-Node-Token": self.cfg.get("node_token", "")},
                                    timeout=15)
                out.raise_for_status()
                d = out.json()
                log.info("peer verify: %s %s (max_err %.2e) — %d/%d distinct passes, now '%s'",
                         job["node_id"], "PASSED" if verdict["passed"] else "FAILED",
                         verdict.get("max_err", 0.0), d.get("distinct_passes", 0),
                         d.get("quorum", 0), d.get("standing"))
            except requests.RequestException as e:
                log.warning("peer verify: could not report verdict for %s: %s",
                            job["node_id"], e)

    def challenge_peer(self, job):
        """Run proof-of-compute against another node. Returns {'passed', 'max_err'}."""
        from security import proof_of_compute as poc
        lo, hi, total = job["layer_start"], job["layer_end"], job["total_layers"]
        if hi == total - 1:
            res = poc.attest(job["host"], job["port"], lo, total)
        else:
            res = poc.attest_middle(job["host"], job["port"], lo, hi + 1)
        return {"passed": bool(res["passed"]), "max_err": float(res["max_err"])}

    # -- personal Chat UI (agent/local_chat.py) ------------------------------ #
    def start_local_chat(self):
        """Best-effort: a broken/slow local Chat UI must never stop this machine from
        serving the network (that's the agent's primary job) -- local_chat.start() already
        swallows its own errors, this just decides whether to call it at all."""
        if not self.cfg.get("local_chat", True):
            self.local_chat_state = "disabled"
            return
        if not self.cfg.get("model_id"):
            log.warning("local chat skipped: model_id not known yet (setup() hasn't run)")
            self.local_chat_state = "failed"
            return
        driver_slice_dir = os.path.join(HERE, "driver_slice")
        port = self.cfg.get("local_chat_port", local_chat.DEFAULT_PORT)
        self.local_chat_server = local_chat.start(
            self.base, self.cfg["model_id"], driver_slice_dir,
            port=port, oauth_cfg=self.cfg.get("oauth"),
            node_id=self.cfg.get("node_id"), node_token=self.cfg.get("node_token"))
        if self.local_chat_server is None:
            self.local_chat_state = "failed"
            # Report the ACTUAL exception. This used to assert "check whether that port is
            # already in use" no matter what went wrong, which is a confident wrong diagnosis:
            # the real failure on a packaged build was `No module named '_sqlite3'`, and the
            # message sent everyone to inspect a port that was free.
            self.local_chat_error = local_chat.LAST_ERROR
            log.error("local Chat UI failed to start on port %d (%s) — the node is still "
                      "serving the network", port,
                      self.local_chat_error or "no reason recorded")
        else:
            self.local_chat_state = "running"
            log.info("local Chat UI ready on http://127.0.0.1:%d", port)

    def heartbeat_loop(self):
        beat = 0
        last_line = None            # so a steady state is logged once, not every 30 s
        while not self._stop.is_set():
            beat += 1

            def beat_log(msg, *args, level=logging.INFO):
                """Log a heartbeat outcome ON CHANGE, plus a periodic proof of life.

                Every beat used to log at INFO. At ~30 s that is 2,833 lines and 141 KB a day
                of `heartbeat ok — active`, and `logtail.MAX_BYTES` is 64 KB — so the tail the
                coordinator collects held under 11 hours and, on an idle node, was ~100%
                heartbeat. `neuron_logs.py` exists so that diagnosing a stranger's node does
                not depend on their attention; a window full of "ok" defeats exactly that.

                Same shape as the fix in `verify_service.py`: quiet while nothing changes,
                an alive line every ALIVE_EVERY beats so SILENCE STILL MEANS DEAD, and every
                transition logged the moment it happens.
                """
                nonlocal last_line
                line = msg % args if args else msg
                changed = line != last_line
                last_line = line
                if changed or beat % HEARTBEAT_ALIVE_EVERY == 0:
                    log.log(level, "%s", line + ("" if changed else "  (still)"))
                else:
                    log.debug("%s", line)
            reasons = ["paused by user"] if self.user_paused.is_set() else self.guard.reasons_to_pause()
            # Prove the PUBLIC path, not just the local one. The tunnel can die silently while
            # this process is perfectly healthy (the control socket stays ESTABLISHED and blocked
            # in recv until the OS keepalive gives up — 2 hours on Windows), and a relayed node
            # is the only kind a stranger can run. If it is dead, restart it and skip this
            # heartbeat so the coordinator stops routing to a black hole in the meantime.
            relay = self.relay or self.cfg.get("relay")
            if relay and self.server is not None and beat % RELAY_PROBE_EVERY == 0 \
                    and not reasons and not self.relay_reachable():
                log.error("relay tunnel is not carrying traffic — restarting it and holding "
                          "off the heartbeat until it answers")
                self.state.update(status="error", detail="relay tunnel unreachable — restarting")
                try:
                    self.start_tunnel(relay)
                except Exception as e:
                    log.warning("could not restart the relay tunnel: %s", e)
                self._stop.wait(PING_SECONDS)
                continue
            # "Online" has to mean "serving", or routing sends real requests into a black
            # hole ([P21]). A dead listener is not a pause: stop pinging so the coordinator
            # marks this node offline and routes around it.
            if self.server is not None and not self.server.listening.is_set():
                detail = f"node server not listening ({self.server.bind_error or 'stopped'})"
                self.state.update(status="error", detail=detail)
                log.error("%s — not advertising availability", detail)
                self._stop.wait(PING_SECONDS)
                continue
            try:
                if reasons:
                    self.state.update(status="idle", detail="; ".join(reasons))
                    beat_log("paused (%s) — not advertising availability", "; ".join(reasons))
                else:
                    self.ping()
                    if self.standing == "probationary":
                        # "active" would be a lie here: the coordinator excludes probationary
                        # nodes from routing, so this beat advertises availability nobody can
                        # use. Say which of the two it is ([P24]).
                        self.state.update(status="active",
                                          detail="awaiting verification — not yet earning")
                        beat_log("heartbeat ok — probationary, not yet serving or earning")
                    else:
                        self.state.update(status="active", detail="earning")
                        beat_log("heartbeat ok — active")
            except requests.RequestException as e:
                self.state.update(status="error", detail=f"coordinator unreachable: {e}")
                beat_log("heartbeat failed: %s", e, level=logging.WARNING)
            self._stop.wait(PING_SECONDS)

    # -- model migration (Build 3, node-side): download the coordinator's chosen target
    # tier's slice in the BACKGROUND while still serving the current model, report ready,
    # then hot-swap only once the coordinator confirms cutover actually happened. ---------- #
    def migration_loop(self):
        prepared = None   # {model_id, layer_start, layer_end, total_layers, slice_dir, ready}
        while not self._stop.is_set():
            if self.server is not None:
                try:
                    asg = requests.get(
                        f"{self.base}/node/{self.cfg['node_id']}/migration", timeout=15).json()
                    if asg.get("migrating"):
                        is_new_target = prepared is None or (
                            prepared["model_id"] != asg["model_id"] or
                            prepared["layer_start"] != asg["layer_start"] or
                            prepared["layer_end"] != asg["layer_end"])
                        if is_new_target:
                            prepared = self._prepare_migration_target(asg)
                        elif not prepared["ready"]:
                            self._report_migration_ready(prepared)
                    elif prepared is not None:
                        # migration ended — either cut over to our target, or aborted first
                        self._maybe_cutover(prepared)
                        prepared = None
                except requests.RequestException as e:
                    log.warning("migration poll failed: %s", e)
            self._stop.wait(MIGRATION_POLL_SECONDS)

    def _prepare_migration_target(self, asg):
        """Download the target tier's slice into a SEPARATE dir — this node keeps answering
        requests on the OLD model for the entire download, so preparing never costs coverage."""
        slice_dir = os.path.join(HERE, "model_slice_migrating")
        shutil.rmtree(slice_dir, ignore_errors=True)      # drop any stale prior target
        total = asg["total_layers"]
        is_first, is_last = asg["layer_start"] == 0, asg["layer_end"] == total - 1
        log.info("migration: preparing target %s layers %d-%d (of %d)",
                 asg["model_id"], asg["layer_start"], asg["layer_end"], total)
        prepared = {"model_id": asg["model_id"], "layer_start": asg["layer_start"],
                   "layer_end": asg["layer_end"], "total_layers": total,
                   "slice_dir": slice_dir, "ready": False}
        try:
            slice_downloader.download_slice(
                asg["model_id"], asg["layer_start"], asg["layer_end"], slice_dir,
                is_first_node=is_first, is_last_node=is_last)
        except Exception as e:
            log.warning("migration: target slice download failed, will retry next poll: %s", e)
            return None
        self._report_migration_ready(prepared)
        return prepared

    def _report_migration_ready(self, prepared):
        try:
            requests.post(f"{self.base}/node/{self.cfg['node_id']}/migration-ready",
                         headers={"X-Node-Token": self.cfg["node_token"]}, timeout=15
                         ).raise_for_status()
            prepared["ready"] = True
            log.info("migration: target slice ready — reported to coordinator, "
                     "still serving the current model until cutover")
        except requests.RequestException as e:
            log.warning("migration: failed to report ready (will retry): %s", e)

    def _maybe_cutover(self, prepared):
        """Migration is no longer active — either it cut over to OUR target, or it aborted
        (capacity dropped, target reverted) before we ever reported ready. Confirm what the
        coordinator actually ended up serving before swapping — an abort must not reload us
        onto a model the network isn't running."""
        if not prepared["ready"]:
            shutil.rmtree(prepared["slice_dir"], ignore_errors=True)
            return
        try:
            net = requests.get(f"{self.base}/network/model", timeout=15).json()
        except requests.RequestException as e:
            log.warning("migration: could not confirm cutover, leaving prepared slice: %s", e)
            return
        if net.get("serving", {}).get("model_id") != prepared["model_id"]:
            log.info("migration: aborted before cutover — discarding prepared slice")
            shutil.rmtree(prepared["slice_dir"], ignore_errors=True)
            return
        self._swap_to(prepared)

    def _swap_to(self, prepared):
        log.info("migration: cutting over to %s layers %d-%d", prepared["model_id"],
                 prepared["layer_start"], prepared["layer_end"])
        self.server.reload(prepared["slice_dir"], prepared["layer_start"],
                           prepared["layer_end"], prepared["total_layers"])
        old_slice_dir = os.path.join(HERE, os.path.normpath(self.cfg["slice_dir"]))
        shutil.rmtree(old_slice_dir, ignore_errors=True)
        os.rename(prepared["slice_dir"], old_slice_dir)
        self.cfg["model_id"] = prepared["model_id"]
        self.cfg["layer_start"], self.cfg["layer_end"] = prepared["layer_start"], prepared["layer_end"]
        self._save()
        self.state.update(layers=[prepared["layer_start"], prepared["layer_end"]])
        log.info("migration: now serving %s layers %d-%d", prepared["model_id"],
                 prepared["layer_start"], prepared["layer_end"])

    def bind_payout_address(self):
        """Give this node's earnings an on-chain destination (MIGRATION_PLAN.md blocker 1).

        Best-effort and non-fatal: an unbound node serves and earns exactly as before, it just
        has nowhere to be paid if the ledger ever moves on-chain. Runs after setup() because it
        needs a registered node_id and a live token.
        """
        try:
            from agent import payout_key
            payout_key.ensure_bound(
                self.base, self.cfg["node_id"], self.cfg["node_token"],
                state_dir=os.path.dirname(self.config_path) or HERE,
                configured_address=self.cfg.get("payout_address"))
        except Exception as e:                                      # noqa: BLE001
            log.debug("payout binding skipped (%s: %s)", e.__class__.__name__, e)

    def _serving_now(self):
        """Is this machine mid-request? The updater's guard against replacing the app under a
        live inference — a node that disappears mid-chain kills the answer for everyone on it."""
        try:
            from agent import node_server
            return node_server.is_busy()
        except Exception:                                       # noqa: BLE001
            return True          # unknown means busy: defer rather than risk it

    def update_loop(self):
        """Startup check, then every 24h. The whole point of the project reaching strangers:
        they will not reinstall to pick up a fix, so fixes have to reach them.

        `auto_update` defaults to True — a node that never updates is a node running whatever
        bugs it shipped with, forever. But this process downloads and executes a binary on
        someone else's machine unattended, and an operator who would rather decide that for
        themselves should not have to firewall us to do it."""
        from agent import updater
        updater.update_loop(self.base, stop=self._stop, busy=self._serving_now,
                            enabled=self.cfg.get("auto_update", True),
                            on_result=self._record_update_check)

    def _record_update_check(self, verdict):
        """Remember the last update verdict so the next registration can report it.

        Kept in memory rather than written to config.json: it is a fact about this RUN, and
        persisting it would outlive the condition it describes -- a `download-failed` from last
        week, replayed after a restart that fixed it, is the stale-field mistake this project has
        now made three times ([P34], [P37], and the ms/layer display)."""
        self._last_update_check = verdict

    def remeasure_loop(self):
        """Re-time this node's own segment periodically and report it.

        The measurement used to be taken exactly once, at startup, and believed forever. One
        taken while the machine was thrashing -- during a slice download, a migration, or simply
        with the owner compiling something -- became permanent, and the coordinator sizes stages
        and picks replicas from it. Live 2026-08-10: 4150 ms/layer against 8-22 for its peers,
        answers at 0.24 tok/s, and no path back because a node excluded from routing never gets
        traffic that could revise it (PROBLEMS.md [P34]).

        Skipped while paused or while the guard is holding this node back: timing a segment on a
        machine that is deliberately not serving measures the pause, not the node.
        """
        while not self._stop.wait(REMEASURE_INTERVAL_S):
            try:
                if self.user_paused.is_set() or self.server is None:
                    continue
                info = {"layer_start": self.cfg.get("layer_start"),
                        "layer_end": self.cfg.get("layer_end")}
                if info["layer_start"] is None or info["layer_end"] is None:
                    continue
                self.report_speed(info)
            except Exception as e:
                # Never take the agent down for a benchmark. A stale figure is a routing
                # nuisance; a dead node is an outage.
                log.debug("re-measure failed, keeping the previous figure: %s", e)

    def run(self):
        self.setup()
        self.bind_payout_address()
        threading.Thread(target=self.remeasure_loop, daemon=True).start()
        threading.Thread(target=self.update_loop, daemon=True).start()
        threading.Thread(target=self.migration_loop, daemon=True).start()
        threading.Thread(target=self.peer_verify_loop, daemon=True).start()
        threading.Thread(target=self.start_local_chat, daemon=True).start()
        self.heartbeat_loop()

    def stop(self):
        self._stop.set()


def main():
    # FIRST STATEMENT, before argparse and before the config is read. It used to run after
    # both, because the log level comes from the config -- so a config this build could not
    # read took the agent down with no log file, and [P24]'s "0.18 produced no log at all"
    # had no way to be diagnosed remotely. The level is re-applied from the config below;
    # starting at INFO and adjusting is strictly better than starting at nothing.
    _setup_logging()
    # Decided at import (it had to be — see _apply_device_preference), reported here, once
    # there is somewhere for it to go. A node that is told it will use a GPU and then does not
    # should be able to find out why from its own log.
    if _DEVICE_NOTE:
        log.info("%s", _DEVICE_NOTE)
    # Decided before the heavy imports (see _CPU) and reported here, once there is somewhere
    # for it to go. Logged on EVERY node, not only on a refusal: the fleet's real instruction-set
    # floor is a thing to learn from the machines that join, not to assume.
    log.info("%s", cpu_check.summary(_CPU))
    ap = argparse.ArgumentParser(description="Run a NEURON node agent.")
    # One machine could only ever run ONE agent, because the config path was a module
    # constant. That is fine for a stranger donating one PC, and wrong for anyone holding the
    # network up: with a 3-stage pipeline you need three online nodes, so losing one machine
    # (a laptop that sleeps -- [P4]) took the whole network down even when the remaining
    # machines had ample spare capacity. Each --config gets its own node_id, port and slice.
    ap.add_argument("--config", default=CONFIG_PATH,
                    help=f"config file to run from (default {CONFIG_PATH})")
    ap.add_argument("--donation-mode", default=None,
                    choices=sorted(resource_guard.DONATION_MODES),
                    help="override donation_mode for this run. 'idle' (the default for a "
                         "stranger's PC) stops serving the moment the owner touches the "
                         "machine; use 'generous' or 'max' on a machine that is meant to "
                         "hold the network up.")
    ap.add_argument("--layers", default=None, metavar="START-END",
                    help="serve this exact layer range instead of asking for a placement")
    ap.add_argument("--port", type=int, default=None, help="node server port")
    ap.add_argument("--node-id", default=None, help="register under this node id")
    ap.add_argument("--relay", dest="relay", action="store_true", default=None,
                    help="be reachable through the public relay (the default): peers get a "
                         "public endpoint instead of a LAN/Tailscale address")
    ap.add_argument("--no-relay", dest="relay", action="store_false",
                    help="advertise this machine's own address instead — only correct when "
                         "every peer is on the same LAN or tailnet")
    ap.add_argument("--no-local-chat", action="store_true",
                    help="do not start this agent's own Chat UI (a second agent on the same "
                         "machine must not fight the first one for the chat port)")
    args = ap.parse_args()

    try:
        path = ensure_config(args.config)
        cfg = load_config(path)
    except Exception:
        # The prime suspect for a version that starts and vanishes: an in-place upgrade reads
        # the previous version's config file. Unreadable JSON, a path that is not writable in
        # the frozen app's state dir, a half-written file -- all of it used to be silent.
        crash_log(f"could not read the config at {args.config}")
        raise
    dirty = False
    if args.donation_mode:
        cfg["donation_mode"], dirty = args.donation_mode, True
    if args.layers:
        lo, _, hi = args.layers.partition("-")
        cfg["layer_start"], cfg["layer_end"], dirty = int(lo), int(hi), True
        # An operator who names a range means it; the agent's own re-placement (which moves a
        # probationary node off a redundant slice) must leave it alone.
        cfg["layers_pinned"] = True
    if args.port:
        cfg["port"], dirty = args.port, True
    if args.node_id:
        cfg["node_id"], dirty = args.node_id, True
    if args.relay is not None:
        cfg["behind_nat"], dirty = args.relay, True
    if args.no_local_chat:
        cfg["local_chat"], dirty = False, True
    if dirty:
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2)

    _setup_logging(cfg.get("log_level", "INFO"))     # now apply the level the config asks for
    log.info("NEURON agent v%s starting | config=%s | coordinator=%s | mode=%s",
             _version(), path, cfg.get("coordinator", DEFAULT_CONFIG["coordinator"]),
             cfg.get("donation_mode"))
    # A config written by an older build is missing whatever this one added. That is not an
    # error -- every lookup falls back to DEFAULT_CONFIG -- but it belongs in the log, because
    # "upgraded in place over an older install" is the first thing to know when a version that
    # worked stops working ([P24]).
    missing = [k for k in DEFAULT_CONFIG if k not in cfg]
    if missing:
        log.info("config predates this build; using built-in defaults for: %s",
                 ", ".join(sorted(missing)))
    try:
        agent = Agent(config_path=path)
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
        log.info("agent stopped")
    except BaseException:
        # Logging is up by now, so this reaches agent.log — but crash_log also puts it in the
        # file when the logging stack itself is what broke, and costs nothing.
        log.exception("the agent stopped with an unhandled error")
        crash_log("the agent stopped with an unhandled error")
        raise


if __name__ == "__main__":
    main()
