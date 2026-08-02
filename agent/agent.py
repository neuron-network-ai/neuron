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
import uuid

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
from agent import local_chat                               # noqa: E402
from agent import resource_guard                          # noqa: E402
from agent.node_server import NodeServer                  # noqa: E402
import slice_downloader                                   # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_PATH = os.path.join(HERE, "agent.log")
RETRY_SECONDS = 60
PING_SECONDS = 30
MIGRATION_POLL_SECONDS = 20
# How long to wait for the node server to actually bind before treating setup as failed.
# A bind either works immediately or fails immediately; the only slow case is a previous
# copy of the agent still holding the port on its way out.
BIND_TIMEOUT_S = 20
# How often to prove this node is reachable at its PUBLIC relay endpoint, and how long to wait
# for that proof. Every fourth heartbeat (~2 min) rather than every one: it opens a real
# connection through the relay and back into our own server, so it is not free.
RELAY_PROBE_EVERY = 4
RELAY_PROBE_TIMEOUT_S = 20
# How often a verified node looks for a newcomer to vouch for. Slow on purpose: verifying is
# a favour to the network, not this node's job, and a newcomer waiting an extra minute costs
# nothing next to needing a human to be awake.
PEER_VERIFY_POLL_SECONDS = 60

# Written on first run if no config exists (so a freshly-installed app just works): open join,
# auto-placement, green idle donation, relay on. Matches agent/config.json.
DEFAULT_CONFIG = {
    "coordinator": "https://neuronnet.duckdns.org",
    "node_id": None, "node_token": None, "model_id": None,
    "layer_start": None, "layer_end": None,
    "slice_dir": "./model_slice/",
    "donation_mode": "idle", "idle_threshold_seconds": 60,
    "behind_nat": True, "log_level": "INFO",
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
        self.cfg = json.load(open(config_path))
        self.base = self.cfg["coordinator"].rstrip("/")
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

    # -- config persistence -------------------------------------------------- #
    def _save(self):
        json.dump(self.cfg, open(self.config_path, "w"), indent=2)

    # -- coordinator calls --------------------------------------------------- #
    def ensure_placement(self):
        """Zero-config open join (S20): if config has no layer range, ask the coordinator
        where we fit (fills a gap, else replicates the last segment) and persist it."""
        if self.cfg.get("layer_start") is not None and self.cfg.get("layer_end") is not None:
            return
        r = requests.get(f"{self.base}/node/placement", timeout=15)
        r.raise_for_status()
        p = r.json()
        self.cfg["layer_start"], self.cfg["layer_end"] = p["layer_start"], p["layer_end"]
        self._save()
        log.info("auto-placed on layers %d-%d (%s: %s)", p["layer_start"], p["layer_end"],
                 p.get("role"), p.get("reason"))

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
            common._run_layers(model, layers, x, cache, 0)      # warm: first pass allocates
            t0 = time.perf_counter()
            for i in range(iters):
                common._run_layers(model, layers, x, cache, 1 + i)
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

    def register(self):
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
            "cores": os.cpu_count(),
            "ram_gb": int(psutil.virtual_memory().total // 10**9),
            # With cores/ram_gb this is the coarse hardware signature the coordinator groups
            # on to spot one machine registering many node ids. It is a signal an operator
            # reviews, never a block, and it says nothing a `User-Agent` header would not.
            "platform": _platform.platform(),
            # When true the coordinator ignores the address above and stores this node at a
            # relay endpoint instead, so every peer that asks for a chain is handed a public
            # host:port. That is the whole of "a stranger can be in the pipeline".
            "behind_nat": self.use_relay(),
        }
        # Feeds coordinator/balancer.py. None on the first registration (the slice is not
        # loaded yet, so there is nothing to time); report_speed() re-registers with the real
        # figure once the node server is up, and the upsert COALESCEs so a later None never
        # erases it.
        if self.cfg.get("ms_per_layer") is not None:
            body["ms_per_layer"] = self.cfg["ms_per_layer"]
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
        standing = data.get("standing", "trusted")
        log.info("registered as %s [%s], assigned layers %s (%d cores, %d GB, %s)",
                 body["node_id"], standing, data["assigned_layers"],
                 body["cores"], body["ram_gb"], ip)
        if standing == "probationary":
            log.info("PROBATIONARY: serving challenges only — a verifier must confirm this "
                     "node (proof-of-compute) before it receives live requests or earns NRN")
        return data["assigned_layers"]

    def slice_info(self):
        r = requests.get(f"{self.base}/node/{self.cfg['node_id']}/slice-info", timeout=30)
        r.raise_for_status()
        return r.json()

    def ping(self):
        requests.get(f"{self.base}/node/{self.cfg['node_id']}/ping",
                     headers={"X-Node-Token": self.cfg["node_token"]}, timeout=10).raise_for_status()

    # -- slice download ------------------------------------------------------ #
    @staticmethod
    def slice_layers_on_disk(weights):
        """Which decoder layers a downloaded slice actually contains.

        Read from the safetensors header (an 8-byte little-endian length then JSON), so it
        reflects the bytes on disk rather than what some config claims about them.
        Returns (lo, hi) or None if it cannot be determined.
        """
        try:
            with open(weights, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                if not 0 < n < 100 * 1024 * 1024:
                    return None
                header = json.loads(f.read(n).decode())
            idx = {int(k.split(".")[2]) for k in header
                   if k.startswith("model.layers.") and k.split(".")[2].isdigit()}
            return (min(idx), max(idx)) if idx else None
        except (OSError, ValueError, KeyError, IndexError):
            return None

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
            if have == want:
                log.info("slice already present (%s) — skipping download", slice_dir)
                return slice_dir
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
                self._save()
                slice_dir = self.ensure_slice(info)
                port = self.cfg.get("port", 50999)
                if self.server is None:
                    self.server = NodeServer(slice_dir, info["layer_start"], info["layer_end"],
                                             info.get("total_layers", 28))
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
            port=port, oauth_cfg=self.cfg.get("oauth"))
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
        while not self._stop.is_set():
            beat += 1
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
                    log.info("paused (%s) — not advertising availability", "; ".join(reasons))
                else:
                    self.ping()
                    self.state.update(status="active", detail="earning")
                    log.info("heartbeat ok — active")
            except requests.RequestException as e:
                self.state.update(status="error", detail=f"coordinator unreachable: {e}")
                log.warning("heartbeat failed: %s", e)
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

    def run(self):
        self.setup()
        self.bind_payout_address()
        threading.Thread(target=self.migration_loop, daemon=True).start()
        threading.Thread(target=self.peer_verify_loop, daemon=True).start()
        threading.Thread(target=self.start_local_chat, daemon=True).start()
        self.heartbeat_loop()

    def stop(self):
        self._stop.set()


def main():
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

    path = ensure_config(args.config)
    cfg = json.load(open(path))
    dirty = False
    if args.donation_mode:
        cfg["donation_mode"], dirty = args.donation_mode, True
    if args.layers:
        lo, _, hi = args.layers.partition("-")
        cfg["layer_start"], cfg["layer_end"], dirty = int(lo), int(hi), True
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

    _setup_logging(cfg.get("log_level", "INFO"))
    log.info("NEURON agent starting | config=%s | coordinator=%s | mode=%s",
             path, cfg["coordinator"], cfg.get("donation_mode"))
    agent = Agent(config_path=path)
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
        log.info("agent stopped")


if __name__ == "__main__":
    main()
