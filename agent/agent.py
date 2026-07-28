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
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import time

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

# Written on first run if no config exists (so a freshly-installed app just works): open join,
# auto-placement, green idle donation, relay on. Matches agent/config.json.
DEFAULT_CONFIG = {
    "coordinator": "http://150.230.22.250:8001",
    "node_id": None, "node_token": None, "model_id": None,
    "layer_start": None, "layer_end": None,
    "slice_dir": "./model_slice/",
    "donation_mode": "idle", "idle_threshold_seconds": 60,
    "behind_nat": True, "log_level": "INFO",
    # every installed agent also runs its own personal Chat UI (agent/local_chat.py) --
    # your own front door to the network, on your own machine, off by default to the
    # internet (127.0.0.1 only). Independent of donation_mode: pausing compute-sharing
    # when you're active shouldn't also take away your own ability to use the network.
    "local_chat": True, "local_chat_port": 8080,
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
    log.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


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

    def register(self):
        self.ensure_placement()
        ip = detect_tailscale_ip()
        body = {
            "node_id": self.cfg.get("node_id") or f"agent-{socket.gethostname().lower()}",
            "tailscale_ip": ip,
            "port": self.cfg.get("port", 50999),
            "layer_start": self.cfg["layer_start"],
            "layer_end": self.cfg["layer_end"],
            "cores": os.cpu_count(),
            "ram_gb": int(psutil.virtual_memory().total // 10**9),
            "behind_nat": self.cfg.get("behind_nat", False),
        }
        # Open join (Session 12): register with NO secret by default — anyone can join.
        # Only send the header if the operator explicitly set one (that path marks the
        # node TRUSTED and is for the founder's own dev nodes, not strangers).
        headers = {}
        secret = self.cfg.get("register_secret")
        if secret:
            headers["X-Register-Secret"] = secret
        r = requests.post(f"{self.base}/node/register", json=body, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        self.cfg["node_id"] = body["node_id"]
        self.cfg["node_token"] = data["node_token"]
        self.relay = data.get("relay")          # set when behind_nat -> auto-tunnel
        if self.relay:
            self.cfg["relay"] = self.relay
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
    def ensure_slice(self, info):
        slice_dir = os.path.join(HERE, os.path.normpath(self.cfg["slice_dir"]))
        weights = os.path.join(slice_dir, "model.safetensors")
        if os.path.exists(weights):
            log.info("slice already present (%s) — skipping download", slice_dir)
            return slice_dir
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
                if not self.cfg.get("node_id") or not self.cfg.get("node_token"):
                    self.register()
                info = self.slice_info()
                self.state.update(node_id=self.cfg["node_id"],
                                  layers=[info["layer_start"], info["layer_end"]])
                self.cfg["model_id"] = info["model_id"]      # what we're actually serving now
                self._save()
                slice_dir = self.ensure_slice(info)
                self.server = NodeServer(slice_dir, info["layer_start"], info["layer_end"],
                                         info.get("total_layers", 28))
                threading.Thread(target=self.server.run,
                                 args=("0.0.0.0", self.cfg.get("port", 50999)),
                                 daemon=True).start()
                log.info("node server started on port %d", self.cfg.get("port", 50999))
                # Session 12: if we're behind NAT the coordinator handed us relay params
                # at registration — start the outbound tunnel so peers can reach us with
                # NO inbound port. One-click NAT traversal; nothing for the user to do.
                relay = self.relay or self.cfg.get("relay")
                if relay:
                    import tunnel_client
                    threading.Thread(
                        target=tunnel_client.run_tunnel,
                        kwargs=dict(node_id=self.cfg["node_id"], public_port=relay["public_port"],
                                    relay_host=relay["host"], control_port=relay["control_port"],
                                    data_port=relay["data_port"], local_host="127.0.0.1",
                                    local_port=self.cfg.get("port", 50999), stop=self._stop,
                                    ticket=relay.get("ticket")),
                        daemon=True).start()
                    log.info("relay tunnel started — reachable via %s:%d (NAT-friendly)",
                             relay["host"], relay["public_port"])
                return
            except requests.RequestException as e:
                self.state.update(status="error", detail=f"coordinator unreachable: {e}")
                log.warning("coordinator unreachable, retrying in %ds: %s", RETRY_SECONDS, e)
                self._stop.wait(RETRY_SECONDS)

    # -- personal Chat UI (agent/local_chat.py) ------------------------------ #
    def start_local_chat(self):
        """Best-effort: a broken/slow local Chat UI must never stop this machine from
        serving the network (that's the agent's primary job) -- local_chat.start() already
        swallows its own errors, this just decides whether to call it at all."""
        if not self.cfg.get("local_chat", True):
            return
        if not self.cfg.get("model_id"):
            log.warning("local chat skipped: model_id not known yet (setup() hasn't run)")
            return
        driver_slice_dir = os.path.join(HERE, "driver_slice")
        local_chat.start(self.base, self.cfg["model_id"], driver_slice_dir,
                         port=self.cfg.get("local_chat_port", local_chat.DEFAULT_PORT))

    def heartbeat_loop(self):
        while not self._stop.is_set():
            reasons = ["paused by user"] if self.user_paused.is_set() else self.guard.reasons_to_pause()
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

    def run(self):
        self.setup()
        threading.Thread(target=self.migration_loop, daemon=True).start()
        threading.Thread(target=self.start_local_chat, daemon=True).start()
        self.heartbeat_loop()

    def stop(self):
        self._stop.set()


def main():
    ensure_config(CONFIG_PATH)
    cfg = json.load(open(CONFIG_PATH))
    _setup_logging(cfg.get("log_level", "INFO"))
    log.info("NEURON agent starting | coordinator=%s", cfg["coordinator"])
    agent = Agent()
    try:
        agent.run()
    except KeyboardInterrupt:
        agent.stop()
        log.info("agent stopped")


if __name__ == "__main__":
    main()
