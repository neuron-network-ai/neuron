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
import socket
import subprocess
import sys
import threading
import time

import psutil
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                 # repo root
from agent import resource_guard                          # noqa: E402
from agent.node_server import NodeServer                  # noqa: E402
import slice_downloader                                   # noqa: E402

CONFIG_PATH = os.path.join(HERE, "config.json")
LOG_PATH = os.path.join(HERE, "agent.log")
DEFAULT_REGISTER_SECRET = "neuron-dev-secret"
RETRY_SECONDS = 60
PING_SECONDS = 30

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
        self.cfg = json.load(open(config_path))
        self.base = self.cfg["coordinator"].rstrip("/")
        self.guard = resource_guard.ResourceGuard(
            self.cfg.get("max_cpu_pct", 2), self.cfg.get("idle_threshold_seconds", 60))
        # live state the tray reads: status in {starting, downloading, active, idle, error}
        self.state = {"status": "starting", "node_id": self.cfg.get("node_id"),
                      "layers": None, "coordinator": self.base, "detail": ""}
        self._stop = threading.Event()
        self.user_paused = threading.Event()   # set by the tray's Pause button

    # -- config persistence -------------------------------------------------- #
    def _save(self):
        json.dump(self.cfg, open(self.config_path, "w"), indent=2)

    # -- coordinator calls --------------------------------------------------- #
    def register(self):
        ip = detect_tailscale_ip()
        body = {
            "node_id": self.cfg.get("node_id") or f"agent-{socket.gethostname().lower()}",
            "tailscale_ip": ip,
            "port": self.cfg.get("port", 50999),
            "layer_start": self.cfg["layer_start"],
            "layer_end": self.cfg["layer_end"],
            "cores": os.cpu_count(),
            "ram_gb": int(psutil.virtual_memory().total // 10**9),
        }
        secret = self.cfg.get("register_secret", DEFAULT_REGISTER_SECRET)
        r = requests.post(f"{self.base}/node/register", json=body,
                          headers={"X-Register-Secret": secret}, timeout=15)
        r.raise_for_status()
        data = r.json()
        self.cfg["node_id"] = body["node_id"]
        self.cfg["node_token"] = data["node_token"]
        self._save()
        log.info("registered as %s, assigned layers %s (%d cores, %d GB, %s)",
                 body["node_id"], data["assigned_layers"], body["cores"], body["ram_gb"], ip)
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
                slice_dir = self.ensure_slice(info)
                server = NodeServer(slice_dir, info["layer_start"], info["layer_end"],
                                    info.get("total_layers", 28))
                threading.Thread(target=server.run,
                                 args=("0.0.0.0", self.cfg.get("port", 50999)),
                                 daemon=True).start()
                log.info("node server started on port %d", self.cfg.get("port", 50999))
                return
            except requests.RequestException as e:
                self.state.update(status="error", detail=f"coordinator unreachable: {e}")
                log.warning("coordinator unreachable, retrying in %ds: %s", RETRY_SECONDS, e)
                self._stop.wait(RETRY_SECONDS)

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

    def run(self):
        self.setup()
        self.heartbeat_loop()

    def stop(self):
        self._stop.set()


def main():
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
