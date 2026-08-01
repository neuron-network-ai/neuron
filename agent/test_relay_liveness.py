"""agent/test_relay_liveness.py — run: python -m agent.test_relay_liveness

Covers the relay-tunnel liveness check. The bug it guards against was observed live: a node's
relay port accepted TCP and then answered nothing for ~2 hours (the control socket sat
ESTABLISHED, blocked in recv, until the OS keepalive gave up — 2h by default on Windows),
while the agent kept logging `heartbeat ok — active`. The coordinator therefore listed the node
online and routed real requests into a black hole.

So the load-bearing case is precisely "accepts the connection, never speaks" — a plain TCP
connect check would have passed it.
"""
import json
import os
import socket
import tempfile
import threading

from agent import agent as agentmod

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


class _Server:
    """A fake relay endpoint. mode='silent' accepts and never replies (the real failure);
    mode='answer' completes the config handshake like a healthy node behind a live tunnel."""

    def __init__(self, mode):
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import common
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            if self.mode == "silent":
                continue                      # hold it open, say nothing — the observed failure
            try:
                common.recv_msg(conn)
                common.send_msg(conn, {"ok": True, "layers": 28, "s1": 0, "s2": 10})
            except Exception:
                pass


class _FakeServer:
    lo, hi, n = 0, 9, 28


def _agent(tmpdir, port):
    path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="t", node_token="tok", layer_start=0, layer_end=9,
               relay={"host": "127.0.0.1", "public_port": port,
                      "control_port": 8010, "data_port": 8011, "ticket": "x"})
    json.dump(cfg, open(path, "w"))
    a = agentmod.Agent(config_path=path)
    a.server = _FakeServer()
    return a


def main():
    agentmod.RELAY_PROBE_TIMEOUT_S = 3        # keep the silent case quick
    tmpdir = tempfile.mkdtemp(prefix="neuron-relay-live-")

    silent = _Server("silent")
    a = _agent(tmpdir, silent.port)
    check("a port that ACCEPTS but never answers is reported unreachable "
          "(a TCP-connect check would wrongly pass here)", a.relay_reachable() is False)

    healthy = _Server("answer")
    b = _agent(tmpdir, healthy.port)
    check("a completed handshake is reported reachable", b.relay_reachable() is True)

    # no relay configured -> nothing to prove, must not raise or report a false failure
    path = os.path.join(tmpdir, "norelay.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="t", node_token="tok", behind_nat=False)
    json.dump(cfg, open(path, "w"))
    c = agentmod.Agent(config_path=path)
    c.server = _FakeServer()
    check("a node with no relay endpoint reports reachable (no false alarm)",
          c.relay_reachable() is True)

    # a dead tunnel must RESTART the tunnel and NOT heartbeat: advertising availability while
    # unreachable is the whole failure this exists to prevent.
    d = _agent(tmpdir, silent.port)
    restarts, pings = [], []
    d.start_tunnel = lambda relay: restarts.append(relay)
    d.ping = lambda: pings.append(1)
    d.guard.reasons_to_pause = lambda: []
    agentmod.RELAY_PROBE_EVERY = 1            # probe on the first beat
    stop_after = {"n": 0}

    real_wait = d._stop.wait

    def wait_once(_t):
        stop_after["n"] += 1
        if stop_after["n"] >= 1:
            d._stop.set()
        return real_wait(0)

    d._stop.wait = wait_once
    d.heartbeat_loop()
    check("a dead tunnel restarts the tunnel", len(restarts) == 1)
    check("a dead tunnel does NOT send a heartbeat (stops advertising availability)",
          pings == [])

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
