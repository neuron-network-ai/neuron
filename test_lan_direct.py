"""test_lan_direct.py — run: python -m test_lan_direct

Two machines in the same house should not send every activation to Amsterdam ([P57]: dialling
this PC's OWN node through the relay costs 88 ms where a LAN neighbour costs 5.3). The obvious
fix — publish node addresses instead of the relay's — was refused for two good reasons: it makes
a private mesh a dependency, and **the relay is a privacy feature**. Peers see the relay and
never where a volunteer lives; `/node/list` already hides node addresses from public callers for
exactly that reason.

So the disclosure is inverted. The DRIVER names the private /24s it is already on, and a node
answers only from inside one of them. This file pins the properties that make that safe, because
every one of them is the kind that fails silently:

  * an address is offered ONLY into a subnet the caller already said it was on;
  * only RFC1918 — a public address can never be offered or accepted;
  * **100.64/10 is excluded on purpose** (Tailscale / CGNAT), because being fast must not
    require a mesh VPN;
  * the caller re-checks the answer, so a node cannot name some OTHER machine on the caller's
    network and have activations sent there;
  * every failure falls back to the relay, so this can speed a request up and never break one.
"""
import socket
import sys
import threading
import types

import torch

import common
import lan_direct
from agent import node_server

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class _FakeBatcher:
    def submit(self, hidden, cache, past):
        return torch.zeros(hidden.shape[0], hidden.shape[1], 64)


class _Server(node_server.NodeServer):
    def __init__(self, port=50999):
        self.lo, self.hi, self.n = 10, 27, 28
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=64))
        self._batchers = {}
        self._batcher_lock = threading.Lock()
        self.paused = threading.Event()
        self.listening = threading.Event()
        self.bind_error = None
        self.listen_port = port

    def _batcher(self, role, lo, hi):
        return _FakeBatcher()


def ack_for(cfg, addresses, port=50999):
    """Drive a real serve() with `cfg` while the node claims to hold `addresses`."""
    real = lan_direct.local_addresses
    lan_direct.local_addresses = lambda: addresses
    try:
        a, b = socket.socketpair()
        srv = _Server(port)
        t = threading.Thread(target=lambda: srv.serve(b, ("test", 0)), daemon=True)
        t.start()
        common.send_msg(a, cfg)
        ack = common.recv_msg(a)
        a.close()
        t.join(timeout=5)
        return ack
    finally:
        lan_direct.local_addresses = real


def main():
    # ---- 1. what counts as a private address ---------------------------------- #
    for ip in ("10.0.0.4", "172.16.5.9", "172.31.255.254", "192.168.1.11"):
        check(f"{ip} is private", lan_direct.is_private(ip) is True)
    check("100.79.125.112 (Tailscale) is NOT — a mesh VPN must not be the fast path",
          lan_direct.is_private("100.79.125.112") is False)
    check("150.230.22.250 (the relay's public IP) is NOT",
          lan_direct.is_private("150.230.22.250") is False)
    for bad in ("127.0.0.1", "172.32.0.1", "8.8.8.8", "", None, "not-an-ip", "192.168.1"):
        check(f"{bad!r} is not offerable", lan_direct.is_private(bad) is False)
    check("prefix_of takes the /24", lan_direct.prefix_of("192.168.1.11") == "192.168.1")

    # ---- 2. an address is only ever offered INTO the caller's own subnet ------- #
    mine = ["192.168.1.11", "10.0.0.7"]
    check("answers on a subnet the caller named",
          lan_direct.address_for(["192.168.1"], mine) == "192.168.1.11")
    check("...and on a second one",
          lan_direct.address_for(["10.0.0"], mine) == "10.0.0.7")
    check("SILENT for a subnet we share nothing with (the privacy property)",
          lan_direct.address_for(["192.168.9"], mine) is None)
    for bad_hint in (None, [], "192.168.1", [None], [1, 2], {}):
        check(f"no answer for a malformed hint {bad_hint!r}",
              lan_direct.address_for(bad_hint, mine) is None)
    check("a machine with only a public address offers nothing",
          lan_direct.address_for(["203.0.113"], ["203.0.113.5"]) is None)
    check("a machine with only a Tailscale address offers nothing",
          lan_direct.address_for(["100.79.125"], ["100.79.125.112"]) is None)

    # ---- 3. the CALLER re-checks, so a node cannot steer it -------------------- #
    good = {"ip": "192.168.1.11", "port": 50999}
    check("a matching offer is usable", lan_direct.usable(good, ["192.168.1"]) is True)
    check("an offer OUTSIDE what we asked about is refused — this is the steering guard",
          lan_direct.usable({"ip": "10.9.9.9", "port": 50999}, ["192.168.1"]) is False)
    check("a public address is refused even if we somehow asked",
          lan_direct.usable({"ip": "8.8.8.8", "port": 53}, ["8.8.8"]) is False)
    for bad in ({"ip": "192.168.1.11"}, {"ip": "192.168.1.11", "port": 0},
                {"ip": "192.168.1.11", "port": "50999"}, {}, None, "x"):
        check(f"malformed offer {bad!r} refused", lan_direct.usable(bad, ["192.168.1"]) is False)

    # ---- 4. through a real serve() -------------------------------------------- #
    base = {"type": "config", "s2": 10, "stage": "last", "wire": ["f32"]}
    ack = ack_for({**base, "lan_hint": ["192.168.1"]}, ["192.168.1.11"])
    check("a node on the caller's LAN answers with its address",
          ack.get("direct") == {"ip": "192.168.1.11", "port": 50999}, ack)
    ack = ack_for({**base, "lan_hint": ["192.168.9"]}, ["192.168.1.11"])
    check("a node on a DIFFERENT network answers with no address at all",
          "direct" not in ack, ack)
    ack = ack_for(base, ["192.168.1.11"])
    check("a caller that did not ask is told nothing (no volunteering)",
          "direct" not in ack, ack)
    ack = ack_for({**base, "lan_hint": ["100.79.125"]}, ["100.79.125.112", "192.168.1.11"])
    check("a Tailscale-only hint gets nothing, so the mesh is never the route",
          "direct" not in ack, ack)
    check("...and the ack still carries everything it carried before",
          ack.get("ok") is True and ack.get("s2") == 10 and ack.get("wire") == "f32", ack)

    # ---- 5. the relay stays the address of record ----------------------------- #
    src = open("neuron_driver.py", encoding="utf-8").read()
    check("the driver ASKS rather than the node volunteering",
          '"lan_hint": lan_direct.local_prefixes()' in src)
    check("the driver re-checks the answer before trusting it",
          "lan_direct.usable(ack.get(\"direct\")" in src)
    check("a failed direct dial keeps the relay connection, never drops the request",
          "return s, c                       # keep the relay connection" in src)
    check("a peer that moved is forgotten rather than retried forever",
          "self._direct.pop(node, None)" in src)
    ns = open("agent/node_server.py", encoding="utf-8").read()
    check("every config ack carries the answer", ns.count("**ack_wire, **ack_direct}") == 3)
    check("the node declines when it has no port to advertise",
          "if not hint or not self.listen_port:" in ns)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
