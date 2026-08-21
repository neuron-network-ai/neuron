"""agent/test_middle_hop_takes_the_lan.py — run:
       python -m agent.test_middle_hop_takes_the_lan

**[P59]: the relay detour `lan_direct` was built to remove, on the one hop it never covered.**
A caller skips the relay by naming the private /24s it sits on; the peer answers with a direct
address only if it holds one inside them. That worked driver->node and stopped there, because
the first hop was the only one anybody had two machines for. A MIDDLE node dialled whatever
`host_b` the coordinator had handed the driver — the relay — and its onward `config` carried
no `lan_hint` at all, so the next node was never asked and could never offer.

Live on 2026-08-21 that was `192.168.1.11` and `192.168.1.10`, two machines on one switch,
routing every token through a relay in Amsterdam: 88 ms where a neighbour costs 5.3 ([P57]).
The driver could not fix it for them — it sits on a different network, so its own hint matches
neither. **Only the middle node knows it shares a LAN with its next hop, because only the
middle node is on that LAN.**

These run over real sockets against a real RFC1918 address on this machine, so
`lan_direct.usable` is doing its actual job rather than being stubbed — the trust check is the
part that must not rot.

What is pinned, in order of what would hurt most if it broke:

  * the relay is the address of record: no offer, an untrusted offer, or a refused direct dial
    all keep the connection we already have. This may make a request faster and must never be
    able to stop one.
  * an offer naming an address OUTSIDE the subnets we asked about is refused even though the
    peer answered — that is what stops a peer steering somebody's activations at another
    machine on our network.
  * a [P52]-sealed hop is never re-dialled, because a grant is single-use and presenting it
    twice is indistinguishable from a replay.
  * the onward config still says `stage: "last"` outright ([P55]/[P56]) — the LAN question is
    an addition to that message, not a replacement for it.
"""
import os
import socket
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import common
import lan_direct
from agent.node_server import NodeServer

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class _FakeSelf:
    """Just enough NodeServer for `_dial_next_hop`: the hidden size it offers codecs for, the
    layer count it forwards, and its own class-level direct-peer cache."""
    _direct_peers = {}

    class model:
        class config:
            hidden_size = 1536

    n = 28


def listener(bind_ip, ack):
    """A next hop that answers one config with `ack` and records what it was sent."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_ip, 0))
    srv.listen(1)
    seen = {}

    def serve():
        try:
            srv.settimeout(20)
            conn, _ = srv.accept()
            try:
                seen["config"] = common.recv_msg(conn)
                common.send_msg(conn, ack)
                conn.settimeout(20)
                try:
                    common.recv_msg(conn)      # hold it open until the caller is done
                except Exception:
                    pass
            finally:
                conn.close()
        except Exception as e:
            seen["error"] = e

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return srv, srv.getsockname()[1], seen, t


def dial(msg, s2=19):
    return NodeServer._dial_next_hop(_FakeSelf(), msg, s2)


def main():
    global ok, fail
    _FakeSelf._direct_peers.clear()

    private = lan_direct.local_addresses()
    if not private:
        print("  SKIP  this machine has no RFC1918 address, so there is no LAN to prefer")
        print("\n0 passed, 0 failed")
        return 0
    lan_ip = private[0]
    hint = lan_direct.local_prefixes()
    print(f"-- using this machine's own LAN address {lan_ip} (prefixes {hint})")

    print("\n-- the next hop is ASKED about the LAN, which is the whole of [P59]")
    direct_srv, direct_port, direct_seen, dt = listener(lan_ip, {"ok": True, "wire": "f32"})
    relay_srv, relay_port, relay_seen, rt = listener(
        "127.0.0.1", {"ok": True, "wire": "f32",
                      "direct": {"ip": lan_ip, "port": direct_port}})
    try:
        conn, ack = dial({"host_b": "127.0.0.1", "port_b": relay_port, "s1": 10, "s2": 19})
        conn.close()
    finally:
        rt.join(timeout=5)
        dt.join(timeout=5)
        relay_srv.close()
        direct_srv.close()

    cfg = relay_seen.get("config", {})
    check("the onward config carries lan_hint", cfg.get("lan_hint") == hint, str(cfg))
    check("...and still says stage: last outright", cfg.get("stage") == "last", str(cfg))
    check("...and still carries n and a wire preference",
          cfg.get("n") == 28 and isinstance(cfg.get("wire"), list), str(cfg))
    check("the direct address was actually dialled", "config" in direct_seen,
          "the offer was made and ignored — the relay detour is still being paid")
    check("the connection handed back is the DIRECT one",
          direct_seen.get("config", {}).get("lan_hint") == hint)
    check("the peer is cached, so the next request does not re-discover it",
          _FakeSelf._direct_peers.get(("127.0.0.1", relay_port)) == (lan_ip, direct_port),
          str(_FakeSelf._direct_peers))

    print("\n-- an offer OUTSIDE the subnets we asked about is refused, though it answered")
    _FakeSelf._direct_peers.clear()
    relay_srv, relay_port, relay_seen, rt = listener(
        "127.0.0.1", {"ok": True, "wire": "f32",
                      # A real private address, on a network we did not ask about.
                      "direct": {"ip": "10.99.99.99", "port": 50999}})
    try:
        conn, ack = dial({"host_b": "127.0.0.1", "port_b": relay_port, "s1": 10, "s2": 19})
        conn.close()
    finally:
        rt.join(timeout=5)
        relay_srv.close()
    check("an address outside our hint is not dialled",
          not _FakeSelf._direct_peers,
          "this is the check that stops a peer steering activations at another machine")

    print("\n-- no offer at all is simply the relay, which is the address of record")
    _FakeSelf._direct_peers.clear()
    relay_srv, relay_port, relay_seen, rt = listener("127.0.0.1", {"ok": True, "wire": "f32"})
    try:
        conn, ack = dial({"host_b": "127.0.0.1", "port_b": relay_port, "s1": 10, "s2": 19})
        conn.close()
    finally:
        rt.join(timeout=5)
        relay_srv.close()
    check("a peer that offers nothing is used over the relay", not _FakeSelf._direct_peers)
    check("...and it was still asked", relay_seen.get("config", {}).get("lan_hint") == hint)

    print("\n-- an offer we cannot reach keeps the relay connection we already have")
    _FakeSelf._direct_peers.clear()
    dead = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    dead.bind((lan_ip, 0))
    dead_port = dead.getsockname()[1]
    dead.close()                                   # nothing is listening there now
    relay_srv, relay_port, relay_seen, rt = listener(
        "127.0.0.1", {"ok": True, "wire": "f32", "direct": {"ip": lan_ip, "port": dead_port}})
    try:
        conn, ack = dial({"host_b": "127.0.0.1", "port_b": relay_port, "s1": 10, "s2": 19})
        alive = conn.fileno() != -1
        conn.close()
        check("a refused direct dial does not fail the request", alive)
        check("...and nothing is cached for a peer we could not reach",
              not _FakeSelf._direct_peers, str(_FakeSelf._direct_peers))
        # The offer arrives on EVERY request — the peer cannot know we failed to reach it — so
        # without a memory of the failure this dials and times out every single time. Measured
        # live at +1.5 s on time-to-first-token against a peer whose firewall scopes its port
        # to another interface, which made the optimisation a straight loss.
        check("the failure is REMEMBERED, so the next request goes straight to the relay",
              lan_direct.recently_unreachable(("127.0.0.1", relay_port)),
              "an offer that never works must not be retried at DIRECT_TIMEOUT_S a time")
    finally:
        rt.join(timeout=5)
        relay_srv.close()

    print("\n-- ...but not remembered forever, because firewalls get fixed")
    key = ("127.0.0.1", relay_port)
    lan_direct._unreachable[key] = lan_direct.time.monotonic() - lan_direct.DIRECT_RETRY_S - 1
    check("a cooled-off failure is worth one more try",
          not lan_direct.recently_unreachable(key))
    lan_direct._unreachable.clear()

    print("\n-- a [P52]-sealed hop is never re-dialled")
    src = open(os.path.join(HERE, "node_server.py"), encoding="utf-8").read()
    i = src.find("def _dial_next_hop")
    # To the end of the FUNCTION, not a fixed byte count. A window of "the next 5000 chars"
    # silently stopped covering the line it was checking the moment the comments above it grew,
    # and a source-text check that has slid off its target reports PASS for the wrong reason.
    body = src[i:src.index("\n    def ", i + 1)]
    check("the direct dial is gated on there being no grant",
          "not grant and lan_direct.usable(" in body,
          "a grant is single-use — presenting it twice is indistinguishable from a replay, "
          "and the receiving node is right to refuse it")
    check("...and why is written down where the next reader will look",
          "_grants_seen" in body and "single-use" in body)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
