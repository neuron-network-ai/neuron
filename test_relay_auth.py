"""test_relay_auth.py — HMAC ticket that lets the relay verify a node's tunnel registration
without a DB or a callback to the coordinator (post-launch-audit fix). Also covers
recv_json's hardening against garbage length-prefixes, found live on the deployed relay
(the open internet was already sending it malformed data that crashed handler threads with
MemoryError). Run: python test_relay_auth.py
"""
import socket
import struct

import relay
import relay_auth

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    secret = "test-secret"
    t = relay_auth.make_ticket(secret, "node-a", 9001)

    check("a correctly-minted ticket verifies", relay_auth.verify_ticket(secret, "node-a", 9001, t))
    check("wrong node_id is rejected", not relay_auth.verify_ticket(secret, "node-b", 9001, t))
    check("wrong port is rejected (can't replay onto another port)",
          not relay_auth.verify_ticket(secret, "node-a", 9002, t))
    check("wrong secret is rejected", not relay_auth.verify_ticket("other-secret", "node-a", 9001, t))
    check("missing ticket is rejected", not relay_auth.verify_ticket(secret, "node-a", 9001, None))
    check("empty-string ticket is rejected", not relay_auth.verify_ticket(secret, "node-a", 9001, ""))
    check("tampered ticket (one char flipped) is rejected",
          not relay_auth.verify_ticket(secret, "node-a", 9001, t[:-1] + ("0" if t[-1] != "0" else "1")))
    check("same node_id+port+secret is deterministic (relay recomputes independently)",
          relay_auth.make_ticket(secret, "node-a", 9001) == t)
    check("different port for the same node -> different ticket (no cross-port reuse)",
          relay_auth.make_ticket(secret, "node-a", 9002) != t)

    # -- recv_json hardening: a garbage/huge length prefix must not crash the handler -- #
    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", 0xFFFFFFFF))   # ~4GB claimed length -> used to MemoryError
        check("huge bogus length -> None, not a crash", relay.recv_json(b) is None)
    finally:
        a.close(); b.close()

    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", 5) + b"\x00\x01\xff\xfe\xfd")  # undecodable bytes
        check("undecodable payload -> None, not a crash", relay.recv_json(b) is None)
    finally:
        a.close(); b.close()

    a, b = socket.socketpair()
    try:
        a.sendall(struct.pack("!I", 8) + b"not-json")   # valid length, invalid JSON
        check("malformed JSON payload -> None, not a crash", relay.recv_json(b) is None)
    finally:
        a.close(); b.close()

    a, b = socket.socketpair()
    try:
        relay.send_json(a, {"node_id": "x", "public_port": 9001})
        check("a real message still round-trips", relay.recv_json(b) == {"node_id": "x", "public_port": 9001})
    finally:
        a.close(); b.close()

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
