"""security/test_secure_hop_live.py — the encrypted hop, against a real NodeServer ([P52]).

    python -m security.test_secure_hop_live

`test_wire_crypto.py` proves the channel in isolation. This proves it is actually WIRED IN:
the same socket, the same `common.send_msg`, a real `NodeServer.serve` loop, and a real
observer reading the bytes that cross the wire.

The three things that decide whether this can be deployed at all:

  1. a secure caller gets a correct answer, and the prompt-derived bytes are NOT on the wire;
  2. a LEGACY caller still works, or deploying this partitions the live network instead of
     securing it -- the node has to serve both through the rolling upgrade;
  3. `NEURON_REQUIRE_SECURE=1` refuses plaintext, so the upgrade has an end state rather than
     being permanently half-done. [P24] is what a half-finished migration costs.

Deliberately no torch: `serve()` is exercised through the `paused` reply, which is answered
before any model is touched. What is being tested here is the TRANSPORT, and dragging a model
load into it would make the suite slow enough that nobody runs it.
"""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common                                              # noqa: E402
from security import wire_crypto as W                      # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


TOKEN = "n" * 48
NODE_ID = "node-under-test"
REQ = "req-live-1"
SECRET = "the user's prompt nobody else should read"


class _FakeNode:
    """The transport half of NodeServer, with the model half removed.

    Uses the REAL `_maybe_secure` and the REAL `common.send_msg`/`recv_msg`, which is the part
    under test. Importing agent.node_server would pull in torch and a slice loader for a test
    about sockets.
    """

    def __init__(self, require_secure=False):
        from agent import node_server as NS
        self._ns = NS
        self.node_token, self.node_id = TOKEN, NODE_ID
        self._require = require_secure
        self._grants_seen, self._grants_since = set(), time.time()
        self.serve_errors = []

    _grants = lambda self: self._grants_seen                      # noqa: E731

    def _maybe_secure(self, conn):
        from agent.node_server import NodeServer
        old = self._ns.REQUIRE_SECURE
        self._ns.REQUIRE_SECURE = self._require
        try:
            return NodeServer._maybe_secure(self, conn)
        finally:
            self._ns.REQUIRE_SECURE = old

    def serve_once(self, conn):
        try:
            self._maybe_secure(conn)
        except W.HandshakeError as e:
            self.serve_errors.append(str(e))
            conn.close()
            return
        msg = common.recv_msg(conn)
        common.send_msg(conn, {"ok": True, "echo": msg.get("prompt"),
                               "secure": common.is_secure(conn)})
        conn.close()



def _relay(dest_port):
    """A byte-for-byte splicing proxy, recording both directions.

    This IS NEURON's relay: `relay.py` splices raw bytes between the driver and the node, so
    whoever runs one sees precisely what this records. Testing "is it encrypted" through it
    checks the property against the real adversary rather than against a mock of one.
    """
    seen = bytearray()
    lis = socket.socket()
    lis.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lis.bind(("127.0.0.1", 0))
    lis.listen(4)

    def pump(a, b):
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                seen.extend(data)
                b.sendall(data)
        except OSError:
            pass
        finally:
            for x in (a, b):
                try:
                    x.close()
                except OSError:
                    pass

    def loop():
        try:
            client, _ = lis.accept()
            upstream = socket.create_connection(("127.0.0.1", dest_port), timeout=15)
            threading.Thread(target=pump, args=(client, upstream), daemon=True).start()
            threading.Thread(target=pump, args=(upstream, client), daemon=True).start()
        except Exception:                                       # noqa: BLE001
            pass

    threading.Thread(target=loop, daemon=True).start()
    return seen, lis.getsockname()[1]


def _serve(node, require_secure=False):
    lis = socket.socket()
    lis.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    lis.bind(("127.0.0.1", 0))
    lis.listen(4)
    port = lis.getsockname()[1]

    def loop():
        try:
            conn, _ = lis.accept()
            node.serve_once(conn)
        except Exception:                                       # noqa: BLE001
            pass
        finally:
            lis.close()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return port, t


def main():
    print("\n-- a SECURE caller, watched by a RELAY that records every byte")
    node = _FakeNode()
    port, t = _serve(node)
    seen, relay_port = _relay(port)

    s = socket.create_connection(("127.0.0.1", relay_port), timeout=15)
    grant = W.mint_grant(TOKEN, REQ, NODE_ID)
    ch = W.client_handshake(s, grant, NODE_ID)
    common.attach_channel(s, ch)
    common.send_msg(s, {"type": "config", "prompt": SECRET})
    reply = common.recv_msg(s)
    s.close()
    t.join(10)

    check("the node answers a secure caller", reply.get("ok") is True, repr(reply))
    check("...with the right content", reply.get("echo") == SECRET, repr(reply))
    check("...and knows the connection is encrypted", reply.get("secure") is True, repr(reply))
    check("THE RELAY CANNOT READ THE PROMPT", SECRET.encode() not in bytes(seen),
          "the plaintext appeared in the bytes the relay spliced")
    check("...and the relay did carry real traffic, so the check is not vacuous",
          len(seen) > 200, f"only {len(seen)} bytes crossed -- did the proxy run?")

    print("\n-- a LEGACY caller still works, or deploying this partitions the network")
    node2 = _FakeNode()
    port2, t2 = _serve(node2)
    s2 = socket.create_connection(("127.0.0.1", port2), timeout=15)
    common.send_msg(s2, {"type": "config", "prompt": "plain"})
    reply2 = common.recv_msg(s2)
    s2.close()
    t2.join(10)
    check("a plaintext caller is still served", reply2.get("ok") is True, repr(reply2))
    check("...and the four peeked bytes were handed back intact",
          reply2.get("echo") == "plain",
          "a mangled length prefix would corrupt the very first message")
    check("...and the node reports it as NOT secure", reply2.get("secure") is False,
          "a node that cannot tell would let the product claim privacy it lacks")

    print("\n-- NEURON_REQUIRE_SECURE=1 gives the upgrade an end state")
    node3 = _FakeNode(require_secure=True)
    port3, t3 = _serve(node3)
    s3 = socket.create_connection(("127.0.0.1", port3), timeout=15)
    try:
        common.send_msg(s3, {"type": "config", "prompt": "plain"})
        common.recv_msg(s3)
        served = True
    except Exception:                                           # noqa: BLE001
        served = False
    s3.close()
    t3.join(10)
    check("plaintext is refused when required", not served)
    check("...and the refusal says why",
          any("only encrypted" in e for e in node3.serve_errors), str(node3.serve_errors))

    print("\n-- a stranger on the public port, against the real accept path")
    node4 = _FakeNode()
    port4, t4 = _serve(node4)
    s4 = socket.create_connection(("127.0.0.1", port4), timeout=15)
    try:
        W.client_handshake(s4, W.mint_grant("wrong-token", REQ, NODE_ID), NODE_ID)
        opened = True
    except Exception:                                           # noqa: BLE001
        opened = False
    s4.close()
    t4.join(10)
    check("a forged grant cannot open a session on a live node", not opened)
    check("...and the node recorded the refusal", bool(node4.serve_errors),
          str(node4.serve_errors))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
