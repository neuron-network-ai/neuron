"""agent/test_rpc_bridge_endtoend.py — run: python -m agent.test_rpc_bridge_endtoend

[P30] phase 3, both halves, spliced together over a real [P52] channel.

`node_server._serve_rpc_bridge` hands an authenticated connection to this machine's ggml
engine. `rpc_bridge_client.RemoteEngine` is the far end: a loopback port llama.cpp can be
pointed at, which carries every byte into that channel. Neither half means anything alone, and
the failure mode of getting it wrong is not a crash — it is a memory protocol reachable by
somebody who should not reach it.

So the tests here are mostly about REFUSAL:

  * a plaintext caller is refused, and told why. ggml-rpc has no authentication whatsoever, so
    the encrypted channel is the only thing standing between a stranger and a write primitive
    on a volunteer's PC. Upstream's own words: "never run the RPC server on an open network".
  * the refusal happens BEFORE any engine is started, so a hostile dial cannot even make the
    node spawn a process.
  * bytes survive the round trip intact in both directions, because a bridge that corrupts one
    byte in a tensor upload produces wrong answers rather than an error, and proof-of-compute
    is what would have to catch that.

A fake engine stands in for `ggml-rpc-server` so this runs on any machine, with no binary and
no model. What is under test is the TRANSPORT — the real binary is exercised by
`test_rpc_engine.py`'s live cases.
"""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common                                                    # noqa: E402
from agent import rpc_bridge_client                              # noqa: E402
from security import wire_crypto as W                            # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def echo_server():
    """Stands in for ggml-rpc-server: echoes every byte, uppercased so direction is provable."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            def handle(c=c):
                try:
                    while True:
                        d = c.recv(65536)
                        if not d:
                            break
                        c.sendall(d.upper())
                except OSError:
                    pass
                finally:
                    try:
                        c.close()
                    except OSError:
                        pass
            threading.Thread(target=handle, daemon=True).start()
    threading.Thread(target=loop, daemon=True).start()
    return srv, srv.getsockname()[1]


def secure_pair(node_id="node-x", token="tok-secret"):
    """A connected, mutually-encrypted socket pair, exactly as a real hop produces."""
    grant = W.mint_grant(token, "req-1", node_id)
    a, b = socket.socketpair()
    out = {}

    def server():
        try:
            # magic_consumed=False: unlike node_server, nothing here peeked the first four
            # bytes to decide the protocol, so the handshake still has to eat MAGIC itself.
            ch, _ = W.server_handshake(b, token, node_id, seen=set(), magic_consumed=False)
            common.attach_channel(b, ch)
            out["server"] = b
        except Exception as e:              # pragma: no cover - surfaces as a test failure
            out["error"] = e

    t = threading.Thread(target=server, daemon=True)
    t.start()
    ch = W.client_handshake(a, grant, node_id)
    common.attach_channel(a, ch)
    t.join(10)
    if "error" in out:
        raise out["error"]
    return a, b


def main():
    from agent import rpc_engine, node_server

    srv, engine_port = echo_server()
    print(f"\n-- a fake ggml engine on 127.0.0.1:{engine_port}")

    print("\n-- refusal comes FIRST, before any engine is started")
    started = []
    real_engine_cls = rpc_engine.RpcEngine

    class SpyEngine(real_engine_cls):
        def start(self):
            started.append(1)
            self.port = engine_port
            return True

        def healthy(self):
            return bool(started)

    rpc_engine.RpcEngine = SpyEngine
    try:
        ns = node_server.NodeServer.__new__(node_server.NodeServer)
        ns._rpc = None
        ns._rpc_lock = threading.Lock()

        plain_a, plain_b = socket.socketpair()
        t = threading.Thread(target=ns._serve_rpc_bridge, args=(plain_b,), daemon=True)
        t.start()
        reply = common.recv_msg(plain_a)
        t.join(5)
        check("a plaintext caller is refused", reply.get("type") == "rpc-error", str(reply))
        check("...and told why", "encrypted" in reply.get("detail", ""), str(reply))
        check("...and NO engine process was started for it", not started,
              "a hostile dial must not be able to make a node spawn anything")
        plain_a.close()

        print("\n-- an authenticated caller reaches the engine, both directions")
        a, b = secure_pair()
        ns2 = node_server.NodeServer.__new__(node_server.NodeServer)
        ns2._rpc = None
        ns2._rpc_lock = threading.Lock()
        threading.Thread(target=ns2._serve_rpc_bridge, args=(b,), daemon=True).start()
        ack = common.recv_msg(a)
        check("the node acks before switching protocols", ack.get("type") == "rpc-ready",
              str(ack))
        check("...and only then starts the engine", bool(started))

        remote = rpc_bridge_client.RemoteEngine(open_channel=lambda: a)
        endpoint = remote.start()
        check("the driver exposes a LOOPBACK endpoint for llama.cpp",
              endpoint.startswith("127.0.0.1:"), endpoint)

        client = socket.create_connection(("127.0.0.1", remote.port), timeout=10)
        payload = b"ggml-rpc frame \x00\x01\x02 and some tensor bytes" * 40
        client.sendall(payload)
        got = b""
        deadline = time.time() + 20
        while len(got) < len(payload) and time.time() < deadline:
            chunk = client.recv(65536)
            if not chunk:
                break
            got += chunk
        check("every byte survived the round trip through the encrypted channel",
              got == payload.upper(),
              f"sent {len(payload)}B, got {len(got)}B")
        client.close()
        remote.stop()

        print("\n-- a channel that cannot be opened fails the dial instead of hanging")
        def boom():
            raise ConnectionRefusedError("node is gone")
        dead = rpc_bridge_client.RemoteEngine(open_channel=boom)
        dead.start()
        c2 = socket.create_connection(("127.0.0.1", dead.port), timeout=10)
        c2.settimeout(10)
        check("the local socket is closed rather than left hanging", c2.recv(16) == b"",
              "a hang here would look exactly like a slow machine")
        c2.close()
        dead.stop()
    finally:
        rpc_engine.RpcEngine = real_engine_cls
        try:
            srv.close()
        except OSError:
            pass

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
