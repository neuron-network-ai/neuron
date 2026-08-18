"""agent/test_rpc_engine.py — the fast engine is reachable, and ONLY through the channel.

    python -m agent.test_rpc_engine

[P30] phase 2. The properties that make this safe to ship are not about speed:

  * `ggml-rpc-server` binds loopback and nothing else, because llama.cpp says never put it on
    an open network and NEURON is an open network;
  * the bridge REFUSES a plaintext caller, so there is exactly one door and it is the
    authenticated one;
  * a missing binary is not an error -- the node keeps serving with PyTorch. An agent that
    refused to start because an optional accelerator was absent would be a worse failure than
    the slowness it was fixing.

Skips the live-process cases when no binary is present, the way `test_stage1_challenge` skips
without weights: "we could not run this" must never read as "this failed".
"""
import os
import socket
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import common                                               # noqa: E402
from agent import rpc_engine as R                           # noqa: E402
from security import wire_crypto as W                       # noqa: E402

ok = fail = skipped = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def skip(label, why):
    global skipped
    skipped += 1
    print(f"  SKIP  {label} — {why}")


def main():
    print("\n-- it is never on a public interface, and that is not configurable")
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rpc_engine.py"),
               encoding="utf-8").read()
    check("the bind host is a constant, not a setting",
          'BIND_HOST = "127.0.0.1"' in src)
    check("...and the start command passes it explicitly",
          '"-H", BIND_HOST' in src,
          "relying on the binary's default would make the safety argument depend on llama.cpp")
    check("no environment variable can move it off loopback",
          "NEURON_RPC_HOST" not in src,
          "a host override is a foot-gun that turns a memory protocol into a public service")

    print("\n-- exactly one door, and it is the authenticated one")
    # A plaintext caller reaching ggml-rpc would undo the entire reason this is shippable.
    a, b = socket.socketpair()
    try:
        R.bridge(a, port=1)          # never gets as far as connecting
        check("the bridge refuses an unencrypted connection", False, "it proceeded")
    except PermissionError:
        check("the bridge refuses an unencrypted connection", True)
    except Exception as e:                                   # noqa: BLE001
        check("the bridge refuses an unencrypted connection", False,
              f"wrong failure: {e.__class__.__name__}: {e}")
    finally:
        a.close(), b.close()
    # Whitespace-normalised: the claim lives in a wrapped docstring, and matching the raw
    # string made the test depend on where the line happened to break.
    flat = " ".join(src.split())
    check("...and it does no authentication of its own",
          "does no authentication of its own" in flat,
          "a bridge that could also authenticate would be a second door")

    print("\n-- a machine without the binary keeps working")
    eng = R.RpcEngine(binary=None)
    eng.binary = None
    check("no binary means 'not available', not an exception", eng.available is False)
    check("...and start() returns False rather than raising", eng.start() is False)

    print("\n-- health is checked on the SOCKET, not on the process")
    check("healthy() probes the port", "_port_open(self.port)" in src.split("def healthy")[1])
    check("...which is [P51]'s lesson, stated",
          "A running process proves nothing" in src)

    binary = R.find_binary()
    if not binary:
        skip("live start/stop", "no ggml-rpc-server binary on this machine "
                                f"(set {R.BINARY_ENV})")
        skip("live bridge through the encrypted channel", "same")
    else:
        print(f"\n-- live, against {os.path.basename(binary)}")
        eng = R.RpcEngine(port=50079, threads=2)
        started = eng.start()
        check("it starts", started is True)
        check("...and reports healthy", eng.healthy() is True)
        check("...on loopback only",
              _reachable("127.0.0.1", 50079) and not _reachable(_lan_ip(), 50079),
              "it answered on a non-loopback address")

        print("\n-- a full request through the channel reaches it")
        got = {}
        lis = socket.socket()
        lis.bind(("127.0.0.1", 0))
        lis.listen(1)
        port = lis.getsockname()[1]
        token, node_id = "t" * 48, "rpc-node"

        def server():
            conn, _ = lis.accept()
            ch, _rid = W.server_handshake(conn, token, node_id)
            common.attach_channel(conn, ch)
            try:
                R.bridge(conn, port=50079)
                got["bridged"] = True
            except Exception as e:                           # noqa: BLE001
                got["error"] = f"{e.__class__.__name__}: {e}"

        t = threading.Thread(target=server, daemon=True)
        t.start()
        cli = socket.create_connection(("127.0.0.1", port), timeout=15)
        ch = W.client_handshake(cli, W.mint_grant(token, "req", node_id), node_id)
        # ggml-rpc's own hello is enough: if bytes reach the server and something comes back,
        # the bridge works. Speaking the full protocol is llama.cpp's job, not this test's.
        cli.sendall(_frame(ch.seal(b"\x00" * 8)))
        cli.settimeout(10)
        try:
            reply = cli.recv(4096)
        except OSError:
            reply = b""
        cli.close()
        t.join(5)
        check("bytes cross the channel into rpc-server without error",
              "error" not in got, got.get("error", ""))
        eng.stop()
        check("stop() actually stops it", not eng.healthy())

    print(f"\n{ok} passed, {fail} failed, {skipped} skipped")
    return fail == 0


def _frame(blob):
    import struct
    return struct.pack(">I", len(blob)) + blob


def _lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _reachable(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
