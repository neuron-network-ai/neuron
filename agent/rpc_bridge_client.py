"""agent/rpc_bridge_client.py — the DRIVER half of [P30] phase 3.

`node_server` will hand an authenticated connection to its local ggml engine (`rpc-bridge`).
This is the other end: a loopback listener that llama.cpp can be pointed at with `--rpc`, which
forwards every byte into a [P52] channel to that node.

**Why a local listener and not "just connect llama.cpp to the node".** Two reasons, and both
are the point of the design:

  1. **llama.cpp must never speak to a remote host itself.** ggml-rpc is a memory protocol with
     no authentication of any kind; upstream says never to put it on an open network. Pointing
     it at a public port would re-open [P19] deliberately. It only ever dials `127.0.0.1`.
  2. **The grant is per-request and NEURON's to mint.** The coordinator seals a grant to the
     target node's token; only this side can present it. llama.cpp knows nothing about any of
     that and should not have to.

So the shape is: `llama-server --rpc 127.0.0.1:PORT` -> this listener -> NEURON's encrypted
channel -> the node's `rpc-bridge` -> `127.0.0.1:50077` on that machine. Nothing is exposed at
either end.

**One connection per dial, deliberately.** ggml's client opens more than one socket to a device
and multiplexing them onto a single channel would mean inventing a stream id and a framing
layer NEURON does not have. Each dial opens its own channel, which is one grant use each and
costs a handshake — measured cheap next to a tensor upload, and far cheaper than a bug in a
hand-rolled multiplexer.
"""
import logging
import socket
import struct
import threading

import common

log = logging.getLogger("neuron.rpc.bridge")

BIND_HOST = "127.0.0.1"
CHUNK = 65536


def _frame(blob):
    return struct.pack(">I", len(blob)) + blob


def _read_frame(sock):
    hdr = b""
    while len(hdr) < 4:
        b = sock.recv(4 - len(hdr))
        if not b:
            return None
        hdr += b
    n = struct.unpack(">I", hdr)[0]
    buf = b""
    while len(buf) < n:
        b = sock.recv(min(CHUNK, n - len(buf)))
        if not b:
            return None
        buf += b
    return buf


def _close(*socks):
    for s in socks:
        try:
            s.close()
        except OSError:
            pass


def splice(local_sock, secure_conn):
    """Carry bytes between a local ggml client socket and an open NEURON channel."""
    ch = common.channel_for(secure_conn)

    def up():
        try:
            while True:
                data = local_sock.recv(CHUNK)
                if not data:
                    break
                secure_conn.sendall(_frame(ch.seal(data)))
        except OSError:
            pass
        finally:
            _close(local_sock, secure_conn)

    t = threading.Thread(target=up, daemon=True)
    t.start()
    try:
        while True:
            blob = _read_frame(secure_conn)
            if blob is None:
                break
            local_sock.sendall(ch.open(blob))
    except OSError:
        pass
    finally:
        _close(local_sock, secure_conn)
    t.join(5)


class RemoteEngine:
    """A loopback address that is really a remote node's ggml engine.

    `open_channel` is injected rather than imported so this can be tested without a coordinator,
    a grant, or a second machine. It must return a socket with a [P52] channel already attached
    and the node's `rpc-ready` ack already consumed.
    """

    def __init__(self, open_channel, host=BIND_HOST):
        self._open_channel = open_channel
        self.host = host
        self.port = None
        self._srv = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def endpoint(self):
        return f"{self.host}:{self.port}" if self.port else None

    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, 0))
        self._srv.listen(8)
        self.port = self._srv.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        log.info("ggml device bridged at %s", self.endpoint)
        return self.endpoint

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                sock, _ = self._srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve_one, args=(sock,), daemon=True).start()

    def _serve_one(self, sock):
        try:
            secure = self._open_channel()
        except Exception as e:
            # The node refused, is gone, or has no engine. Close the local socket so llama.cpp
            # sees a dead device and reports it, rather than hanging on a connection that will
            # never answer — a silent hang here would look exactly like a slow machine.
            log.warning("could not open a channel to the remote engine: %s", e)
            _close(sock)
            return
        splice(sock, secure)

    def stop(self):
        self._stop.set()
        _close(self._srv)
