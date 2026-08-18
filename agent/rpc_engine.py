"""agent/rpc_engine.py — expose this machine's compute as a ggml backend, safely ([P30] phase 2).

**What this replaces, and why it is a bridge rather than a rewrite.** `node_server` computes a
layer range with PyTorch fp32, measured at 3.09 tok/s against llama.cpp's 26.93 on the same
machine ([P30]). llama.cpp already knows how to split a model across machines -- `ggml-rpc-server`
plus `--rpc` on the driver -- and the released Windows binaries ship it, so nothing here has to
be built or written.

**The reason it was not simply adopted.** llama.cpp's own `tools/rpc/README.md`:

    "the functionality is fragile and insecure. Never run the RPC server on an open network or
     in a sensitive environment!"

NEURON is an open network of strangers' machines by definition, and ggml-rpc is a *memory*
protocol -- allocate a buffer, write bytes into it, execute a graph. Publishing that on a relay
port would be [P19] reintroduced deliberately, with a worse payload.

**So it is never published.** `ggml-rpc-server` binds `127.0.0.1` and nothing else can reach it.
The only path in is `node_server`'s authenticated, encrypted channel ([P52]): a caller must
present a grant sealed to THIS node's token, which only the coordinator can mint, and every byte
afterwards is AES-GCM under a key derived from an ephemeral exchange. Measured end to end at
**28.07 tok/s** through that channel, against 32.11 plaintext and a 34.97 no-RPC control.

**The residual risk, stated because it does not disappear.** The channel authenticates the
CALLER. It does not make ggml-rpc safe against an authenticated PEER -- a chain member is
coordinator-selected, not trusted. The exposed surface shrinks from "anyone on the internet" to
"a machine the coordinator placed in this chain", which is a large reduction and not an
elimination. Whether that is acceptable is a decision about who may join a chain.

**And the constraint this architecture adds, which today's does not have.** In ggml's RPC design
the CLIENT reads the model file and uploads tensors to each device; the server holds no model of
its own (`-c` caches what it has been sent). With mmap the driver does not need the model in
RAM, but it does need the whole file on DISK -- where today it downloads only its own shard.
That is a real change to what a driver must provide and it is not yet decided.
"""
from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import threading
import time

log = logging.getLogger("neuron.rpc")

# Bound to loopback, always. Not a default to be overridden -- the whole safety argument above
# rests on this process being unreachable except through the authenticated channel, so the host
# is not a parameter at all.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = int(os.environ.get("NEURON_RPC_PORT", "50077"))
START_TIMEOUT_S = 30.0

# Where the binary lives. Packaged beside the agent; overridable for a source checkout.
BINARY_ENV = "NEURON_RPC_SERVER"
BINARY_NAME = "ggml-rpc-server.exe" if os.name == "nt" else "rpc-server"


def find_binary():
    """The ggml-rpc-server executable, or None.

    None is a legitimate answer, not an error: a node without it simply keeps serving with
    PyTorch. An agent that refused to start because an optional accelerator was missing would
    be a worse failure than the slowness it was trying to fix.
    """
    explicit = os.environ.get(BINARY_ENV)
    if explicit and os.path.exists(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, BINARY_NAME),
                 os.path.join(os.path.dirname(here), "bin", BINARY_NAME)):
        if os.path.exists(cand):
            return cand
    return shutil.which(BINARY_NAME)


def _port_open(port, host=BIND_HOST, timeout=0.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class RpcEngine:
    """Supervises a local ggml-rpc-server.

    Deliberately a supervisor rather than a library binding: the process is llama.cpp's, its
    lifetime is ours, and the boundary between them is a socket on loopback. That keeps the
    C++ out of this process entirely -- a crash in ggml takes down a child, not the agent that
    is holding this node's identity and its earnings.
    """

    def __init__(self, port=DEFAULT_PORT, threads=None, cache=True, binary=None):
        self.port = int(port)
        self.threads = threads or max(1, (os.cpu_count() or 4) - 1)
        self.cache = cache
        self.binary = binary or find_binary()
        self.proc = None
        self._lock = threading.Lock()

    @property
    def available(self):
        return bool(self.binary)

    def start(self):
        """Start it if it is not already up. Returns True if the engine is usable.

        Idempotent, and tolerant of an already-running instance on the port: an agent that
        restarts should attach rather than fight for the port, which is the failure [P24] is
        full of.
        """
        with self._lock:
            if not self.binary:
                log.info("no ggml-rpc-server binary found — this node keeps serving with "
                         "PyTorch. Set %s to point at one.", BINARY_ENV)
                return False
            if self.proc and self.proc.poll() is None:
                return True
            if _port_open(self.port):
                log.info("ggml-rpc-server already listening on %s:%d — attaching rather than "
                         "starting a second one", BIND_HOST, self.port)
                return True
            cmd = [self.binary, "-H", BIND_HOST, "-p", str(self.port), "-t", str(self.threads)]
            if self.cache:
                cmd.append("-c")     # keep uploaded tensors, so a reconnect does not re-ship them
            log.info("starting ggml-rpc-server on %s:%d with %d threads (loopback only)",
                     BIND_HOST, self.port, self.threads)
            try:
                self.proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(self.binary) or None)
            except OSError as e:
                log.warning("could not start ggml-rpc-server: %s — continuing with PyTorch", e)
                self.proc = None
                return False
            threading.Thread(target=self._drain, daemon=True).start()
            deadline = time.time() + START_TIMEOUT_S
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    log.warning("ggml-rpc-server exited immediately (code %s) — continuing "
                                "with PyTorch", self.proc.returncode)
                    return False
                if _port_open(self.port):
                    log.info("ggml-rpc-server ready on %s:%d", BIND_HOST, self.port)
                    return True
                time.sleep(0.25)
            log.warning("ggml-rpc-server did not open %s:%d within %.0fs — continuing with "
                        "PyTorch", BIND_HOST, self.port, START_TIMEOUT_S)
            self.stop()
            return False

    def _drain(self):
        """Forward the child's output into our log.

        Not cosmetic: without it the child's stdout pipe fills and the process blocks, which
        looks exactly like a hang. And a node whose accelerator died silently is [P51] again.
        """
        try:
            for line in iter(self.proc.stdout.readline, b""):
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    log.debug("rpc-server: %s", text)
        except Exception:                                    # noqa: BLE001
            pass

    def healthy(self):
        """Is it actually accepting connections? Checked on the socket, not on the process.

        A running process proves nothing -- [P51] is 81 minutes of exactly that -- so the test
        is whether the thing it exists to provide actually answers.
        """
        if self.proc is not None and self.proc.poll() is not None:
            return False
        return _port_open(self.port)

    def stop(self):
        with self._lock:
            p, self.proc = self.proc, None
            if p is None or p.poll() is not None:
                return
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()


def bridge(secure_conn, port=DEFAULT_PORT, chunk=65536):
    """Splice an already-authenticated NEURON connection to the local rpc-server.

    `secure_conn` must ALREADY have a [P52] channel attached: this function does no
    authentication of its own, deliberately, so there is exactly one place that decides who may
    reach ggml-rpc and it is the same place that decides everything else. A bridge that could
    also authenticate would be a second door.

    Returns when either side closes.
    """
    import common
    if not common.is_secure(secure_conn):
        # Refusing is the whole point. Bridging a plaintext caller into a memory protocol is
        # the failure this module's docstring exists to prevent.
        raise PermissionError("refusing to bridge an unencrypted connection to ggml-rpc")
    up = socket.create_connection((BIND_HOST, port), timeout=30)
    ch = common.channel_for(secure_conn)

    def down():
        try:
            while True:
                data = up.recv(chunk)
                if not data:
                    break
                secure_conn.sendall(_frame(ch.seal(data)))
        except OSError:
            pass
        finally:
            _close(up, secure_conn)

    t = threading.Thread(target=down, daemon=True)
    t.start()
    try:
        while True:
            blob = _read_frame(secure_conn)
            if blob is None:
                break
            up.sendall(ch.open(blob))
    except OSError:
        pass
    finally:
        _close(up, secure_conn)
    t.join(5)


def _frame(blob):
    import struct
    return struct.pack(">I", len(blob)) + blob


def _read_frame(sock):
    import struct
    head = b""
    while len(head) < 4:
        b = sock.recv(4 - len(head))
        if not b:
            return None
        head += b
    (n,) = struct.unpack(">I", head)
    body = b""
    while len(body) < n:
        b = sock.recv(n - len(body))
        if not b:
            return None
        body += b
    return body


def _close(*socks):
    for s in socks:
        try:
            s.close()
        except OSError:
            pass
