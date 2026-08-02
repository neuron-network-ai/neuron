"""
agent/node_server.py — one generalized NEURON node server for ANY layer range.

Loads this node's downloaded SLICE (not the full model) and serves whichever role
the incoming config implies, staying compatible with node_a.py's existing wire
protocol so it drops straight into the chain:
  - MIDDLE relay  (config carries host_b/port_b): run my layers, forward the hidden
    to the next hop, relay its result back  (the node_c role)
  - LAST stage    (config carries s2/n only):     run my layers + final norm, return
    the normed hidden  (the node_b role)

Reuses common.py (first/mid/last_stage, KV cache, TCP framing) and the slice
loader — does NOT modify any existing node script. A per-machine compute lock
serialises this node's own math (pipelining across concurrent requests).

Usage (normally launched by agent.py):
  python node_server.py --slice-dir ./model_slice --layer-start 10 --layer-end 18 --port 50999
"""
import argparse
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import batching                                  # noqa: E402
import common                                    # noqa: E402
import wire_codec                                # noqa: E402
from slice_downloader import load_slice_model    # noqa: E402

# Held only for swapping the model pointer during a migration reload. It used to wrap every
# forward pass too, which meant a machine served exactly ONE request at a time no matter how
# many cores it had -- the single biggest reason a volunteer node was worth so much less than
# its hardware suggested (TOKENOMICS.md §12.5). Concurrent compute is now handled by
# batching.MicroBatcher, which serves a whole batch of requests in one forward pass instead
# of serialising them.
compute_lock = threading.Lock()

# Is this machine in the middle of serving somebody? Used by the auto-updater, which must never
# replace the app underneath a request in flight -- a dropped hop shows up to the driver as
# "socket closed mid-message" and the whole inference fails, for every user on that chain.
#
# compute_lock is NOT the signal: it only guards a slice reload now (the serving path moved to
# batching.MicroBatcher), so it is almost always free even mid-request. Counting live
# connections is what actually reflects work, and the timestamp covers the gaps between a
# chain's per-token round trips, when the connection is open but momentarily idle.
_serving_lock = threading.Lock()
_serving_conns = 0
_last_activity = 0.0


def _serving_enter():
    global _serving_conns, _last_activity
    with _serving_lock:
        _serving_conns += 1
        _last_activity = time.time()


def _serving_exit():
    global _serving_conns, _last_activity
    with _serving_lock:
        _serving_conns = max(0, _serving_conns - 1)
        _last_activity = time.time()


def is_busy(idle_seconds=120):
    """True if a connection is open, or one closed less than `idle_seconds` ago.

    The grace period is deliberate: a driver holds a chain across many token round trips and
    may reconnect between them, so "no open socket right now" does not mean "nobody is using
    this node". Erring towards busy only delays an update by a couple of minutes; erring the
    other way breaks somebody's answer.
    """
    with _serving_lock:
        if _serving_conns > 0:
            return True
        return (time.time() - _last_activity) < idle_seconds


class NodeServer:
    def __init__(self, slice_dir, layer_start, layer_end, total_layers, paused_flag=None):
        self.lo = self.hi = self.n = None
        self.model = None
        self._batchers = {}
        self._batcher_lock = threading.Lock()
        self.paused = paused_flag if paused_flag is not None else threading.Event()  # set = paused
        # Whether this server is actually ACCEPTING connections. agent.py runs run() in a
        # daemon thread, so a failed bind used to kill that thread silently while the agent
        # kept heartbeating "active" forever -- the coordinator then advertised a healthy
        # 28/28 network and routed real requests into a node refusing every connection
        # ([P21], observed live on the Pavilion). Nothing may advertise availability without
        # checking this first.
        self.listening = threading.Event()
        self.bind_error = None
        self.reload(slice_dir, layer_start, layer_end, total_layers)

    def reload(self, slice_dir, layer_start, layer_end, total_layers):
        """Hot-swap the served slice in place (model migration, Build 3 node-side). The new
        slice is loaded OUTSIDE compute_lock (I/O + weight materialization is the slow part);
        only the pointer swap is locked, so it can't land mid-forward-pass of an in-flight
        request. Existing connections keep using self.model/self.n by reference, so the very
        next request after the swap is served by the new slice with no reconnect needed."""
        print(f"[node] loading slice from {slice_dir} (layers {layer_start}-{layer_end}) ...")
        t0 = time.time()
        model = load_slice_model(slice_dir)
        with compute_lock:
            self.model = model
            self.lo, self.hi, self.n = layer_start, layer_end, total_layers
            # Batchers close over the OLD model, so they must go with it. The next request
            # rebuilds one against the new slice.
            for b in getattr(self, "_batchers", {}).values():
                b.stop()
            self._batchers = {}
        print(f"[node] slice ready in {time.time()-t0:.1f}s | serving layers {layer_start}-{layer_end}")

    def _batcher(self, role, lo, hi):
        """One MicroBatcher per (role, layer range). Keyed rather than global because a
        batch's slots must all run the SAME layers -- the range arrives in the caller's
        config, so it is not safe to assume every connection asked for the same one.

        Named lo/hi, not s1/s2, deliberately. They were s1/s2, and the LAST role is called as
        `_batcher("last", s2, self.n)` -- so the parameter named `s1` held the real s2 while
        the closure used `s2`, which was self.n. `layers[self.n:]` is EMPTY, so the last node
        ran zero layers and just normed whatever arrived. The chain still produced fluent-
        looking tokens, which is why it took an end-to-end read of the actual text to catch:
        "There noinspectionably..." instead of "The sky is blue because...".
        """
        key = (role, lo, hi)
        with self._batcher_lock:
            b = self._batchers.get(key)
            if b is None:
                model = self.model
                if role == "last":
                    def run(h, cache, lengths):
                        return batching.last_stage_batched(model, lo, h, cache, lengths)
                else:
                    def run(h, cache, lengths):
                        return batching.mid_stage_batched(model, lo, hi, h, cache, lengths)
                b = batching.MicroBatcher(run)
                self._batchers[key] = b
            return b

    def serve(self, conn, addr):
        cache, past, role, s1, s2, bconn = None, 0, None, None, None, None
        codec = bcodec = None
        try:
            while True:
                msg = common.recv_msg(conn)
                mtype = msg.get("type")

                if mtype == "config":
                    cache, past = common.new_cache(), 0
                    is_true_last = (self.hi == self.n - 1)
                    # Negotiated per hop and per connection: the caller lists what it can
                    # decode, we answer with our pick (or omit the field, which keeps an
                    # un-upgraded caller on the legacy format). See wire_codec.
                    codec = wire_codec.negotiate(msg.get("wire"))
                    ack_wire = {"wire": codec} if codec else {}
                    if "host_b" in msg:                      # MIDDLE relay role (real pipeline traffic)
                        role, s1, s2 = "middle", msg["s1"], msg["s2"]
                        bconn = socket.create_connection((msg["host_b"], msg["port_b"]),
                                                         timeout=common.COLD_CONNECT_TIMEOUT_S)
                        common.send_msg(bconn, {"type": "config", "s2": s2, "n": msg.get("n", self.n),
                                                "wire": wire_codec.preference(self.model.config.hidden_size)})
                        back = common.recv_msg(bconn)
                        assert back.get("ok"), f"next hop refused: {back}"
                        bcodec = wire_codec.negotiate([back["wire"]] if back.get("wire") else None)
                        bconn.settimeout(common.HOT_TIMEOUT_S)
                        common.send_msg(conn, {"ok": True, "layers": self.n, "s1": s1, "s2": s2,
                                               **ack_wire})
                    elif is_true_last:                        # LAST stage role (real pipeline traffic)
                        role, s2 = "last", msg["s2"]
                        common.send_msg(conn, {"ok": True, "layers": msg.get("n", self.n), "s2": s2,
                                               **ack_wire})
                    else:
                        # PROBE role (security/proof_of_compute.py): a config with no host_b,
                        # on a node whose own range does NOT reach the model's final layer, can
                        # only mean a verifier challenging this node's layers in isolation --
                        # calling last_stage() here would be WRONG (and likely crash: this
                        # shard was downloaded without norm/later layers, which stay on the
                        # meta device, uninitialized). Uses OUR OWN self.lo/self.hi, never the
                        # caller's claimed s1/s2 -- this tests what we actually loaded, not
                        # what a challenger asserts.
                        # self.hi is the INCLUSIVE last layer this node owns, but s2 is used
                        # as a Python slice bound (`layers[s1:s2]`, see common.mid_stage), so
                        # it has to be hi+1. Passing hi ran one layer too few and advertised a
                        # range the verifier rejects -- which meant proof-of-compute could
                        # never promote a probationary node on any segment except the last,
                        # and auto-placement puts a joining stranger wherever the GAP is.
                        role, s1, s2 = "probe", self.lo, self.hi + 1
                        common.send_msg(conn, {"ok": True, "layers": self.n, "s1": s1, "s2": s2,
                                               **ack_wire})

                elif mtype == "act":
                    hidden = msg["hidden"]
                    q = hidden.shape[1]
                    # Submitting instead of locking is the whole change: concurrent requests
                    # now ride the SAME forward pass rather than queueing for the machine.
                    # The reported *_compute_ms therefore includes any time spent waiting to
                    # fill a batch (capped at NEURON_BATCH_WINDOW_MS) -- it is what the hop
                    # actually cost the caller, which is what the driver's net_ms accounting
                    # and the coordinator's balancer both want.
                    if role == "middle":
                        tc = time.time()
                        h2 = self._batcher("middle", s1, s2).submit(hidden, cache, past)
                        c_ms = (time.time() - tc) * 1000
                        past += q
                        common.send_msg(bconn, {"type": "act", "hidden": h2}, codec=bcodec)
                        resp = common.recv_msg(bconn)
                        common.send_msg(conn, {"hidden": resp["hidden"], "c_compute_ms": c_ms,
                                               "b_compute_ms": resp["b_compute_ms"]}, codec=codec)
                    elif role == "probe":
                        tc = time.time()
                        h2 = self._batcher("probe", s1, s2).submit(hidden, cache, past)
                        c_ms = (time.time() - tc) * 1000
                        past += q
                        common.send_msg(conn, {"hidden": h2, "c_compute_ms": c_ms}, codec=codec)
                    else:  # last
                        tb = time.time()
                        out = self._batcher("last", s2, self.n).submit(hidden, cache, past)
                        b_ms = (time.time() - tb) * 1000
                        past += q
                        common.send_msg(conn, {"hidden": out, "b_compute_ms": b_ms}, codec=codec)

                elif mtype == "bye":
                    if bconn:
                        common.send_msg(bconn, {"type": "bye"})
                    return
        finally:
            if bconn:
                try:
                    bconn.close()
                except OSError:
                    pass

    def run(self, host, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, port))
            srv.listen(16)
        except OSError as e:
            # Reported, not raised: this usually runs in a daemon thread, where an exception
            # is swallowed and the caller never learns the node is deaf. The agent polls
            # bind_error/listening and refuses to advertise this node until it clears.
            self.bind_error = f"{e.__class__.__name__}: {e}"
            print(f"[node] FAILED to bind {host}:{port} — {self.bind_error}")
            srv.close()
            return
        self.bind_error = None
        self.listening.set()
        print(f"[node] listening on {host}:{port}  (Ctrl-C to stop)")
        try:
            while True:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        finally:
            self.listening.clear()
            srv.close()

    def _handle(self, conn, addr):
        _serving_enter()
        try:
            self.serve(conn, addr)
        # TimeoutError is a sibling of ConnectionError under OSError, not caught by it --
        # see node_c.py's handle() for why this matters (a slow next-hop cold-start would
        # otherwise die as an uncaught thread exception and silently slam this connection
        # shut, surfacing upstream as an unexplained "socket closed mid-message").
        except (ConnectionError, TimeoutError, EOFError) as e:
            print(f"[node] conn {addr} ended: {e}")
        finally:
            _serving_exit()
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--total-layers", type=int, default=28)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50999)
    args = ap.parse_args()
    NodeServer(args.slice_dir, args.layer_start, args.layer_end, args.total_layers).run(
        args.host, args.port)


if __name__ == "__main__":
    main()
