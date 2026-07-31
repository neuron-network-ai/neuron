"""
node_ns.py — a NEURON node server whose Linear layers run on the NeuronScript AVX2 int8
kernel instead of PyTorch.

Speaks the EXACT wire protocol of agent/node_server.py (config / act / bye, wire_codec
negotiation, middle / last / probe roles), so it is a drop-in for any chain position and any
driver -- node_a.py, neuron_driver.py -- talks to it unchanged.

Usage (on a Linux node with an AVX2 CPU):
    gcc -O3 -mavx2 -march=native -shared -fPIC neuronscript_simd.c -o libns.so
    NEURON_NS_LIB=./libns.so python node_ns.py --slice-dir ./model_slice \
        --layer-start 10 --layer-end 18 --total-layers 28 --port 51099

    --engine torch   run the identical server on PyTorch, for an A/B on one machine

WHAT IS AND IS NOT ACCELERATED
------------------------------
Only this node's Linear GEMMs, and only on single-token decode. Prefill falls back to
PyTorch (the kernel is mat-vec; N scalar calls lose to one batched GEMM), and RMSNorm, RoPE,
softmax attention over the K/V cache and SwiGLU are untouched. So the tok/s gain is strictly
smaller than the 2.11x measured on the GEMMs alone -- `/stats` reports the real split.

WHY THIS DOES NOT USE THE NEURONSCRIPT COMPILER
-----------------------------------------------
It cannot: the compiler's output encoding and the kernel's expected input encoding are
different things, and nothing in the C sources reads the former. Measured on a real
gate_proj it is also 2.7x LARGER than fp32 (147 MB vs 55 MB; dense int8 is 13.8 MB), because
it only compresses on sparse matrices and transformer weights are dense. So the weights are
packed to dense int8 here (ns_engine.pack) and the compiler is not in the path.
"""
import argparse
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import batching                                  # noqa: E402
import common                                    # noqa: E402
import ns_engine                                 # noqa: E402
import wire_codec                                # noqa: E402
from slice_downloader import load_slice_model    # noqa: E402

compute_lock = threading.Lock()


class NSNodeServer:
    def __init__(self, slice_dir, layer_start, layer_end, total_layers, engine="ns"):
        self.lo, self.hi, self.n = layer_start, layer_end, total_layers
        self.engine = engine
        self.converted = 0
        self.tok_count = 0
        self.compute_ms = 0.0
        self._batchers = {}
        self._batcher_lock = threading.Lock()

        print(f"[ns] loading slice {slice_dir} (layers {layer_start}-{layer_end}) ...")
        t0 = time.time()
        self.model = load_slice_model(slice_dir)
        if engine == "ns":
            lib = ns_engine.load()
            if lib is None:
                print(f"[ns] WARNING: kernel not found at {ns_engine.DEFAULT_LIB} -- "
                      f"falling back to PyTorch. Set NEURON_NS_LIB.")
                self.engine = "torch"
            else:
                self.model, self.converted = ns_engine.convert(
                    self.model, layer_start, layer_end, lib)
        print(f"[ns] ready in {time.time()-t0:.1f}s | engine={self.engine} | "
              f"{self.converted} Linear layers on the int8 kernel")

    def _batcher(self, role, lo, hi):
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
                    codec = wire_codec.negotiate(msg.get("wire"))
                    ack_wire = {"wire": codec} if codec else {}
                    if "host_b" in msg:
                        role, s1, s2 = "middle", msg["s1"], msg["s2"]
                        bconn = socket.create_connection((msg["host_b"], msg["port_b"]),
                                                         timeout=common.COLD_CONNECT_TIMEOUT_S)
                        common.send_msg(bconn, {"type": "config", "s2": s2,
                                                "n": msg.get("n", self.n),
                                                "wire": wire_codec.preference(
                                                    self.model.config.hidden_size)})
                        back = common.recv_msg(bconn)
                        assert back.get("ok"), f"next hop refused: {back}"
                        bcodec = wire_codec.negotiate(
                            [back["wire"]] if back.get("wire") else None)
                        bconn.settimeout(common.HOT_TIMEOUT_S)
                        common.send_msg(conn, {"ok": True, "layers": self.n,
                                               "s1": s1, "s2": s2, **ack_wire})
                    elif is_true_last:
                        role, s2 = "last", msg["s2"]
                        common.send_msg(conn, {"ok": True, "layers": msg.get("n", self.n),
                                               "s2": s2, **ack_wire})
                    else:
                        role, s1, s2 = "probe", self.lo, self.hi
                        common.send_msg(conn, {"ok": True, "layers": self.n,
                                               "s1": s1, "s2": s2, **ack_wire})

                elif mtype == "act":
                    hidden = msg["hidden"]
                    q = hidden.shape[1]
                    t0 = time.time()
                    if role == "last":
                        out = self._batcher("last", s2, self.n).submit(hidden, cache, past)
                    else:
                        out = self._batcher(role, s1, s2).submit(hidden, cache, past)
                    ms = (time.time() - t0) * 1000
                    self.compute_ms += ms
                    self.tok_count += q
                    past += q

                    if role == "middle":
                        common.send_msg(bconn, {"type": "act", "hidden": out}, codec=bcodec)
                        resp = common.recv_msg(bconn)
                        common.send_msg(conn, {"hidden": resp["hidden"], "c_compute_ms": ms,
                                               "b_compute_ms": resp["b_compute_ms"]},
                                        codec=codec)
                    elif role == "probe":
                        common.send_msg(conn, {"hidden": out, "c_compute_ms": ms}, codec=codec)
                    else:
                        common.send_msg(conn, {"hidden": out, "b_compute_ms": ms}, codec=codec)

                elif mtype == "stats":
                    # tok/s this node sustained, so a driver or the coordinator can compare
                    # engines without trusting a claim.
                    s = ns_engine.stats(self.model)
                    s.update(engine=self.engine, converted=self.converted,
                             tokens=self.tok_count, compute_ms=round(self.compute_ms, 2),
                             tok_per_s=round(self.tok_count / max(self.compute_ms / 1000, 1e-9), 2))
                    common.send_msg(conn, s)

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
        srv.bind((host, port))
        srv.listen(16)
        print(f"[ns] listening on {host}:{port} | engine={self.engine}")
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def _handle(self, conn, addr):
        try:
            self.serve(conn, addr)
        except (ConnectionError, TimeoutError, EOFError) as e:
            print(f"[ns] conn {addr} ended: {e}")
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--total-layers", type=int, default=28)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=51099)
    ap.add_argument("--engine", choices=["ns", "torch"], default="ns",
                    help="'torch' runs the identical server on PyTorch, for a same-machine A/B")
    args = ap.parse_args()
    NSNodeServer(args.slice_dir, args.layer_start, args.layer_end,
                 args.total_layers, engine=args.engine).run(args.host, args.port)


if __name__ == "__main__":
    main()
