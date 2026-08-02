"""
ns_engine.py — drive the NeuronScript AVX2 int8 kernel from Python.

Contains no kernel source: it dlopen's a shared library built from the (untracked) C and
calls it through ctypes. Everything proprietary stays out of the repo; this file is just the
adapter, so it is safe to track.

BUILD THE LIBRARY FIRST (Linux, AVX2 CPU):
    gcc -O3 -mavx2 -march=native -shared -fPIC neuronscript_simd.c -o libns.so
    export NEURON_NS_LIB=/path/to/libns.so

WHAT IT REPLACES
----------------
Only the Linear GEMMs inside a decoder layer. RMSNorm, RoPE, softmax attention over the K/V
cache and the SwiGLU elementwise work stay in PyTorch -- the kernel has no opinion about
them. So a layer's end-to-end speedup is strictly less than its GEMM speedup.

MEASURED (Pavilion, idle, real Qwen2.5-1.5B layer-10 weights, PyTorch single-thread):
    q_proj    1536x1536   torch 0.810ms  int8 0.371ms  2.18x   rel_err 0.034
    o_proj    1536x1536   torch 0.819ms  int8 0.372ms  2.20x   rel_err 0.041
    gate_proj 8960x1536   torch 3.532ms  int8 2.321ms  1.52x   rel_err 0.058
    down_proj 1536x8960   torch 3.499ms  int8 1.041ms  3.36x   rel_err 0.070
    GEMM total            torch 8.660ms  int8 4.105ms  2.11x

TWO LIMITS THAT SHAPE THE CODE BELOW
------------------------------------
1. It is a mat-VEC kernel: one input vector per call. Prefill (q>1) would need one call per
   token, which is slower than the batched fp32 GEMM PyTorch already does -- so prefill falls
   back to PyTorch and only single-token decode uses the kernel. Decode is ~99% of the work
   in a long answer, so this keeps nearly all of the win.

2. The accumulator is int32. `_mm256_madd_epi16` sums `w_i16 * x_i16` across `in_dim` terms,
   so the worst case must stay under 2^31. With in_dim=1536 and int8 weights near +-127 that
   caps |x_int16| near 1e4 -- the fixed input_scale=128 the C comment suggests would overflow
   on real activations, whose measured absmax is ~6620 (see wire_codec.py). `_input_scale`
   derives it from the actual input instead, per call.
"""
import ctypes
import os

import numpy as np
import torch

DEFAULT_LIB = os.environ.get("NEURON_NS_LIB", "./libns.so")
TILER_LIB = os.environ.get("NEURON_NS_TILER_LIB", "./libns_tiler.so")
BITMASK_LIB = os.environ.get("NEURON_NS_BITMASK_LIB", "./libns_bitmask.so")

# simd    -- mat-vec kernel on decode, PyTorch on prefill (the shipped behaviour)
# tiler   -- the cache-tile scheduler on EVERY forward pass, prefill included
# hybrid  -- tiler for prefill (batch>1), the bitmask predictor for decode (batch=1)
MODE = os.environ.get("NEURON_NS_MODE", "simd")

_lib = None
_tiler = None
_bitmask = None


def load(path=None):
    """dlopen the kernel. Returns None if unavailable, so callers can fall back to PyTorch
    rather than crash -- a node with no AVX2 build must still be able to serve."""
    global _lib
    if _lib is not None:
        return _lib
    p = path or DEFAULT_LIB
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(os.path.abspath(p))
    sig = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
           ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    lib.matmul_i8_avx2.argtypes = sig
    lib.matmul_i8_avx2_4x.argtypes = sig
    _lib = lib
    return _lib


def available(path=None):
    return load(path) is not None


# --------------------------------------------------------------------------- #
# Tiler / predictor kernels
#
# Each kernel is its own shared library rather than one combined build, because the three
# sources define `make_layer`, `free_layer`, `bench` and `main` at file scope -- linking
# them together is a duplicate-symbol error. Separate libraries need no source edits.
# --------------------------------------------------------------------------- #
class NSLayerStruct(ctypes.Structure):
    """Mirrors the `NSLayer` / `Layer` struct, which is byte-identical in both sources."""
    _fields_ = [("W", ctypes.c_void_p),
                ("scales", ctypes.c_void_p),
                ("od", ctypes.c_int),
                ("id", ctypes.c_int),
                ("id32", ctypes.c_int),
                ("tile_rows", ctypes.c_int),
                ("n_tiles", ctypes.c_int)]


_PP = ctypes.POINTER(ctypes.c_void_p)


def load_tiler(path=None):
    """dlopen the cache-tile scheduler (`tiler_run`). None if it is not built here."""
    global _tiler
    if _tiler is not None:
        return _tiler
    p = path or TILER_LIB
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.tiler_run.argtypes = [_PP, ctypes.c_int, _PP, _PP, ctypes.c_int, _PP, _PP, _PP]
    lib.tiler_run.restype = None
    _tiler = lib
    return _tiler


def load_bitmask(path=None):
    """dlopen the bitmask row predictor (`full_system`). None if it is not built here."""
    global _bitmask
    if _bitmask is not None:
        return _bitmask
    p = path or BITMASK_LIB
    if not os.path.exists(p):
        return None
    lib = ctypes.CDLL(os.path.abspath(p))
    lib.full_system.argtypes = [_PP, ctypes.c_int, _PP, _PP, ctypes.c_int, _PP, _PP, _PP,
                                ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_long)]
    lib.full_system.restype = None
    _bitmask = lib
    return _bitmask


# The tile size the C picks from its own compiled-in L3 constant (33 MB in both sources).
# Recomputed here only so a node can REPORT what it actually got -- see `tile_report`.
_L3_BYTES = 33792 * 1024
_TILE_BUDGET = _L3_BYTES // 2


def tile_rows_for(in_dim):
    rows = (_TILE_BUDGET // max(in_dim, 1)) // 4 * 4
    return max(rows, 4)


def pack_rows(W):
    """fp32 weight -> (int8 rows padded to a multiple of 32, PER-ROW dequant scales).

    The tiler/predictor ABI carries one scale per output row, where the mat-vec kernel takes
    a single scale for the whole tensor. Per-row is the more accurate of the two -- an output
    channel with a small dynamic range no longer has to share a scale with the largest one.
    """
    W = W.detach().float()
    od, idim = W.shape
    id32 = (idim + 31) & ~31
    amax = W.abs().amax(dim=1)
    scale = torch.where(amax > 0, amax / 127.0, torch.ones_like(amax))
    q = torch.round(W / scale[:, None]).clamp(-127, 127).to(torch.int8)
    packed = torch.zeros(od, id32, dtype=torch.int8)
    packed[:, :idim] = q
    return (np.ascontiguousarray(packed.numpy()),
            np.ascontiguousarray(scale.numpy().astype(np.float32)))


def pack(W):
    """fp32 weight -> (dense int8 rows padded to a multiple of 32, scale).

    The kernel indexes rows at stride `id32`, so the padding is part of the ABI, not an
    optimisation. Note this is dense int8 (1 byte/weight) -- NOT the compiler's own
    encoding, which measured 147 MB against 55 MB of fp32 on a real gate_proj.
    """
    W = W.detach().float()
    od, idim = W.shape
    id32 = (idim + 31) & ~31
    scale = W.abs().max().item() / 127.0
    if scale <= 0:
        scale = 1.0
    q = torch.round(W / scale).clamp(-127, 127).to(torch.int8)
    packed = torch.zeros(od, id32, dtype=torch.int8)
    packed[:, :idim] = q
    return np.ascontiguousarray(packed.numpy()), float(scale)


def _input_scale(x_abs_max, in_dim):
    """Largest input scale that keeps the int32 accumulator safe (see module docstring)."""
    a = max(float(x_abs_max), 1e-9)
    by_int16 = 32767.0 / a
    by_accum = 2.0e9 / (127.0 * max(in_dim, 1) * a)
    return min(by_int16, by_accum)


class _TiledWeight:
    """One Linear packed for the tiler/predictor ABI, plus the scratch those entry points
    require the caller to own (x16 staging, ping/pong, and the pointer arrays).

    THE INPUT SCALE PROBLEM, AND THE FIX THAT NEEDS NO SOURCE EDIT
    -------------------------------------------------------------
    `tiler_run` and `full_system` both hard-code `isc = 256.f` and quantise the activation as
    `(int)(x*256)`, clamped to +-32767. Real junction activations here reach absmax ~6620
    (measured, see wire_codec.py), so at a fixed 256 every one of them saturates, and the
    int32 accumulator would overflow long before that. The kernels take no scale argument,
    so the scale cannot be passed in.

    It does not have to be. Feeding `x * (s/256)` makes the kernel's own quantiser compute
    `(int)(x*s)` for any `s` we choose, and the result then comes back scaled by exactly `s`:

        y_kernel = sum_j Wq[i][j] * (x[j]*s) * row_scale[i]  =  s * (W @ x)[i]

    so dividing the output vector by `s` recovers the true value. One multiply in, one
    divide out, both O(dim) against an O(od*id) GEMM. `_input_scale` picks the largest `s`
    that keeps the accumulator inside int32 -- the same bound the mat-vec path uses.
    """

    def __init__(self, weight):
        self.q, self.scales = pack_rows(weight)
        self.od, self.idim = int(self.q.shape[0]), int(weight.shape[1])
        self.id32 = int(self.q.shape[1])
        self.tile_rows = min(tile_rows_for(self.idim), self.od)
        self.n_tiles = (self.od + self.tile_rows - 1) // self.tile_rows

        self.layer = NSLayerStruct(W=self.q.ctypes.data, scales=self.scales.ctypes.data,
                                   od=self.od, id=self.idim, id32=self.id32,
                                   tile_rows=self.tile_rows, n_tiles=self.n_tiles)
        self._layers = (ctypes.c_void_p * 1)(ctypes.addressof(self.layer))
        self._bufs = {}

    def _buffers(self, n):
        """Scratch for a batch of n token vectors. Cached per n: decode always asks for 1,
        a prefill asks once for its length, so this settles after the first of each."""
        b = self._bufs.get(n)
        if b is not None:
            return b
        wide = max(self.idim, self.od)
        xin = np.zeros((n, self.idim), dtype=np.float32)
        yout = np.zeros((n, self.od), dtype=np.float32)
        x16 = np.zeros((n, self.id32), dtype=np.int16)   # tail past idim stays 0 forever
        ping = np.zeros((n, wide), dtype=np.float32)
        pong = np.zeros((n, wide), dtype=np.float32)

        def rows(a):
            stride = a.strides[0]
            return (ctypes.c_void_p * n)(*[a.ctypes.data + i * stride for i in range(n)])

        b = (xin, yout, x16, ping, pong,
             rows(xin), rows(yout), rows(x16), rows(ping), rows(pong))
        self._bufs[n] = b
        return b

    def run(self, x2d, lib, predictor=False):
        """x2d: [n, in_features] float32 numpy. Returns [n, out_features] float32."""
        n = x2d.shape[0]
        xin, yout, _x16, _pi, _po, p_in, p_out, p_x16, p_ping, p_pong = self._buffers(n)

        s = _input_scale(float(np.abs(x2d).max()), self.idim)
        np.multiply(x2d, s / 256.0, out=xin)

        if predictor:
            comp, tot = ctypes.c_long(0), ctypes.c_long(0)
            lib.full_system(self._layers, 1, p_in, p_out, n, p_x16, p_ping, p_pong,
                            ctypes.byref(comp), ctypes.byref(tot))
            self.rows_computed = comp.value
            self.rows_total = tot.value
        else:
            lib.tiler_run(self._layers, 1, p_in, p_out, n, p_x16, p_ping, p_pong)
        return yout / s


class NSLinear(torch.nn.Module):
    """Drop-in for nn.Linear that uses the int8 kernel for single-vector decode.

    Keeps the original fp32 weight so prefill (and any batch>1) can fall back to PyTorch --
    which costs memory. That is a deliberate trade for this experiment: it makes the A/B
    exact, because both paths run against identical weights in one process. A production
    build would drop the fp32 copy and pay a slower prefill.
    """

    def __init__(self, linear, lib, mode=None, tiler=None, bitmask=None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self._lib = lib
        self.mode = mode or MODE
        self._tiler = tiler
        self._bitmask = bitmask
        self._tiled = None
        if self.mode in ("tiler", "hybrid"):
            # Both modes need the tiler: "hybrid" runs it on prefill and only swaps the
            # decode path for the predictor.
            if self._tiler is None:
                self.mode = "simd"
            else:
                self._tiled = _TiledWeight(linear.weight)
                if self.mode == "hybrid" and self._bitmask is None:
                    self.mode = "tiler"
        self._q, self._scale = pack(linear.weight)
        self._y = np.zeros(self.out_features, dtype=np.float32)
        self.ns_calls = 0
        self.fallback_calls = 0
        self.tiler_calls = 0
        self.predictor_calls = 0

    def _tiled_forward(self, x, predictor):
        """Every position in the block is an independent vector for a Linear, so a [B, q, in]
        block flattens to B*q rows -- which is exactly the `batch` the tiler wants, and the
        only shape in which it has anything to amortise a tile load over."""
        shape = x.shape
        flat = x.reshape(-1, self.in_features).to(torch.float32).contiguous()
        x2d = np.ascontiguousarray(flat.numpy())
        lib = self._bitmask if predictor else self._tiler
        y = self._tiled.run(x2d, lib, predictor=predictor)
        out = torch.from_numpy(y.copy()).reshape(*shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias.float()
        if predictor:
            self.predictor_calls += 1
        else:
            self.tiler_calls += 1
        return out

    def forward(self, x):
        if self._tiled is not None:
            n = 1
            for d in x.shape[:-1]:
                n *= d
            # "tiler": every pass, prefill included. "hybrid": predictor on single-token
            # decode, tiler on the wider prefill block.
            return self._tiled_forward(x, predictor=(self.mode == "hybrid" and n == 1))

        # Only [.., 1, in] single-vector decode goes to the kernel; anything wider is a
        # prefill and PyTorch's batched GEMM beats N scalar calls.
        if x.dim() == 3 and x.shape[0] == 1 and x.shape[1] == 1:
            vec = x.reshape(-1).to(torch.float32).contiguous()
            xin = np.ascontiguousarray(vec.numpy())
            self._lib.matmul_i8_avx2_4x(
                self._q.ctypes.data, xin.ctypes.data, self._y.ctypes.data,
                self.out_features, self.in_features, self._scale,
                _input_scale(float(vec.abs().max()), self.in_features))
            out = torch.from_numpy(self._y.copy()).reshape(1, 1, self.out_features)
            if self.bias is not None:
                out = out + self.bias.float()
            self.ns_calls += 1
            return out
        self.fallback_calls += 1
        return torch.nn.functional.linear(x, self.weight.float(),
                                          None if self.bias is None else self.bias.float())


def convert(model, layer_lo, layer_hi, lib=None, head=False, mode=None):
    """Swap every nn.Linear inside layers[layer_lo:layer_hi+1] for an NSLinear.

    Scoped to the node's OWN layers on purpose, so a node converts only what it serves.

    `head=True` also converts `lm_head`, which lives OUTSIDE model.model.layers and would
    otherwise be silently skipped -- easy to miss, and expensive to miss: it is by far the
    largest GEMM in the pipeline (151936x1536 for Qwen2.5-1.5B) and the single biggest cost
    on the driver. Measured on Windows/AVX2: 37.78 ms -> 16.09 ms, 2.35x, rel_err 0.013.
    Only the driver holds it; every other node passes head=False.
    """
    lib = lib or load()
    if lib is None:
        return model, 0
    mode = mode or MODE
    tiler = load_tiler() if mode in ("tiler", "hybrid") else None
    bitmask = load_bitmask() if mode == "hybrid" else None
    kw = {"mode": mode, "tiler": tiler, "bitmask": bitmask}
    n = 0
    for idx in range(layer_lo, layer_hi + 1):
        try:
            layer = model.model.layers[idx]
        except IndexError:
            continue
        for mod in layer.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mod, name, NSLinear(child, lib, **kw))
                    n += 1
    if head and isinstance(getattr(model, "lm_head", None), torch.nn.Linear):
        model.lm_head = NSLinear(model.lm_head, lib, **kw)
        n += 1
    return model, n


def tile_report(model):
    """What the tiler actually decided, per distinct Linear shape.

    Worth printing rather than assuming: both C sources compile in a fixed 33 MB L3, so on a
    machine with less cache the "tile" they choose is larger than the cache it is meant to
    fit, and on a small matrix it covers every row -- one tile, i.e. no tiling at all.
    """
    seen = {}
    for m in model.modules():
        if isinstance(m, NSLinear) and m._tiled is not None:
            t = m._tiled
            seen[(t.od, t.idim)] = (t.tile_rows, t.n_tiles)
    return {f"{od}x{idim}": {"tile_rows": tr, "n_tiles": nt}
            for (od, idim), (tr, nt) in sorted(seen.items())}


def stats(model):
    ns = fb = 0
    for m in model.modules():
        if isinstance(m, NSLinear):
            ns += m.ns_calls
            fb += m.fallback_calls
    return {"kernel_calls": ns, "fallback_calls": fb}
