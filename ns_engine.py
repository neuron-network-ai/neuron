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

_lib = None


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


class NSLinear(torch.nn.Module):
    """Drop-in for nn.Linear that uses the int8 kernel for single-vector decode.

    Keeps the original fp32 weight so prefill (and any batch>1) can fall back to PyTorch --
    which costs memory. That is a deliberate trade for this experiment: it makes the A/B
    exact, because both paths run against identical weights in one process. A production
    build would drop the fp32 copy and pay a slower prefill.
    """

    def __init__(self, linear, lib):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self._lib = lib
        self._q, self._scale = pack(linear.weight)
        self._y = np.zeros(self.out_features, dtype=np.float32)
        self.ns_calls = 0
        self.fallback_calls = 0

    def forward(self, x):
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


def convert(model, layer_lo, layer_hi, lib=None, head=False):
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
    n = 0
    for idx in range(layer_lo, layer_hi + 1):
        try:
            layer = model.model.layers[idx]
        except IndexError:
            continue
        for mod in layer.modules():
            for name, child in list(mod.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mod, name, NSLinear(child, lib))
                    n += 1
    if head and isinstance(getattr(model, "lm_head", None), torch.nn.Linear):
        model.lm_head = NSLinear(model.lm_head, lib)
        n += 1
    return model, n


def stats(model):
    ns = fb = 0
    for m in model.modules():
        if isinstance(m, NSLinear):
            ns += m.ns_calls
            fb += m.fallback_calls
    return {"kernel_calls": ns, "fallback_calls": fb}
