"""
NEURON — wire_codec.py

How activations are encoded for the hop between two nodes. Two jobs, one format:

1. **Get pickle off the wire.** `common.recv_msg` used to `torch.load(..., weights_only=False)`
   whatever arrived on the socket. `torch.save` payloads are pickles, and unpickling is
   arbitrary code execution — so any peer in the chain (or anyone who could reach a node's
   port, which since Session 12 is a public relay port) could run code on the next machine
   in the pipeline. On an OPEN-JOIN network of strangers that is the whole ballgame. This
   format is a length-prefixed JSON header plus raw tensor bytes: nothing in it can execute.

2. **Stop shipping fp32.** Measured on the real 3-stage chain (Qwen2.5-1.5B, H=1536), the
   old `torch.save` fp32 path cost 12,508 bytes per message, 1,153 of it pickle framing.
   That is paid once per token per hop, and since Session 12 every relayed hop crosses the
   public VM twice. `i8h` brings it to 2,946 — **4.25x** — with all 6 benchmark answers
   still token-identical to the fp32 baseline. Petals halves this; we do better than halve
   it, because we had to (see below). Reproduce with `bench_wire.py`.

WHY int8 IS NOT ENOUGH ON ITS OWN — and what the Hadamard rotation buys
-----------------------------------------------------------------------
Transformer hidden states are dominated by a handful of outlier channels. Measured at
NEURON's own junctions (2026-07-29, layer-9 boundary, real prompt):

    absmax 6620, std 42, worst channel / median channel = 753x

An absmax quantizer sets its whole scale from that one channel, so every other value
collapses into a couple of quantization levels. That is [P9] showing up again, this time
on the wire instead of in the weights. Two things follow directly from the measurement:

  * **fp8 e4m3 is unusable here.** Its max representable value is 448; the real activations
    reach 6620, so they overflow to inf and the generation becomes NaN. Measured: 0/3
    prompts survived.
  * **Rotating first is free and fixes it.** QuaRot's result is that multiplying by a
    Hadamard matrix spreads every outlier evenly over the block without changing the vector's
    length (it is orthogonal), so the absmax scale becomes representative again. Unlike
    QuaRot we need no weight surgery and no calibration: this is a *transport* rotation,
    applied by the sender and undone by the receiver, so the model never sees it.

Measured on real junction activations, same byte cost:

    int8 blockwise-256                  1.01 B/elem   rel_l2 0.0276
    int8 blockwise-64                   1.03 B/elem   rel_l2 0.0152
    int8 HADAMARD-256 + blockwise-256   1.01 B/elem   rel_l2 0.0037   <-- 7x better, and smaller

int4 was measured too (~0.53 B/elem, rel_l2 ~0.09) and is deliberately NOT offered: at a
~9% relative error per hop it is fine for a 3-node chain and not fine for the 20-node chain
that a 70B model implies, and the wire is the one place where being wrong is silent.

Reference: QuaRot, Ashkboos et al. 2024 (arXiv:2404.00456); Petals, Borzunov et al. 2023
(arXiv:2209.01188) §"dynamic blockwise quantization".
"""
import json
import os
import struct

import torch

MAGIC = b"NRNW"
VERSION = 1

# Rotation and quantization both work on blocks of this many channels. 256 is a power of two
# (so the fast Walsh-Hadamard transform applies directly) and divides 1536, 2048, 4096, 5120
# and 8192 — i.e. every hidden size in the Llama/Qwen family we might serve.
BLOCK = 256

# Codec ids as they appear on the wire. Ordered best-first; this is also the preference list
# a driver offers during the config handshake.
CODECS = ("i8h", "f16", "f32")
DEFAULT_CODEC = "i8h"


# --------------------------------------------------------------------------- #
# Hadamard transform
# --------------------------------------------------------------------------- #
_HMAT = {}


def _hadamard_matrix(n):
    """Sylvester-construction Hadamard matrix of order n (a power of two), scaled by
    1/sqrt(n) so it is orthogonal AND self-inverse: H @ H == I exactly. Both entries are
    +-1/sqrt(256) = +-1/16 at BLOCK=256, which is exactly representable in binary floating
    point, so encoder and decoder agree to the bit on any machine."""
    if n not in _HMAT:
        h = torch.ones(1, 1)
        while h.shape[0] < n:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HMAT[n] = h / (n ** 0.5)
    return _HMAT[n]


def fwht(x):
    """Normalized Hadamard transform over the last dim (a power of two). Self-inverse, so
    the decoder calls this same function rather than a separate inverse.

    Implemented as one matmul against a cached matrix rather than the textbook log-n
    butterfly loop. The butterfly is fewer FLOPs but runs as ~8 separate Python-level tensor
    ops; the matmul is a single BLAS call and measured 28-41x faster here. That margin is
    the difference between the rotation paying for itself and not: at 8192 wide the loop
    cost 1.26 ms per call against ~6.4 ms of wire time saved, and this costs 0.045 ms.
    """
    return x @ _hadamard_matrix(x.shape[-1]).to(x.dtype)


# --------------------------------------------------------------------------- #
# per-codec tensor encode / decode
# --------------------------------------------------------------------------- #
def _pad_rows(t):
    """(..., D) -> (rows, Dp) zero-padded up to a multiple of BLOCK. Returns (padded, D)."""
    d = t.shape[-1]
    flat = t.reshape(-1, d).to(torch.float32)
    pad = (-d) % BLOCK
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat, d


# Scales travel as fp32, not fp16. It costs 4 bytes per 256 values -- 1.016 B/elem instead
# of 1.008, i.e. nothing -- and it removes a silent-corruption mode: the rotation preserves
# the block's L2 norm, so a block whose norm exceeds 65504 would overflow an fp16 scale to
# inf, and every value in that block would decode as zero with no error raised anywhere.
# Real activations measured here peak around 6.6e3, but the wire is the one place where
# being wrong is silent, so it is not worth the 0.8%.
def _encode_i8h(t):
    flat, d = _pad_rows(t)
    rows, dp = flat.shape
    rot = fwht(flat.reshape(rows, dp // BLOCK, BLOCK)).reshape(-1, BLOCK)
    scale = rot.abs().amax(dim=1).clamp_min(1e-12)
    q = torch.round(rot / scale[:, None] * 127.0).clamp(-127, 127).to(torch.int8)
    return scale.numpy().tobytes() + q.numpy().tobytes(), {"d": d, "rows": rows}


def _decode_i8h(blob, shape, extra):
    d, rows = extra["d"], extra["rows"]
    dp = d + (-d) % BLOCK
    nblocks = rows * (dp // BLOCK)
    scale = torch.frombuffer(bytearray(blob[: 4 * nblocks]), dtype=torch.float32)
    q = torch.frombuffer(bytearray(blob[4 * nblocks:]), dtype=torch.int8).to(torch.float32)
    rot = (q.view(nblocks, BLOCK) * scale[:, None] / 127.0)
    flat = fwht(rot.reshape(rows, dp // BLOCK, BLOCK)).reshape(rows, dp)
    return flat[:, :d].reshape(shape)


def _encode_raw(t, dtype):
    c = t.to(dtype).contiguous()
    view = c.view(torch.int16) if dtype is torch.float16 else c
    return view.numpy().tobytes(), {}


def _decode_raw(blob, shape, dtype):
    buf = torch.frombuffer(bytearray(blob), dtype=torch.int16 if dtype is torch.float16 else dtype)
    if dtype is torch.float16:
        buf = buf.view(torch.float16)
    return buf.to(torch.float32).reshape(shape)


_ENCODE = {
    "i8h": _encode_i8h,
    "f16": lambda t: _encode_raw(t, torch.float16),
    "f32": lambda t: _encode_raw(t, torch.float32),
}
_DECODE = {
    "i8h": _decode_i8h,
    "f16": lambda b, s, e: _decode_raw(b, s, torch.float16),
    "f32": lambda b, s, e: _decode_raw(b, s, torch.float32),
}


# --------------------------------------------------------------------------- #
# message framing
# --------------------------------------------------------------------------- #
def encode(obj, codec=DEFAULT_CODEC):
    """dict -> bytes. Tensor values are split out into raw binary; everything else must be
    JSON-representable (the pipeline only ever sends str/int/float/bool alongside tensors)."""
    if codec not in _ENCODE:
        raise ValueError(f"unknown wire codec {codec!r}")
    meta, tensors, blobs = {}, [], []
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            blob, extra = _ENCODE[codec](v.detach().to(torch.float32))
            tensors.append({"k": k, "c": codec, "s": list(v.shape), "n": len(blob), "e": extra})
            blobs.append(blob)
        else:
            meta[k] = v
    header = json.dumps({"m": meta, "t": tensors}, separators=(",", ":")).encode("utf-8")
    return b"".join([MAGIC, struct.pack("<BI", VERSION, len(header)), header] + blobs)


def decode(data):
    """bytes -> dict. Raises ValueError on anything that is not a well-formed NRNW frame."""
    if not data.startswith(MAGIC):
        raise ValueError("not a NEURON wire frame")
    ver, hlen = struct.unpack_from("<BI", data, len(MAGIC))
    if ver != VERSION:
        raise ValueError(f"unsupported wire version {ver}")
    off = len(MAGIC) + 5
    head = json.loads(data[off: off + hlen])
    off += hlen
    out = dict(head["m"])
    for spec in head["t"]:
        blob = data[off: off + spec["n"]]
        off += spec["n"]
        dec = _DECODE.get(spec["c"])
        if dec is None:
            raise ValueError(f"unknown wire codec {spec['c']!r}")
        out[spec["k"]] = dec(blob, tuple(spec["s"]), spec.get("e", {}))
    return out


def is_frame(data):
    return data[:4] == MAGIC


def negotiate(offered, supported=CODECS):
    """Pick the first codec both ends know, preferring the offerer's order. Returns None if
    the peer offered nothing we understand — the caller then stays on the legacy path."""
    for c in offered or ():
        if c in supported:
            return c
    return None


# Below this hidden size, int8 on the wire is NOT offered — see preference().
I8H_MIN_HIDDEN = 1536


def preference(hidden_size=None):
    """The codec list a driver/node offers, best first. `NEURON_WIRE_CODEC` overrides it:
    a codec name pins that one, `legacy` disables the new framing for sending entirely
    (receiving still accepts both, and still refuses hostile pickles).

    **Small models do not get i8h, because the drift is measurably worse on them.** Same
    benchmark, same 3 junctions, 6 prompts, greedy decode, against the fp32 baseline:

        Qwen2.5-1.5B (H=1536)   i8h  6/6 identical   max|dlogit| 0.21
        Qwen2.5-0.5B (H=896)    i8h  3/6 identical   max|dlogit| 0.50
        both sizes              f16  6/6 identical   max|dlogit| <0.01

    Be precise about what the 3/6 is: the diverging 0.5B answers stay correct and stay on
    topic — they re-word, usually 100+ characters in ("ensuring smooth and efficient
    communication between nodes" vs "improving the performance and responsiveness of
    applications"). This is drift, not the [P9]-style collapse that unrotated int8 produces.
    But it is drift the larger model does not show, which is the expected direction: fewer
    parameters means less redundancy to absorb the noise. Both pressures point the same way
    — small models are the fragile ones AND the cheap ones to ship uncompressed — so there
    is nothing to trade off, and f16 still buys 2.3x on them.

    The threshold sits between the two sizes actually measured. It is two data points, not a
    curve — measure a third model before trusting it far from 896/1536.

    The env knob exists because the CPU/bandwidth trade also depends on the link. Measured at
    H=8192: i8h costs 0.54 ms of encode+decode per hop and saves 6.4 ms of transmission on a
    10 Mbit/s home upload, a 12x payoff. On a LAN the same 0.54 ms buys ~0.06 ms and f16 is
    the better trade. The default assumes NEURON's actual deployment: volunteers at home.
    """
    forced = os.environ.get("NEURON_WIRE_CODEC", "").strip().lower()
    if forced:
        if forced in ("legacy", "off", "none"):
            return []
        if forced not in CODECS:
            raise ValueError(f"NEURON_WIRE_CODEC={forced!r} is not one of {CODECS} or 'legacy'")
        return [forced]
    if hidden_size is not None and hidden_size < I8H_MIN_HIDDEN:
        return [c for c in CODECS if c != "i8h"]
    return list(CODECS)
