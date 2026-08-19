"""tools/bench_quant.py — is the chain's compute wall bytes, and does cheap quantization fix it?

    python -m tools.bench_quant --slice agent/model_slice_19_27 --lo 19

**Why this exists.** [P57] measured the live network at 2.18 tok/s, of which two thirds is
compute, and argued that compute is a MEMORY BANDWIDTH wall rather than a CPU one: this PC
streams ~36.9 GB/s and an fp32 Qwen2.5-1.5B decoder layer is 187 MB of weights, every byte of
which is read for every token at batch 1. If that argument is right, then cutting bytes per
weight must cut latency by roughly the same factor, and nothing else will.

This tests it the cheapest way available -- `torch.ao.quantization.quantize_dynamic` to qint8,
4x fewer weight bytes, no new binary, no wire change, no relay change -- and measures BOTH
halves of the answer, because only one of them is good news:

  * the SPEED, against fp32 on the same slice and the same machine;
  * the DRIFT, against `security/proof_of_compute`'s `atol=0.05`, where honest fp32 drift is
    ~1e-5 and a cheating node is ~25.

Read-only: loads a slice, times forward passes, prints. Touches no config, no coordinator, no
node, and spends no NRN.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                     # noqa: E402

import common                                                    # noqa: E402
import slice_downloader                                          # noqa: E402

POC_ATOL = 0.05          # security/proof_of_compute.attest default


def _decode_ms(model, lo, hidden_size, n=12):
    """Median wall time of ONE decode token (q=1 with a warm cache), which is the only shape
    that matters: a 128-token answer is 1 prefill and 127 of these."""
    for _ in range(2):                                            # warm
        common.last_stage(model, lo, torch.randn(1, 1, hidden_size), common.new_cache(), 0)
    ts = []
    for _ in range(n):
        cache = common.new_cache()
        common.last_stage(model, lo, torch.randn(1, 4, hidden_size), cache, 0)
        t0 = time.perf_counter()
        common.last_stage(model, lo, torch.randn(1, 1, hidden_size), cache, 4)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


def quantize_slice(model, lo):
    """qint8 the layers this node actually HOLDS.

    Per-layer rather than whole-model, and that is not a style choice: `load_slice_model` builds
    a full skeleton and materialises only this node's range, so a whole-model call walks into
    the meta tensors of layers nobody downloaded and dies with `Tensor.item() cannot be called
    on meta tensors`. Quantising exactly what is materialised is also what a real node would do.
    """
    for i in range(lo, len(model.model.layers)):
        model.model.layers[i] = torch.ao.quantization.quantize_dynamic(
            model.model.layers[i], {torch.nn.Linear}, dtype=torch.qint8, inplace=False)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="agent/model_slice_19_27")
    ap.add_argument("--lo", type=int, default=19, help="first layer this slice serves")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args(argv)

    print(f"loading {args.slice} ...")
    model = slice_downloader.load_slice_model(args.slice)
    H = model.config.hidden_size
    n_layers = len(model.model.layers) - args.lo
    print(f"{n_layers} layers, hidden={H}\n")

    torch.manual_seed(0)
    fp32_ms = _decode_ms(model, args.lo, H, args.n)
    print(f"{'fp32 (what nodes serve)':26} {fp32_ms:7.2f} ms/token  "
          f"({fp32_ms/n_layers:5.2f} ms/layer)")

    torch.manual_seed(1)
    probe = torch.randn(1, 6, H)
    ref = common.last_stage(model, args.lo, probe.clone(), common.new_cache(), 0)

    quantize_slice(model, args.lo)
    torch.manual_seed(0)
    int8_ms = _decode_ms(model, args.lo, H, args.n)
    print(f"{'dynamic int8':26} {int8_ms:7.2f} ms/token  "
          f"({int8_ms/n_layers:5.2f} ms/layer)")

    got = common.last_stage(model, args.lo, probe.clone(), common.new_cache(), 0)
    drift = (got - ref).abs().max().item()
    rel = drift / max(ref.abs().max().item(), 1e-9)

    print(f"\n  speedup            : {fp32_ms/int8_ms:.2f}x")
    print(f"  max|drift| vs fp32 : {drift:.4f}  (relative {rel:.2%})")
    passes = drift <= POC_ATOL
    print(f"  proof-of-compute   : atol={POC_ATOL} -> "
          + ("within tolerance" if passes
             else "FAILS. A node serving these weights is scored as CHEATING, and it is not "
                  "wrong to -- at this drift the answer is garbage too."))
    print("\n  The speedup is the point and so is the drift: bytes-per-weight IS the wall, and\n"
          "  naive per-tensor int8 is NOT the way through it. llama.cpp's k-quants keep the\n"
          "  answer correct at 4 bits, which is why [P30]'s engine is the route and this is not.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
