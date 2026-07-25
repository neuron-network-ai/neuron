"""
benchmark.py — a node measures its own inference speed for auto-balancing  [Session 14]

Times REAL Qwen2 decoder-layer forward passes (a decode step) to get ms/layer, and the
lm_head to get the driver's fixed head cost. The coordinator feeds these into
coordinator/balancer.py to assign each node an optimal layer slice — so no more manual
--s1/--s2. Reuses common (loads only a few probe layers, ~one-time cost). ARM-safe.

Run:  python benchmark.py            # non-driver node: ms/layer only
      python benchmark.py --driver   # node_a: also measure head_ms
"""
import argparse
import json
import time

import torch

import common


def measure(is_driver=False, probe_layers=4, steps=25, warmup=5):
    """Load a few real layers (+ embed/head if driver) and time a decode step."""
    _tok, model, n = common.load_model_shard(0, probe_layers, embed=is_driver, head=is_driver)
    H = model.config.hidden_size

    def decode_hidden():
        return torch.randn(1, 1, H, dtype=common.DTYPE) * 0.1

    # ms / layer: time `steps` decode passes over `probe_layers` real layers
    cache, past = common.new_cache(), 0
    for _ in range(warmup):
        common.mid_stage(model, 0, probe_layers, decode_hidden(), cache, past)
        past += 1
    cache, past = common.new_cache(), 0
    t0 = time.perf_counter()
    for _ in range(steps):
        common.mid_stage(model, 0, probe_layers, decode_hidden(), cache, past)
        past += 1
    ms_per_layer = (time.perf_counter() - t0) * 1000 / (steps * probe_layers)

    # head cost (driver only): the lm_head GEMM over the ~152k vocab
    head_ms = 0.0
    if is_driver:
        hid = decode_hidden()
        for _ in range(warmup):
            common.apply_lm_head(model, hid)
        t0 = time.perf_counter()
        for _ in range(steps):
            common.apply_lm_head(model, hid)
        head_ms = (time.perf_counter() - t0) * 1000 / steps

    return {"ms_per_layer": round(ms_per_layer, 3), "head_ms": round(head_ms, 3),
            "total_layers": n, "hidden": H, "is_driver": is_driver}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", action="store_true", help="also measure lm_head cost (node_a)")
    ap.add_argument("--probe-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=25)
    args = ap.parse_args()
    print(json.dumps(measure(args.driver, args.probe_layers, args.steps)))


if __name__ == "__main__":
    main()
