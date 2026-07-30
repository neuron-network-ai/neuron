"""test_weight_dtype.py — does half-precision STORAGE with fp32 compute actually pay?

The claim being tested: a node can hold its weights at 2 bytes/param instead of 4, keep every
GEMM in fp32 (the only precision these CPUs are fast at), and get the same answer. If true,
Llama-3.1-8B drops from 9.3 GB per node to 4.7 GB and fits on a 6 GB laptop -- which is the
difference between NEURON's premise being testable and not.

Three things have to hold, and all three are easy to get wrong:
  1. resident memory really halves (not "the file is smaller" -- actual live bytes)
  2. speed does not collapse (the cast is not free; [P2] measured fp16 COMPUTE at ~8x slower,
     so if this accidentally computes in fp16 it will show up here)
  3. the output does not change

Run: python test_weight_dtype.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

CHILD = r'''
import os, sys, time, json
import torch
sys.path.insert(0, r"{here}")
import common

LO, HI = 0, 6
tok, model, n = common.load_model_shard(LO, HI, embed=True, head=True)
h = model.config.hidden_size
resident = common.resident_bytes(model)
# Count params over UNIQUE STORAGES, the same dedup resident_bytes() uses. Two traps here:
# unused layers sit on the meta device (numel but no storage), and CastLinear wraps lm_head
# in a fresh Parameter so .parameters() stops de-duplicating it against the tied embedding --
# the storage is still shared, but a naive numel() sum counts it twice and reports 1.39
# bytes/param for a 2-byte dtype.
_seen, nparams = set(), 0
for _p in model.parameters():
    if _p.is_meta:
        continue
    _ptr = _p.untyped_storage().data_ptr()
    if _ptr in _seen:
        continue
    _seen.add(_ptr)
    nparams += _p.numel()

ids = torch.tensor([[9707, 11, 1879, 0, 1246, 525, 498]])
cache = common.new_cache()
out = common.first_stage(model, HI, ids, cache, 0)
torch.manual_seed(0)
t0 = time.perf_counter()
for i in range(8):
    common.first_stage(model, HI, torch.tensor([[100 + i]]), cache, 7 + i)
ms = (time.perf_counter() - t0) / 8 * 1000

print("RESULT" + json.dumps({{
    "dtype": str(common.WEIGHT_DTYPE),
    "bytes_per_param": resident / nparams,
    "resident_gb": resident / 1e9,
    "ms_per_pass": ms,
    "hidden_checksum": float(out.double().sum()),
    "hidden_absmax": float(out.abs().max()),
}}))
'''.format(here=HERE)


def run(dtype):
    env = dict(os.environ, NEURON_WEIGHT_DTYPE=dtype,
               NEURON_MODEL_ID=os.environ.get("NEURON_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct"))
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            import json
            return json.loads(line[6:])
    print(p.stdout[-1500:])
    print(p.stderr[-1500:])
    raise SystemExit(f"child failed for {dtype}")


def main():
    ok = fail = 0

    def check(name, cond):
        nonlocal ok, fail
        ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    print("loading the same shard twice, once per storage dtype ...\n", flush=True)
    a = run("fp32")
    b = run("fp16")

    print(f"{'':10s} {'B/param':>9s} {'resident':>10s} {'ms/pass':>9s}")
    for r in (a, b):
        print(f"{r['dtype'].replace('torch.',''):10s} {r['bytes_per_param']:9.2f} "
              f"{r['resident_gb']:9.2f}G {r['ms_per_pass']:9.1f}")
    print()

    check(f"fp32 stores 4 B/param (got {a['bytes_per_param']:.2f})",
          abs(a["bytes_per_param"] - 4.0) < 0.05)
    check(f"fp16 stores 2 B/param (got {b['bytes_per_param']:.2f})",
          abs(b["bytes_per_param"] - 2.0) < 0.05)
    check(f"resident memory halves ({a['resident_gb']:.2f}G -> {b['resident_gb']:.2f}G)",
          b["resident_gb"] < a["resident_gb"] * 0.55)

    # The cast is NOT free: it adds a read of the fp16 weight plus a write of an fp32 copy
    # on every forward, and a batch-1 decode GEMM is already pure weight-read. Measured
    # 2.86x slower at batch 1 -- but the cast happens once per forward regardless of batch
    # size, so it amortises: 1.64x at batch 8, which is the loaded case that matters and the
    # reason this composes with batching.py. What must NOT happen is an accidental fp16
    # GEMM, which [P2] measured at ~8x; this bound catches that.
    ratio = b["ms_per_pass"] / max(a["ms_per_pass"], 1e-9)
    print(f"      speed cost at batch 1: {ratio:.2f}x (expected ~2.9x; ~1.6x at batch 8)")
    check(f"cost is the cast, not an accidental fp16 GEMM (<4x, got {ratio:.2f}x)", ratio < 4.0)

    # bf16 -> fp16 is a real precision change, so bit-equality is not on offer. What matters
    # is that the hidden state is the same vector, not a subtly different one.
    rel = abs(b["hidden_checksum"] - a["hidden_checksum"]) / max(abs(a["hidden_checksum"]), 1e-9)
    print(f"      hidden-state checksum drift: {rel:.2e}")
    check(f"output is materially unchanged (rel drift < 1e-2)", rel < 1e-2)
    check("no overflow/NaN in the fp16 path",
          b["hidden_absmax"] == b["hidden_absmax"] and b["hidden_absmax"] < 1e6)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
