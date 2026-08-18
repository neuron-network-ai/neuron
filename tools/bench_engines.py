"""tools/bench_engines.py — how much faster is the fast engine, on THIS machine? ([P30])

    python -m tools.bench_engines

[P30] is filed on a quoted ~17x. That figure decides whether replacing the network's compute
engine is worth a C++ dependency on every volunteer's PC, so it should be measured here rather
than cited — the same reasoning that retired [P47]'s skip and [P2]'s int8 assumption.

**What this compares, and what it does not.** Both engines run the SAME model on the SAME
machine, decoding tokens one at a time, which is the operation the product is bound by
(prefill is unmeasured — [P13]). It is deliberately NOT apples to apples on precision:
`node_server` runs PyTorch fp32 because that is what the network actually does today, and
llama.cpp runs q4_k_m because that is what it is for. The gap therefore mixes kernel quality
with memory bandwidth, and that is the honest comparison — the question is not "which matmul is
better" but "how fast could a volunteer's machine answer", which is what a user feels.

Quality is NOT measured here and the difference is not free ([P9]: naive int8 destroyed output).
k-quants are not naive int8, but this script makes no claim about that.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROMPT = "Explain in one short paragraph why distributed inference is hard."


def _fmt(label, tok, secs, note=""):
    rate = tok / secs if secs else 0.0
    print(f"  {label:<34} {tok:>4} tok in {secs:>6.2f}s  =  {rate:>6.2f} tok/s"
          + (f"   {note}" if note else ""))
    return rate


def bench_torch(model_id, n_tokens, threads):
    """PyTorch fp32 decode — what agent/node_server.py runs today."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval()
    load = time.time() - t0
    ids = tok(PROMPT, return_tensors="pt").input_ids
    with torch.no_grad():
        # Prefill once, then time DECODE alone: decode is the bandwidth-bound loop the
        # product's latency is made of, and folding prefill in would flatter whichever
        # engine has the better prefill.
        out = model(ids, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        t1 = time.time()
        n = 0
        for _ in range(n_tokens):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1:].argmax(-1)
            n += 1
        dt = time.time() - t1
    return n, dt, load


def bench_llama(gguf_path, n_tokens, threads):
    """llama.cpp q4_k_m decode — what engine/local_gguf.py runs."""
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=gguf_path, n_ctx=2048, n_threads=threads, verbose=False)
    load = time.time() - t0
    t1 = time.time()
    n = 0
    for _ in llm.create_completion(PROMPT, max_tokens=n_tokens, stream=True):
        n += 1
    dt = time.time() - t1
    return n, dt, load


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--gguf", default=None, help="path to the same model as gguf")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--threads", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args(argv)

    print(f"\nmodel   : {args.model}")
    print(f"threads : {args.threads}   (of {os.cpu_count()} logical CPUs)")
    print(f"decode  : {args.tokens} tokens, greedy, single machine\n")

    rates = {}
    print("PyTorch fp32 — what the NETWORK runs (agent/node_server.py)")
    try:
        n, dt, load = bench_torch(args.model, args.tokens, args.threads)
        rates["torch"] = _fmt("torch fp32", n, dt, f"(load {load:.1f}s)")
    except Exception as e:                                    # noqa: BLE001
        print(f"  FAILED: {e.__class__.__name__}: {e}")

    print("\nllama.cpp q4_k_m — what the LOCAL engine runs (engine/local_gguf.py)")
    gguf = args.gguf
    if not gguf:
        try:
            from engine import local_gguf
            gguf = local_gguf.model_path(args.model)
        except Exception:                                     # noqa: BLE001
            gguf = None
    if not gguf or not os.path.exists(gguf):
        print("  SKIP: no gguf for this model — pass --gguf")
    else:
        try:
            n, dt, load = bench_llama(gguf, args.tokens, args.threads)
            rates["llama"] = _fmt("llama.cpp q4_k_m", n, dt, f"(load {load:.1f}s)")
        except Exception as e:                                # noqa: BLE001
            print(f"  FAILED: {e.__class__.__name__}: {e}")

    if len(rates) == 2:
        mult = rates["llama"] / rates["torch"] if rates["torch"] else 0
        print(f"\n  llama.cpp is {mult:.1f}x the decode rate of PyTorch fp32 on this machine.")
        print("  [P30] is filed on a quoted ~17x. This is the number that decides whether")
        print("  replacing the network's engine is worth a C++ dependency on every volunteer PC.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
