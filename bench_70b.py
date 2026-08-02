"""
bench_70b.py -- llama.cpp baseline on Llama 3.3 70B Q4_K_M, this machine.

Mirrors the llama-cli invocation:
    llama-cli -m <gguf> -p "Why is the sky blue" -n 100 --no-warmup

llama-cpp-python drives the same llama.cpp engine (no llama-cli.exe here).
Raw completion, not the chat template, to match `-p`.
"""
import argparse
import glob
import os
import sys
import time

DEFAULT_GLOB = r"C:\Users\optin\models\llama70b\**\*.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--prompt", default="Why is the sky blue")
    ap.add_argument("-n", "--max-tokens", type=int, default=100)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--ctx", type=int, default=2048)
    args = ap.parse_args()

    path = args.model
    if not path:
        hits = sorted(glob.glob(DEFAULT_GLOB, recursive=True))
        # a split GGUF is loaded by pointing at part 00001; llama.cpp finds the rest
        hits = [h for h in hits if "00002-of-" not in h and "00003-of-" not in h]
        if not hits:
            sys.exit(f"no gguf under {DEFAULT_GLOB}")
        path = hits[0]

    size_gb = sum(os.path.getsize(p) for p in
                  glob.glob(os.path.join(os.path.dirname(path), "*.gguf"))) / 1e9
    print(f"model   : {path}")
    print(f"on disk : {size_gb:.1f} GB")
    print(f"threads : {args.threads}   ctx: {args.ctx}   max_tokens: {args.max_tokens}")

    from llama_cpp import Llama

    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=args.ctx, n_threads=args.threads, verbose=False)
    load_s = time.time() - t0
    print(f"load    : {load_s:.1f} s\n")

    t0 = time.time()
    out = llm(args.prompt, max_tokens=args.max_tokens, temperature=0.0, echo=False)
    wall = time.time() - t0

    text = out["choices"][0]["text"]
    usage = out.get("usage", {})
    n_out = usage.get("completion_tokens", 0)
    n_in = usage.get("prompt_tokens", 0)

    print("ANSWER")
    print("-" * 66)
    print(text.strip())
    print("-" * 66)
    print(f"prompt tokens     : {n_in}")
    print(f"generated tokens  : {n_out}")
    print(f"wall (incl prefill): {wall:.2f} s")
    print(f"tok/s (total)     : {n_out / wall:.3f}")
    print(f"ms/token          : {1000 * wall / max(n_out, 1):.1f}")
    print(f"finish reason     : {out['choices'][0].get('finish_reason')}")


if __name__ == "__main__":
    main()
