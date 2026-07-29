"""
NEURON — bench_wire.py

What does the pipeline actually pay to move one activation between two nodes, and does
shrinking it change the answer?

Runs the real 3-stage split IN-PROCESS (no sockets, no second machine) and puts the shipped
`wire_codec` at all three junctions the pipeline crosses per token:

    node_a -> node_c   hidden after layer s1
    node_c -> node_b   hidden after layer s2
    node_b -> node_a   the normed hidden coming back for the lm_head

so quantization error compounds exactly as it does on the wire. Compares every codec's
output against the fp32 baseline token-for-token.

    python bench_wire.py                  # the shipped codecs
    python bench_wire.py --all            # + the rejected ones, to show why they were rejected

Measured 2026-07-29 (Qwen2.5-1.5B-Instruct, H=1536, split 9/9/10, 6 prompts x 48 tokens):

    codec                    B/msg    vs before  identical   max|dlogit|
    torch.save fp32 (was)    12508       1.00x       6/6         0.0000
    f32 (NRNW framing)       11355       1.10x       6/6         0.0000
    f16                       5723       2.19x       6/6         0.0069
    i8h                       2946       4.25x       6/6         0.2054

and the schemes NOT shipped (exploratory sweep, 3 prompts, so identity is out of 3):

    fp8 e4m3                  2786       4.43x       0/3            nan   <- overflows, see below
    int8 per-tensor           2792       4.42x       0/3        30.4636
    int8 blockwise-256        2812       4.39x       2/3         1.1270   <- Petals' scheme
    int8 blockwise-64         3014       4.09x       1/3         0.5866
    int4 blockwise-32         1577       7.82x       0/3         4.5185

Run it against a SECOND model before changing any default. Qwen2.5-0.5B (H=896, `--s1 8
--s2 16`) disagrees with the table above -- i8h scores 3/6 there, which is why
`wire_codec.preference()` gates i8h on hidden size:

    torch.save fp32 (was)     7815       1.00x       6/6         0.0000
    i8h                       1998       3.91x       3/6         0.5034
    f16                       3375       2.32x       6/6         0.0075
    f32                       6661       1.17x       6/6         0.0000

fp8 fails because e4m3 tops out at 448 and real hidden states here reach 6620 -- it
overflows to inf and the generation becomes NaN. int8 without a rotation fails because one
channel is ~750x the median, so the absmax scale is set by that channel and everything else
quantizes to nearly nothing. Rotating first (i8h) fixes both at the same byte cost. This is
[P9] restated on the wire: naive quantization does not fail loudly, it fails plausibly.
"""
import argparse
import statistics
import sys
import time

import torch

import common
import wire_codec

S1, S2 = 9, 18
PROMPTS = [
    "Explain how a rainbow forms.",
    "Write a Python function that reverses a linked list.",
    "What is the capital of Australia, and why is it not Sydney?",
    "Summarise the causes of the 1929 stock market crash.",
    "Convert 72 degrees Fahrenheit to Celsius and show the working.",
    "Name three ways to reduce latency in a distributed system.",
]


def _legacy_codec():
    """The pre-Session-21 wire: torch.save/torch.load of the whole message."""
    import io

    def enc(msg):
        buf = io.BytesIO()
        torch.save(msg, buf)
        return buf.getvalue()

    def dec(b):
        return torch.load(io.BytesIO(b), weights_only=True)
    return enc, dec


@torch.no_grad()
def generate(model, tok, prompt, codec, max_new, stats):
    """Greedy decode through the 3-stage split, encoding at each junction."""
    def hop(t):
        if codec is None:
            return t
        if codec == "legacy":
            enc, dec = _legacy_codec()
            blob = enc({"type": "act", "hidden": t})
        else:
            enc, dec = wire_codec.encode, wire_codec.decode
            blob = enc({"type": "act", "hidden": t}, codec)
        stats["bytes"] += len(blob)
        stats["msgs"] += 1
        return dec(blob)["hidden"]

    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids
    ca, cc, cb = common.new_cache(), common.new_cache(), common.new_cache()
    past, cur, out = 0, ids, []
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for step in range(max_new):
        h = hop(common.first_stage(model, S1, cur, ca, past))
        h = hop(common.mid_stage(model, S1, S2, h, cc, past))
        h = hop(common.last_stage(model, S2, h, cb, past))
        logits = common.apply_lm_head(model, h)
        if step == 0:
            stats["first_logits"] = logits.clone()
        past += cur.shape[1]
        nxt = int(torch.argmax(logits, dim=-1))
        if nxt in (tok.eos_token_id, eos):
            break
        out.append(nxt)
        cur = torch.tensor([[nxt]])
    return tok.decode(out, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--s1", type=int, default=None, help="first/middle layer boundary")
    ap.add_argument("--s2", type=int, default=None, help="middle/last layer boundary")
    ap.add_argument("--all", action="store_true",
                    help="also run the codecs that were measured and rejected")
    args = ap.parse_args()

    print(f"loading {common.MODEL_ID} fp32 ...", flush=True)
    tok, model = common.load_model()
    h = model.config.hidden_size
    global S1, S2
    n = model.config.num_hidden_layers
    S1 = args.s1 if args.s1 is not None else round(n / 3)
    S2 = args.s2 if args.s2 is not None else round(2 * n / 3)
    print(f"H={h}  layers={model.config.num_hidden_layers}  split {S1}/{S2 - S1}/"
          f"{model.config.num_hidden_layers - S2}\n", flush=True)

    runs = [("BASELINE fp32 (no wire)", None), ("torch.save fp32 (pre-S21)", "legacy")]
    runs += [(c, c) for c in wire_codec.CODECS]

    base_text, base_logits, base_bpm = {}, {}, None
    print(f"{'codec':26s} {'B/msg':>8s} {'vs today':>9s}  identical  max|dlogit|")
    for name, codec in runs:
        texts, deltas, agg = {}, [], {"bytes": 0, "msgs": 0}
        for p in PROMPTS:
            st = {"bytes": 0, "msgs": 0, "first_logits": None}
            texts[p] = generate(model, tok, p, codec, args.max_new, st)
            agg["bytes"] += st["bytes"]
            agg["msgs"] += st["msgs"]
            if codec is None:
                base_logits[p] = st["first_logits"]
            elif st["first_logits"] is not None:
                deltas.append((st["first_logits"] - base_logits[p]).abs().max().item())
        if codec is None:
            base_text = texts
            print(f"{name:26s} {'-':>8s} {'-':>9s}      {len(PROMPTS)}/{len(PROMPTS)}       0.0000")
            continue
        bpm = agg["bytes"] / agg["msgs"]
        if codec == "legacy":
            base_bpm = bpm
        same = sum(1 for p in PROMPTS if texts[p] == base_text[p])
        ratio = f"{base_bpm / bpm:.2f}x" if base_bpm else "-"
        print(f"{name:26s} {bpm:8.0f} {ratio:>9s}      {same}/{len(PROMPTS)}       "
              f"{max(deltas):.4f}", flush=True)
        if same < len(PROMPTS):
            for p in PROMPTS:
                if texts[p] != base_text[p]:
                    print(f"    diverged: {p!r}\n      fp32: {base_text[p][:110]}\n"
                          f"      {codec:4s}: {texts[p][:110]}")

    print("\nPer-token wire cost of one hop at this hidden size, and what it implies for a "
          "70B model\n(H=8192, 20 stages, one decode token crossing every stage):")
    for label, bpe in (("fp32 + torch.save (today)", 4.2), ("f16", 2.0), ("i8h", 1.02)):
        per_hop = 8192 * bpe
        print(f"  {label:26s} {per_hop/1024:6.1f} KB/hop   "
              f"{per_hop * 20 / 1e6:5.2f} MB per token across 20 stages")


if __name__ == "__main__":
    sys.exit(main())
