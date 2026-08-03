"""
measure_amb.py — Phase 1 of research/AMB_THEORY.md.

MEASUREMENT ONLY. This script implements nothing from the AMB protocol; it
measures whether the assumptions the protocol rests on are true of real
Qwen2.5-1.5B activations. Section 6 sets a decision gate on the output:

    accuracy < 40%  ->  the idea needs revision, stop
    accuracy > 60%  ->  proceed to Phase 2

Everything measured here comes from Section 5:

  5.1  correlation(a_N, a_N+1) at layers 9, 14, 18, 27
  5.2  momentum prediction accuracy at alpha 0.85/0.90/0.95, buffer depth 2/3/5
  5.3  output quality: does a predicted activation change the emitted token?
  5.4  empirical Lipschitz constant of the consuming layer
  5.5  per-token generation time (the denominator of the required buffer depth)

The layer indices are the NEURON pipeline's own wire boundaries: node_a holds
layers 0-9, node_c holds 10-18, node_b holds 19-27. So the output of layer 9
is what node_a puts on the wire, the output of layer 18 is what node_c puts on
the wire, and the output of layer 27 is node_b's. Layer 14 is a mid-slice probe
that never crosses the network -- it is here to show whether smoothness is a
property of the boundaries or of the residual stream generally.

Two things this script does that Section 5 does not ask for, because without
them the headline number cannot be read honestly:

  * A HOLD baseline (predict a_N+1 = a_N; momentum fixed at zero). The
    Section 5.2 metric divides by ||real||, and activations have a large
    component that barely moves between tokens, so ANY predictor that returns
    something in the right neighbourhood scores well. If momentum does not
    beat hold, the momentum term in Section 2.2 is decoration.

  * DELTA CAPTURE: 1 - ||pred - real|| / ||real - a_N||, the fraction of the
    actual token-to-token change the predictor explains. Zero means "no better
    than hold". Negative means "worse than doing nothing".

Usage:
  python tools/measure_amb.py                        # full run, 20 prompts
  python tools/measure_amb.py --tokens 16 --limit 4  # quick smoke run
  python tools/measure_amb.py --no-counterfactual    # skip 5.3/5.4 (the slow part)

Output: a printed report, plus --json <path> for the raw numbers.
"""

import argparse
import copy
import json
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# The wire boundaries (see module docstring). The value is the index of the
# layer whose OUTPUT is the activation under test.
BOUNDARY_LAYERS = [9, 14, 18, 27]

ALPHAS = [0.85, 0.90, 0.95]
BUFFER_DEPTHS = [2, 3, 5]
MAX_K = max(BUFFER_DEPTHS)

# Momentum needs a few real deltas before its prediction means anything. With
# alpha=0.95 the EMA is still 86% zero after 3 updates, so scoring the first
# tokens would measure the warmup, not the predictor.
WARMUP = 3

# Section 6: "20 diverse prompts (conversational, code, structured)". Section
# 4.3 predicts the categories will separate -- smooth prose best, code worse,
# structured worst -- so the report breaks the numbers out by category rather
# than reporting one average that hides it.
PROMPTS = [
    # --- conversational (7) ---
    ("conversational", "Why is the sky blue?"),
    ("conversational", "Explain what photosynthesis does, in plain language."),
    ("conversational", "What is the difference between weather and climate?"),
    ("conversational", "Tell me about the history of the port of Rotterdam."),
    ("conversational", "How does a bicycle stay upright when it is moving?"),
    ("conversational", "What should I look for when buying a second-hand laptop?"),
    ("conversational", "Describe the taste of a ripe mango to someone who has never had one."),
    # --- code (7) ---
    ("code", "Write a Python function that reverses a linked list."),
    ("code", "Write a Python function that returns the nth Fibonacci number iteratively."),
    ("code", "Show me a SQL query that finds the second highest salary in a table."),
    ("code", "Write a bash one-liner that finds the ten largest files under a directory."),
    ("code", "Write a Python class that implements a fixed-size ring buffer."),
    ("code", "Write a regular expression that matches an IPv4 address, and explain it."),
    ("code", "Write a Python function that merges two sorted lists into one sorted list."),
    # --- structured (6) ---
    ("structured", "Return a JSON object describing a book: title, author, year, isbn."),
    ("structured", "List the first 15 prime numbers, comma separated, nothing else."),
    ("structured", "Produce a CSV with columns name,age,city and four rows of sample data."),
    ("structured", "Write a YAML config for a web server with port, host, tls and log level."),
    ("structured", "Give me a markdown table comparing TCP and UDP across five properties."),
    ("structured", "Output the numbers 1 to 20 as a JSON array."),
]


# --------------------------------------------------------------------------- #
# model plumbing
# --------------------------------------------------------------------------- #
def load(model_id, dtype, threads):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    model.eval()
    n_layers = model.config.num_hidden_layers
    for layer in BOUNDARY_LAYERS:
        if layer >= n_layers:
            sys.exit(f"boundary layer {layer} does not exist: model has {n_layers} layers")
    return tok, model


def consumer_module(model, layer):
    """The module that eats the activation produced at `layer`.

    For an interior boundary that is the next decoder layer. For the last layer
    it is the final norm -- and note that HF's output_hidden_states does NOT
    expose the raw output of the last layer (its final entry is post-norm),
    which is why every capture in this script goes through hooks instead.
    """
    if layer + 1 < model.config.num_hidden_layers:
        return model.model.layers[layer + 1]
    return model.model.norm


def clone_cache(cache):
    """Snapshot the KV cache. A counterfactual forward must run against the K/V
    the REAL activations wrote for positions < t -- only the current token's
    activation is predicted. Cloning is cheap here (a few MB at these lengths)."""
    return copy.deepcopy(cache)


def last_vec(t):
    return t[0, -1, :].detach().to(torch.float32).clone()


class Capture:
    """Records, per boundary layer, the layer's own output and its consumer's
    output for one forward pass."""

    def __init__(self, model):
        self.model = model
        self.produced = {}
        self.consumed = {}
        self._handles = []

    def __enter__(self):
        for layer in BOUNDARY_LAYERS:
            self._handles.append(
                self.model.model.layers[layer].register_forward_hook(
                    self._mk(self.produced, layer)))
            self._handles.append(
                consumer_module(self.model, layer).register_forward_hook(
                    self._mk(self.consumed, layer)))
        return self

    def __exit__(self, *_):
        for h in self._handles:
            h.remove()
        self._handles = []

    @staticmethod
    def _mk(store, layer):
        def hook(_module, _args, output):
            store[layer] = last_vec(output[0] if isinstance(output, tuple) else output)
        return hook


def substitute_hook(vector):
    """Replace the current token's hidden state on the way INTO a module.

    This is exactly the failure semantics of Section 3.4: the node downstream of
    the dead node receives a predicted activation for this token, recomputes its
    own K/V from it, and attends to real K/V for every earlier token.
    """
    def hook(_module, args):
        hs = args[0].clone()
        hs[0, -1, :] = vector.to(hs.dtype)
        return (hs,) + tuple(args[1:])
    return hook


# --------------------------------------------------------------------------- #
# 5.1 / 5.2 -- offline, from the recorded activation sequence
# --------------------------------------------------------------------------- #
def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return float("nan")
    return float((a @ b) / denom)


def correlation_stats(seq):
    """5.1 -- correlation(a_N, a_N+1) over consecutive activations."""
    pear, cos = [], []
    for j in range(len(seq) - 1):
        pear.append(pearson(seq[j], seq[j + 1]))
        cos.append(float(torch.nn.functional.cosine_similarity(
            seq[j], seq[j + 1], dim=0)))
    return pear, cos


def momentum_series(seq, alpha):
    """The Section 2.2 state, replayed over a recorded activation sequence.
    Returns momentum[j] = the EMA available AFTER observing seq[j]."""
    mom = torch.zeros_like(seq[0])
    out = [mom.clone()]
    for j in range(1, len(seq)):
        mom = alpha * mom + (1.0 - alpha) * (seq[j] - seq[j - 1])
        out.append(mom.clone())
    return out


def prediction_stats(seq, alpha):
    """5.2 -- for every valid (origin j, lookahead k), score momentum and hold.

    accuracy      : Section 5.2's metric, 1 - ||pred-real|| / ||real||
    hold_accuracy : same metric for the do-nothing predictor
    delta_capture : 1 - ||pred-real|| / ||real-seq[j]||, the fraction of the
                    actual change the predictor explains. This is the number
                    that says whether momentum earns its place.
    """
    moms = momentum_series(seq, alpha)
    rows = []
    for j in range(WARMUP, len(seq) - 1):
        for k in range(1, MAX_K + 1):
            if j + k >= len(seq):
                break
            real = seq[j + k]
            pred = seq[j] + k * moms[j]
            hold = seq[j]
            rn = float(real.norm())
            change = float((real - hold).norm())
            err = float((pred - real).norm())
            rows.append({
                "k": k,
                "accuracy": 1.0 - err / rn,
                "hold_accuracy": 1.0 - float((hold - real).norm()) / rn,
                "delta_capture": (1.0 - err / change) if change > 0 else float("nan"),
                "cosine": float(torch.nn.functional.cosine_similarity(pred, real, dim=0)),
            })
    return rows


# --------------------------------------------------------------------------- #
# generation + counterfactual
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_prompt(model, tok, prompt, max_tokens, alpha, do_counterfactual):
    """Greedy-decode `prompt`, recording the boundary activation sequence, and
    at each step re-run that step with the boundary activation replaced by the
    momentum prediction (5.3/5.4)."""
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt")

    seqs = {layer: [] for layer in BOUNDARY_LAYERS}
    cf_rows = []
    step_times = []

    # Prefill. The whole prompt crosses the wire as one tensor; the activation
    # the predictor starts from is its last position.
    with Capture(model) as cap:
        out = model(ids, past_key_values=DynamicCache(), use_cache=True)
    for layer in BOUNDARY_LAYERS:
        seqs[layer].append(cap.produced[layer])
    cache = out.past_key_values
    nxt = int(out.logits[0, -1].argmax())

    moms = {layer: torch.zeros_like(seqs[layer][0]) for layer in BOUNDARY_LAYERS}

    for step in range(max_tokens):
        if nxt == tok.eos_token_id:
            break
        tok_in = torch.tensor([[nxt]])
        snapshot = clone_cache(cache) if do_counterfactual else None

        t0 = time.perf_counter()
        with Capture(model) as cap:
            out = model(tok_in, past_key_values=cache, use_cache=True)
        step_times.append(time.perf_counter() - t0)

        real_logits = out.logits[0, -1].to(torch.float32)
        real_tok = int(real_logits.argmax())
        cache = out.past_key_values

        # Predictions are formed from state available BEFORE this step, then
        # scored against the activation this step actually produced.
        if do_counterfactual and step >= WARMUP:
            for layer in BOUNDARY_LAYERS:
                real_act = cap.produced[layer]
                prev = seqs[layer][-1]
                for name, predicted in (("momentum", prev + moms[layer]),
                                        ("hold", prev.clone())):
                    cf = counterfactual_step(
                        model, tok_in, snapshot, layer, predicted)
                    err_in = float((predicted - real_act).norm())
                    err_out = float((cf["consumed"] - cap.consumed[layer]).norm())
                    cf_rows.append({
                        "predictor": name,
                        "layer": layer,
                        "input_error": err_in,
                        # 5.4: ||f(a) - f(b)|| / ||a - b|| in the one direction
                        # that matters operationally -- the prediction error.
                        "lipschitz": (err_out / err_in) if err_in > 0 else float("nan"),
                        "consumer_cosine": float(torch.nn.functional.cosine_similarity(
                            cf["consumed"], cap.consumed[layer], dim=0)),
                        "argmax_same": bool(cf["token"] == real_tok),
                        "kl": kl_divergence(real_logits, cf["logits"]),
                    })

        for layer in BOUNDARY_LAYERS:
            act = cap.produced[layer]
            moms[layer] = alpha * moms[layer] + (1.0 - alpha) * (act - seqs[layer][-1])
            seqs[layer].append(act)

        nxt = real_tok

    return seqs, cf_rows, step_times


@torch.no_grad()
def counterfactual_step(model, tok_in, snapshot, layer, predicted):
    """Re-run one decode step with `predicted` substituted for the real
    activation at `layer`. Runs against a copy of the pre-step KV cache so the
    real run is unaffected and the past stays real."""
    cache = clone_cache(snapshot)
    consumer = consumer_module(model, layer)
    handle = consumer.register_forward_pre_hook(substitute_hook(predicted))
    store = {}

    def grab(_m, _a, output):
        store["consumed"] = last_vec(output[0] if isinstance(output, tuple) else output)

    grab_handle = consumer.register_forward_hook(grab)
    try:
        out = model(tok_in, past_key_values=cache, use_cache=True)
    finally:
        handle.remove()
        grab_handle.remove()
    logits = out.logits[0, -1].to(torch.float32)
    return {"logits": logits, "token": int(logits.argmax()), "consumed": store["consumed"]}


def kl_divergence(real_logits, cf_logits):
    """KL(real || counterfactual) over the next-token distribution, in nats.
    A token can match on argmax while the distribution behind it has moved a
    long way; this catches that."""
    p = torch.log_softmax(real_logits, dim=-1)
    q = torch.log_softmax(cf_logits, dim=-1)
    return float((p.exp() * (p - q)).sum())


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def mean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return statistics.fmean(xs) if xs else float("nan")


def pct(x):
    return "  n/a " if x != x else f"{100 * x:6.1f}%"


def report(results, args):
    layers = BOUNDARY_LAYERS
    cats = sorted({r["category"] for r in results})

    print("\n" + "=" * 78)
    print("AMB PHASE 1 -- MEASUREMENT (research/AMB_THEORY.md Section 5)")
    print("=" * 78)
    print(f"model      : {args.model}")
    print(f"dtype      : {args.dtype}   threads: {args.threads}")
    print(f"prompts    : {len(results)}   max new tokens: {args.tokens}")
    print(f"boundaries : layers {layers}  (9=node_a->node_c, 18=node_c->node_b, "
          f"27=node_b->head; 14=mid-slice probe)")
    toks = [n for r in results for n in [len(r['seqs'][layers[0]]) - 1]]
    print(f"activations: {sum(toks)} decode steps recorded "
          f"({min(toks)}-{max(toks)} per prompt)")

    # ---- 5.1 ----
    print("\n--- 5.1  ACTIVATION CORRELATION  corr(a_N, a_N+1) " + "-" * 26)
    print("required > 0.50   expected > 0.70")
    print(f"{'layer':>6} {'pearson':>9} {'cosine':>9}   " +
          "  ".join(f"{c[:12]:>12}" for c in cats))
    for layer in layers:
        allp = [v for r in results for v in r["corr"][layer]["pearson"]]
        allc = [v for r in results for v in r["corr"][layer]["cosine"]]
        per = []
        for c in cats:
            per.append(mean([v for r in results if r["category"] == c
                             for v in r["corr"][layer]["pearson"]]))
        print(f"{layer:>6} {mean(allp):>9.4f} {mean(allc):>9.4f}   " +
              "  ".join(f"{v:>12.4f}" for v in per))

    # ---- 5.2 ----
    print("\n--- 5.2  MOMENTUM PREDICTION ACCURACY  1 - ||pred-real||/||real|| " + "-" * 9)
    print("required > 40%   expected > 60%   (gate: <40% stop, >60% proceed)")
    print("HOLD is the same metric for predicting a_N+1 = a_N. DELTA is the")
    print("fraction of the real token-to-token change momentum explains; <=0")
    print("means momentum adds nothing over HOLD.\n")
    for alpha in ALPHAS:
        print(f"  alpha = {alpha}")
        print(f"  {'layer':>6} " + " ".join(f"{'B='+str(b):>27}" for b in BUFFER_DEPTHS))
        print(f"  {'':>6} " + " ".join(
            f"{'acc':>8}{'hold':>9}{'delta':>10}" for _ in BUFFER_DEPTHS))
        for layer in layers:
            cells = []
            for depth in BUFFER_DEPTHS:
                rows = [row for r in results for row in r["pred"][alpha][layer]
                        if row["k"] <= depth]
                cells.append(f"{pct(mean([x['accuracy'] for x in rows])):>8}"
                             f"{pct(mean([x['hold_accuracy'] for x in rows])):>9}"
                             f"{pct(mean([x['delta_capture'] for x in rows])):>10}")
            print(f"  {layer:>6} " + " ".join(cells))
        print()

    print("  one-step-ahead (k=1) by category, alpha=%.2f:" % args.alpha)
    print(f"  {'layer':>6} " + " ".join(f"{c[:14]:>22}" for c in cats))
    print(f"  {'':>6} " + " ".join(f"{'acc':>7}{'hold':>7}{'delta':>8}" for _ in cats))
    for layer in layers:
        cells = []
        for c in cats:
            rows = [row for r in results if r["category"] == c
                    for row in r["pred"][args.alpha][layer] if row["k"] == 1]
            cells.append(f"{pct(mean([x['accuracy'] for x in rows])):>7}"
                         f"{pct(mean([x['hold_accuracy'] for x in rows])):>7}"
                         f"{pct(mean([x['delta_capture'] for x in rows])):>8}")
        print(f"  {layer:>6} " + " ".join(cells))

    # ---- 5.3 / 5.4 ----
    cf = [row for r in results for row in r["cf"]]
    if cf:
        print("\n--- 5.3  OUTPUT QUALITY UNDER PREDICTION ERROR " + "-" * 31)
        print(f"one-step-ahead substitution at alpha={args.alpha}; the question is")
        print("whether a predicted activation still produces the right token.\n")
        print(f"  {'layer':>6} {'predictor':>10} {'same token':>11} "
              f"{'consumer cos':>13} {'KL(real||cf)':>13}")
        for layer in layers:
            for name in ("momentum", "hold"):
                rows = [x for x in cf if x["layer"] == layer and x["predictor"] == name]
                if not rows:
                    continue
                print(f"  {layer:>6} {name:>10} "
                      f"{pct(mean([x['argmax_same'] for x in rows])):>11} "
                      f"{mean([x['consumer_cosine'] for x in rows]):>13.4f} "
                      f"{mean([x['kl'] for x in rows]):>13.4f}")

        print("\n  same-token rate by category (momentum):")
        print(f"  {'layer':>6} " + " ".join(f"{c[:14]:>14}" for c in cats))
        for layer in layers:
            cells = []
            for c in cats:
                rows = [x for r in results if r["category"] == c for x in r["cf"]
                        if x["layer"] == layer and x["predictor"] == "momentum"]
                cells.append(f"{pct(mean([x['argmax_same'] for x in rows])):>14}")
            print(f"  {layer:>6} " + " ".join(cells))

        print("\n--- 5.4  EMPIRICAL LIPSCHITZ CONSTANT  ||f(a)-f(b)||/||a-b|| " + "-" * 17)
        print("measured along the prediction-error direction, on the consuming module.\n")
        print(f"  {'layer':>6} {'consumer':>22} {'mean K':>9} {'max K':>9} "
              f"{'mean ||err_in||':>16}")
        for layer in layers:
            rows = [x for x in cf if x["layer"] == layer and x["predictor"] == "momentum"]
            if not rows:
                continue
            ks = [x["lipschitz"] for x in rows if x["lipschitz"] == x["lipschitz"]]
            name = ("layer %d" % (layer + 1)) if layer + 1 < 28 else "final norm"
            print(f"  {layer:>6} {name:>22} {mean(ks):>9.4f} {max(ks):>9.4f} "
                  f"{mean([x['input_error'] for x in rows]):>16.3f}")

    # ---- 5.5 ----
    times = [t for r in results for t in r["step_times"]]
    if times:
        print("\n--- 5.5  RECOVERY TIME / BUFFER DEPTH " + "-" * 40)
        tpt = mean(times)
        print(f"  single-machine decode step: {1000 * tpt:.1f} ms "
              f"({1 / tpt:.2f} tok/s, this box, batch 1, no network)")
        print("  B >= ceil(recovery_time / token_time). The recovery-time half needs")
        print("  the live 3-node network and is Phase 4 work; for reference:")
        for rt in (0.5, 1.0, 2.0, 5.0, 10.0):
            print(f"    recovery {rt:>5.1f}s  ->  B >= {int(-(-rt // tpt)):>3}")

    print("\n" + "=" * 78)


def verdict(results, args):
    """The Section 6 decision gate, read off the k=1 numbers at the two real
    wire boundaries (layers 9 and 18)."""
    wire = [9, 18]
    rows = [row for r in results for layer in wire
            for row in r["pred"][args.alpha][layer] if row["k"] == 1]
    acc = mean([x["accuracy"] for x in rows])
    hold = mean([x["hold_accuracy"] for x in rows])
    delta = mean([x["delta_capture"] for x in rows])
    print("DECISION GATE (Section 6, k=1 at wire boundaries 9 and 18, "
          f"alpha={args.alpha})")
    print(f"  momentum accuracy : {pct(acc)}")
    print(f"  hold accuracy     : {pct(hold)}")
    print(f"  delta capture     : {pct(delta)}")
    if acc < 0.40:
        print("  -> BELOW 40%: the idea needs revision. Do not proceed to Phase 2.")
    elif acc > 0.60:
        print("  -> ABOVE 60%: gate passed on the Section 5.2 metric.")
    else:
        print("  -> BETWEEN 40% and 60%: no verdict; Section 6 defines neither.")
    if delta <= 0:
        print("  -> WARNING: delta capture <= 0. Momentum does not beat predicting")
        print("     no change at all, so the gate is passing on the metric's")
        print("     denominator, not on the predictor.")
    print("=" * 78 + "\n")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--tokens", type=int, default=32, help="max new tokens per prompt")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N prompts")
    ap.add_argument("--alpha", type=float, default=0.90,
                    help="momentum decay used for the 5.3/5.4 substitution runs")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="float32 matches common.py's compute dtype")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-counterfactual", action="store_true",
                    help="skip 5.3/5.4 (they cost 8 extra forwards per token)")
    ap.add_argument("--json", default="", help="write raw measurements here")
    args = ap.parse_args()

    if args.threads <= 0:
        args.threads = max(1, (torch.get_num_threads() or 4))
    if args.alpha not in ALPHAS:
        ALPHAS.append(args.alpha)

    prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    print(f"loading {args.model} ({args.dtype}) ...", flush=True)
    t0 = time.time()
    tok, model = load(args.model, dtype, args.threads)
    print(f"loaded in {time.time() - t0:.1f}s "
          f"({model.config.num_hidden_layers} layers, hidden {model.config.hidden_size})")

    results = []
    for i, (category, prompt) in enumerate(prompts, 1):
        t0 = time.time()
        seqs, cf, step_times = run_prompt(
            model, tok, prompt, args.tokens, args.alpha, not args.no_counterfactual)
        rec = {
            "category": category,
            "prompt": prompt,
            "seqs": seqs,
            "cf": cf,
            "step_times": step_times,
            "corr": {},
            "pred": {a: {} for a in ALPHAS},
        }
        for layer in BOUNDARY_LAYERS:
            pear, cos = correlation_stats(seqs[layer])
            rec["corr"][layer] = {"pearson": pear, "cosine": cos}
            for alpha in ALPHAS:
                rec["pred"][alpha][layer] = prediction_stats(seqs[layer], alpha)
        results.append(rec)
        print(f"  [{i:>2}/{len(prompts)}] {category:<14} "
              f"{len(seqs[BOUNDARY_LAYERS[0]]) - 1:>3} tokens  "
              f"{time.time() - t0:>6.1f}s   {prompt[:44]}", flush=True)

    report(results, args)
    verdict(results, args)

    if args.json:
        blob = {
            "config": vars(args),
            "boundary_layers": BOUNDARY_LAYERS,
            "warmup": WARMUP,
            "prompts": [{
                "category": r["category"],
                "prompt": r["prompt"],
                "tokens": len(r["seqs"][BOUNDARY_LAYERS[0]]) - 1,
                "step_times": r["step_times"],
                "corr": {str(k): v for k, v in r["corr"].items()},
                "pred": {str(a): {str(k): v for k, v in per.items()}
                         for a, per in r["pred"].items()},
                "cf": r["cf"],
            } for r in results],
        }
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=1)
        print(f"raw measurements -> {args.json}")


if __name__ == "__main__":
    main()
