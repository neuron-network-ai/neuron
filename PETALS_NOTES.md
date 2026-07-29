# Petals — extracted architecture, and what NEURON is missing

Source: [Petals: Collaborative Inference and Fine-tuning of Large Models](https://aclanthology.org/2023.acl-demo.54.pdf)
(ACL 2023 demo) · [arXiv:2209.01188](https://arxiv.org/pdf/2209.01188) ·
[github.com/bigscience-workshop/petals](https://github.com/bigscience-workshop/petals)

Extracted 2026-07-29. `SCALING.md` already names Petals as the proven reference; this is the
concrete mechanism list, so the design isn't re-derived from guesswork.

---

## The five mechanisms

### 1. Placement — servers pick the weakest link
Each server announces its active blocks to a Kademlia DHT along with its measured throughput.
A joining server chooses the **contiguous** interval whose blocks have the *lowest* total
throughput, thereby removing the current bottleneck.

- `block throughput` = sum of throughputs of all servers hosting that block
- `server throughput` = **min(network, compute)** throughput, measured empirically *before*
  the server joins
- **Contiguous intervals only** — "hosting disjointed blocks would harm the inference latency"
- All nodes periodically re-check whether rebalancing would improve total throughput, and
  switch layers if so; this is also what closes gaps when peers serving certain blocks leave

### 2. Client-side routing — latency-aware beam search
> "clients have to ping nearby servers to measure latency and then find the path with minimal
> time via beam search"

Single-token generation is dominated by **network latency**, not compute. On server failure the
client drops it from consideration and re-runs routing.

### 3. Fault tolerance — partial restart via junction caching
Servers hold attention K/V for their own blocks. Critically, the **client caches the
intermediate activations at every "junction" between servers** (what it received from the
previous server and sent to the next).

When a server fails, the client finds another server holding the same blocks and **re-sends
only that junction's cached activation**. No restart of the whole generation. The paper is
explicit that naive full restart scales badly: longer sequences restart more often, and more
participants means higher chance any one fails.

### 4. Two separate quantizations
- **Weights — 8-bit mixed matrix decomposition** (LLM.int8, Dettmers et al. 2022a): ~0.1% of
  values kept as 16-bit outliers, ~99.9% at 8-bit. Roughly halves memory vs 16-bit.
  Quality cost is negligible (BLOOM-176B avg 70.3 at 8-bit vs 70.1 at 16-bit).
  Speed cost ~5% at batch size 1, negligible at larger batches.
- **Activations on the wire — dynamic blockwise quantization** (Dettmers et al. 2022b), applied
  to hidden states *before* pipeline-parallel communication. **Halves bandwidth requirements**
  with no noticeable effect on generation quality.

### 5. Reference numbers
| | |
|---|---|
| BLOOM-176B at 16-bit | 352 GB ⇒ ~44 nodes at 8 GB each |
| Client RAM | ≥12 GB (mostly the 3.6B embedding params) |
| Bandwidth | ≥25 Mbit/s bidirectional recommended |
| Throughput (8×A100, batch 1) | 3.95 tok/s at 8-bit vs 4.18 at 16-bit |

Infrastructure: the `hivemind` library plus custom fault-tolerant algorithms. Users interact
through `model.inference_session()`, which forms server chains, holds cache and recovers from
failures transparently.

---

## NEURON vs Petals

| Petals mechanism | NEURON today |
|---|---|
| Contiguous interval placement | ✅ `router.suggest_placement` — but balances by **replica count**, not measured throughput |
| Periodic rebalancing / gap closing | ✅ `migration.self_heal` |
| DHT discovery | ➖ central coordinator. Fine for Phase 1–2; `SCALING.md` plans DHT for Phase 3 |
| Latency-aware beam-search routing | ❌ `router.build_chain` uses `random.choice` among replicas |
| Client-side junction cache | ❌ **a node failing mid-generation kills the whole request** |
| 8-bit weights | ❌ bf16 on disk, **fp32 resident** — see the correction below |
| Compressed activations between nodes | ✅ **done 2026-07-29** — `wire_codec.py`, and it needed more than Petals' scheme |

### Correction to the weight-size claim above
The first version of this file said the download was fp32 and "~7× the download". Checked
against the actual bytes: HuggingFace ships Qwen2.5-1.5B as **BF16** (3.09 GB / 1.54 B params
= 2.00 bytes/param), and `slice_downloader.py` copies safetensors byte ranges **verbatim** —
so the download was never fp32. The fp32 is `load_slice_model`, which upcasts on load. So:

| | download | resident |
|---|---|---|
| today | 2.00 B/param | 4.00 B/param |
| Q4_K_M | ~0.55 B/param | ~0.55 B/param |
| measured, one 9-layer middle slice | 0.84 GB | 1.68 GB |

For 70B over 20 nodes that is **7 GB downloaded and 14 GB resident per node**, against ~2 GB
for both at Q4. The 14 GB is the recruiting blocker — a 16 GB laptop has no 14 GB to give —
and it is a **RAM** problem, not a bandwidth one. Worth stating precisely, because the two
have completely different fixes.

### The obvious fix for weights is blocked, and it is worth knowing why now
`PROBLEMS.md` [P9] pencilled in "llama.cpp GGUF + its RPC backend" as the production-engine
pivot: quantized *and* distributed, exactly NEURON's shape. The RPC backend does work and is
actively maintained — but its own documentation says it is "currently in a proof-of-concept
development stage… fragile and insecure. **Never run the RPC server on an open network or in
a sensitive environment!**" An open network of strangers is precisely the disqualifying case,
so this is not a drop-in for the public path. It remains viable for a *trusted* cluster (a
LAN, or one operator's own machines). Decide it on that basis, not on "Petals-like, therefore
fine". → [llama.cpp/tools/rpc](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc)

### Gaps in impact order

1. ~~**Compress activations on the wire.**~~ **Done 2026-07-29** — 4.3× smaller, measured, and
   the interesting part is that **Petals' own scheme was not good enough here.** Blockwise int8
   without a rotation diverged from the fp32 answer on 1 of 3 prompts, because NEURON's real
   junction activations have one channel ~750× the median. Adding a Hadamard rotation at the
   transport layer (QuaRot's trick, no weight surgery, no calibration) cut the error ~7× at
   identical byte cost. Details in `wire_codec.py`; numbers in `PROBLEMS.md` [P20],
   reproducible with `bench_wire.py`. The same work removed a pickle-deserialisation RCE on
   the wire ([P19]) — the wire was carrying executable content, which no document had noticed.
2. **Quantize the weights.** The 14 GB-per-node figure above, not the download. Blocked on a
   method: the llama.cpp RPC route is disqualified for the open path (above), and naive int8
   is [P9]. Weight-only int4 with dequant-on-the-fly is the candidate worth measuring next —
   it cuts RAM even if it does not speed up the matmul, and RAM is the actual blocker.
3. **Junction caching.** Without it one flaky laptop kills a long generation. Petals treats
   this as essential, not an optimization.
4. **Latency-aware routing.** `random.choice` will pick a node on another continent as readily
   as one next door, and single-token inference is latency-bound.

### What NEURON is NOT trying to invent
Per `SCALING.md`: the differentiator is **packaging** — 1-click, no-GPU, no crypto-staking,
earn-while-idle — *not* new distributed-systems primitives. Petals proves the networking works
at 100B+. Lean on proven libraries (`libp2p`, `hivemind`) rather than rebuilding them.
