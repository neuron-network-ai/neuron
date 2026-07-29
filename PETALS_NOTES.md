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
| 8-bit weights | ❌ fp32 (`common.DTYPE`) — ~7× the download, 6.7× slower than Q4 |
| Compressed activations between nodes | ❌ `common.send_msg` does `torch.save` of raw fp32 tensors over TCP |

### Gaps in impact order

1. **Quantize the network path.** 70B fp32 = 282 GB total / 14 GB per node at 20 nodes.
   At 4-bit that is 40 GB total / **2 GB per node** — the difference between recruiting
   volunteers and not. The local engine (`engine/local_gguf.py`) already proves Q4_K_M keeps
   quality at 6.7× speed; the pipeline never got the same treatment.
2. **Compress activations on the wire.** Petals halves bandwidth here. NEURON pickles
   full-precision tensors once per token per hop — on home connections this plausibly dominates
   latency, and it is currently unmeasured.
3. **Junction caching.** Without it one flaky laptop kills a long generation. Petals treats
   this as essential, not an optimization.
4. **Latency-aware routing.** `random.choice` will pick a node on another continent as readily
   as one next door, and single-token inference is latency-bound.

### What NEURON is NOT trying to invent
Per `SCALING.md`: the differentiator is **packaging** — 1-click, no-GPU, no crypto-staking,
earn-while-idle — *not* new distributed-systems primitives. Petals proves the networking works
at 100B+. Lean on proven libraries (`libp2p`, `hivemind`) rather than rebuilding them.
