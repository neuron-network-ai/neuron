# NEURON — Scaling Architecture (prototype → worldwide)

**The question this answers:** the current setup (one cloud coordinator + one relay VM,
~100 relay ports) is fine for the first strangers — does it scale to a worldwide network
of millions of nodes? **No — and it isn't meant to.** You never build planet-scale infra
before the first 10 users. This doc is the honest plan for how NEURON gets from a 3-node
prototype to a global network, and where today's work fits.

Companion docs: `ROADMAP.md` (session plan), `PROBLEMS.md` (risks/decisions), `sessions.md`
(build log). This file = the long-horizon scaling design.

---

## Where we are today (the prototype ceiling)

Everything routes through **one free micro-VM**: the coordinator (registry/router/ledger)
and the reverse-tunnel relay (`relay.py`, ~100 public ports = ~100 relayed nodes). Nodes
that are on Tailscale talk peer-to-peer; NAT'd/stranger nodes reach each other via the relay.

- ✅ Correct for **Phase 1** (first strangers, dozens of nodes).
- ❌ Single point of failure + bottleneck; hard cap on relayed nodes; all relayed traffic
  funnels through one box in one region.

**This is expected.** The prototype exists to prove the mechanism and get the first stranger
earning. The scaling layers below are grown into as real demand appears.

---

## The three things that must change for scale

### 1. Connectivity — stop routing everything through the relay
Relaying-through-one-VM is a **fallback**, not the main path. At scale:
- Most node↔node traffic goes **direct, peer-to-peer**, via NAT hole-punching
  (STUN / ICE / WebRTC / `libp2p`). Direct connections succeed for the majority of home
  NATs.
- Relays become the **rare exception** (symmetric/hard NAT), and there are **many** relays
  worldwide (a relay fabric), not one. Connectivity load spreads across the network itself.
- Interim step before full P2P: single-port relay multiplexing (one port for all nodes,
  routed by node_id) or fold the relay into the coordinator's existing port 8001 over
  WebSocket — removes the per-port firewall limit entirely (never open another port).

### 2. Coordination — one coordinator can't be the brain of the planet
- **Near term:** a few **regional coordinator instances** behind a load balancer; stateless
  API + a replicated/sharded DB. Removes the single point of failure.
- **Real scale:** **decentralized peer discovery via a DHT** (distributed hash table, à la
  BitTorrent/Kademlia) — nodes find each other and assemble pipelines with **no central
  server**. This is where the blockchain/decentralization roadmap (S17) belongs: the ledger
  and discovery go on-chain / into the DHT rather than a single SQLite file.

### 3. Topology — many small pipelines, not one giant one
The key insight: **a million nodes ≠ a million-stage pipeline.** Deep pipelines are slower
(more hops) and fragile. Instead:
- The network is **hundreds of thousands of independent ~3–8-node pipelines running in
  parallel** (replicas of the model split across small groups).
- **Aggregate throughput scales ~linearly** with node count — that's the whole thesis
  (capacity, not single-user speed; see PROBLEMS.md [P1], [P8]).
- The coordinator's job becomes: **match each request to a nearby, healthy pipeline**
  (load balancing across the swarm) and **continuously re-form pipelines** as nodes
  join / sleep / leave.

---

## Hard problems that come with scale

- **Churn:** millions of volunteer machines constantly join/leave/sleep. Pipelines must
  re-form dynamically; requests reroute mid-flight; replication provides redundancy.
- **Latency-aware assembly:** pipelines should be built from network-close nodes (same
  region) — network is already a dominant per-token cost (PROBLEMS.md [P3]).
- **Trust / correctness at scale:** strangers' nodes could return garbage. Needs
  proof-of-compute + reputation (ROADMAP S16) so bad nodes don't poison results or steal NRN.
- **Model quality vs. size vs. speed:** bigger models need more nodes per pipeline and are
  slower (PROBLEMS.md [P6]) — a permanent three-way trade to manage per model.

---

## This is a solved model (not speculative)

**Petals** (BigScience) already runs 100B+ parameter models across volunteer machines
worldwide using exactly this shape: a **DHT for discovery + direct P2P connections + relay
fallback**, with hundreds of real nodes. So the networking at scale is proven engineering.
NEURON's differentiator is the **packaging** — 1-click, no-GPU, no crypto-staking, earn-while-
idle — **not** inventing new distributed-systems primitives. When we build the scale layer,
we lean on proven libraries (`libp2p`, `hivemind`) rather than from scratch.

---

## Phased plan

| Phase | Nodes | Connectivity | Coordination | Status |
|-------|-------|--------------|--------------|--------|
| **1 — Prototype (now, S12)** | 1–50 | Tailscale + single relay VM | one cloud coordinator | ✅ coordinator live; relay built |
| **2 — Growth** | 50–5,000 | single-port / port-8001 relay mux | a few regional coordinators, latency-aware assembly | planned |
| **3 — Scale** | 5k–millions | P2P hole-punching, relay fabric | DHT peer discovery (no central coordinator), on-chain ledger | future (leans on libp2p/hivemind) |

Rough mapping to `ROADMAP.md`: S12 = Phase 1 (first stranger); S16 (security/proof-of-compute)
+ S17 (on-chain NRN) + S18/S19 (launch + scale testing) = the Phase 2→3 groundwork.

---

## The one rule of sequencing

**Don't build the worldwide layer now.** It is real, substantial engineering (weeks–months)
and is the *wrong* investment before there are real users. Get the **first stranger earning**
(proves people will run it at all), watch how real nodes actually behave, and design the
scaling architecture from that evidence. Prototype → learn → scale, in that order.

---

## The consumer product & the engine keystone (llama.cpp)  *(noted 2026-07-25)*

Founder's product direction — the "normal person joins via an app" goal and the single
architectural decision it all depends on. **Captured for later; not scheduled as a numbered
session yet.**

### The goal
A non-technical person joins NEURON by clicking an **app** — not `python agent.py`, not editing
`config.json`, not Termux — and uses NEURON (chat) just as easily. This is the real front door for
strangers (ROADMAP S12) and the whole consumer story.

### One keystone unlocks everything: llama.cpp
Every consumer requirement converges on replacing the hand-rolled **PyTorch** engine with
**llama.cpp** (which the v2.0 plan doc already named — "Python + llama.cpp"):

| Requirement | Why llama.cpp is the answer |
|---|---|
| Phones charging (Android) | runs natively on ARM/Android; PyTorch does not |
| Idle GPUs | GPU via CUDA / Vulkan / Metal, built in |
| A tiny installable app | a few-MB binary, not 200MB+ of PyTorch |
| Fast inference (speed, [P1]) | optimized quantized kernels |
| Quality quantization ([P9]) | GGUF Q4_K_M / Q8_0 preserve quality |
| Split across machines | llama.cpp **RPC backend** distributes layers |

You **cannot mix engines in one pipeline** (llama.cpp-quantized activations don't cleanly hand off to
fp32 PyTorch nodes), so adding phones/GPUs/a-light-app is **one network-wide engine decision**, not a
per-node bolt-on. The current PyTorch system is a working *proof of concept*; the consumer product is
gated on this rebuild.

### GPU harvesting (highest-leverage; fits the current arch)
Idle consumer GPUs are the biggest **speed + big-model** unlock — GPU inference is 10–100× CPU. Unlike
Android, basic CUDA support is a *small* change to the current PyTorch stack (move model + tensors to
GPU), so it can even land before a full llama.cpp move. S14's auto-balancer is what makes a mixed
CPU+GPU network work (the GPU node auto-gets most layers). Caveats: VRAM-aware balancing; GPU-idle
detection (a gaming GPU is busy gaming); **green nuance — a GPU draws real power (200–400 W), so it's
"reuse existing hardware, no new datacenters" green, NOT "negligible energy" green.**

### Android / phones (the best green story)
Billions of phones charging overnight on WiFi = arguably the **purest green node** — charger already
drawing power, 1% of an efficient ARM CPU ≈ negligible marginal energy (far greener than GPUs). Needs
llama.cpp. Caveats: battery heat/wear (charging-only + WiFi-only + ~1% cap + thermal throttle); slice
must fit phone RAM (GGUF makes it fit); Android background limits (foreground service / Termux:Boot);
iOS forbids background compute → **Android-first**.

### Honest "1MB agent" framing (don't overclaim)
"1MB" is the **app**, not the total footprint:
- App (code + llama.cpp binary): **~1–10 MB** ✅
- Model slice (weights this node runs): **hundreds of MB → ~1 GB+**, streamed in chunks, cached on disk
- RAM while active: ≈ slice size + small overhead · CPU: 1–2% idle

Say *"a ~1MB app that streams only the slice it needs"* — never *"NEURON is 1MB total."* The layer-split
keeps each phone's slice small (a few layers, not the whole model) — which is exactly what makes phones
viable for huge models. PyTorch can't make an honest small-app claim at all; llama.cpp is what rescues it.

### Recommendation
Before committing to the rebuild, do a cheap **llama.cpp spike**: prove one GGUF model split across 2
machines via the RPC backend (and/or running on a phone). Decide with data, like the quantization spike.
This llama.cpp / consumer-app track is **cross-cutting** — schedule it deliberately; it is NOT a quick
session, and Android (S13) stays deferred until it exists.

---

### Engine deep-dive: adapt llama.cpp vs. build "neuron.cpp"  *(discussion 2026-07-25 — PARKED, needs dedicated time)*

Founder asked whether to write our own C++ engine ("neuron.cpp") for speed, motivated by the green
vision. **Conclusion of the discussion: do NOT write kernels from scratch — but a NEURON-specific engine
LAYER on top of llama.cpp's kernels is the legitimate path.** Parked for a dedicated session; recorded
so we can resume without re-deriving. Priority stays the first stranger, not the engine.

**Why not a from-scratch engine:** speed on CPU comes from quantization + SIMD/GPU kernels, which are a
commodity llama.cpp already nails (years of work, hundreds of contributors). A custom engine would be, at
best, *as fast, years later*. The green differentiation lives in NEURON's **network** (slicing +
coordinator + economics + proof-of-compute), NOT the kernels. Build only the differentiated layer.

**If/when we adopt llama.cpp, the right shape = a NEURON engine on `ggml`, not on `libllama`:**
- `ggml` = llama.cpp's low-level tensor+kernel library (the fast quantized CPU/GPU kernels — the commodity we want).
- `libllama` = the high-level layer that loads a *complete* GGUF and builds the *full* graph (this is what refuses "just layers 19–27").
- The **legitimate "neuron.cpp"** = a thin engine on ggml that (1) loads ONLY this node's slice tensors and (2) wires the Qwen2 decoder block (RMSNorm → GQA attention + RoPE → SwiGLU) for ONLY its layers — i.e. our current `first/mid/last_stage` re-expressed as a ggml graph, **cribbed from llama.cpp's own Qwen2 graph code.** Graph+loader work, NOT kernel/quant work.
- Founder's "give id" idea = already half-built: the coordinator's `slice-info` assigns each node its `layer_start/end`; we'd just emit **GGUF slices** from `slice_downloader` instead of safetensors slices.

**The two HARD, NEURON-specific problems (these are the real work; only we can do them):**
1. **Partial-model / slice loading** — solved by the ggml engine above (partial graph from slice tensors). Keeps the "no machine holds the whole model" property; stock llama.cpp RPC breaks it (its orchestrator loads the full model).
2. **Proof-of-compute is coupled to bit-exactness** — S16 PoC recomputes layers in torch and compares to ~1e-5. Under int4 + heterogeneous hardware, outputs legitimately spread, so the atol-based honest/cheat boundary must be re-derived or PoC redesigned. **Solving the slice does NOT solve the trust check.**

**Manageable problems (friction, not blockers):** Python↔C/C++ boundary + per-platform native builds
(Windows/Linux/ARM/Android NDK, code-signing/AV); llama.cpp **RPC backend is built for a trusted cluster,
not trustless strangers** (its `rpc-server` is an RCE surface — must sandbox/bridge to our coordinator +
relay + reputation); **all-or-nothing cutover** (engines can't mix in one pipeline → flag-day, or run a
parallel network and cut over); S14 rebalancing granularity; KV-cache mapping onto ggml.

**Cheaper speed path to test FIRST (zero C++):** **GPTQ / AWQ int4** inside the *existing PyTorch split*
— quality-preserving (unlike the naive int8 that broke output, [P9]), stays 100% Python, keeps the slice
property AND the current PoC. Downside: PyTorch CPU int4 isn't as fast as llama.cpp's hand-tuned kernels
→ *partial* speedup, not the full win. Two speed tiers to weigh:

| Path | Effort | Speed | Keeps slice + PoC? |
|---|---|---|---|
| GPTQ/AWQ int4 in the PyTorch split | low (Python) | partial | yes |
| ggml NEURON engine ("neuron.cpp") | high (C/ggml) | full + GPU + phones | slice yes; **PoC needs rework** |

**When we give this dedicated time, the spike must answer BOTH:** (a) how fast + how good is GPTQ/AWQ
int4 on the *current* split right now (might already be "fast enough" and buys months); (b) prove ONE
ggml slice-node runs only its layers with no node loading the full GGUF (the feasibility gate for the big
engine). Decide with numbers, not faith.
