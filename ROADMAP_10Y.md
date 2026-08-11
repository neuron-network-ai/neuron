# NEURON — The Ten-Year Roadmap

**Written 2026-08-10.** Companion to `ROADMAP.md` (the session plan, which build rule 2 says
never to modify), `SCALING.md` (the infrastructure ladder), `TOKENOMICS.md` (the economics) and
`PROBLEMS.md` (what has actually gone wrong). This document is the long-horizon strategy: what
to build, in what order, and — more usefully — what to refuse to build until a specific fact is
known.

---

## How to read this

A ten-year plan for a project at this stage is honest for about eighteen months and increasingly
fictional after that. Two facts decide almost everything downstream and **neither is known yet**:

1. **Can a volunteer CPU network serve a useful model at a tolerable speed?** ([P30])
2. **Will strangers actually run the agent?** As of today, **no stranger's machine has ever
   reached the coordinator** ([P24]) — every node that has ever run is the founder's.

So the horizons below get deliberately less specific as they get further out. Horizon 1 is a
task list. Horizon 4 is a set of *scenarios* with the trigger that selects between them. Anything
more precise than that at year five would be decoration.

The one fixed commitment is the emission schedule: `TOKENOMICS.md` allocates 60% of a fixed
1,000,000,000 NRN supply to node rewards **over ten years, halving every two**. That is already a
ten-year promise, and it is the spine this roadmap hangs on. Era 0 is paying out now
(`EMISSION_BASE_NRN_PER_HOUR = 1.0`), which means the clock is already running whether or not
anyone is here to earn it.

---

## Where the project actually is (2026-08-10)

Stated plainly, because every plan below depends on the starting point being honest.

| | state |
|---|---|
| Nodes online | **2**, both the founder's |
| Strangers ever registered | **0** ([P24] — two IP addresses have ever hit `/node/register`, both ours) |
| Model served | Qwen2.5-1.5B-Instruct, 28 layers, fp32 |
| Chain | 2 stages `[[0,9],[10,27]]`, routable, healthy (recovered today, [P37]) |
| Measured speed | 8–11 ms/layer → **~2.6 tok/s** best-path ([P34]) |
| Engine on the network path | **PyTorch fp32** — the ~17× llama.cpp engine is unreachable ([P30]) |
| GPU path | **has never executed** ([P31]); shipped torch is `2.4.1+cpu` |
| Users able to buy NRN | **none** — faucet is one-time, no top-up route ([P29]) |
| Node earnings reaching a human | **impossible** — `nodes` has no owner column ([P29]) |
| Total requests served, all time | 78 |

The engineering is far ahead of the adoption. That asymmetry is the single most important fact
in this document, and most of Horizon 1 exists to correct it.

---

## What the outside world says (research, 2026-08-10)

Four findings, each of which changes a decision already recorded in this repo.

### 1. llama.cpp's RPC backend is LAN technology. [P30] route 1 is the wrong route.

The RPC backend **serialises each tensor operation** and ships it over TCP to a `rpc-server`
process, which runs the op and returns the result. Every guide describes it on a local switch —
"a $90 2.5 GbE switch unlocks 70B-class models" — and the official docs warn it is **not secure
by default and must never be exposed to the open internet**.

Per-*operation* round trips are fine at 2.5 Gb/s and 0.1 ms. NEURON's links are home broadband:
pavilion measured **3.9 MB/s** during today's slice download, and Tailscale ping to it swings
44–148 ms. A per-op protocol over that is not slow, it is non-functional.

**Consequence:** [P30] lists three routes. Route 1 (a llama-cpp-python build exposing
`rpc_servers`) should be **struck** — it would import a LAN protocol into a WAN product. Route 2
(embed ggml, keep NEURON's own layer-range wire protocol) is confirmed as the correct one, and
now for a stronger reason than "it owns the right layer": NEURON's protocol — **one activation
per layer-range hop** — is already the WAN-appropriate design, and llama.cpp's is not. The thing
worth borrowing is the *kernel*, not the transport.

### 2. Tensor parallelism needs microseconds. The "cube" cannot have that axis.

`exo` implements both pipeline and tensor parallelism across consumer devices — but achieves
usable tensor parallelism only over **Thunderbolt 5 RDMA at 1–2 μs**, against 100+ μs for
ordinary network protocols. Their own framing: that ~99% latency reduction is what "transforms
tensor parallelism from theoretically possible to practically usable."

**Consequence:** splitting *within* a layer across volunteer machines is permanently unavailable
to NEURON. The grid is **depth (stages) × width (replicas) × tier (model)**. Depth is capped at
3 by latency; width is the only axis that grows safely; tier is the last one to build.

### 3. Speculative decoding does not rescue a WAN pipeline.

Worth recording as a **negative result**, because it looks like an obvious shortcut and is not.
SpecPipe (arXiv 2504.04104) combines speculative decoding with pipeline parallelism to fill
pipeline bubbles — and is explicitly optimised for InfiniBand/NVLink data centres. Its
requirements are low inter-stage latency and roughly *homogeneous* stage capacity. NEURON has
neither. Do not spend a session on it.

### 4. Petals is alive, and it is a GPU network.

Actively maintained, serving Llama 3.1 405B, Mixtral 8x22B, Falcon 180B, BLOOM 176B, at up to
**6 tok/s for a 70B**. That is the honest benchmark for "distributed inference over the
internet, done well."

**Consequence:** Petals has proven the *mechanism* NEURON is also using, on GPUs. NEURON's
distinct claim is not the pipeline — it is **CPU-only, one-click, no staking, no GPU**. That
claim is what must be proven, and it is exactly what has not been proven yet. If NEURON ends up
requiring a GPU, it has re-implemented Petals with a worse swarm.

### 5. Model families have moved on

The tier ladder is pinned to **Qwen2.5**. The current small-model default is the **Qwen3** family
(Apache 2.0, 128K+ context, strong 4B class), with **SmolLM3-3B** notable for beating
Llama-3.2-3B and Qwen2.5-3B at its size. Not urgent — but a **3B tier between 1.5B and 7B** is
now a real option, and it is a much better fit for 8 GB machines than 7B is.

---

## The three questions that gate everything

Every phase below is really a bet on one of these. State them plainly so a future session can
tell whether the bet paid.

**Q1 — Speed.** Can NEURON serve a model people want at a speed people tolerate?
*Measured today:* 2.6 tok/s on 1.5B. *Threshold:* ~5 tok/s is comfortable reading; below ~2 is
abandonment. 7B on today's fleet lands near **0.8 tok/s** — a downgrade, not an upgrade.
*Resolved by:* fp16 measurement (weeks), then the engine (months), then GPU (conditional).

**Q2 — Economics.** Can a volunteer earn something they can spend, and a user buy something they
want? *Today: no, in both directions* ([P29]). Everything about tiered pricing, NRN, and the
blockchain is downstream of this, and it is a **missing database column**, not a hard problem.

**Q3 — Adoption.** Will strangers run it? *Zero evidence either way after a year of building.*
This is the riskiest of the three because it cannot be resolved by writing code, and the project
has spent most of its effort on the one of the three that can.

---

## Horizon 1 — Next 6 months: make the floor trustworthy, and get one stranger

The goal of this horizon is **not** more capability. It is to make the existing capability
truthful, and then to find out whether anyone wants it.

### H1.1 Correctness floor (weeks, small code)

- **Route on what a node HOLDS, not what it is ASSIGNED.** The [P37] structural fix. Today
  `assigned_layers` is both the instruction and the fact; when they diverge the network routes
  into a node that cannot serve and then blames it. The `holds` field now exists on the wire —
  make the router prefer it. This subsumes "auto-repair never tells the node": a node that has
  not caught up is simply skipped, instead of being flagged.
- **Wire auto-repair into the migration handshake.** Tier migration already does remote reloads
  correctly (prepare → download → ready → cut over). Auto-repair writes the DB and tells nobody.
  One should use the other.
- **Ship the pending halves.** The coordinator side of [P37] (`standing` in the attest reply,
  `reputation-reset`) and the agent side (`holds`, typed `range_mismatch` refusal) are written
  and undeployed. Agent 0.20.
- **Depth 2, not depth 3, for small fleets.** `canonical_assignment` does
  `n_stages = min(max_stages - 1, len(rest))` — it maximises stages before it makes a replica.
  At 3 machines that is 12% chain availability against 37% for 2 stages + 1 replica, and one
  more network hop. `PIPELINE_STAGES` should be a ceiling, not a target.
- **Expose the model pin.** `/network/model` does not report whether the capacity ladder is live
  or overridden by an operator pin. That is invisible state of exactly the kind [P33] was about.

### H1.2 Economics floor ([P29] — blocking everything commercial)

- **`nodes.owner_wallet`.** Prove ownership the way `/node/{id}/payout-address` already does —
  node token plus a signature. This is the missing relationship.
- **A route from node earnings to a user wallet**, or a recurring allowance. Today a user is
  dead after ~158 messages with no action available, while availability emission accrues NRN to
  node rows no human can reach. **Paying volunteers in something they cannot spend is worse than
  not paying them.**
- Only then: per-model pricing. It is the right idea and it is worthless before the above.

### H1.3 Speed, cheaply and measured

- **Measure fp16 end to end**: quality, tok/s, memory. `NEURON_WEIGHT_DTYPE` defaults to fp32
  and nothing sets it. If decode is bandwidth-bound, halving the bytes should approach a 2×
  speedup *and* halve `gb_per_layer` — one change, both problems.
  **Hold it as a hypothesis:** `cast_linears` converts each weight at forward time and its own
  comment says the cast is "amortised across a whole batch" — decode is batch 1, so there is
  nothing to amortise. Hours to measure, and the answer swings two decisions.
- **Do not migrate to 7B before this resolves.** On today's fleet 7B is ~0.8 tok/s: better
  answers, arriving three times slower. Consider a **3B tier** instead — it fits 8 GB machines
  comfortably and the 2026 3B class is strong.

### H1.4 The GPU path, if and only if a card arrives

The 2026-08-08 decision was to rent GPU time because no project machine has a card. If a
GPU-holding volunteer joins, that blocker dissolves — and **that machine's first job is
verification, not serving.** In the order already recorded, which is not arbitrary:
4a on CUDA → 4b (the one-line device move) → reconcile `selftest_shard.py` (build rule 6 is
currently *unsatisfiable* on a GPU box) → `GPU_EXECUTION` back on → **packaging last**.

Packaging is its own project: ~2.4 GB of CUDA torch against a 207 MB installer, a 600 s
timeout, no resume, no disk check, no rollback, and one installer means **every volunteer pays
that download, including the ones with no GPU**.

### H1.5 The milestone that actually matters

**One stranger, installing from the public installer, registering, passing proof-of-compute, and
earning — without the founder touching their machine.** `ROADMAP.md` says the project is "not
finished until Session 12 (first stranger node)". That is still true, and it is the only item in
Horizon 1 that tests Q3.

Everything else here is preparation for being worth installing. If this milestone keeps slipping
while the engineering advances, that is the signal to stop building and start recruiting.

**Exit criteria for Horizon 1:** possession-based routing live · a volunteer can spend what they
earn · fp16 answered yes or no · ≥1 stranger node earning · chain stays routable unattended for
30 days.

---

## Horizon 2 — 6 to 24 months: the engine, and the first 50 nodes

Conditional on Horizon 1's exit criteria. If strangers did not come, **do not build this** — go
to the Horizon 4 pivot section instead.

### H2.1 The engine ([P30] route 2)

The largest single piece of work in this roadmap, and the only one that changes the 200B story.
Embed ggml/llama.cpp as a **kernel library**, driven by NEURON's own layer-range wire protocol.
Explicitly **not** llama.cpp's RPC transport (finding 1). This also removes Python and PyTorch
from volunteer machines, which shrinks the installer and the attack surface at the same time.

Do not write another matmul. This repo has measured a hand-written AVX2 int8 kernel at **1.44×**
against llama.cpp's ~17×, on the same CPU, in the same language.

Gate: attempt this only after fp16 is measured. If fp16 delivers 1.5–2×, the urgency drops and
the sequencing changes.

### H2.2 Quality-preserving quantization

Logged since 2026-07-24 and still owed. int8 measured **3.46× faster** and **broke quality**
([P9]). GPTQ/AWQ or GGUF via the new engine, with a real eval, not a vibe check.

### H2.3 Connectivity off the single relay

`SCALING.md`'s ladder, in its stated order: single-port relay multiplexing (removes the ~100
port cap without opening another firewall port) → NAT hole-punching (STUN/ICE) so direct P2P is
the common case → relays become the rare fallback, and there are many.

### H2.4 Coordinator redundancy

One free micro-VM is the brain, the router, the ledger and the relay. Regional stateless
instances behind a load balancer, replicated DB. This is a prerequisite for taking anyone's
money or trust, and it is a known single point of failure today.

### H2.5 Multi-model, at the right fleet size

Two full chains with replicas is **~12–14 machines**. Below that, partitioning starves both
models. When the fleet supports it: a node's identity becomes `(model, layer_start, layer_end)`,
coverage and routability become per-model, `reported_model_id` becomes load-bearing rather than
diagnostic, and replica targets become per-model (a 7B replica costs 5× a 1.5B one — a flat
`EMISSION_TARGET_REPLICAS = 3` is wrong across tiers).

**The middle path, available much earlier:** the network serves one model distributed, while
capable machines serve bigger models **locally** via `local_gguf` at ~17×. That delivers user
model-choice with no new distributed machinery, and maps cleanly onto pricing — local is free,
distributed is metered.

### H2.6 NRN on-chain

Already gated at **50 nodes / 500 MAU** (NEURON Chain, not Polygon; contract prepped, deployed
nowhere). Keep that gate. A token before a network is the failure mode this project has
explicitly refused, and refusing it is a strength.

**Exit criteria for Horizon 2:** ≥50 nodes with ≥25 not the founder's · a 7B-class model at
≥5 tok/s · no single machine whose loss takes the network down · a volunteer has withdrawn
earnings.

---

## Horizon 3 — 2 to 5 years: decentralise the brain

Only meaningful if Horizon 2 produced a real swarm. The shape is already written in
`SCALING.md`; this is the timeline for it.

- **DHT discovery.** Nodes find each other and assemble pipelines with no central server —
  Kademlia, as Petals uses. The coordinator's remaining job becomes matching a request to a
  nearby healthy pipeline, and eventually that dissolves too.
- **Many small pipelines, not one big one.** Hundreds of independent 3–8 node chains running in
  parallel. Aggregate throughput scales ~linearly with node count — that is the whole thesis
  ([P1], [P8]). Single-user speed does not improve; capacity does. Say this out loud in the
  product, because users will otherwise expect the opposite.
- **Ledger decentralisation.** Follows the DHT, not the other way round.
- **Regional relay fabric**, run by volunteers, paid from emission.
- **Reputation without a central verifier.** Peer attestation exists (`PEER_VERIFY_QUORUM`) and
  is the seed of this. [P37] is the cautionary tale: any global, cumulative, monotonic verdict
  needs an exit, or a bug becomes a permanent sentence.

At this horizon the emission schedule is in **era 2–3** (0.25–0.125 NRN/device-hour). If the
network is not large by then, the halving does the arguing: late volunteers earn a fraction of
what early ones did, and recruitment gets harder every two years. **The economics assume growth
in the first four years or they do not work.** That is worth confronting now rather than in
year five.

---

## Horizon 4 — 5 to 10 years: three scenarios, not one plan

By year five, Q1 and Q3 are answered. Which of these is true determines everything, and no
amount of planning today substitutes for the measurement.

### Scenario A — The thesis holds
CPU volunteers at scale, aggregate capacity is real, NRN has value because the network does.
Then: 200B-class serving across ~30–80 CPU machines per pipeline, thousands of pipelines, and an
on-chain ledger.

**Note what this scenario does NOT restore.** [P16] already retracted the green claim:
consumer-CPU inference costs **more** energy per token than a datacenter GPU, and the ~0.1 W
figure described *idle*, not inference. The defensible claims are **no new hardware,
sovereignty, and privacy by architecture** — and `ROADMAP.md:20` still reads "No new power draw.
Green AI by design," which is the retracted version. Fixing that line is a Horizon 1 task, not a
year-five one: it is the pitch a stranger reads before deciding whether to install.

What *is* worth measuring, and never has been, is the honest version: **watt-hours per request**
(`TOKENOMICS.md` §11.5 already commits to publishing it). Publishing a number that loses on one
axis and wins on another is a stronger position than a slogan that loses quietly.

### Scenario B — GPUs win, and that is fine
The CPU path stays too slow for models people want, but the swarm mechanism works. NEURON
becomes a GPU volunteer network — which is Petals with better onboarding, economics, and a
one-click installer. **That is a legitimate product**, and the honest version of it drops the
"no GPU required" claim rather than quietly keeping it in the README.

### Scenario C — Distributed inference loses to local
Small models keep improving faster than distributed inference gets faster (a 3B in 2028 beating
a 70B of 2025 is not a wild extrapolation). The distributed network becomes the *fallback* for
machines that cannot run local, and NEURON's centre of gravity moves to the one-click local
agent, federated privately across a household or an organisation.

**The trigger that selects between them** is a single measurement taken every six months:
*tokens per second, for the largest model the network can serve, against the best model that
fits on a median volunteer's own machine.* When the second number wins consistently, C is true —
and pretending otherwise wastes years.

---

## What to refuse to build

Recorded because each of these is tempting and each would cost a session or more.

- **Tensor parallelism / a true compute cube.** Needs microsecond interconnect (finding 2).
- **Speculative decoding across WAN pipeline stages.** Explicitly data-centre technology
  (finding 3).
- **llama.cpp's RPC transport.** LAN-only, insecure by design over the internet (finding 1).
- **Another hand-written kernel.** Measured 1.44× against llama.cpp's ~17×.
- **A token launch before a network.** Gate stands at 50 nodes / 500 MAU.
- **Any fix that requires touching a specific volunteer's machine.** [P32] option A was rejected
  for exactly this reason: it has a ceiling of "the number of machines one person can babysit."
  Today's pavilion recovery needed SSH into the founder's own house — that is a warning, not a
  precedent.
- **Deeper pipelines to use more machines.** More stages is slower, not bigger. Spend extra
  machines on width.

---

## The one-page version

| horizon | the bet | the proof it worked |
|---|---|---|
| **0–6 mo** | truthfulness and adoption, not capability | a stranger earns, unattended, for 30 days |
| **6–24 mo** | the engine and the first 50 nodes | 7B-class at ≥5 tok/s, no single point of failure |
| **2–5 yr** | decentralise the brain | the network survives the coordinator being switched off |
| **5–10 yr** | whichever scenario measurement selects | network tok/s beats the best model on a median volunteer's own machine |

And the rule that has not changed since Session 1, which the whole of Horizon 1 exists to serve:

> The network has no value without nodes. Nodes have no value without users. Users have no value
> without the network. **Build the agent. Get the first stranger. Everything else follows.**

A year of engineering later, the first stranger is still the open item. That is the finding this
roadmap is built around.

---

## Sources

- [llama.cpp RPC backend](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md) ·
  [distributed llama.cpp guide](https://sharedllm.org/blog/llama-cpp-rpc-distributed-inference.html)
- [Petals](https://github.com/bigscience-workshop/petals) ·
  [Petals paper (arXiv:2209.01188)](https://arxiv.org/abs/2209.01188) · [petals.dev](https://petals.dev/)
- [exo — distributed inference on consumer hardware](https://blog.exolabs.net/day-1/) ·
  [exo deep dive](https://medium.com/@leif.markthaler/deep-dive-exo-distributed-ai-inference-on-consumer-hardware-068e341d8e3c)
- [SpecPipe: speculative decoding with pipeline parallelism (arXiv:2504.04104)](https://arxiv.org/pdf/2504.04104)
- [Best small language models 2026](https://www.bentoml.com/blog/the-best-open-source-small-language-models) ·
  [SLM guide 2026](https://localaimaster.com/blog/small-language-models-guide-2026)
