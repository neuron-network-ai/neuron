# NEURON — A Volunteer Distributed Inference Network

**NEURON Labs · August 2026 · v0.17.1 (early alpha)**

Every figure in this document was measured on the three machines named in section 5, or is
stated as unbuilt. Where something is designed but not implemented, it says so.

---

## 1. Abstract

NEURON is a distributed AI inference network in which ordinary consumer computers each hold a
slice of a language model's transformer layers and run inference collectively. A model too
large for any single machine runs across several; no machine holds the whole model, and no
machine except the one the user is sitting at ever sees readable text — the intermediate nodes
receive only opaque numeric activations. The agent installs in one step on Windows, requires no
GPU, no port forwarding, no wallet and no staked cryptocurrency, and pays the operator in NRN
for the compute it contributes while the machine is otherwise idle. This matters because the
capability to run large language models is concentrating in a handful of companies who can read
what is sent to them, price it as they choose, and withdraw it. NEURON is an attempt to build
the same capability out of hardware that already exists and is already switched on. It is an
early alpha: the mechanism is proven end to end on real hardware, the network is currently a
handful of machines, and NRN has no cash value.

---

## 2. Problem

**Centralisation.** Large-model inference is effectively available from three companies, on
their terms and at prices they set. That is not a conspiracy — it is what the capital cost of
GPU clusters implies. But it means the most consequential software of the decade has three
gatekeepers.

**Energy and hardware.** Serving that demand means new datacenters and newly manufactured
accelerators, while the world already contains billions of general-purpose processors running
at a few percent of capacity, most of the time.

**Privacy.** A prompt sent to a hosted API is readable by the company receiving it. For legal,
medical, financial or personal use that is a real constraint, and it is not one the user can
verify away — it rests on a policy document.

**Access.** Hosted inference is metered, can be geographically restricted, refused, or revoked.
No version of it keeps working if the provider decides otherwise.

**Existing distributed projects don't close the gap.** Bittensor, Gensyn and io.net all assume
GPU hardware and, in the first two cases, crypto staked upfront and non-trivial operator setup.
Petals — the closest technical relative, and the proven reference for this architecture — runs
100B-class models across volunteer machines without a GPU, but has no incentive layer and no
one-click install. The unfilled space is the ordinary person with an ordinary laptop willing to
contribute if it costs them nothing and takes one click.

---

## 3. Solution

NEURON splits a transformer model by layer across volunteer machines. Each node loads only its
own layers, runs them on the activations arriving from the previous node, and passes the result
on. Concretely:

- **Consumer CPU, no GPU required.** The nodes measured in this paper are a 16-core Windows
  desktop, a 6-core Dell OptiPlex and a 4-core HP Pavilion.
- **One-click install on Windows.** `NEURON-Setup-0.17.1.exe`; the node registers itself, is
  assigned a layer range, downloads only that range, and starts serving. Linux and macOS are a
  source install today.
- **Earn while idle.** A resource guard gates availability, not raw CPU: the default `idle` mode
  yields above 15% system CPU, pauses when the user is active or on battery, and always pauses
  below 500 MB free RAM. Higher donation levels are an explicit operator choice.
- **Privacy as an architectural property, not a policy.** Only the driver — the machine holding
  the embedding and `lm_head`, which is the user's own — handles plaintext. Middle and last
  stages never touch a tokenizer; they cannot read what they compute. This is why the
  alternative design of "every node runs the whole model" was rejected: it would put readable
  prompts on volunteers' machines. The honest caveat is in section 6.
- **No new hardware.** Every node is a machine that already exists and is already powered.

---

## 4. Technical Architecture

```
   User
    │  1. POST /infer {prompt}
    ▼
 ┌─────────────┐   returns the chain (which nodes / which layers / where) + request_id
 │ Coordinator │   registry · health-checks · routing · NRN ledger · dashboard
 │ neuronnet   │   neuronnet.duckdns.org
 │ .duckdns.org│
 └─────────────┘
    │  2. the client runs the returned pipeline, activations over TCP:
    ▼
  node_a ─────────► node_c ─────────► node_b
  layers 0–9        layers 10–18      layers 19–27
  embed + lm_head   (middle relay)    + final norm
    ▲                                     │
    └────────── final hidden state ◄──────┘   node_a applies lm_head, picks the
    │                                          next token, and loops until done
    │  3. POST /infer/{id}/complete  → coordinator credits NRN to each node
    ▼
   User  (receives the generated text)
```

**Layer splitting.** `common.py` builds a model on the `meta` device and materialises only the
node's own layers from the safetensors shards. Roles differ only by extras — embedding and
`lm_head` on the first stage, the final norm on the last. The split is bit-exact against running
the whole model on one machine, verified by `selftest_shard.py`.

**Coordinator.** A FastAPI service over SQLite: node registry, health sweeps, chain assembly,
layer placement, model tiering, standing (probationary / verified / trusted / flagged) and the
NRN ledger. It holds no PyTorch and never computes inference itself.

**NAT relay.** Home machines cannot accept inbound TCP. A node makes only outbound connections
to a public relay, which exposes a port on its behalf and reverse-tunnels traffic to it — a
small self-hosted equivalent of ngrok. Registration with `behind_nat` assigns a relay port
automatically and the agent starts the tunnel from the coordinator's response, with no operator
steps. Liveness is checked by dialling the node's own public endpoint and completing a real
handshake, because a relay accepts TCP on a public port whether or not it can still reach the
node behind it.

**Wire codec.** Activations cross the network as a length-prefixed JSON header plus raw tensor
bytes — nothing executable. The `i8h` codec applies a Hadamard rotation before blockwise int8
quantisation; the rotation is orthogonal, so the sender rotates and the receiver rotates back,
with no weight surgery and no calibration. It exists because real junction activations measure
absmax 6620 with the worst channel roughly 750× the median, which an unrotated absmax scale
cannot survive. Codecs are negotiated per hop and degrade to the legacy format, and `i8h` is
offered only at hidden size ≥ 1536 — below that it drifts.

**Proof of compute.** A verifier sends a node a challenge input for its declared layer range,
runs the same layers locally, and compares. Honest work matches to about 1e-5; garbage or
echo-the-input cheating is off by 25 or more, so `atol=0.05` separates them cleanly. Last-stage
nodes are challenged directly; middle nodes via a probe mode that runs their layers in isolation.
The probe acknowledgement echoes the node's own actual range, so a node whose real range
disagrees with its registration fails loudly.

**Slice downloader.** A safetensors file opens with a JSON header listing every tensor's byte
range. The downloader fetches that header over HTTP Range, merges the spans for this node's
layers, downloads only those, and reassembles a smaller valid safetensors file. It is
shard-aware for multi-file models.

**Auto-placement and auto-balancing.** A joining node never picks layer numbers: the coordinator
fills the first coverage gap, or — if the chain is complete — places it as a replica of the last
segment, which is both verifiable and additive to throughput. Separately, each node self-measures
its per-layer time and the coordinator solves in closed form for the split that equalises stage
time, accounting for the fixed `lm_head` cost on the driver.

**KV cache and generation.** Each node keeps its own cache, keyed by the layer's native index,
and tracks token positions itself. Prefill runs the whole prompt; each decode step ships one
hidden-state vector per hop.

**GPU nodes.** A node detects an NVIDIA GPU (via `torch.cuda`, falling back to `nvidia-smi`)
and reports it at registration, and the pipeline resolves an execution device at load time,
moving its shard onto CUDA when one is present and falling back to CPU otherwise. Layer
assignment is then VRAM-aware: a GPU node is sized by its VRAM rather than its system RAM,
because that is where its weights sit — so 8 GB of VRAM outranks 4 GB of free system RAM, and
GPU-capable nodes take larger layer counts. Every interface stays CPU-side: the stage functions
take and return CPU tensors, so the wire codec, the relay and proof-of-compute are unchanged by
this. Two deliberate constraints: TF32 is disabled, because its ~1e-3 drift from CPU arithmetic
would eat into the tolerance proof-of-compute uses to separate honest work from cheating; and a
node also yields while its GPU is busy, so a game or a render is never competing with it. **This
path is written and unverified — see section 11.**

---

## 5. Real Results

All measured on the three machines below — a 16-core Windows desktop (63 GB), a 6-core Dell
OptiPlex (15 GB) and a 4-core HP Pavilion (11 GB) — running Qwen2.5-1.5B-Instruct in fp32.

| Measurement | Result | Where |
|---|---|---|
| Single machine, all 28 layers | ~3.2 tok/s | Session 3; 3.18 measured in the [P2] spike |
| 2 nodes, concurrent requests | **4.61 tok/s**, 2.06× pipeline overlap | Session 4 |
| 3 nodes, concurrent requests (N=8) | **6.16 tok/s**, **3.82× pipeline overlap** | Session 5 |
| 3 nodes, serial baseline | 1.64 tok/s | Session 5 |
| Wire codec `i8h` | 12,508 → **2,946 bytes/message = 4.25×**, 6/6 outputs identical to fp32 | Session 21 |
| Auto-verification of a new node | **254 ms** | Session 27 |
| Node reboot survival | **13 seconds** from boot to serving, nobody logged in | Session 26 |
| Sharded inference correctness | **bit-exact** vs. one machine (`selftest_shard.py`) | Sessions 1, 5 |
| Slice download per node | 1.40 GB / 0.84 / 0.84 vs. 3.09 GB full — network total 1× instead of 3× | Session 8 |
| Stranger onboarding, clean first run | 22 s from registration to earning-eligible, no human | Session 29 |

**Read the throughput numbers correctly.** Distribution scales *aggregate throughput* — more
simultaneous users — and *capacity* — models no single machine can hold. It does not make one
answer arrive faster: a serial pipeline is bounded by its slowest stage, and hops add latency.
Scaling is sub-linear here (3 nodes ≈ 1.9× one node) because the nodes are heterogeneous and
the driver carries the fixed head cost.

**Where the time actually goes.** In a fully relayed run with Tailscale disabled — every hop
across the public internet, which is the real stranger path — a 24-token answer took 45.5 s at
0.53 tok/s: 33.6 s (74%) wire, 11.1 s compute across all three nodes. The codec's 4.25× is the
obvious lever against that number and is **not yet deployed** to the remote nodes.

**For models that fit on one machine, distribution is the wrong tool.** The local llama.cpp
engine measures 27.9 tok/s on Qwen2.5-1.5B and 7.8 tok/s on Qwen2.5-7B (Q4_K_M, 16 cores).
Llama 3.3 70B Q4_K_M runs on that same machine at 0.62 tok/s — reachable and unusable, which is
precisely the case distribution exists for.

---

## 6. Security Model

**Proof of compute** (above) is the primitive. A node that returns garbage to farm NRN is
detected because the verifier recomputes the same layers.

**Reputation.** The coordinator tracks passes and failures per node. A node with at least 3
samples and a pass rate below 0.6 is flagged and excluded from routing, coverage and earning.
Verification is peer-driven: any verified node can be assigned a target, runs the same
challenge, and reports a verdict signed with its own token. Votes are keyed on
(verifier, target), so one machine voting repeatedly counts once; a quorum of distinct
agreements promotes a newcomer. An unreachable node records nothing — a failed attestation is
permanent, and a mid-restart node is not a cheat.

**Payout address with signature proof.** A node claims an EVM address by signing a message
naming its own node id, the address and a single-use nonce; the coordinator recovers the signer
and requires it to equal the claimed address. Rebinding additionally requires a signature from
the address currently on file, so a `node_token` copied off a volunteer's disk is not enough to
redirect their earnings. The agent generates an ordinary secp256k1 key on first run and binds it
automatically, so an operator needs no wallet — a hot key protected by file permissions, which
is appropriate for an address that only receives.

**Relay authentication.** Relay tickets are HMAC-authenticated; the node's own liveness probe
completes a real handshake rather than trusting a TCP accept.

**The wire carries no executable content.** The frame is a JSON header plus raw tensor bytes.
This replaced `torch.load(..., weights_only=False)`, which is pickle — any peer, in either
direction, and reachable from the public relay ports rather than only from the chain, could
execute code in the receiving process. The declared message length is capped at 512 MB so a
port scanner cannot make a 1 GB relay VM allocate an arbitrary buffer.

**Zero personal data.** A node sends its id, layer range, core and RAM counts, OS string and IP.
Nothing else. Two honest caveats from `SAFETY.md`: the coordinator no longer persists prompt
text but the prompt still passes transiently through its process on every `/infer` call, and
content moderation is a keyword blocklist — a first line of defence, not a trust-and-safety
system.

---

## 7. NRN Token Economics

Fixed supply of **1,000,000,000 NRN**, split at genesis **60 / 20 / 15 / 5** — node rewards,
NEURON Labs, ecosystem grants, liquidity. Per completed request: **1.0 NRN**, of which 0.1 is
the coordinator fee and 0.9 is divided across the chain in proportion to layers held, so a node
holding `L` of 28 layers earns `0.9 · L/28`.

The ledger is **transfer-only and mints nothing**. Genesis seeds the four buckets; settlement
moves NRN out of escrow; `SUM(balance) == 1e9` is asserted after every operation. A 10-year
emission schedule halving every two years is **designed but not implemented** — today the
ledger settles a flat 1.0 NRN per request out of the emission pool.

**Phase 1 (current): SQLite on the coordinator.** After the Session 38 cleanup the live ledger
holds exactly 1,000,000,000 NRN, and the only accounts holding any are node_a (9.011876),
node_c (8.107126), node_b (6.082124), the coordinator fee account (2.867800) and two earlier
node identities (0.187746 each). All of that is real compute on real hardware.

**Phase 2 (planned): NEURON Chain.** An ERC-20 (`NRN.sol`, OpenZeppelin 5.1, fixed 1B,
60/20/15/5, rewards *released* from a locked pool rather than minted) is written, compiles, and
passes 60 local checks against a throwaway EVM, alongside a migration script rehearsed against a
real ledger snapshot. **It is deployed nowhere; no transaction has been sent to any public
chain.** Polygon is skipped entirely — not deferred — in favour of NEURON Chain, which does not
exist yet.

The gates are **50+ external nodes and 500+ monthly active users**. External nodes today: zero.
Even then, on-chain settlement cannot replace SQLite — one transaction per node per request is
unworkable — so the chain would be a settlement and withdrawal layer over the hot ledger. That
removes the risk of the record dying with one VM; it does not remove the coordinator's authority
over who earned what.

---

## 8. Decentralisation Roadmap

Phase 1 is built. Everything after it is planned.

| Phase | Trigger | Contents | Status |
|---|---|---|---|
| **1** | now | single coordinator + relay + SQLite on one VM; peer quorum verification | **built** |
| **2** | ~50 nodes | second VM, PostgreSQL, Cloudflare in front of the existing name | planned |
| **3** | ~1,000 nodes | DHT peer discovery (libp2p), NAT hole-punching, relay fabric | planned |
| **4** | ~10,000 nodes | NEURON Chain, Proof of Compute consensus | planned |
| **5** | 100,000+ | full decentralisation, community governance | planned |

Phase 2's groundwork is shipped: the coordinator's public address is returned on every
heartbeat, and an agent adopts a new one only after probing it successfully — so the network can
be moved without stranding every installed node. The public name deliberately stays put;
Cloudflare goes behind it. PostgreSQL, the second VM and the load balancer are **not built** —
the coordinator still opens SQLite directly, in one region.

What remains centralised, plainly: the coordinator decides routing, placement and standing; the
relay is on the same VM and every NAT'd hop crosses it; the ledger is a SQLite file on that VM.

---

## 9. Comparison to Existing Projects

| | Bittensor | Gensyn | io.net | Petals | NEURON |
|---|---|---|---|---|---|
| GPU required | yes | yes | yes | no | no |
| Crypto staked upfront | yes | yes | yes | no | no |
| Setup | complex | complex | moderate | moderate | one installer (Windows) |
| Incentive to contribute | yes | yes | yes | none | NRN |
| Automatic slice download | no | no | no | no | yes |
| Proof of compute | no | no | no | no | yes |
| Opaque-tensor privacy | no | no | no | partial | yes |

Two caveats. Petals is the technically closest system and the proof that this networking model
works at scale — DHT discovery, direct P2P with relay fallback, hundreds of real nodes,
100B-class models. NEURON's differentiator is packaging and incentive, not new
distributed-systems primitives, and the eventual scale layer will lean on `libp2p`/`hivemind`
rather than reinvent them. And NEURON is at v0.17.1 with a handful of machines while the other
four are deployed networks: this compares designs, not maturity.

---

## 10. Environmental Argument

**No new hardware is manufactured.** Every node already exists, is already purchased, and is
already powered on. Typical consumer CPU utilisation sits at 2–5%, and NEURON's default mode
takes only capacity below a 15% ceiling that the owner is not using. There is no new datacenter,
no new grid connection, no new cooling plant — the work spreads across power infrastructure that
is already built.

**And inference draws real power.** This is where the claim has to stop. A busy CPU draws
roughly 15–45 W above idle; a consumer GPU node would draw 200–400 W. NEURON does not run on
zero energy, and per token it is *less* efficient than a datacenter GPU. What it avoids is the
**embodied** cost — the manufacturing, the building, the new capacity — not the marginal
electricity. At high donation levels an operator may spend more on electricity than the NRN is
presently worth, which is why NEURON is positioned as compute-barter and not as income.

---

## 11. Current Status and Limitations

- **Early alpha, v0.17.1.** The installer is Windows-only and **unsigned**, so SmartScreen warns
  that it is unrecognised. Code signing needs a certificate and is a pre-distribution step.
- **The network is very small** — a handful of machines, all belonging to the project. The live
  coordinator right now reports **2 nodes online covering 21 of 28 layers**, so the chain is
  incomplete and `network_healthy` is `false`: one node holds 0–13, the other 14–20, and nothing
  is serving 21–27. Lifetime totals are 38 requests and 25.81 NRN distributed. **No person
  outside the project has ever run a node.** That is the milestone that has not happened.
- **NRN has no cash value.** No exchange, no sale, no promise of either. It is a record of
  compute contributed.
- **One coordinator, one relay, one SQLite ledger, one VM, one region.** Backed up hourly, still
  one machine. Removing that is the current work.
- **The CPU pipeline is slower per token than a GPU datacenter.** Physics, not a bug, and
  acceptable: distribution buys capacity and aggregate throughput. For models that fit on one
  machine, NEURON runs them locally instead.
- **The wire codec is not deployed** to the remote nodes, so that 74%-of-wall-clock wire cost is
  still being paid in full.
- **The GPU execution path has never run.** The device resolution, the shard move onto CUDA and
  the VRAM-aware layer assignment described in section 4 are written and covered by tests, but
  every machine in this project is CPU-only — the installed torch is a `+cpu` build — so no line
  of the CUDA branch has ever executed. What *is* proven is that it changes nothing on a CPU
  machine: `selftest_shard.py` still reports `max|delta| = 0.000e+00` against the unsharded
  model. The speedup a GPU node would bring is therefore unmeasured, and no figure is claimed
  for it. The first real GPU node is the test.
- **No legal review.** Neither the content policy nor the token economics has been read by a
  lawyer. Under EU MiCA that review is required before public transferability, and it gates
  mainnet.
- **Also unbuilt:** PostgreSQL and coordinator redundancy, DHT discovery, NAT hole-punching, the
  Android agent, int4 weights in the pipeline (needed for 72B on few nodes), and a policy for
  accounts that never bind a payout address.

---

## 12. Conclusion

**The mechanism works, and it was proven rather than asserted.** Layer-split inference is
bit-exact against a single machine. Three heterogeneous consumer computers serve 6.16 tok/s
aggregate at 3.82× pipeline overlap. A node behind NAT joins over a relay with no port
forwarding, is verified by its peers in 254 ms with no human involved, and comes back 13 seconds
after a power cut. Activations cross the wire 4.25× smaller with identical output, in a format
that cannot carry code.

**The economics are designed and internally consistent** — fixed supply, transfer-only ledger,
an invariant asserted after every operation, earnings bound to an address the operator proves
they control. What is designed is not deployed: the emission halving is unimplemented and the
chain does not exist.

**The path to decentralisation is written down honestly.** One VM today, redundancy at fifty
nodes, peer discovery at a thousand, a chain at ten thousand. Each step is planned, none is
built, and authority genuinely sits in one place until they are.

Everything follows from growth. The coordinator's single point of failure only matters when
there is traffic to lose; the chain only matters when there are balances worth withdrawing; the
DHT only matters when there are too many nodes to register centrally. The next milestone is not
technical. It is one person outside this project installing NEURON and earning their first NRN.

---

*NEURON — Network of Existing Utilised Resources — Open Nodes*
*NEURON Labs, 2026 · Apache License 2.0 · github.com/neuron-network-ai/neuron*
