# NEURON — Session Log

A distributed AI inference network: a tiny agent on any laptop contributes a
little CPU to run LLM inference collectively. Each machine runs a *slice* of the
model's transformer layers and passes activations to the next machine over TCP.

This file logs what was built each session, why, the results, and the traps hit
along the way.

---

## Setup (three machines) — pipeline: node_a → node_c → node_b → node_a

| | node_a (driver) | node_c (middle) | node_b (last) |
|---|---|---|---|
| Role | embed + first layers + `lm_head` | middle layers (relay) | last layers + norm |
| Hardware | Windows 11, 63 GB, 16 cores | HP Pavilion, Ryzen, 4 cores, 11 GB | Dell OptiPlex, 6 cores, 15 GB |
| Python | 3.11 venv `C:\Users\optin\neuron\.venv` | 3.12 venv `~/neuron/.venv` | 3.10 venv `~/neuron/.venv` |
| Reach (Tailscale) | — | `ssh <node-c-host>` | `ssh <node-b-host>` |
| Port | connects out | listens **50999** | listens **50999** |

Stack on all three: `torch==2.4.1` (CPU), `transformers==4.44.2`, `accelerate`. The
Python minor version need NOT match across nodes — only the torch/transformers
versions must, since that's what makes the pickled tensors compatible over TCP.

> Note: the build tooling runs **natively on Windows** here, not in WSL2. The OptiPlex's
> LAN IP `192.168.1.10` is often unreachable (Machine 1 roams onto a phone
> hotspot), so we use its Tailscale IP `100.114.189.46`.

Files in `C:\Users\optin\neuron\`:
- `common.py` — shared: model loading/sharding, the manual layer driver, the KV
  cache, and length-prefixed tensor framing over TCP.
- `node_a.py` — Machine 1 client/driver: embeds tokens, runs its layers + `lm_head`
  (S3), and drives generation. Parallel multi-request driver (S4) with per-request
  latency, aggregate throughput, and node-utilisation reporting.
- `node_b.py` — last stage: runs its layers + norm, returns the normed hidden
  state (S3). Threaded — one thread per connection, shared shard (S4).
- `node_c.py` — middle relay (S5): server to node_a AND client to node_b. Runs its
  layers, forwards to node_b, relays the result back. Threaded.
- `selftest.py` — proves the split is bit-exact vs. the full model, and that the
  KV cache produces identical tokens to brute-force generation.
- `selftest_shard.py` — same checks, but for the sharded (partial-load) path.
- `coordinator/` — the network brain (S6): FastAPI + SQLite registry, health,
  routing, and the NRN ledger. See `coordinator/README.md`.
- `slice_downloader.py` — a node downloads ONLY its layers' weights via safetensors
  byte-range requests (S8), not the whole model. Pairs with `/node/{id}/slice-info`.

---

## Session 1 (2026-07-22) — first end-to-end split

**Goal:** split a small model across the two machines and run one inference E2E.

**Decisions**
- **transformers + safetensors, not GGUF.** A GGUF can't be cut by layers for our
  own Python; that's llama.cpp's runtime format.
- Model: **Qwen2.5-0.5B-Instruct** (24 layers, split 12/12).
- Split trick: both nodes load the *whole* model but neutralise (pass-through) the
  layers they shouldn't run, so transformers still builds the causal mask + rotary.
- Greedy, **no KV cache** — recompute the whole growing sequence each token.

**Result:** `"Hello"` → `"Hello! How can I assist you today?"`. Bit-exact vs. a
single-machine forward. ~494 ms/token round-trip (A 110 / net 105 / B 279 ms).

---

## Session 2 (2026-07-23) — 1.5B + KV cache + light nodes

**Goals:** upgrade to Qwen2.5-1.5B, add a KV cache, add an autoregressive loop
(until EOS or 200 tokens), measure tokens/sec, keep selftest bit-exact.
Plus a follow-on: make each node "light" by loading only its own layers.

**What changed**
1. **Model → Qwen2.5-1.5B-Instruct** (28 layers, split **14/14** = the original
   0–13 / 14–27 plan).
2. **Manual layer driver.** Session 1's "neutralise + let transformers do
   everything" trick does *not* survive a KV cache, so `common.py` now drives the
   decoder layers directly: each node keeps its own cache and tracks token
   positions itself.
3. **KV cache + autoregressive generation.** Prefill runs the whole prompt; each
   decode step ships only a `[1,1,1536]` hidden-state tensor. Stops at EOS or
   `--max-new-tokens` (default 200).
4. **Light nodes (sharding).** `common.load_model_shard()` builds the model on the
   `meta` device (no memory) and materialises **only this node's layers** from the
   safetensors shards. node_b resident dropped to **~4.6 GB** (vs. ~7 GB full-load).

**Result:** correct Rayleigh-scattering answer; 38 tokens, stopped at EOS.
**1.85 tokens/sec** decode; prefill ~950 ms; per-token A 118 / net 66 / B 356 ms.
The KV cache is doing its job: 1.5B *with* cache (~540 ms/tok) ≈ 0.5B *without*
cache in Session 1 (~494 ms/tok), despite 3× the parameters.

**Traps hit (transformers 4.44.2, Qwen2)**
- No `model.model.rotary_emb` — rotary lives inside each attention module and is
  computed from `position_ids`. Don't pass `position_embeddings`.
- HF `DynamicCache.update` assumes layers fill sequentially from index 0, so it
  `IndexError`s on node_b's layers 14–27. Replaced with `common.SplitCache`, a
  dict keyed by the layer's native index.
- A single decode token needs **no causal mask** (it may attend to everything);
  the mask is only built for multi-token prefill.
- `sed "s/\r$//"` on copied files silently strips a trailing `r` from every line —
  never do it; the Write tool already emits Unix line endings.

**On "can a node be 1 MB?"** The *agent code* already is (~20 KB of Python). The
*model weights* can't be — one transformer layer is ~100–215 MB, and the vocab
embedding (~0.9 GB, tied to `lm_head`) is needed on *both* ends. Sharding shrinks a
node toward that floor, but never to 1 MB.

---

## Session 3 (2026-07-23) — move lm_head, rebalance, bf16 test

**Goals:** move `lm_head` from node_b to node_a (target >3 tok/s), try bf16, keep
selftest bit-exact.

**What changed**
1. **`lm_head` moved to node_a.** node_b now returns its *normed final hidden
   state*; node_a applies `lm_head` and picks the token. No token echo back is
   needed — node_b's KV cache is updated purely by running its layers on the
   incoming hidden (it never embeds anything). Bonus: node_b no longer loads the
   0.9 GB embedding at all.
2. Bit-exactness held after a subtle fix: apply `lm_head` to the whole hidden
   block and take the last row (`model.lm_head(hidden)[:, -1, :]`). Slicing to
   `[1,1,H]` first changes the GEMM shape and perturbs logits by ~1e-5.

**Results (prompt: "why is the sky is blue", 38 tokens, identical output at every split)**

| config | A ms | B ms | head ms | tok/s |
|---|---|---|---|---|
| S2 baseline (split 14, head on B) | 118 | 356 | — | 1.85 |
| split 14, head on A | 119 | 309 | 33 | **2.08** |
| split 20 (time-balanced) | 177 | 175 | 36 | 2.46 |
| split 24 | 208 | 113 | 37 | 2.66 |
| split 26 (OptiPlex ~idle) | 250 | 39 | 42 | 2.90 |

**Verdict on >3 tok/s:** not reached while genuinely distributed. Moving `lm_head`
helped only +12% because B's cost is its **transformer layers**, not the head —
the OptiPlex is ~2.6× slower per layer than the Windows CPU (fp32). Rebalancing
the split helps but hits diminishing returns; even handing the OptiPlex just 2
layers gives 2.90. Running all 28 layers on Windows alone would be ~3.2 tok/s.
**Lesson:** a serial fast→slow pipeline is bottlenecked by the slow node and can't
beat the fast node alone on single-request latency. Distribution pays off for
*capacity* (model too big for one node) and *throughput* (overlapping requests),
not single-stream speed. `node_a --split N` tunes the balance; ~20 balances *time*
on this pair (both nodes ~175 ms).

**bf16 (goal 2): tested and REJECTED.** `DTYPE=torch.bfloat16` halves layer RAM,
but these CPUs have no bf16 GEMM (no AVX512-BF16 / AMX), so torch falls back to a
slow path — a forward pass that takes seconds in fp32 didn't finish in 5 minutes.
Several× *slower*. Kept fp32. (True low-precision speedup on CPU = the GGUF/
llama.cpp int4/int8 path, which can't do this hand-written split.)

**New traps hit**
- `pkill -f node_b.py` matches the ssh command's OWN cmdline (it contains
  "node_b.py") and kills the session → exit 255. Use the self-exclusion pattern
  `pkill -f '[n]ode_b.py'`, and keep the kill in a separate ssh call from any
  command that mentions the script literally.
- Moving an op across the split can break bit-exactness purely via matmul GEMM
  shape (the `[1,1,H]` vs `[1,L,H]` lm_head issue above), even with identical
  weights and math.

---

## Session 4 (2026-07-23) — request pipelining / throughput

**Goal:** prove Session 3's claim — distribution scales *throughput* (simultaneous
users), not single-stream latency. Serve N concurrent requests; both nodes busy at
once.

**What changed**
1. **node_b is threaded** — one thread per TCP connection, all sharing ONE loaded
   shard; each connection has its own KV cache, so requests are independent.
2. **node_a is a parallel driver** — N requests, each its own thread + connection
   + cache. Built-in 4 prompts; `--serial` runs the baseline.
3. **Per-machine compute lock** on each node. Each node is CPU-bound, so letting N
   requests run `forward` at once would oversubscribe the cores (N × torch's GEMM
   threads) and thrash. The lock lets one request compute at a time *with full
   cores*, while the other threads overlap their **network waits**. That is the
   pipeline: node_a computes stage-A for request 2 while node_b computes stage-B
   for request 1. Bonus: since each node serialises its own compute, `sum(compute)
   / wall` is a clean utilisation number. node_a warms up node_b's shard before
   timing so the one-time load doesn't skew results.

**Results (4 prompts, 80 tokens each = 320 tokens, greedy so identical either way)**

| run | split | throughput | node_a busy | node_b busy | overlap | latency/req |
|---|---|---|---|---|---|---|
| serial baseline | 20 | 2.13 tok/s | 55% | 40% | 1.00× | 37.6 s |
| **parallel** | 20 | **4.02 tok/s** | 100% | 52% | 1.60× | 79.5 s |
| **parallel** | 16 | **4.61 tok/s** | 99% | 82% | 2.06× | — |

**Verdict: PROVEN.** Parallel throughput ~**2× the serial baseline** on 2 nodes,
approaching the 2× ceiling of a 2-stage pipeline (split 16: overlap 2.06×, both
nodes >80% busy simultaneously). Per-request latency doubles — expected and fine;
the win is aggregate tokens served, i.e. more simultaneous users. This is how
NEURON scales: more nodes → more concurrent users, not faster single answers.

**Insight:** the best split for *throughput* (≈16) differs from the best for
*latency* (≈20–24). Throughput is bottlenecked by the busier node's per-token
compute, and since `lm_head` lives on node_a, node_a should carry fewer layers to
balance. At split 20 node_a was pinned at 100% (its compute lock capped throughput
at 320 tok / 79.3 s = 4.03); shifting 4 layers to node_b balanced it to 99%/82%.

**Traps hit**
- The `pkill` self-match bites again, this time via a *verification* command:
  `... "pkill -f '[n]ode_b.py'; grep -c x ~/neuron/node_b.py"` — the `grep`
  argument contains the literal `node_b.py`, which the pattern matches, killing the
  ssh session (exit 255). Rule: the kill's ssh command must contain **no literal**
  `node_b.py` anywhere — keep verification in a *separate* ssh call.

---

## Session 5 (2026-07-23) — third node, 3-stage pipeline

**Goal:** add a 3rd machine (HP Pavilion) as a middle stage and show throughput
keeps scaling. Chain: `node_a → node_c → node_b → node_a(lm_head)`.

**What changed**
1. **`common.py` generalised to arbitrary layer ranges** — `load_model_shard(lo,
   hi, embed, norm, head)` and `first_stage` / `mid_stage` / `last_stage`. Any node
   can own any contiguous slice; roles differ only by embed/norm/head extras.
2. **`node_c.py`** — the middle relay: a threaded *server* to node_a and a *client*
   to node_b. Per request it runs its layers, forwards the hidden to node_b, and
   relays node_b's result back to node_a. node_a connects only to node_c and passes
   node_b's address in the config, so node_c dials node_b itself.
3. **`node_a.py`** now takes `--host-c --host-b --s1 --s2` and reports a 3-node
   utilisation breakdown. `node_b.py` config carries `s2, n`.
4. Default split **10/9/9** as requested; **9/9/10** balances best on this trio.

**Results (4 prompts × 80 tok = 320 tok; N=8 = the 4 prompts twice)**

| run | split | N | throughput | a% | c% | b% | overlap |
|---|---|---|---|---|---|---|---|
| serial baseline | 9/9/10 | 4 | 1.64 tok/s | 22 | 23 | 38 | 1.00× |
| parallel | 10/9/9 | 4 | 5.46 tok/s | 93 | 79 | 74 | 3.17× |
| **parallel** | 9/9/10 | 4 | **5.86 tok/s** | 93 | 78 | 79 | 3.23× |
| parallel | 8/10/10 | 4 | 5.58 tok/s | 79 | 88 | 81 | 3.68× |
| **parallel** | 9/9/10 | **8** | **6.16 tok/s** | 98 | 78 | 87 | 3.82× |

**Verdict: throughput keeps scaling.** 3 machines busy at once (all >74%), overlap
up to **3.82×**, and parallel beats serial by ~3.6×. The **>6 tok/s** target is met
at N=8 (6.16). Scaling across sessions: **single ~3.2 → 2-node 4.61 → 3-node 6.16
tok/s.** Identical correct outputs; `selftest_shard.py` (now the 3-stage chain) is
bit-exact, ALL PASS.

**Honest caveat — scaling is sub-linear, not 3×.** The two added nodes are *slower*
than the Windows box (Pavilion 4-core ~14.7 ms/layer, OptiPlex 6-core ~13.5, Windows
~11.8), and node_a carries the fixed `lm_head` + orchestration overhead. So 3
*heterogeneous* nodes give ~1.9× a single node, not 3×. Linear scaling would need
*equal* nodes. Two levers seen: **balance for the weakest node** (node_a's head is
fixed, so it should hold fewer layers, but its neighbour node_c is the slowest CPU
and can't absorb them — hence 9/9/10, not fewer on node_a); and **more concurrency**
(N 4→8 lifted 5.86→6.16 by filling the deeper pipeline).

**Traps hit**
- A brand-new node needs (a) your SSH pubkey in its `authorized_keys` and (b) a
  one-time `sudo apt install python3.12-venv` — neither is automatable without
  password/sudo, so provisioning a fresh node always needs one manual hand-off.
- Ubuntu 24.04 ships **Python 3.12**; `torch==2.4.1` has 3.12 wheels, so no need to
  install 3.11. Nodes can run different Python minors as long as torch/transformers
  versions match (that's what keeps the TCP tensor pickles compatible).

---

## Session 6 (2026-07-24) — the coordinator (network brain)

**Goal:** build the NEURON coordinator — a FastAPI + SQLite service for node
registry, health-checking, request routing, and an NRN ledger — so nodes stop
connecting by hardcoded address and instead ask the coordinator for the chain.

**What was built** (new `coordinator/` package; deps: `fastapi`, `uvicorn`):
- `main.py` — FastAPI app: `/node/register|list|{id}|{id}/ping`, `/infer`,
  `/infer/{id}/complete`, `/ledger/{id}`, `/status`, `/dashboard` (auto-refresh
  HTML), token auth, and a 60 s background health sweep that logs offline nodes.
- `models.py` — SQLite (nodes / ledger / requests), no ORM; status computed from
  `last_seen` so it's accurate between sweeps.
- `router.py` — assemble a contiguous 0..27 chain from online nodes; report gaps.
- `ledger.py` — NRN split. `config.py` — settings. `register_nodes.py` — register
  the 3 nodes + a liveness-probing heartbeat. `README.md`.
- `node_a.py` gained **`--coordinator URL`**: it asks `/infer` for the chain, runs
  it, and POSTs `/complete` so NRN is credited. `--host-c/--host-b` still work as a
  direct fallback. (No changes to `common.py`, `node_b.py`, `node_c.py`.)

**Economics reconciliation.** The spec said both "a node gets layers/28 of 1.0"
*and* "coordinator keeps 10% always" — which conflict. The `/status` example
resolves it (47 requests → 42.3 NRN distributed = 47 × 0.9): the **10% fee comes
off the top**, and nodes split the remaining **0.9 by layer share** (`0.9·L/28`).

**Health for unmodified node servers.** Offline detection needs a liveness signal,
but the brief said don't modify `node_b.py`/`node_c.py`. So `register_nodes.py`
runs a **liveness-probing heartbeat**: it checks whether each server node's port is
really listening and pings the coordinator on its behalf (node_a, the driver, has
no server port so it's always pinged). Kill a node's server → port down → pings
stop → offline after 90 s. (A decentralised build would have each node self-ping.)

**Test results — all 7 steps pass:**

| # | test | result |
|---|------|--------|
| 1 | start coordinator | ✓ up on `:8000`, SQLite auto-created |
| 2 | register 3 nodes | ✓ node_a 0–9, node_c 10–18, node_b 19–27 |
| 3 | `GET /status` | ✓ 3/3 online, 28 layers covered, healthy |
| 4 | `POST /infer` | ✓ returns ordered `a→c→b` chain + `request_id` |
| 5 | `node_a --coordinator` inference | ✓ correct Rayleigh answer, completion reported |
| 6 | `GET /ledger` | ✓ a=0.3214, c=0.2893, b=0.2893, fee=0.10, distributed 0.9 |
| 7 | node offline → routing fails | ✓ node_c offline after 90 s (logged), `/infer` → **503 "incomplete chain - missing layers 10-18"** |
| — | security | ✓ register w/o secret → 401; ping wrong token → 401; correct token → 200 |

> Pavilion (node_c) was powered off this session, so node_c ran **locally on
> Windows (port 51000)** as a host-agnostic stand-in — the coordinator doesn't care
> which machine hosts a layer range. The offline test was therefore shown on node_c
> ("missing 10-18") rather than node_b ("missing 19-27"); identical mechanism.
> Tomorrow, point node_c back at the Pavilion (`register_nodes.py` defaults to it).

---

## Session 7 (2026-07-24) — coordinator to an always-on host + public on GitHub

**Goal:** move the coordinator off the laptop onto an always-on host, and publish the
whole project to GitHub.

### Part 1 — deploy the coordinator (OptiPlex, `:8001`)

Oracle Cloud wasn't set up, so per the brief the **always-on OptiPlex** hosts the
coordinator (it already runs node_b, so a separate port 8001 keeps them apart).
- `scp` the `coordinator/` package (`.py` + README only — **not** the local `neuron.db`
  or `node_tokens.json`) to `~/neuron-coordinator/coordinator/`; installed `fastapi` +
  `uvicorn` into the OptiPlex's existing `~/neuron/.venv`.
- `uvicorn coordinator.main:app --host 0.0.0.0 --port 8001`; `sudo ufw allow in on
  tailscale0 to any port 8001`. Reachable from Windows: `GET /status` → 200.
- `register_nodes.py` default coordinator URL updated to `http://100.114.189.46:8001`.

**Full flow verified through the cloud coordinator** (Pavilion still off → node_c ran
locally on Windows `:51000` again): 3/3 online + healthy; `node_a --coordinator
http://100.114.189.46:8001` produced the correct answer; ledger credited a=0.3214 /
c=0.2893 / b=0.2893, fee 0.10, distributed 0.9; dashboard renders HEALTHY. Because the
host is always on, the network stays reachable without the laptop.

### Part 2 — public on GitHub

- **README** rewritten to stand alone for a stranger: what NEURON is, an ASCII
  architecture diagram (User → Coordinator → [node_a → node_c → node_b] → Coordinator →
  User), scaling table, **how to run a node**, **how to run the coordinator**, dashboard
  description, **what hardware you need**, **how to earn NRN** (1.0/req, 10% fee, 0.9 by
  layer share), repo layout, status.
- `.gitignore` hardened: excludes `*.db` (+ `-wal/-shm`) and `node_tokens.json` so the
  token-bearing SQLite is never pushed, alongside `.venv/`, `__pycache__/`,
  `*.safetensors`, `models--*/`.
- Branch renamed `master → main`; committed as **v0.2** and pushed.
- **Public repo: https://github.com/raman011sharma-code/neuron**

Trap: probing the repo URL with `git ls-remote` triggered **Git Credential Manager's**
GitHub OAuth sign-in (a browser popup) — expected first-push behaviour on Windows;
authorising GCM stores the credential so `git push` works non-interactively after.

### Part 3 — stranger check
Read the README cold against the five questions (what is it / run a node / run the
coordinator / hardware / earn NRN) — all answered; no fixes needed.

---

## Session 8 (2026-07-24) — slice downloader (only download your layers)

**Goal:** a node should download ONLY the weights for its assigned layers, not the
whole 3 GB model. This is the mechanism behind the "1 MB agent".

**THE TRAP — the model is not sharded.** Qwen2.5-1.5B-Instruct on HF is a *single*
`model.safetensors` (3.087 GB). There is **no `model.safetensors.index.json`** and
no `model-0000N-of-M` files — so the whole premise of "pick which shard files to
download" (and `hf_hub_download`, which only fetches whole files) does not apply.
There is one shard and it holds all 28 layers.

**The fix — per-tensor byte-range download.** A safetensors file begins with an
8-byte length + a JSON header that lists *every tensor's exact byte range*
(`data_offsets`). So `slice_downloader.py`:
1. Fetches just the header (~38 KB) via HTTP Range — HF serves `Accept-Ranges:
   bytes` (206), which is how `huggingface_hub` does resumable downloads.
2. Picks the tensors for this node's layers (+ embedding on the first node, + norm
   on the last), merges their byte ranges into contiguous spans, and Range-downloads
   only those spans.
3. Reassembles a small, valid `model.safetensors` (new header + concatenated data)
   and downloads `config.json`/`generation_config.json` (+ tokenizer on node_a).
This is *more* granular than shards — per tensor.

**Shard map for Qwen2.5-1.5B-Instruct:** 1 file, **338 tensors**, 38 KB header.
Naming: `model.embed_tokens.weight` (~0.9 GB, tied to `lm_head` → no separate
`lm_head.weight`), `model.layers.{0..27}.{input_layernorm, post_attention_layernorm,
self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}.weight`, `model.norm.weight`.
Each decoder layer ≈ **94 MB** (bf16). node_c/node_b's layer blocks are one
contiguous span each; node_a is 3 spans (embedding sits apart from its layer block).

**Download sizes per node (verified byte-identical to the full model, and node_a
verified functionally identical — same hidden state):**

| download | size | % of full |
|---|---|---|
| full model (old way, every node) | 3.087 GB | 100% |
| node_a (0–9 + embed + tokenizer) | **1.403 GB** | 45% |
| node_c (10–18) | **0.842 GB** | 27% |
| node_b (19–27 + norm) | **0.842 GB** | 27% |
| **sum across the 3-node network** | **3.087 GB (1×)** | vs **9.26 GB (3×)** downloading full on each |

So slice-downloading cuts total network to **1×** the model (from 3×), and each node
fetches **up to 3.7× less** than before.

**Traps hit:**
- Single-file safetensors (no index.json) — the whole "download selected shards"
  plan is impossible; byte-range is the only real slice for this model.
- `lm_head` is **tied to the embedding** (absent as its own tensor), so the FIRST
  node (which owns the head in NEURON) needs `embed_tokens.weight`; there is no
  separate head tensor to fetch on the last node.
- Reassembling a valid safetensors slice: write `<u64 header-len><JSON header with
  recomputed contiguous data_offsets><data>`; keep offsets gap-free.

**Coordinator (Task 4):** new `GET /node/{id}/slice-info` (in `coordinator/
sliceinfo.py`, stdlib + requests, header cached) returns `{model_id, layer_start,
layer_end, shards_needed:["model.safetensors"], tensors_needed, tokenizer_needed,
lm_head_needed, norm_needed, is_first/last_node, estimated_download_gb,
full_model_gb}`. Deployed to the always-on OptiPlex coordinator (`:8001`).

**How it's used (Task 5) — the command a fresh node runs:**
```
python slice_downloader.py --coordinator http://100.114.189.46:8001 --node-id node_a --output-dir ./model_slice
```
It asks the coordinator what it owns, downloads only that, verifies byte-identity,
and writes a ready-to-load slice dir. (Manual mode: `--model-id --layer-start
--layer-end [--first] [--last] --output-dir`.) `common.py`, `node_*.py` unchanged.

---

## Session 9 (2026-07-24) — the agent (turn any machine into a node)

**Goal:** a background agent that auto-configures a machine into a NEURON node.
Pushed v0.3 first (byte-range slice downloader).

**Built (`agent/`, ARM-compatible — pure Python + psutil/requests/pystray/PIL/ctypes,
no x86-specific code):**
- `agent.py` — main loop: read config → register (sends cores/RAM/Tailscale IP) →
  `GET /node/{id}/slice-info` → auto-download only its slice → start the server →
  heartbeat every 30 s (gated by the resource guard) → log to `agent.log`.
- `node_server.py` — **one generalized server for ANY layer range.** Loads the
  downloaded *slice* (not the full model) and, from the incoming config, acts as a
  MIDDLE relay (`host_b` present → node_c role) or LAST stage (`s2/n` → node_b role).
  Reuses `common.py`; stays compatible with node_a.py's wire protocol.
- `resource_guard.py` — only use TRULY idle capacity. Pauses if system CPU > 2%,
  user active (Win `GetLastInputInfo` / Linux `xprintidle`, headless = always idle),
  on battery, or < 500 MB free. "Pause" = stop heartbeating so the coordinator routes
  elsewhere; in-flight requests finish.
- `tray.py` — pystray/Pillow tray: green=active, grey=idle, yellow=downloading,
  red=error; menu shows NRN balance, Pause/Resume, Open Dashboard, Quit.
- `updater.py` — polls `GET /agent/version`; self-downloads + restarts if newer.
- `install.py` / `uninstall.py` — one-command setup / clean removal (Windows HKCU
  Run key or Linux systemd `--user`; deregisters + deletes slice + config on uninstall).
- `config.json` — template. Coordinator gains `GET /agent/version` and `total_layers`
  in slice-info. **No existing node/common scripts modified.**

**Design decisions:**
- Registration needs a layer range (auto-assignment is Session 14), so the agent's
  config carries `layer_start/layer_end`; the installer sets them. Register-secret
  defaults to `neuron-dev-secret` in code so the config template stays clean.
- The resource guard gates the *heartbeat* (availability), not mid-request compute —
  the honest, non-disruptive way to "pause" a pipeline node.

**Tests — ALL 5 PASS:**

| # | test | result |
|---|------|--------|
| 1 | fresh install | agent registered as node_c, auto-downloaded its 0.84 GB slice, served on :51000, heartbeat active; `install.py` writes correct config |
| 2 | resource guard | correctly paused on a busy machine (`cpu 7%>2%, user active 3s`) |
| 3 | coordinator unreachable | enters `error` state, retries every 60 s |
| 4 | uninstall | deregistered node_c, stopped agent, deleted slice + config, printed lifetime NRN |
| 5 | inference through agent | agent participated in the live chain; NRN **0.2893 → 0.5786** (served 1 → 2) |

**Traps / honest limits:**
- Single-file model (from S8): the agent downloads its slice by byte-range, ~2 min.
- `uninstall.py` kills agent processes by cmdline match (`agent.agent`/`agent.py`);
  if the agent shares a process group with the caller this can signal the caller
  (cosmetic exit 15) — the cleanup still completes.
- The **tray icon can't be visually verified in this headless session** (it builds
  and imports cleanly); and I did **not** add auto-start to the real Windows machine
  or run a permanent background service — install/uninstall were tested in a sandbox
  and via a permissive test config.
- After the uninstall test, **node_c is deregistered** — the live network is now
  node_a + node_b only (DEGRADED). Restore by running a node_c (Pavilion when on, or
  a local stand-in / the agent again).

---

## Session 10 (2026-07-24) — chat UI (talk to the network in a browser)

**Goal:** a web page where anyone types a prompt and gets a streamed answer from
the node network — no local model on the user's side.

**Built (`ui/`, nothing in common/node_*/coordinator modified):**
- `ui/app.py` — FastAPI server that **is** the node_a driver: loads the
  embed+layers 0..S1-1+lm_head shard once, and per prompt asks the coordinator
  `/infer` for a live chain, runs the autoregressive loop, and **streams each token
  over SSE**. Reuses `node_a.coord_get_chain` / `coord_complete` and `common`'s
  stage primitives, so it credits NRN via `/infer/{id}/complete` exactly like
  node_a.py. Endpoints: `GET /` (page), `GET /network` (live node count/health,
  fetched from the coordinator server-side so the browser never talks to it),
  `POST /chat` (SSE frames: `meta` → `token`* → `done`, or `error`).
- `ui/static/chat.html` — clean single-file chat (no external assets): streams
  tokens into the bubble; header shows **"Powered by N nodes worldwide"** + a
  health dot; a banner warns when the chain is incomplete; each answer shows
  **Cost: 1.0 NRN** + tokens + tok/s. Light/dark aware.

**Design:** the driver holds the model, so ui/app.py must run on the node_a machine
(this Windows PC). SSE (not websockets) — a plain sync generator yielded through
StreamingResponse (Starlette runs it in a threadpool); node_a's own compute is
serialised with a lock. Incremental decode = decode the growing id list each step
and emit the new text suffix (robust across BPE merges). Env overrides:
`NEURON_COORDINATOR`, `NEURON_S1`, `NEURON_MAX_TOKENS`.

**Verified (against the LIVE coordinator on OptiPlex :8001):**

| # | check | result |
|---|-------|--------|
| 1 | `GET /` | serves chat.html |
| 2 | `GET /network` | live: 2 nodes online, 19/28 layers, healthy=false; lists node_a+node_b |
| 3 | `POST /chat` (degraded) | clean SSE `error`: "coordinator /infer 503: incomplete chain - missing layers 10-18" |
| 4 | frontend (in-browser) | header "Powered by 2 nodes worldwide", red dot, degraded banner "19/28 layers", NRN hint all rendered |

**LIVE DEMO — success metric MET (Pavilion reconnected same session):** the spec
wants a *streamed response from the 3-node network*, and it now works end-to-end.
Restored node_c (`node_c.py --port 50999` on the Pavilion, killed the two stale
Windows heartbeats, re-registered the trio) → network healthy, 3 nodes, 28/28 layers.
Streamed real answers through the Chat UI:
- "Why is the sky blue? Answer in two sentences." → correct 35-token answer arriving
  token-by-token (meta: 3 nodes → 35 `token` frames → done). First request 35 s @
  1.0 tok/s — node_c + node_b **cold-started their shards** on first connect.
- Warm: "Name three primary colors." → "red, blue, and yellow." (12 tok, 9.5 s). Short
  answers look slow because prefill + per-request chain setup amortise over few tokens.
NRN credited via `/complete`: network 2→4 requests served, 1.8→3.6 NRN distributed;
**node_c ledger 0 → 1.157 NRN** (served 4). Coordinator/node/common code all unchanged,
so selftest_shard.py is unaffected. Note: the Pavilion suspends/roams off Tailscale
when idle — keep it awake for a persistent public node.

---

## Session 11 (2026-07-24) — OpenAI-compatible API (switch by changing one URL)

**Goal:** any app built on OpenAI's API works against NEURON by changing only
`base_url`. Zero code changes for the developer.

**Refactor first (DRY):** extracted the node_a driver loop into a shared
`neuron_driver.py` — a `DRIVER` singleton that loads the shard once and exposes
`stream(input_ids, max_new, coordinator, router_prompt)` yielding `meta`/`token`/
`done`/`error` events. Rewired `ui/app.py` (Session 10) to use it; chat UI behavior
unchanged (regression: /chat still streams meta→token→done). Nothing in common.py /
node_*.py touched, so selftest_shard.py is unaffected.

**Built `api/openai_compat.py`** (also mounted into ui.app at the same /v1 paths, so
one process + one model load serves the chat page AND the API):
- `GET /v1/models` — lists `Qwen/Qwen2.5-1.5B-Instruct` + alias `neuron`.
- `POST /v1/chat/completions` — OpenAI chat shape; `stream` supported (role chunk →
  content chunks → finish chunk → `data: [DONE]`), `stream_options.include_usage`
  honored. Applies the chat template to the messages array.
- `POST /v1/completions` — legacy text completion; `stream` supported; raw tokenize.
- Auth: `Authorization: Bearer <NRN wallet>` (required; OpenAI-shaped 401 if absent).
  Each request = 1.0 NRN, reported in `usage.nrn_cost` + `X-NRN-Cost`/`X-NRN-Wallet`
  headers; credits nodes via the coordinator `/complete` (same path as node_a).
- `GET /docs` (standalone) / `/api-docs` (ui) — self-contained usage page (curl +
  Python SDK examples). Pydantic bodies `extra="ignore"` so real payloads never 422;
  greedy generation, so temperature/top_p are accepted and ignored.

**Verified — success metric MET (`pip install openai`, openai 2.48.0 in the venv):**

| # | check | result |
|---|-------|--------|
| 1 | real OpenAI SDK, base_url only | `models.list()`, `chat.completions.create()`, and `stream=True` all work unchanged |
| 2 | non-stream chat | correct "sky is blue" answer, finish=stop, usage 26+36=62 |
| 3 | stream chat | "Red, Blue, Green" arrives token-by-token via the SDK |
| 4 | legacy /v1/completions | "The capital of France is" → " Paris. ..." usage 5+8=13 |
| 5 | auth | missing key → OpenAI-shaped 401 `invalid_api_key` |
| 6 | standalone app | `uvicorn api.openai_compat:app` serves /v1/* + /docs (HTTP 200) |
| 7 | Chat UI regression | /chat still streams after the driver refactor |

**Honest limits:** generation is greedy (no sampling); `n>1`, logprobs, tool/function
calling, and vision content are not implemented. The wallet is recorded and the cost
reported, but a **per-wallet balance debit is not persisted** — that's coordinator-side
economics (ties to S17 on-chain NRN). Quirk: this FastAPI version stores an included
router as a nested `_IncludedRouter` (routes resolve at request time; they don't appear
flattened in `app.routes` — introspection only, endpoints all respond).

---

## Session 12 (IN PROGRESS, 2026-07-24) — coordinator to the cloud (first-stranger groundwork)

**Goal (S12):** first stranger node. Found the hard blocker first: the whole network assumed
ONE Tailscale net (coordinator Tailscale-only; nodes dial each other's `tailscale_ip`), so no
outside machine could join — not a bug, an architecture gap. See `PROBLEMS.md` [P10]/[P11].

**Decision + done this session — coordinator moved to a free CLOUD VM** so the founder's personal
OptiPlex/Pavilion are never the public front door:
- Oracle **Always Free** VM, Amsterdam, **x86 `VM.Standard.E2.1.Micro`** (ARM A1.Flex was out of
  capacity — chronic in AMS), Ubuntu 22.04, 1 GB RAM. Public IP **`150.230.22.250:8001`**.
- Deployed `~/neuron/coordinator/` via a venv (fastapi/uvicorn/requests, **no torch**) as a
  **systemd service** (auto-restart). **Strong `NEURON_REGISTER_SECRET`** (saved locally, gitignored
  `.env.coordinator`). Opened 8001 in BOTH the VM **iptables** (Oracle Ubuntu REJECTs non-22) and the
  Oracle VCN **security list** (two separate firewalls — a classic gotcha). New `coordinator/DEPLOY.md`
  + `coordinator/requirements.txt`. `register_nodes.py` now reads the secret from env (was hardcoded).
- **Migrated the live network onto it and PROVED inference end-to-end:** registered
  node_a/node_c/node_b against the cloud coordinator (healthy, 28/28), ran a prompt via
  `node_a.py --coordinator http://150.230.22.250:8001` → correct answer; NRN credited on the CLOUD
  ledger (a 0.3214 / c 0.2893 / b 0.2893, fee 0.10). Nodes still use Tailscale for pipeline traffic;
  only the coordinator moved. `/dashboard` is public at `http://150.230.22.250:8001/dashboard`.

**Stranger-NAT relay — BUILT, PROVEN, DEPLOYED (this session).** New `relay.py` (public host) +
`tunnel_client.py` (node), pure stdlib / ARM-safe / protocol-agnostic byte-splice → **zero changes to
node_*/common**. A node makes only OUTBOUND connections to the relay (behind NAT); the relay exposes a
public port and reverse-tunnels to it (a tiny self-hosted ngrok). Local selftest PASS (50 KB binary +
8 concurrent, byte-exact). Deployed on the cloud VM as systemd `neuron-relay` (control 8010, data 8011,
public 9000-9100). **LIVE PROOF PASSED:** opened `8010-9100` in the Oracle security list, ran
`tunnel_client` on node_b (OptiPlex, outbound-only), and ran a real inference with the node_c→node_b hop
forced through the relay (`node_a.py --host-b 150.230.22.250 --port-b 9002`) — over the **public
internet, no Tailscale** for that hop → correct answer. That's the exact path a stranger's NAT'd machine
uses. Test tunnel torn down after; node_b back to normal Tailscale.

**Scaling plan captured in new `SCALING.md`** (prototype → worldwide: P2P + relay fabric, regional →
DHT coordination, many-small-pipelines-not-one; Petals as the proven reference; rule: don't build the
scale layer before the first stranger). The 1-VM coordinator+relay is a Phase-1 prototype (~100 relayed
nodes ceiling), which is correct for now.

**Relay onboarding AUTOMATED (this session).** Coordinator `/node/register` accepts `behind_nat` →
auto-assigns a relay port from the pool (config `RELAY_*`) + stores the node at the relay endpoint +
returns a `relay` block; `agent.py` auto-starts `tunnel_client` from it (via new
`tunnel_client.run_tunnel()`, persisted in config for re-runs) → **a NAT'd node self-configures with
zero manual steps**. Isolated test PASS (behind_nat → port 9000 → reachable via the cloud relay
byte-exact using only the coordinator's response); redeployed to the cloud coordinator (DB persisted,
3 nodes intact, `behind_nat` register live-verified).

**Still pending for S12 (a real stranger):** an actual outside person installs the agent — plus the
open-join model (registration still needs a shared secret; a real open network wants proof-of-compute /
reputation, ROADMAP S16). Repo still PRIVATE. Model output quality (small 1.5B) still to discuss.
Speed: int8 3.46× but naive breaks quality (`PROBLEMS.md` [P2]/[P9]).

---

## Session 14 (2026-07-25) — heterogeneity-aware auto-balancing

**Goal:** the coordinator assigns each node an optimal layer slice from its MEASURED speed —
no more manual `--s1/--s2`. Nodes differ: self-benchmarked node_a **8.87** ms/layer (+**38.3**
ms head), node_b **12.22**, node_c **12.41**.

**Built:**
- `coordinator/balancer.py` — closed-form solver. Node i does k_i layers at s_i ms/layer plus
  fixed cost H_i (lm_head on the driver); equalize stage times T = s_i·k_i + H_i with Σk_i = L
  ⇒ **T = (L + Σ H_i/s_i) / Σ 1/s_i**, then round to ints summing to L (largest remainder).
  Pure Python, no torch.
- `benchmark.py` — a node self-measures ms/layer (times real Qwen2 decoder-layer decode passes)
  + head_ms (lm_head GEMM), reports JSON. Reuses common; ARM-safe.
- Coordinator: `/node/register` accepts `ms_per_layer`/`head_ms` (models.py schema + migration
  for existing DBs); `GET /network/plan` (advisory balanced split + speedup vs equal),
  `POST /network/rebalance` (applies it — updates stored ranges). register_nodes.py sends the
  measured speeds. **node_*/common UNCHANGED.**

**Verified:**
- Solver reproduces the hand-tuned **9/9/10** from measured speeds (sanity cases pass too).
- End-to-end (isolated coordinator): 3 nodes register with real speeds → `/network/plan` =
  9/9/10 (node_a 0-8 / node_c 9-17 / node_b 18-27, stages 112–122 ms), bottleneck 122 vs
  equal-split 127 ms = **1.04× faster**; `/rebalance` applied it with full 0-27 coverage. PASS.
- **LIVE on the cloud coordinator:** redeployed (DB migrated, 3 nodes intact); nodes
  re-registered with speeds; live `/network/plan` returns the balanced 9/9/10. Left the running
  split at 10/9/9 (working) — applying a new split live needs the driver/nodes to reload with the
  new ranges (proven in the isolated `/rebalance` test; on the live net = restart the driver with
  the new S1).

**Honest:** the 1.04× gain is small on this near-homogeneous trio (all similar CPUs); the win
grows with heterogeneous hardware (a fast GPU node + slow phones). The real value is REMOVING
manual tuning — the coordinator now derives the optimum that took hand-tuning in Session 5. Full
dynamic re-balance-on-join (auto-reload) is the natural extension.

---

## Session 15 (2026-07-25) — model registry + RAG (current info despite the cutoff)

**Goal:** NEURON isn't locked to the model's training cutoff — retrieve current web context
before inference; and track available models so more can be added.

**Built:**
- `rag/retriever.py` — before inference, DuckDuckGo web search (via `ddgs`, no API key) →
  compact context → inject into the prompt. Fails **soft** (no internet/results → original
  prompt, inference still runs). `retrieve_and_augment(prompt) -> (augmented, sources)`.
- `coordinator/model_registry.py` — config-driven catalog (id, layers, description);
  `list/get/resolve`; env `NEURON_EXTRA_MODELS` to add more. Coordinator **`GET /models`**;
  API **`/v1/models` now registry-driven**.
- Wired RAG into the Chat UI: a **🌐 Web search** toggle; the driver retrieves + augments when
  on, streams a `sources` event, chat.html shows "grounded on: [links]". **node_*/common
  UNCHANGED.** New dep: `ddgs`.

**Verified — success metric MET:** with web search ON, *"What are the latest AI model releases?"*
→ grounded on real July-2026 sources → answered *"…include Claude Opus 5 by Anthropic, released
July 24 2026…"* — info the 1.5B model (≈2023 cutoff) **could not know** without retrieval. Retriever
+ registry tested standalone; `/models` + `/v1/models` live (cloud coordinator redeployed, 3 nodes
intact). RAG directly helps the small model's weak/incomplete replies by grounding it.

**Scope note:** the model registry is the catalog + selection surface; nodes actually SERVING
multiple models (per-request routing, extra RAM) is the extension — the network still serves the one
default model. RAG uses search snippets (fast); full page-fetch + reranking is a later upgrade.

---

## Session 16 (2026-07-25) — security hardening (proof-of-compute, reputation, rate limiting)

**Goal:** safe for strangers to install, safe for users to trust — catch nodes that return
garbage to farm NRN, and add basic abuse protection.

**Built:**
- `security/proof_of_compute.py` — a verifier challenges a node (known input for its layer
  range), runs the same layers locally, and compares. **Honest work matches ~1e-5; garbage or
  lazy (echo-input) cheating is off by ~25+** (`atol=0.05` separates). Challenges a last-stage
  node (`layers[s2:n]` + norm) over the wire protocol; reuses common. Middle-node = extension.
- **Coordinator reputation:** `challenges_passed/failed` per node (models.py + migration);
  reputation = pass-rate; a node with ≥3 samples and pass-rate <0.6 is **flagged** and excluded
  from routing AND coverage (router.py + `_network_summary`). `POST /node/{id}/attest {passed}`
  (register-secret gated). Config `REPUTATION_MIN_SAMPLES` / `THRESHOLD`.
- **Rate limiting:** per-IP middleware, `RATE_LIMIT_MAX` (120) / `RATE_WINDOW` (60 s) → 429.
- `SECURITY.md` — the trust model + manual pre-launch items. **node_*/common UNCHANGED.**

**Verified:**
- Proof-of-compute LIVE against real node_b: honest passed (max_err **5.5e-05**), garbage failed
  (27.6), lazy-echo failed (25.3).
- Reputation loop (isolated coordinator): node_b failed 3 → flagged → layers 19-27 dropped
  (19/28, unhealthy, excluded from routing); node_a passed 3 → reputation 1.0.
- Rate limit: 60-request burst → 32× 429.
- LIVE end-to-end on the cloud coordinator (redeployed, DB migrated, nodes intact): challenged
  real node_b → passed → `POST /attest` → node_b reputation 1.0 recorded.

**Not built (needs a cert / ops):** agent **code signing** (Authenticode + Linux) — documented
in SECURITY.md as a pre-distribution step. **Open join** (drop the shared secret, gate on
proof-of-compute + reputation) is the natural next step now that the primitives exist.

---

## Session 17 (2026-07-25) — open join (drop the shared secret)

**Goal:** a true stranger can register a node with NO shared secret, but cannot serve live
traffic or earn NRN until proven — the front-door piece of the first-stranger milestone
(FIRST_STRANGER.md Path A, step 3). Uses the S16 proof-of-compute/reputation primitives.

**Design — three node standings:**
- **trusted** — registered *with* the valid `X-Register-Secret`; skips probation (the dev
  trio via `register_nodes.py`). Grandfathered: a migration sets `trusted=1` on every
  pre-existing node so opening the door doesn't demote the live network.
- **probationary** — registered *without* the secret (open join). Reachable/challengeable,
  but **excluded from routing, balancing, coverage, and earning**.
- **verified** — a probationary node that has passed proof-of-compute ≥ `PROBATION_MIN_PASSES`
  (default 1). **flagged** (S16, failed PoC) still overrides everything → excluded.
  Single predicate `eligible = (not flagged) and (trusted or passed)` gates routing + earning.

**Built:**
- `config.py` — `OPEN_JOIN` (env `NEURON_OPEN_JOIN`, default **on**), `PROBATION_MIN_PASSES`.
- `models.py` — `trusted` column + CREATE + migration (backfills existing→1); `register_node(trusted=)`;
  `_node_dict` computes `trusted`/`eligible`/`standing`.
- `main.py` — `/node/register` no longer secret-gated: `classify_registration()` → trusted vs
  probationary (or 401 when `OPEN_JOIN=0`); **hijack guard** (a secret-less register of an
  existing *trusted* id → 409); response carries `standing` (+ note). `_network_summary` and
  `_balanced_plan` now use `eligible`; dashboard shows a colored **standing** column;
  `_network_summary` reports `probationary_nodes`.
- `router.py` / `ledger.py` — route and credit **only `eligible` nodes** (probationary earns 0).
- `agent/agent.py` — registers with **no secret by default** (only sends the header if the
  operator sets `register_secret`); logs its standing; removed the hardcoded dev secret.
- `security/proof_of_compute.py` — new `attest_via_coordinator()` + CLI `--coordinator/--node-id`:
  look a node up in `/node/list`, challenge its last-stage slice, POST `/node/{id}/attest` →
  a passing probationary node is promoted. (Last-stage only, as in S16; middle-node = extension.)

**Verified:** `coordinator/test_open_join.py` — **17/17 PASS** (temp DB, real endpoint/router/
ledger/models fns): secret→trusted, no-secret→probationary, probationary excluded from chain
+ earns 0, PoC pass→verified→routed→earns, trusted-id hijack→409, `OPEN_JOIN=0`→401, flagged
overrides, legacy DB grandfathered→trusted. App imports/builds; all edited files byte-compile;
`register_nodes.py` still sends the secret (trio stays trusted). **node_*/common UNCHANGED.**

**Not done / next:** `/infer/{id}/complete` is still **unauthenticated** and mints from
caller-supplied `node_ids` (PROBLEMS.md [P12]) — open join makes strangers able to reach it, so
authenticating it + settling from the coordinator-recorded plan is the next security step. PoC
still covers **last-stage nodes only** (place the first stranger on the final segment). **NOT
deployed to the live cloud coordinator** — deploy = redeploy code + restart (DB auto-migrates,
grandfathering the 3 nodes as trusted). Then FIRST_STRANGER steps 4-7 (4th-node placement,
package agent, install guide, a friend runs it).

---

## Session 18 (2026-07-25) — the 4th node via REPLICATION (first-stranger Path A, step 4)

**Goal:** let a 4th machine join and earn without deepening the pipeline. Considered three
shapes; chose replication after tracing the code.

**Why replication, not a 4-stage re-split:** a deeper pipeline is SLOWER per single request
(PROBLEMS.md [P8]) and would force changes to `node_c.py`/relay/driver (risking
`selftest_shard` bit-exactness). Replication is the [P8]-correct throughput shape AND -- the
deciding factor -- **each assembled chain stays the usual 3-stage `driver->middle->last`**, so
the drivers' hardwired `len(chain)==3` still holds. Net result: **coordinator-only change;
`node_*.py`, `neuron_driver.py`, `common.py` all UNCHANGED.**

**Built (router only):** `router.build_chain(now, pick=random.choice)` -- when several eligible
nodes cover the SAME farthest segment from a cursor they are REPLICAS; the router picks one per
call (default random), so concurrent requests spread across them and both earn. `pick` is
injectable for deterministic routing/tests. A 4th node is placed simply by **registering with an
existing node's layer range** (e.g. a second `19-27`); open-join means a stranger's replica joins
probationary -> verified via proof-of-compute -> then becomes selectable. No new endpoints; the
dashboard already shows two rows with the same range = replicas.

**Verified:** `coordinator/test_replica.py` -- **9/9 PASS**: both replicas selected across 300
calls (load spreads); every chain complete + 3-stage; injected picker forces either replica;
earnings follow the chosen replica (the other earns nothing that request); a probationary replica
is NEVER routed until a PoC pass; and the chain stays complete when only the stranger's replica
remains. Open-join suite still 17/17; app builds; a live 4-node register yields a 3-node chain the
drivers accept, last slot alternating `b`/`b2`.

**Honest limits:** replication lifts THROUGHPUT under concurrent load (many requests / the
parallel driver / multiple UI users); it does NOT speed a single request (correct, per [P8]). The
auto-balancer (`_balanced_plan`) still assumes a contiguous partition, so replicas are a MANUAL
placement -- don't run `/network/rebalance` on a replicated set (it would re-partition into 4
contiguous stages, which the 3-stage driver can't run). **Not deployed to the live coordinator.**
Next Path-A: package the agent + install guide, then a real stranger runs a last-segment replica
end-to-end (steps 5-7). PoC still last-stage-only, which fits (place the stranger on `19-27`).

---

## Session 19 (2026-07-25) — authenticate /complete + settle from the recorded plan ([P12])

**Goal:** open join made the coordinator stranger-reachable, but `/infer/{id}/complete` was
unauthenticated and paid out to CALLER-SUPPLIED `node_ids` — anyone reaching the coordinator could
mint NRN to any node. Close the security hole WITHOUT the economics rewrite (still-minting +
wallets/debit = TOKENOMICS §11, post-first-stranger).

**Built (coordinator + driver clients):**
- `/infer` now records the chain IT chose (`plan_node_ids`) and issues a per-request
  `complete_token` (returned to the caller). `requests` table gained both columns (+ migration).
- `/complete` requires the token (`secrets.compare_digest`; wrong/missing → 401, request stays
  pending, nothing credited) and **settles from the recorded plan, never `body.node_ids`** — so a
  completion can only ever pay the nodes the coordinator actually routed (incl. the chosen replica).
  `tokens_generated` clamped to `max_tokens`. 404/409 paths unchanged.
- Driver clients threaded the token through: `node_a.coord_get_chain` returns it (8-tuple),
  `node_a.coord_complete(..., complete_token)` sends it; `node_a.run_request`/`_run` and
  `neuron_driver.stream` pass it. So the Chat UI + OpenAI API (both go through `neuron_driver` →
  `node_a`) are covered with no changes to `ui/app.py` or `api/openai_compat.py`. **common.py and
  the node_b/node_c servers UNCHANGED** (node_a is the driver/client, not inference math).

**Verified:** `coordinator/test_complete_auth.py` — **13/13 PASS**: /infer issues a token + persists
the plan; wrong/missing token → 401 with no credit and still-pending; a completion that LIES about
`node_ids` (claims the unchosen replica + a ghost node) pays the recorded plan only (ghost never
credited, unchosen replica earns nothing); tokens clamped; 409 double-complete; 404 unknown. All
three coordinator suites green (open_join 17, replica 9, complete_auth 13); node_a + neuron_driver
import cleanly (tuple arity OK).

**Still open (deferred by design):** the ledger still MINTS 1.0 NRN/request with no user debit and no
fixed-supply enforcement — the §11 economics (genesis buckets, wallets, debit, settle) come AFTER the
first stranger. Server-side token RECOUNT (vs the clamp) is future. **Not deployed to the live
coordinator.** Next Path-A: package the agent + install guide, then a real stranger joins (steps 5-7).

---

## Session 20 (2026-07-25) — zero-config auto-placement for a joining node (first-stranger Path A)

**Goal:** a stranger should never pick layer numbers. Today `agent/config.json` hardcoded
`layer_start/end = 10/18` — a middle segment that collides with node_c AND can't be verified
(proof-of-compute is last-stage only), so a stranger literally couldn't earn.

**Built (coordinator + agent):**
- `router.suggest_placement()` — fill the first coverage GAP if the eligible chain is incomplete;
  else the chain is complete, so **replicate the LAST segment** (verifiable via PoC, adds throughput
  via S18 replica routing). Returns `{layer_start, layer_end, role, reason}`.
- `GET /node/placement` — advisory, no auth (a node calls it before it has a token), rate-limited by
  the middleware. Returns `total_layers` + the placement.
- `agent/agent.py` — `ensure_placement()` (called at the top of `register()`): if config has no layer
  range, fetch `/node/placement`, persist and log it. So the agent self-configures.
- `agent/config.json` — `layer_start/end` → **null** (auto-place) and `behind_nat` → **true** (a public
  stranger is reachable via the relay with no inbound port).

**Verified:** `coordinator/test_placement.py` — empty net → fill-gap over all layers; missing last
segment → fill-gap 19-27; complete chain → replica-last 19-27; a PROBATIONARY node covering the gap does
NOT count as coverage (still a gap) until PoC-verified; endpoint returns total_layers + role. **6/6 PASS;
all four coordinator suites green (open_join 17, replica 9, complete_auth 13, placement 6 = 45).
node_*/common UNCHANGED** (agent only).

**Net effect for the first stranger:** install → run the agent → auto-placed on a verifiable last-segment
replica → founder verifies via proof-of-compute → earns under load. Combined with S17 (open join), S18
(replica), S19 ([P12] complete-auth): the coordinator is now safe to expose and a stranger self-onboards.
**Still uncommitted (S19 + S20). Not deployed to the live coordinator.** Next Path-A: install guide (step 6)
+ package agent (step 5), then a real friend runs it (step 7).

---

## Session 21 (2026-07-29) — the wire: 4.3× smaller, and it stopped executing strangers' code

**Goal:** `PETALS_NOTES.md` ranked "quantize the network path" first. Split it apart and do
the half that was actually blocked on nothing: the activations crossing between nodes.

**What was found before writing any code.**
- `common.recv_msg` called `torch.load(..., weights_only=False)` on whatever arrived on the
  socket. That is pickle: **any peer could execute code in the receiving process**, in both
  directions, and since S12 node ports are published on a public relay so the sender need
  not even be in the chain. Demonstrated with a crafted message. Nothing in `SECURITY.md` or
  the S14 audit had ever named it. → [P19]
- The fp32 wire cost **12,508 bytes per message**, 1,153 of it pickle framing, once per
  token per hop. Never measured until now. → [P20]
- **The notes' weight-size claim was wrong** and is corrected in `PETALS_NOTES.md`: HF ships
  Qwen2.5 as BF16 and `slice_downloader` copies bytes verbatim, so the *download* was never
  fp32. The fp32 is `load_slice_model` upcasting. Download 2.00 B/param, resident 4.00 —
  a **RAM** problem (14 GB/node at 70B), not a bandwidth one, with a different fix.
- [P9] pencilled in llama.cpp's RPC backend as the quantized+distributed pivot. Its own docs
  say "fragile and insecure… **never run the RPC server on an open network**". Fine for a
  trusted cluster, disqualified for open join. Worth knowing before building on it.

**Built:** `wire_codec.py` — a length-prefixed **JSON header + raw tensor bytes** frame
(nothing executable), with three codecs negotiated per hop in the config handshake.
- `i8h` = **Hadamard rotation, then blockwise int8**. Real junction activations measured at
  absmax 6620, std 42, **worst channel ≈ 750× the median** — so an absmax scale is set by
  one channel and everything else collapses. That is [P9] again, on the wire. QuaRot's fix
  applied at the *transport* layer: the rotation is orthogonal, so the sender rotates before
  quantizing and the receiver rotates back — **no weight surgery, no calibration**, the model
  never sees it. Same bytes as unrotated int8, ~7× less error.
- Done as one matmul against a cached Hadamard matrix, not the textbook butterfly: 0.045 ms
  vs 1.26 ms at H=8192. The butterfly cost a fifth of the wire time it was saving.
- Scales travel fp32, not fp16: the rotation preserves block L2 norm, so a large block would
  have overflowed an fp16 scale to inf and decoded as **silent zeros**. Costs 0.8%.
- Negotiation degrades: a peer that offers nothing recognised stays on the legacy format, so
  a half-upgraded fleet keeps working. Proof-of-compute deliberately offers nothing — a
  lossy reply would spend its `atol` budget on transport noise instead of hardware jitter.

**Measured** (`bench_wire.py`, codec at all three junctions so error compounds as on the wire;
6 prompts × 48 tokens, greedy, vs the fp32 baseline):

| codec | B/msg | vs before | identical | max Δlogit |
|---|---|---|---|---|
| `torch.save` fp32 (was) | 12508 | 1.00× | 6/6 | 0.0000 |
| `f16` | 5723 | 2.19× | 6/6 | 0.0069 |
| **`i8h`** | **2946** | **4.25×** | **6/6** | 0.2054 |

Rejected, all at the same size as i8h: fp8 e4m3 → **NaN** (its max is 448; activations reach
6620), int8 per-tensor → Δlogit 30.5, int4 → 4.5. **Petals' own scheme — blockwise int8, no
rotation — diverged on 1 of 3 prompts.** Copying the paper's mechanism verbatim was not
enough; that is the session's real finding.

**Then the same benchmark on a second model disagreed.** Qwen2.5-0.5B (H=896): i8h scores
**3/6**, f16 stays 6/6. The divergences are re-wordings, still correct, typically 100+
characters in — drift, not collapse — but drift the bigger model doesn't show. So
`preference()` offers i8h only at **H ≥ 1536**, f16 below. Small models are both the fragile
ones and the cheap ones to ship uncompressed, so nothing is traded away. Two data points,
not a curve. `NEURON_WIRE_CODEC` pins a codec (a LAN wants f16/f32: i8h's 0.54 ms of CPU
buys 6.4 ms on a 10 Mbit/s home upload but only 0.06 ms on a fast link).

**A false lead, recorded so it isn't re-derived.** An end-to-end socket run seemed to show
i8h giving a factually worse 0.5B answer. It was the test rig: a stray `node_b.py` was still
bound to the port and `SO_REUSEADDR` let a second bind alongside it. One listener → all four
codecs agree. The size gate rests on the in-process benchmark, which has no sockets.

**Verified:** `test_wire_codec.py` **27/27** (round-trips at 5 shapes × 3 codecs, the
orthogonality and outlier-flattening properties, the fp16-scale overflow regression, a
hostile pickle refused, an absurd length prefix refused, a legacy sender still readable),
`test_relay_auth.py` 13/13 still green. End-to-end over **real sockets**, 3 processes: a
driver offering `[i8h,f16,f32]` negotiates i8h (1138 B), `[f16,f32]` → f16 (1882 B),
`[f32]` → f32 (3674 B), and a driver sending **no** `wire` field or an unknown codec falls
back to legacy — all five return a correctly-shaped hidden.

**What this implies at the size NEURON exists for.** 70B is H=8192 over ~20 stages: one
decode token cost 0.69 MB across the chain, ≈0.55 s/token of pure serialisation on a
10 Mbit/s home upload. At i8h it is 0.17 MB and ≈0.13 s. [P3] observed the network dominates
per-token cost; this is one reason why.

**Not deployed.** The Pavilion and OptiPlex still run the old build and will negotiate down
to legacy until updated. Next from `PETALS_NOTES.md`: the weight/RAM half (gap 2), then
junction caching (gap 3).

---

## Session 22 (2026-07-31) — evaluated a hand-written AVX2 int8 kernel as the node engine

**Goal:** replace PyTorch as the node-side inference engine with an experimental CPU-native
int8 compiler + AVX2 kernel, and measure tok/s against the PyTorch baseline.

*(Kernel sources are deliberately untracked — see `.gitignore`. Only the measurements and the
decision are recorded here, since those are what future sessions need.)*

**Measured, on the Pavilion (idle, gcc 13.3, AVX2, PyTorch single-thread), against REAL
Qwen2.5-1.5B layer-10 weights — not random Gaussians:**

| matrix | PyTorch fp32 | int8 AVX2 | speedup | rel err |
|---|---|---|---|---|
| `q_proj` 1536×1536 | 0.810 ms | 0.371 ms | 2.18× | 0.034 |
| `o_proj` 1536×1536 | 0.819 ms | 0.372 ms | 2.20× | 0.041 |
| `gate_proj` 8960×1536 | 3.532 ms | 2.321 ms | 1.52× | 0.058 |
| `down_proj` 1536×8960 | 3.499 ms | 1.041 ms | 3.36× | 0.070 |
| **GEMM total** | **8.660 ms** | **4.105 ms** | **2.11×** | |

**The kernel is real. The headline number in its own harness is not.** That harness reports
9–20× because it benchmarks against a naive C scalar triple loop. Against the baseline NEURON
actually runs — PyTorch's BLAS GEMM — it is **2.11×** on a real layer's Linears. Both figures
are correct; only one is the relevant comparison.

**Three findings that block the integration as originally scoped:**

1. **The compiler and the kernel do not connect.** The compiler's output encoding and the
   kernel's expected input encoding are different things; grepping all three C sources for
   the compiler's format magic returns **zero** hits — nothing reads it.
2. **That encoding is 2.7× LARGER than fp32 on real weights.** At the compiler's own default
   sparsity threshold it keeps 89% of `gate_proj`'s weights → **147 MB vs 55 MB fp32, vs
   13.8 MB for the dense int8 the kernel actually wants.** It only compresses on sparse
   matrices; transformer weights are dense. Raising the threshold to keep 18% shrinks it to
   30 MB but discards 82% of the model.
3. **The Windows driver machine has no C compiler at all** — no gcc, clang, MSVC or Visual
   Studio. The kernel builds and runs only on the two Linux nodes.

**What was built and kept:** the SIMD source compiles cleanly as a shared library
(`-O3 -mavx2 -shared -fPIC`) exporting `matmul_i8_avx2` / `matmul_i8_avx2_4x`, and a ctypes
harness drives it from Python against real safetensors weights. That is the viable
integration path if this is picked up again — dense int8 packing + ctypes, *not* the sparse
compiler.

**Numerical limit worth recording:** the kernel accumulates `madd_epi16` into **int32**, so
`sum(|w_i16 · x_i16|)` over `id` terms must stay below 2³¹. At `id`=1536 with int8 weights
near ±127 that caps `|x_int16|` around 10⁴ — so `input_scale` cannot be the fixed 128 the
source comment suggests, especially given NEURON's measured activation absmax of ~6620
(`wire_codec.py`). The harness picks `input_scale` from the actual input range instead.

**Decision: do NOT make this the node engine.** Not because it doesn't work — it does — but
because `engine/local_gguf.py` already measured **6.7×** over fp32 with quality intact
(Q4_K_M, 36 ms/token vs 240), which is ~3× better than this kernel's 2.11×, and that 2.11×
covers only the Linear GEMMs — a decoder layer also runs RMSNorm, RoPE, softmax attention
over the K/V cache and SwiGLU, none of which the kernel touches, so end-to-end would be
strictly less. Add the 3.4–7.0% per-GEMM error against [P9], where naive int8 made this exact
model answer *"I'm sorry, but I can't provide an answer"*, and the trade is bad. The kernel
source's own closing verdict says the same thing: *the value is the distribution layer, not
the kernel — use llama.cpp as the kernel.*

**BUILT AND MEASURED ANYWAY (founder's call, and the right one).** `node_ns.py` +
`ns_engine.py`: a node server speaking the identical wire protocol, with every Linear in its
own layers swapped for a ctypes call into the kernel (dense int8 packing — the compiler is
not in the path, see above). Live A/B on the real 3-machine chain, same prompt, same 24
tokens, only the middle node's engine changing:

| middle node engine | throughput | answer |
|---|---|---|
| PyTorch fp32 | 1.57 tok/s | "…a phenomenon called Rayleigh…" |
| **int8 AVX2** | **1.77 tok/s** | **identical** |

**+12.7% end-to-end from converting one of three nodes**, answer unchanged. The node's own
`stats` message confirms the kernel really ran rather than silently falling back: **63**
Linear layers converted (9 layers × 7), **1449 kernel calls** = 23 decode tokens × 63, and
**63 fallbacks** = exactly one prefill pass. Prefill deliberately stays on PyTorch — the
kernel is mat-vec, so N scalar calls lose to one batched GEMM.

**Then converted the OptiPlex too — 2 of 3 nodes on the kernel.** Three runs of each config,
same prompt, same 24 tokens, answer identical every time:

| nodes on int8 | runs (tok/s) | mean |
|---|---|---|
| 0 of 3 (all PyTorch) | 1.62, 1.57, 1.63 | **1.61** |
| 1 of 3 (Pavilion) | 1.77 | 1.77 |
| **2 of 3 (+ OptiPlex)** | 1.98, 1.85, 1.97 | **1.93** |

**+20% end-to-end.** The ranges do not overlap (PyTorch max 1.63 < int8 min 1.85), so this is
signal rather than noise. Both nodes verified on the kernel via their `stats` message: 63
Linears converted each, and the OptiPlex's 1449 calls = exactly 23 decode tokens × 63.

**Then the driver too — 3 of 3.** MinGW 16.1.0 installed via Chocolatey (needed an elevated
shell; `winget` is absent on this machine), kernel built as a self-contained Windows DLL with
the `ns_win_compat.h` shim:

| nodes on int8 | runs (tok/s) | mean | vs baseline |
|---|---|---|---|
| 0 of 3 (all PyTorch) | 1.75, 1.66 | **1.71** | — |
| 2 of 3 (remotes only) | 1.98, 1.85, 1.97 | 1.93 | +13% |
| **3 of 3 (+ driver)** | **2.78, 2.75, 2.73** | **2.75** | **+61%** |

**The driver is where the win is**, and by a wide margin — converting it alone moved more
than both remote nodes combined. Because it holds `lm_head`, the largest GEMM in the whole
pipeline: 151936×1536, measured on Windows at **37.78 ms → 16.09 ms (2.35×)**. `convert()`
walks `model.model.layers`, so `lm_head` sits outside it and had to be converted explicitly —
easy to miss, and missing it would have forfeited most of the gain. Driver reports **71**
Linears converted = 10 layers × 7 + the head. Answer identical to fp32 at both 24 and 40
tokens.

**Two build notes worth keeping:**
- The kernel calls C11 `aligned_alloc`, which MinGW-w64 lacks. The shim maps it to plain
  `malloc` rather than `_aligned_malloc`, because the sources release aligned AND ordinary
  pointers through the same `free()` — redirecting `free()` globally would corrupt the heap.
  Safe because every vector access is `_mm256_loadu_si256` (24 unaligned loads across the
  three files, zero aligned ones), so the alignment was never load-bearing.
- A plain MinGW `-shared` DLL will not load under ctypes: it pulls in `libgcc`/`libwinpthread`
  which are not on the Python process's search path. Build with
  `-static -static-libgcc` for a self-contained DLL. And it does not compose with `NEURON_WEIGHT_DTYPE=fp16` — NSLinear
keeps the fp32 weight for prefill fallback, so it costs memory rather than saving it.

---

## Session 23 (2026-07-31) — the rest of the NeuronScript stack, measured: tiler and predictor both rejected

Goal: add `neuronscript_tiler.c` (the L3 tile scheduler) and `neuronscript_bitmask.c` (the
row predictor) on top of the shipped SIMD kernel and measure each step. **Both were
measured and both were rejected.** The SIMD kernel from Session 22 remains the only one in
the path.

### The three numbers

Interleaved simd/tiler runs back to back (see the confound below), 3 runs each, same prompt
("Why is the sky blue"), same 24 tokens, all 3 nodes on the int8 kernel:

| config | runs (tok/s) | mean | vs SIMD | answer vs PyTorch |
|---|---|---|---|---|
| **a. SIMD only** | 2.37, 2.45, 2.15 | **2.32** | — | identical |
| **b. SIMD + tiler** | 2.22, 2.41, 2.16 | **2.26** | **−2.6%** | identical |
| **c. + predictor** | — | **crashes** | — | **FAILS** |

(c) has no number because it never produced one: **exit `0xC0000374`, STATUS_HEAP_CORRUPTION.**

### Why the tiler cannot help (structural, not tuning)

`tile_rows_for(in_dim)` divides a compiled-in 16.5 MB budget by `in_dim`. Every Linear in a
Qwen2.5-1.5B decoder layer is *smaller than one tile*:

| linear | od × in | weight | n_tiles |
|---|---|---|---|
| q/o_proj | 1536×1536 | 2.4 MB | **1** |
| gate/up_proj | 8960×1536 | 13.8 MB | **1** |
| down_proj | 1536×8960 | 13.8 MB | **1** |
| lm_head | 151936×1536 | 233 MB | 14 |

`n_tiles = 1` means one tile = the whole matrix, i.e. the tiler's loop degenerates to
exactly the SIMD kernel's loop plus ping/pong copies. Measured locally on real weights at
batch=1: q_proj 0.197 vs 0.194 ms, gate_proj 1.231 vs 1.239, down_proj 0.899 vs 0.903,
lm_head 19.73 vs 19.86 — identical within noise, every one.

And the tiler's whole premise — amortise a tile load across a batch — is void at batch=1,
which is what decode is. Its one real effect is on **prefill**, where it replaces PyTorch's
blocked GEMM with a row-at-a-time kernel: **0.14×–0.56×, i.e. 2–7× slower** (batch=16:
q_proj 2.60 ms vs torch 0.370, down_proj 11.99 vs 3.695). That is why "tiler on every
forward pass" costs rather than pays — the end-to-end −2.6% is the prefill regression
showing up: driver layer compute 1.87 s → 2.47 s (+32%), head 1.07 → 1.37 (+28%), every run,
not noise.

The tiler is not useless in principle — it would engage on a model whose layers exceed
16.5 MB (70B: 8192×8192 = 67 MB → 4 tiles). It does nothing at 1.5B.

**One thing the tiler does do better:** per-row dequant scales instead of the mat-vec path's
one scale per tensor — lm_head rel_err 0.0109 vs 0.0297. Accuracy, not speed. Worth
harvesting into the SIMD path on its own.

### Why the predictor fails the quality gate

`exec_tile_masked` (neuronscript_bitmask.c:210) **zeroes every row it does not predict**.
That is not a rounding error, it is a deleted logit. On lm_head it forced 56,888 of 151,936
logits to exactly 0.0 while computing 62.6% of rows.

lm_head is also the *only* matrix the predictor can ever touch here — everything else has
n_tiles=1, so `full_system` always takes its `ti==0` full-compute branch and the predictor
never engages. So it can only act at exactly the place where argmax *is* the generated token.
Measured against PyTorch on real hidden states, **7/9 tokens** (simd and tiler both 9/9):

```
  ' sky'        -> '.sky'     WRONG
  ' phenomenon' -> '现象'      WRONG
```

It *is* faster on lm_head (11.5 ms vs 19.9, 1.72×) — by not computing 37% of the answer.

**And it corrupts the heap.** `RowMask` is `uint64_t bits[1024]` = 65,536 rows, but
`mask_set(m, row)` indexes `bits[row>>6]` with absolute row numbers. lm_head has **151,936**
rows → word index up to 2374 → a ~10.8 KB out-of-bounds write past the struct. Crash
confirmed end-to-end (`0xC0000374`). The unit test only survived it because a smaller heap
happened to absorb the overwrite. Any output ≥ 65,536 rows triggers this — every vocabulary
projection of every model we care about.

### The measurement trap that nearly produced a fake +73%

The first SIMD set measured **1.21 / 1.23 / 1.47 tok/s**; the first tiler set measured
**2.24 / 2.27 / 2.22**. Reported naively that is "+73% from the tiler". It is entirely
Wi-Fi: net time 4.8 s → 1.0 s between the two sets, while node compute was unchanged
(node_c 3.5 s in both). **Interleaving the configs run-by-run** collapsed the difference to
−2.6%. Never measure two engine configs in separate blocks on this network.

Root cause: node_c (Pavilion) is on Wi-Fi at **−73 dBm, Link Quality 37/70**, with Tailscale
ping swinging **44–148 ms** to a peer on its own LAN (node_b, wired, is 8 ms). It is also
running the founder's IRIS stack — `iris_voice.py` + `iris_widget.py` at ~52% CPU each on 4
cores, bursty (load 3.14 → 0.73 within minutes). Both left running: they were also up during
Session 22's 2.75 tok/s, so killing them would have made the comparison *less* comparable.

**Session 22's 2.75 tok/s did not reproduce today — best single run was 2.45.** Same code,
same nodes, same kernel. The delta is the Pavilion's link, not the engine.

**The real bottleneck is now the network, not the kernel.** In the worst runs the three
machines were 15%/14%/11% utilised — idle ~85% of the time waiting on the wire. Optimising
GEMMs further is optimising the 20–30% that is compute. Put node_c on Ethernet (`enp2s0` is
DOWN) before any further engine work; that is worth more than any kernel change measured
here.

### State

`ns_engine.py` already carries the full three-mode adapter (`NEURON_NS_MODE=simd|tiler|hybrid`,
`_TiledWeight`, `load_tiler`/`load_bitmask`, `tile_report`) and `node_a.py` gained
`--dump-json`. **The shipped default stays `simd`** — the other two modes are reachable only
by env var, and on this evidence should stay that way. `ns_tiler.dll` / `ns_bitmask.dll`
built locally (MinGW 16.1.0, `-static -static-libgcc`, `-include ns_win_compat.h`); the
remote nodes were never switched off `simd`, so the live network is untouched.

Note the founder's brief named `tiler_only_run()`; the actual exports are `tiler_run()`
(neuronscript_tiler.c) and `tiler_only()` (neuronscript_bitmask.c) — the adapter binds
`tiler_run`.

---

## Session 24 (2026-07-31) — the cube diagonal predictor: built, measured, rejected at Test 3

**Goal:** build `neuronscript_cube.c` — estimate which output rows matter by reading the main
diagonal of four 64×64 corner windows (256 values = 0.002% of the matrix), skip the rows below
0.3× the mean diagonal magnitude, run the rest on the AVX2 4-row kernel. Target: ≥20% of
`down_proj` rows skipped with tokens unchanged.

*(Kernel source deliberately untracked, added to `.gitignore` before anything else was written,
along with the `cube_check.py` / `cube_corr_audit.py` harnesses. Only measurements here.)*

**Tests 1 and 2 pass. Test 3 fails, and the stop rule applies — Tests 4 and 5 were not run.**

### Test 1 — build (Pavilion, gcc 13.3) ✅
Clean with the specified command, and still clean under `-Wall -Wextra`. The session-23 heap
corruption is fixed and regression-tested in the binary: `RowMask` is `calloc((od+63)/64, 8)`
sized from the real `out_dim`, every set/get is range-checked, and a mask at od=**151936**
(lm_head, the shape that crashed) allocates 2374 words and rejects out-of-range indices.

### Test 2 — synthetic weights ✅ (but the predictor never engaged)
Max abs error vs fp32: down_proj 4.12% of output std, gate_proj 3.83%, q_proj 3.35% — all under
the 5% bar. **Rows skipped: 0.0%.** The error measured is therefore pure int8 quantisation, not
prediction error: random Gaussian weights have no diagonal structure by construction, the
measured correlation was 0.007, and the >0.4 correlation gate correctly refused to enable the
predictor. A pass, but not evidence for anything.

### Test 3 — real Qwen2.5-1.5B weights ❌

| matrix (layer 10) | corr (all rows) | enabled | rows skipped | oracle-skippable |
|---|---|---|---|---|
| `down_proj` 1536×8960 | **−0.0152** | no | **0.0%** | **0.0%** |
| `gate_proj` 8960×1536 | −0.0129 | no | 0.0% | 0.0% |
| `up_proj` 8960×1536 | +0.0176 | no | 0.0% | 0.0% |

**Two independent failures, either one fatal.**

**(a) The 0.509 correlation is a sample-size artifact.** `diagonal_check.py` hardcodes
`n_samples = 20`. Porting its correlation math verbatim and varying only that count, on the
same layer-10 weights:

| matrix | n=20 | n=50 | n=100 | n=400 | n=1536 (all rows) |
|---|---|---|---|---|---|
| down_proj | **+0.5095** | +0.2640 | +0.3291 | +0.1804 | **+0.1605** |
| gate_proj | +0.2477 | +0.0499 | +0.1481 | +0.1930 | +0.1880 |
| up_proj | +0.4192 | +0.1024 | +0.0738 | +0.1323 | +0.1456 |

The n=20 column reproduces the briefed 0.248 / 0.509 / 0.419 almost exactly, so the port is
faithful — those numbers are real, they are just twenty points. Measured against every row they
decay to 0.15–0.19, below the 0.4 bar. And 0.16 is the *generous* variant, which reads a window
at every position along the diagonal; the algorithm as specified interpolates four corners, and
that estimate scores **−0.015**, because it is nearly flat: its row estimates span only
1.12× min-to-max while the true row magnitudes span 9.47×.

**The corner-to-center 0.8% prediction is the same illusion.** down_proj's 64-long diagonals:
TL 0.0186, TR 0.0243, BL 0.0200, BR 0.0182, CENTER 0.0190 — and eight windows sampled at
**random**, off any diagonal, land in 0.0179–0.0233, the same band, around a whole-matrix mean
of 0.0210. Corners predict the centre because every 64-element window of this matrix has
roughly the same mean |w|. A random window predicts it equally well. That is homogeneity, and
homogeneity is precisely what leaves nothing to skip.

**(b) The 0.3× rule cannot skip a row of down_proj at any layer — even with perfect knowledge.**
This one does not depend on the estimator at all. Per-row mean |W| across all 28 down_proj:

| | min/mean ratio | rows below 0.3× mean | rows below 0.6× mean |
|---|---|---|---|
| best case for skipping (layer 25) | 0.218 | **0.7%** | 3.0% |
| worst (layer 27) | 0.817 | 0.0% | 0.0% |
| **24 of 28 layers** | ≥ 0.308 | **0.0%** | 0–5.3% |

A row can only be cut when its magnitude falls under 0.3× the mean; the weakest row in the
weakest layer sits at 0.218× and the other 27 layers never get near it. So an **oracle** holding
the true row magnitudes skips **0.0%** on 24 of 28 layers and at most **0.7%** anywhere. The 20%
target is unreachable by construction, and doubling the cut to 0.6× still tops out at 5.5%.

The premise inverted a statistic: "real spike fraction only 0.6% of rows" measured rows *above*
1.5× the mean. It says nothing about a tail *below* 0.3× — and that tail is empty. Trained
transformer weight rows are tightly clustered; the sparsity that makes MLP rows skippable is in
the **activations**, which are input-dependent, not in the weights. This predictor is a function
of the weights alone, so its active-row set is fixed at load time and identical for every token.

### Why Tests 4 and 5 were not run
The stop rule, and they would measure nothing. The correlation gate holds the predictor off
permanently on every real matrix, so `NSLinear` would run today's int8 path plus mask
indirection: `selftest_shard.py` would pass and tok/s would come back at or slightly below the
2.32 baseline, and neither number would be about the cube idea.

### What is worth keeping
- **The RowMask fix and its regression test** — session 23's `STATUS_HEAP_CORRUPTION` is a live
  bug in `neuronscript_bitmask.c` for any output ≥ 65,536 rows. The pattern here is the fix.
- **The two gates did their job.** The correlation gate refused three matrices on their own
  measured structure rather than on an inherited claim, and the quality gate (double-compute,
  disable on max abs err > 0.05) never had to fire because nothing got past the first gate.
  Cheap, and the reason nothing wrong ever reached the model.
- **Per-row dequant scales** are implemented here in the mat-vec path — the accuracy win session
  23 flagged as worth harvesting (lm_head rel_err 0.0109 vs 0.0297). Independent of the
  predictor and still worth folding into `ns_engine.pack`.
- **Method note:** measure a correlation on every row before believing it. n=20 over-reported by
  3.2× here, and 20 points was enough to make a structural claim look STRONG.

Session 23's conclusion is unchanged and still the priority: the bottleneck is the Pavilion's
Wi-Fi (−73 dBm, nodes 11–15% utilised), not the GEMMs. Ethernet before any further engine work.

---

## Session 25 (2026-08-01) — Llama 3.3 70B on one machine: the llama.cpp number, and why the NeuronScript comparison could not be run

**Goal:** the decisive NeuronScript-vs-llama.cpp test at 70B. **Steps 1-2 ran. Steps 3-5 are
not executable on this hardware** — the reasons are arithmetic and were confirmed before the
download started, not discovered after.

### Step 2 — llama.cpp baseline, measured

`bartowski/Llama-3.3-70B-Instruct-GGUF` Q4_K_M, **42,520,398,816 bytes (42.5 GB)**, single file
(not split). Windows driver PC, 63.3 GB RAM, 15 threads, n_ctx 2048, raw completion (matching
`llama-cli -p`, no chat template), prompt `"Why is the sky blue"`, `-n 100`, temperature 0.
Driven through `llama_cpp` 0.3.34 — there is no `llama-cli.exe` on this machine, same engine.

| run | load | wall | tok/s | ms/token |
|---|---|---|---|---|
| 1 (cold) | 118.0 s | 278.10 s | 0.360 | 2781 |
| 2 (warm) | 75.6 s | 159.40 s | **0.627** | 1594 |
| 3 (warm) | 91.2 s | 165.13 s | **0.606** | 1651 |

**Steady state ≈ 0.62 tok/s.** The cold run is 1.7× slower purely from paging 42.5 GB into
52.8 GB of free RAM; anyone quoting a single cold 70B number is quoting their disk. Output was
**byte-identical across all three runs** (greedy), and factually correct: *"a phenomenon called
Rayleigh scattering, which is the scattering of light by small particles or molecules in the
atmosphere."* Raw completion continues the prompt as an article rather than answering directly
— faithful to `-p`, not the shape to ship.

**What this means for the roadmap.** 70B *does* run on one commodity machine — barely, with
10 GB of headroom. But a 100-token answer takes **161 s**, against TOKENOMICS §11.6's "<30 s
answers" gate: **5.4× too slow**. And per [P8] a serial pipeline does not improve single-stream
latency, so splitting this across the trio would not fix it either — distribution buys
*capacity* and *throughput*, never single-answer speed. The honest read: 70B is reachable on
this hardware and unusable at it. Compare `engine/local_gguf.py`'s 1.5B Q4_K_M at 36 ms/token —
**44× faster per token** at 1/47th the parameters.

### Steps 3-5 — not executable, confirmed by measurement

**(a) Nothing consumes the compiler's output.** Grepping every kernel source for the NSProgram
magic `NS03`: `neuronscript_simd.c` 0, `neuronscript_tiler.c` 0, `neuronscript_bitmask.c` 0,
`neuronscript_cube.c` 0. There is no fourth component to build into `libns.dll`. This restates
Session 22's finding; it has not changed.

**(b) The format is 2.67× larger than fp32.** Measured on a real `gate_proj` (8960×1536) at
`NSCompiler`'s own default `sparsity_threshold=0.005`: **89.0%** of weights survive, and
`NSProgram` stores 12 bytes each (`int32 src`, `int32 dst`, `int8` padded to 4).

| Llama 3.3 70B (70.6B params) | size |
|---|---|
| Q4_K_M GGUF (what ran) | 42.5 GB |
| dense int8 (what the kernel wants) | 70.6 GB |
| fp32 | 282.4 GB |
| **NSProgram** | **754.2 GB** |
| **RAM available** | **63.3 GB** |

**(c) The compiler cannot ingest it anyway.** `NSCompiler.compile()` takes
`List[List[float]]` and appends Python ints per surviving weight — 62.8 billion of them, in an
interpreted loop — and it takes fp32 input, so a Q4_K_M GGUF would first have to be
dequantized to 282 GB. Even bypassing the compiler entirely for the dense int8 path the kernel
actually uses, 70.6 GB exceeds RAM, and `NSLinear` additionally retains the fp32 weight for
prefill fallback.

Extra disk does not move any of these; (a) is a code fact and (b)/(c) are RAM.

### State
`bench_70b.py` holds the harness. The GGUF stays at `C:\Users\optin\models\llama70b\`
(42.5 GB) — it is the only artifact here worth keeping, and it makes the tier ladder's "does
not fit / barely fits" branch testable for real instead of hypothetically.

---

## Session 26 (2026-08-01) — the three things standing between here and one stranger joining

**Goal:** [P21] auto-restart, [P10] stranger NAT traversal, and `engine/local_gguf.py` as the
driver's default engine. Nothing else. The measure is one outside person able to join.

### FIX 1 — [P21] a node that survives a reboot

Both remote nodes now run the agent as a `systemd --user` service with **`Restart=always` /
`RestartSec=10`**, installed by `agent/install.py --startup`.

**The documented command did not exist, and the undocumented one was destructive.**
`install.py` had `--no-startup`, not `--startup`, so `PROBLEMS.md`'s own prescription failed on
argparse. Running plain `install.py` instead would have been worse: `write_config()` wrote
DEFAULT_CONFIG *over* the existing file, discarding `node_id`, `node_token`, the layer range the
machine was serving and its `register_secret`, and re-pinning both nodes to layers 10-18. The
fix for [P21] would have de-identified the live network. `write_config()` now merges, and
`--startup` is a non-interactive repair path for a machine that is already a working node.

**A unit file was never going to be enough — and this is the part worth remembering.** A
`systemd --user` service does **not** start at boot unless the user has *lingering* enabled.
Without it the unit is bound to a login session: `WantedBy=default.target` fires when somebody
logs in and never after an unattended reboot, while `systemctl --user is-enabled` cheerfully
reports `enabled` the whole time. Both machines had `Linger=no`. The installer now enables it
(unprivileged first, then `sudo -n`; both machines accepted the unprivileged call) and falls
back to **cron** where neither is permitted — `@reboot` plus a two-minute keepalive
(`agent/neuron-keepalive.sh`), which needs no privileges at all and is therefore the path a
stranger's laptop will actually take. `uninstall.py` removes both, or an uninstalled agent
would be resurrected every two minutes by a cron job nobody remembered.

**The listener check caught a live failure the moment it shipped.** [P21]'s second half —
"a node can report itself healthy while serving nothing" — is now enforced: `NodeServer.run()`
records a failed bind instead of dying silently in its daemon thread, `setup()` waits for the
listener before advertising, and `heartbeat_loop()` refuses to ping while it is down. First run
on both machines: **`OSError: [Errno 98] Address already in use`** — each still had a
hand-started `node_ns.py` from Session 23 holding port 50999, 17h 57m old. Before this change
the agent would have logged `heartbeat ok — active` indefinitely against a node that never
bound, which is exactly the Pavilion symptom [P21] describes. Stale processes stopped; the
service owns the port now.

**Verified:** `systemctl --user kill -s SIGKILL neuron-agent` on the OptiPlex → back online in
the coordinator's `/node/list` in **20 s**, no intervention. Linger is `yes` on both machines,
units `enabled`. **A real power cycle was NOT performed** — see "left for the founder" below.

### FIX 2 — [P10] a node reachable by a stranger, proven with Tailscale stopped

The relay has existed since Session 12 and **no real request had ever used it.** Both remote
configs said `behind_nat: false`, so the coordinator stored their **Tailscale** addresses and
`router.chain_public` handed those to every peer. Any stranger placed anywhere but the final
segment must dial the next hop, and `100.114.189.46` is not routable for anyone outside the
founder's tailnet. The mechanism was built, deployed, and bypassed.

- `agent.use_relay()` defaults `behind_nat` to **True** (it was `.get("behind_nat", False)`,
  contradicting `DEFAULT_CONFIG`'s `true` for any config that merely omitted the key);
  `--relay/--no-relay` overrides. `setup()` now re-registers when a node is in relay mode but
  holds no endpoint, not only when its ticket is stale — otherwise a node switched to relay
  mode after first registration advertises its old direct address forever, since `register()`
  is skipped once credentials exist. Asking for a relay and getting none is now a warning
  naming the address peers were given instead of passing silently.
- Both live nodes re-registered onto relay endpoints: **`150.230.22.250:9002`** (OptiPlex) and
  **`:9003`** (Pavilion).

**Two bugs found only by actually running the stranger path, either one fatal:**

1. **`agent/__init__.py` did not exist.** `INSTALL.md` step 4 is `python agent/agent.py`, and
   that dies with `ImportError: cannot import name 'local_chat' from partially initialized
   module 'agent'`. Python puts `<repo>/agent` on `sys.path` ahead of anything the script adds,
   and a regular module named `agent` beats a namespace-package directory of the same name — so
   agent.py imported itself. **Both live nodes had an `__init__.py` created by hand during
   setup**, which is why eleven sessions never saw it; every stranger would have hit it on the
   first command in the install guide.
2. **Proof-of-compute could not promote a node on any segment except the last.**
   `node_server.py`'s probe role set `s2 = self.hi` — the inclusive last layer — where `s2` is a
   Python slice bound (`layers[s1:s2]`). It ran one layer too few and advertised a range
   `proof_of_compute.attest_middle` rejects outright. Auto-placement puts a joining node
   wherever the coverage GAP is, which here was 0-9, so the realistic case was the broken one.

**The test, run with `tailscale down` on this machine** (both remotes confirmed unreachable at
their `100.x` addresses first, coordinator still reachable):

| step | result |
|---|---|
| fresh copy of the repo, no config, `install.py` | registered as `stranger-test-win`, **probationary** |
| auto-placement | layers **0-9** (`fill-gap`), no layer numbers chosen by hand |
| slice download | 1.40 GB, byte-range, ~6 min |
| listener + tunnel | bound :50999, relay endpoint **`150.230.22.250:9004`** |
| `/node/list` | **online**, off-tailnet |
| proof-of-compute over the relay | **passed, max_err 0.0**, 280 ms round trip → **verified** |
| network | **28/28 layers, `healthy: true`, 3 eligible nodes** |

And a real inference over the relay with Tailscale still down — driver → Pavilion `:9003` →
OptiPlex `:9002`, every hop over the public internet:

```
'Why is the sky blue' -> "The sky appears blue because of a phenomenon called Rayleigh..."
24 tokens, 45.5 s, 0.53 tok/s
node_a 4.6 s (10%) | node_c 3.7 s (8%) | node_b 2.8 s (6%) | net 33.6 s (74%)
```

**Correct, and slow in exactly the way [P3]/[P20] predict.** 74% of wall time is the wire, and a
relayed hop crosses the Amsterdam VM twice, so this is the honest cost of universal
reachability. Direct Tailscale was faster and worked for nobody outside the tailnet. The
S21 wire codec (4.25× smaller activations) is still not deployed to these nodes and is the
obvious next lever on that 33.6 s.

### FIX 3 — `local_gguf` as the driver's default engine

`node_a.py --engine` gains `auto` (**the new default**) and `gguf`; `auto` runs the model
locally through llama.cpp Q4_K_M whenever the machine can hold it, and falls back to the fp32
node pipeline when it cannot. This is the same tiering `ui/app.py` and `api/openai_compat.py`
already applied — node_a was the one driver entry point still defaulting to fp32.

| | tok/s | note |
|---|---|---|
| single prompt, cold process | 14.75 | includes the ~1.5 s model load |
| single prompt, engine-reported | **27.9** | steady state, matches the briefed 28 |
| 4 prompts, wall clock | **25.29** | load amortized |
| the fp32 chain it replaces (relay) | 0.53 | above |

Shard load dropped ~40 s → ~1.5 s, and the answer is the correct Rayleigh one.

**On "confirm NRN credited correctly": the local engine credits 0 NRN, deliberately, and that
is not a gap to close.** No `/infer` call is made because no other machine ran anything —
crediting the network for local compute would mint NRN for work no node did ([P12]'s open hole)
and would be farmable by anyone with a laptop, since local execution is verified by nothing.
The report line says so explicitly rather than leaving a zero to be discovered.

### Left for the founder (both need a human, neither is a code gap)

- ~~A real power cycle.~~ **DONE — the founder rebooted the Pavilion and it passed cleanly.**
  Booted 14:03:14; `neuron-agent` active at **14:03:27 — 13 s after boot**, with nobody logged
  in and no SSH; listening on :50999 and heartbeating; back in the coordinator's `/node/list`
  well inside the 2-minute bar. That is the whole of [P21] closed: linger is what made the
  user-level unit start with no login, which is the case a reboot actually tests.
  (Note for future sessions: **the Pavilion cannot be rebooted remotely** — no passwordless
  sudo, and polkit refuses `systemctl reboot` from an SSH session with `Call to Reboot failed:
  Interactive authentication required`. It needs someone at the machine.)
- **A coordinator-billed inference.** `/infer` requires an OAuth-linked wallet since [P17]
  closed the open faucet, and `node_a.py`'s auto-faucet path can no longer self-fund. Running
  the billed request — and therefore confirming NRN lands on all three nodes — needs a
  `wallet_id` from a real Google/GitHub login, or the operator key.

### State

`stranger-test-win` is still running from `C:\Users\optin\neuron-stranger` and is currently the
only thing covering layers 0-9; stopping it drops the network to 18/28. Both remote agents are
service-managed and will come back on their own. `.venv` note for future sessions: the local
engine needs `C:\Users\optin\neuron\.venv\Scripts\python.exe` — the system `py -3.11` has no
`llama_cpp` and silently reports `local engine unavailable`.

---

## Session 27 (2026-08-01) — the auto-verifier, and what still blocks a real stranger

### TASK 1 — auto-verifier ✅

`verify_service.py`. Sweeps every 60 s, challenges every probationary+online node with
proof-of-compute, attests the result, logs to `verify_service.log`. Running now, and installed
to auto-start (`HKCU\...\Run : NEURONVerifier`, via `agent/install.py --with-verifier`).

**Proven, not asserted:** a node registered with no secret (exactly what a stranger does) joined
`probationary` and was promoted with no human involved —

```
verify-service-test VERIFIED — layers 0-9, max_err 0.00e+00, 254ms → standing now 'verified'
```

Three deliberate departures from the brief's sketch, each with a reason:
- **A timeout is not a failure.** The sketch attests `passed: result` on every outcome. A failed
  attestation is permanent and `REPUTATION_THRESHOLD` 0.6 means a few of them exclude a machine
  from the network for good — so an unreachable node (mid-restart, cold shard, relay hiccup)
  records *nothing*. Passing is attested instantly; condemnation waits for 3 consecutive wrong
  answers. This fired for real during testing and correctly stayed silent.
- **It is not in the default `--startup`.** The verifier needs the operator's
  `NEURON_REGISTER_SECRET` (node addresses are private, `/attest` is secret-gated) and PyTorch.
  Every stranger runs `--startup`; bundling it would ship a permanently-failing service to every
  donor and imply they should hold the registration secret. Hence `--with-verifier`.
- **Challenges are cached per layer range.** Building one loads that range with torch; a 60 s
  loop would otherwise reload the same shard every minute forever.

Two bugs found while wiring the auto-start, both of the same shape — *trusting PATH*:
`shutil.which("pythonw")` resolved to a bare Python 3.14 with no PyTorch, so the Run key would
have died on import at every boot; and the service read its secret only from the environment,
which auto-start does not provide, so it would have started and immediately exited. It now takes
`pythonw` from beside the chosen interpreter and falls back to `.env.coordinator`. Both verified
by reading the registry back and by running with `env -u NEURON_REGISTER_SECRET`.

### TASK 2 — first stranger: prepared, NOT achieved

`STRANGER_INSTALL.md` written — 5 steps, plain English, honest about NRN having no cash value.
`RELEASE_NOTES_v0.1.0.md` written. `INSTALL.md` rewritten and pointed at the correct repo.

**Three steps cannot be done from here and are not done:**
1. **Making the repo public** needs a GitHub login this environment does not have (`gh` is not
   installed). The pre-publication audit *is* done and it is safe: no secret appears in any
   commit, no API keys, and all three coordinator secrets are overridden by real values on the
   VM, so the dev defaults in `config.py` are inert. What does go public: 19 Tailscale IPs and
   6 `ssh user@host` lines across `PROBLEMS.md` / `ROADMAP.md` / `sessions.md`.
2. **Sending the guide to a person** is not something to do on the founder's behalf.
3. **Steps 4-6 (their node_id, time-to-first-NRN, issues hit)** describe observations of a real
   person. There is nothing to record until one exists, and inventing them would defeat the
   entire point of the milestone.

### TASK 3 (added mid-session) — the silent relay-tunnel death, fixed ✅

Root-caused to a number rather than a hunch: `SO_KEEPALIVE` was on, but with the **OS default
idle timer — 7,200,000 ms on Windows**, confirmed absent from the registry, i.e. 2 hours. The
tunnel's control socket is idle by design (it waits for the relay to push `new_conn`), so when
the relay's end went away the socket stayed ESTABLISHED and blocked in `recv` for exactly that
long. The ~2 h dead window was the keepalive timer.

Fixed in two layers: keepalive at 60 s/10 s (`SIO_KEEPALIVE_VALS` on Windows, `TCP_KEEPIDLE`
and friends elsewhere) so the existing reconnect loop actually fires; and
`agent.relay_reachable()`, which every 4th heartbeat dials the node's **own public endpoint**
and completes a real handshake — a TCP connect proves nothing, because the relay accepts on the
public port whether or not it can still reach the node. On failure the tunnel is restarted and
**the heartbeat is withheld**, so the coordinator routes around the node instead of into it.

`agent/test_relay_liveness.py` 5/5 (the load-bearing case is an endpoint that accepts and never
speaks). Deployed to all three nodes: every public endpoint answers in 0.03-0.10 s, **zero false
alarms across 4+ probe cycles**. Full write-up in `PROBLEMS.md` [P22].

### The finding that prompted it

**A relay tunnel dies silently while the node keeps reporting `active`.** Measured: after ~2 h
the Windows node's relay port accepted TCP and then never spoke (20 s timeout), while both Linux
nodes answered the same handshake in 0.10 s. Restarting the agent fixed it instantly.

This is [P21]'s "online means nothing" in a second place. Session 26 made the heartbeat assert
the *local listener* binds; nothing asserts the *tunnel* is alive. So the node stays `online` and
`verified` in `/node/list`, routing sends it real requests, and it serves none of them — and the
auto-verifier cannot promote a node in that state either. Every stranger is relayed, and the
machine it was reproduced on is Windows, which is what most of them will run.

**Fixed — see TASK 3 above.** The shape: the heartbeat dials the node's own public relay endpoint
periodically and restart `tunnel_client` when the handshake fails, the same way `setup()` now
refuses to advertise a listener that did not bind.

### Also fixed this session (found by forcing the network path)
- `neuron_driver.py` used `batching.MicroBatcher` and **never imported `batching`** — every
  distributed generation through the Chat UI *and* the OpenAI API died on `NameError` at the
  first token. Invisible because `local_gguf` short-circuits the network path on any machine that
  can hold the model, and because the tests stub the driver instead of running a generation.
- `ui/app.py` reached for the driver's tokenizer without loading it when startup had skipped the
  load — reachable without the new flag, on any install that is capable-but-not-yet-downloaded.
- New `NEURON_FORCE_NETWORK=1` makes the distributed path testable from the UI at all; without
  it, no machine that can run locally will ever exercise the chain.

---

## Session 28 (2026-08-01) — 7B measured and made the local default; the 72B arithmetic

### 7B: measured, and it meets the target

`Qwen2.5-7B-Instruct` Q4_K_M (4.68 GB, split across two GGUF files upstream — llama.cpp follows
the split once both parts are in one cache dir). Windows driver, 16 cores, 15 threads:

| run | tokens | tok/s |
|---|---|---|
| 1 (cold) | 26 | 6.08 |
| 2 | 60 | **7.85** |
| 3 | 60 | **7.77** |

Model load 5.3 s. **~7.8 tok/s warm, against the brief's "8+" target.** For scale, 1.5B on the
same box is 27.9 tok/s.

**A prediction of mine that the measurement corrected.** Extrapolating linearly from the two
known points (1.5B = 36 ms/token, 70B = 1594 ms/token — 46.7× params for 44.3× time) predicted
182 ms/token = 5.5 tok/s for 7B, and I said to plan for ~5. Measured 7.8. The 70B run was
memory-bandwidth-bound at 42.5 GB, so it drags a linear fit upward; 7B is not. Extrapolating
across a 46× range hid a regime change.

**Wired as the local default** — `local_gguf.best_local_model()` picks the LARGEST model this
machine can hold rather than mirroring the network's serving model, which is capped by the
network's weakest member (a 64 GB desktop was answering with 1.5B because a 4-core laptop
elsewhere sets the tier). `prefetch_best()` pulls a bigger model in the background at startup, and
selection only ever returns already-cached weights, so a first run answers immediately and
upgrades when the download lands. `NEURON_LOCAL_MODEL` pins one. Live in `ui/app.py` and
`node_a.py --engine auto`.

### The landmine in the deployed tier ladder

The **live** coordinator's middle tier is `meta-llama/Llama-3.1-8B-Instruct`, promoting at
**6 nodes**. That repo is **gated** — `slice_downloader` fetches byte ranges with no auth and
gets 401. So the plan "get more strangers to join" would, at 6 nodes, auto-promote the entire
network to a model every node fails to download. The local repo already fixed this (Qwen 7B,
publicly fetchable, min_nodes 3) and it was never deployed.

Checked before touching anything: at today's 3 nodes nothing migrates (`PROMOTE_MARGIN` 0.15
needs 3×1.15 = 3.45 nodes), so deploying the fix is inert now and correct later. **The deploy
itself was blocked** by this environment's guard on restarting the production coordinator —
command left for the founder.

### 72B at 6 nodes is off by ~4×

Per-node RAM for Qwen2.5-72B (72.7B params), by weight dtype:

| weights | total | **6 nodes** | 10 | 20 | 40 |
|---|---|---|---|---|---|
| fp32 (what the pipeline loads today) | 291 GB | **48.5** | 29.1 | 14.5 | 7.3 |
| fp16 (`NEURON_WEIGHT_DTYPE=fp16`) | 145 GB | **24.2** | 14.5 | 7.3 | 3.6 |
| int8 | 73 GB | 12.1 | 7.3 | 3.6 | 1.8 |
| int4 (**not implemented**) | 36 GB | **6.1** | 3.6 | 1.8 | 0.9 |

Nodes today are 68 / 16 / 12 GB; a volunteer laptop is 8-16 GB. So 6×72B needs **int4 weights in
the pipeline**, which is the still-open "quantize the weights" gap in `PETALS_NOTES.md`. At fp16,
which works today, 72B needs ~20 nodes — which is exactly what `model_tiers.py` already says.
`model_tiers.py` is not wrong; the 6-node figure is.

Also worth recording: 72B Q4_K_M is ~47 GB and this machine has 45 GB free, so the single-machine
comparison run that was possible for 70B in Session 25 is no longer possible here.

### Deployed, and a bug the deploy surfaced

Founder deployed `model_tiers.py` and `main.py` and restarted the coordinator. Live tiers are now
`1.5b` / `7b` (Qwen2.5-7B, min_nodes 3) / `70b` (Qwen2.5-72B) — the gated Meta tier is gone.

Restarting an agent against the new build exposed a separate bug, found by reading the agent's
own log rather than by a test: a **verified** node was told

    PROBATIONARY: serving challenges only — a verifier must confirm this node before it
    receives live requests or earns NRN

while sitting at `challenges_passed=2`, `standing=verified`, `eligible=True` — routing and
earning the whole time. `/node/register` derived the reported standing from *"did this call carry
the register secret"* instead of the node's actual state. It fires routinely (a relayed node
re-registers on a ticket refresh, and on any restart that needs one), so **a stranger would be
told on every restart that they had stopped earning** — and it would look like the auto-verifier
had failed when it had not. Fixed to read standing back from the DB; the "you will not earn" note
now only attaches when the node really is probationary. `coordinator/test_open_join.py` 26 (2 new),
plus an end-to-end check on a temp DB: register → probationary, attest → verified, re-register →
verified with no note. Deployed and confirmed present in the running file.

### Not done, and why
"Each node runs full 7B locally, coordinator routes to a free node" changes what a node *is*:
today a node handles opaque tensor slices and never sees text, and that property is the basis of
`SAFETY.md`, `SECURITY.md` and the node-operator story in `INSTALL.md`/`STRANGER_INSTALL.md`
("your node only ever processes opaque numeric tensors, never readable prompts or answers").
Full-model-per-node means volunteers' machines receive readable prompts and produce readable
answers, in their own jurisdictions. That is a product and legal decision, not a refactor, so it
was raised rather than quietly built.

---

## Session 29 (2026-08-02) — the packaged app audited, because a stranger only gets one shot

**Goal:** stop fixing what we trip over and go through the whole stranger-facing surface.
Everything below was found by building 0.16.0 and actually running it, not by reading code.

### Four bugs in the shipped app, all silent

1. **The tray showed `0.00 NRN` forever and could not be told from earning nothing.**
   `_open_my_dashboard` and the balance poll both used the `node_token` the process loaded at
   **startup**. The coordinator re-mints that token on every registration, so any
   re-registration (relay ticket refresh, a second copy, a restart that needs one) leaves the
   tray holding a dead token. Confirmed live: the My Dashboard URL carried a token starting
   `13dea1f0` while `config.json` held `194dfddf`, and `194dfddf` returns HTTP 200. Both now
   re-read the token from disk, where whoever registered last wrote the good one.
2. **The ledger poll swallowed every non-200** (`if r.status_code == 200: ...`). "You earned
   nothing" and "we could not read your balance" rendered identically as `0.00`. For a stranger
   watching a stuck zero, that is indistinguishable from a network that does not pay — the most
   likely reason a first volunteer quits. Non-200 is now recorded, shown in the menu, and logged
   once per distinct status.
3. **A Chat UI that had already failed said "Chat UI (starting…)", greyed, forever.**
   `local_chat.start()` returns None on failure and the tray only checked `is not None`, so
   "failed" and "still starting" were the same state. `agent.local_chat_state` now distinguishes
   pending / running / failed / disabled, and a failed start is logged with the usual cause
   (port already in use).
4. **`--headless` has never worked in any build.** `neuron_app_entry` dispatched on the flag and
   passed it through to `agent.main()`'s argparse: `error: unrecognized arguments: --headless`,
   exit 2, nothing runs. Found by trying to run 0.16.0 headlessly to debug bug 1.

Plus `DISCLOSURE.txt` — the first thing a stranger reads — pointed at
`github.com/raman011sharma-code/neuron-network`, which 404s. Same for `AppSupportURL`.

### The pattern worth naming

Every one of these is invisible on the founder's machine and only appears to a new user, and
every one of them lived in code with **no test at all**: the tray had none, the app's single
entry point had none. Both now do (`agent/test_tray.py` 8 cases with pystray/PIL stubbed;
`packaging/test_app_entry.py` 5 cases). The stranger-facing surface is exactly where "it works
for me" is worth the least.

### Installer

**0.16.0** built and then superseded by **0.16.1**, which carries the four fixes above plus
everything from Sessions 26-28. Installer changes: autostart is now **checked by default** (it
was unchecked, which quietly undid all of [P21]'s reboot work — a stranger will not go hunting
for that checkbox), and the support URL is corrected.

### Verified on the real artifact
Installed 0.16.0 on the founder's PC: registers, fetches placement, downloads its slice, comes
online and heartbeats. The tray "Error" that prompted this audit was a transient retry state
between coordinator polls, not a fault — but chasing it is what surfaced all four bugs.

### Two more found by attempting the clean first-run test — both fatal, both silent

**5. A hostname collision locked a machine out of the network permanently.** `node_id` was
exactly `agent-{hostname}`, which collides deterministically: Windows ships defaults like
DESKTOP-8F3K2P1, machines get called "laptop", and the same machine reinstalling produces the id
it had before. The coordinator refuses a secret-less registration of an id already
trusted/verified (the hijack guard — correct), so a collision meant a 409 on **every** attempt,
forever, advising "restart to pick up the current token" when no copy of the agent held that
token. Anyone who lost their config, reinstalled without uninstalling, or shared a hostname with
an existing node could never join. Generated ids now carry a random suffix
(`agent-<host>-<6 hex>`), persisted once; and a 409 while holding **no** token takes a fresh
identity instead of looping, while a 409 while holding one still means "another copy
re-registered me" and must not rotate. `agent/test_node_id_collision.py` (6).

**6. A re-placed node served another segment's weights.** `ensure_slice()` skipped the download
whenever *a* `model.safetensors` existed, never checking which layers were in it. Delete
config.json and re-register and the coordinator hands out whichever gap needs filling — not the
range you had — so the node loaded the old segment's weights while claiming the new one. Nothing
detects that locally: it answers confidently with wrong activations, fails proof-of-compute, is
flagged after three strikes, and nothing anywhere explains why. The assigned range is now
compared against the layer indices read from the **safetensors header on disk**, so the bytes
decide rather than a config claim. No marker file and no forced re-download for existing slices —
the header was always authoritative. `agent/test_slice_reuse.py` (4).

Both were found by following this file's own "move config.json aside" instruction, which walked
straight into the first and would have hit the second next: this machine's state dir held a 19-27
slice while the clean test places on 0-9.

### The clean first-run test — PASSED

Run from source with the fixes, fresh config, nothing pre-seeded but the slice:

```
10:54:50  auto-placed on layers 0-9 (fill-gap: chain is missing layers 0-9)
10:54:51  registered as agent-optinovate-447583 [probationary]     <- unique id, no collision
10:54:51  slice already present — skipping download                <- header validated as 0-9
10:54:56  node server listening on port 50999
10:54:57  relay tunnel started — reachable via 150.230.22.250:9005
10:55:13  agent-optinovate-447583 VERIFIED — max_err 0.00e+00, 300ms
```

**22 seconds from registration to earning-eligible, no human involved.** Network 28/28, healthy.
That is the whole stranger onboarding path working end to end for the first time.

### Installer
Final artifact: **`NEURON-Setup-0.16.3.exe`** (211 MB). 0.16.0/.1/.2 were superseded during the
audit and deleted so the wrong one cannot ship. Verified in the built exe: `--headless` works,
the bundled chat.html carries the footer fix, and both new agent fixes are in the frozen source.

**Still not done:** the packaged .exe has not itself been run through a clean first-run (the flow
above was proven from source with identical code). And no actual outside person has installed
anything — the repo is still private, so `STRANGER_INSTALL.md` points at a 404.

---

## Session 30 (2026-08-02) — off one machine, off one person, off one name

### Published
Code pushed off this machine for the first time, then moved off the founder's personal account
to an organisation: **`github.com/neuron-network-ai/neuron`**. `main-full` (the real 84-commit
history) is the default branch; the old single-commit curated snapshot is preserved on `main`.
Personal identity stripped from the repo — real name → "NEURON Labs" in LICENSE/README/
ROADMAP/TOKENOMICS, location line dropped, `ssh user@host` and `/home/<user>` genericised in
sessions.md. Commits were already authored under GitHub's noreply address, so no real email
was ever exposed. Installer rebuilt through **0.16.5** as the URLs changed (the disclosure page
and AppSupportURL are compiled into the setup binary, so a link fix needs a rebuild to reach
anyone).

Two stranger-stoppers came out of the renames, both invisible until someone follows the guide
literally: cloning `neuron.git` lands in `neuron/` (the guide said `cd neuron-network`), and the
ZIP extracts to `neuron-main` (the guide said `neuron-network-main`). Either one strands a
newcomer at step 3, right after a long download.

### A live outage, caused by the tier ladder reaching a model it could not fetch
Removing the homeserver node dropped the network to 3 machines, which is exactly
`7b`'s `min_nodes`, so the coordinator auto-promoted the whole network to
**Qwen2.5-7B** — and every node's `/node/{id}/slice-info` returned **502**:

    could not read model header: 404 ... /Qwen2.5-7B-Instruct/resolve/main/model.safetensors

`coordinator/sliceinfo.py` assumed a **single** `model.safetensors`, because Qwen2.5-1.5B is
the only model ever exercised and HF ships it unsharded. Every larger model is sharded
(`model-0000N-of-0000M.safetensors` + an index), so the hardcoded URL 404s. Nobody could learn
what to download, nobody could serve the new tier, and the chain could not heal.
`slice_downloader.shard_files()` had handled this correctly for two sessions; the coordinator
never did. Now shard-aware, verified against both: 7B resolves to the two files actually
holding layers 21-27 (4.35 GB of 15.2), 1.5B still one file. **This is the same shape as the
gated-Meta landmine — a tier the download path cannot fetch takes the network down at the
moment it is reached.** Both were latent until capacity crossed a threshold.

### [P12] was stale and I repeated it — corrected
Said out loud that the ledger "mints 1.0 NRN out of nothing per request with no supply cap".
That is what `PROBLEMS.md` [P12] still says, and it has not been true since Workstream B:
`config.GENESIS_TOTAL_SUPPLY = 1_000_000_000`, `coordinator/genesis.py` seeds the fixed buckets
(60/20/15/5), `ledger.py` opens with *"NRN is transfer-only from here on — this file never
mints"*, settlement transfers out of escrow, and `test_wallet_settlement.py` asserts
`SUM(balance) == 1e9` after every operation. The founder corrected me. Check the code before
repeating a problem entry — `PROBLEMS.md` records what was true when written.

### Decentralisation, step 1: the network verifies its own newcomers
The last human in the join path is gone. Previously a stranger stayed probationary — reachable,
serving nothing, earning nothing — until the founder personally ran `security/proof_of_compute`
against them, so growth required one particular laptop to be on, and the credential that
allowed verification could not be delegated without handing over the network.

Now any verified node pulls `GET /node/verify-assignment`, runs the **same** proof-of-compute,
and reports a verdict signed with its own node token via `POST /node/{id}/peer-attest`.
`PEER_VERIFY_QUORUM` (default 2) *distinct* agreements promote the newcomer.

The consensus rules are the whole risk, so they are explicit and tested:
- votes are keyed `(verifier_id, target_id)` — one machine voting six times counts **once**, so
  a single node cannot wave through its own sybils; two verified machines must collude instead
- a node cannot verify itself (400); only an eligible node may vote or be assigned work (403);
  a bogus token is 401
- **unreachable is not cheating** — the agent reports nothing when it cannot reach a target,
  because a failed attestation is permanent and a few of them exclude an honest machine for good

The operator's `/node/{id}/attest` still works unchanged; peer quorum is a second, human-free
route to the same standing. `coordinator/test_peer_verify.py` 14/14; all six coordinator suites
green.

### What is still centralised (honest list)
1. **The coordinator** — one VM decides routing, placement, standing, and holds the ledger.
   This is now the real single point, since the supply rules are already sound.
2. **The relay** — same VM; every NAT'd hop crosses it. NAT hole-punching removes it.
3. **The ledger** — SQLite on that VM. On-chain is about removing "one person holds the
   database", not about fixing a supply hole.

---

## Session 31 (2026-08-02) — NRN as a real ERC-20, prepared and rehearsed, deployed nowhere

### Decision, settled the same day: Polygon is skipped

- **Phase 1** — the SQLite ledger. Current, working, stays the hot ledger.
- **Phase 2** — **NEURON Chain directly.** Not Polygon, not "Polygon first then migrate".
- **No Polygon deployment until further notice**, mainnet or testnet.
- On-chain work waits for **50+ external nodes and 500+ MAU**. External nodes today: zero.

The work below stands regardless — it is written against the EVM and against the coordinator's
ledger, not against Polygon. If NEURON Chain is EVM-compatible, `NRN.sol` deploys unchanged and
`chain.py` needs one new `NETWORKS` entry; if it is not, the contract is the written
specification of the token's rules (fixed 1B, 60/20/15/5, release-not-mint, owner pause) and
those port even when the Solidity does not. `deploy.py`'s refusal message now cites this
decision, so anyone reaching for a public network is told why it is closed rather than just
that it is.

**Nothing was deployed.** No transaction has been sent to Polygon, Amoy, Mumbai or any other
public network. The whole of `blockchain/` is gitignored, along with the npm toolchain it needs
(`/node_modules`, `/package.json`, `/package-lock.json` — hardhat has to resolve
`@openzeppelin` from above `blockchain/`, or it treats every OpenZeppelin contract as one of
our own source files and refuses to compile).

### The contract

`blockchain/NRN.sol` — OpenZeppelin 5.1 `ERC20` + `ERC20Pausable` + `Ownable`, solc 0.8.24,
"NEURON"/"NRN", 18 decimals, 1,000,000,000 total. Constructor splits it 60/20/15/5.

The brief asked for a fixed 1B supply *and* an owner who can "mint node rewards from the 60%
locked supply". Those only reconcile if the 60% already exists, so **`mintReward` releases, it
does not mint**: the constructor mints the entire supply once, the 60% to the contract's own
address, and `mintReward` transfers out of it against a `rewardsDistributed` counter that
cannot be overdrawn. `totalSupply()` is 1B forever — the same property `ledger.py` has held
since Workstream B, carried on-chain rather than dropped at the boundary. Holding the reserve at
the contract address also means no key can move it; the only exit is `onlyOwner` and capped.

### The migration script, and the two things the schema disagreed with

`blockchain/migrate_ledger.py` reads the coordinator ledger, pays every account with a balance,
logs each one with its transaction hash, then re-reads `balanceOf` and reconciles. Dry run is
the default; `--execute` is the only thing that sends. Two corrections the brief's
`SELECT node_id, wallet_id, balance FROM ledger` did not anticipate:

1. **There is no `wallet_id` column.** `models.py` keys the ledger on `node_id` for all three
   account types. The query adapts to either schema.
2. **User wallets must not be paid from the reward pool.** A user's balance came from the
   faucet, which `wallet_for_oauth` funds out of the **150M ecosystem** bucket. Paying it with
   `mintReward` draws 60%-pool NRN for something the 15% pool already paid for — the on-chain
   ecosystem wallet ends up permanently richer than its SQLite row and the emission pool
   permanently poorer. Caught by a bucket-reconciliation check that compares all four genesis
   buckets against their on-chain holders; `account_type='wallet'` rows are now settled by a
   transfer from the ecosystem wallet. This is the single most useful test in the suite, because
   no per-account balance check can see it — every individual balance was correct.

Genesis buckets themselves are skipped explicitly: the constructor's 60/20/15/5 split *is*
those four rows, and migrating them again would double-count 40% of supply.

### Local test — PASSED, 60 checks

`python blockchain/run_local_test.py` starts `npx hardhat node`, deploys, builds a throwaway
ledger shaped like the real one, migrates it, reconciles, and tears the chain down. 28 harness
checks + 32 in `blockchain/test_nrn.py` (supply, distribution, transfer/approve, mintReward,
owner-only, allocation overdraw, pause). Three real bugs came out of it:

- the resume guard compared `amount_wei` as an int against the string JSON round-trips it to,
  so **a second `--execute` run paid everyone twice**. Now compared as strings, and tested.
- teardown terminated `cmd.exe` and left the actual node holding port 8545, which the next run
  then silently reused with stale chain state. `taskkill /T` on Windows.
- the ecosystem/rewards split above.

### Run against the real ledger (read-only, no chain writes)

Against `backups-offbox/neuron-20260802-115534.db`: **16 accounts hold NRN and none has an EVM
address**, about 228 NRN in total. That is the actual blocker — `ledger.node_id` is a string
like `agent-optinovate-447583`, nothing has ever asked a node for a payout address, and there is
no proof-of-control if it did. It also surfaced that the live ledger still contains
`attacker-demo-1`, `attacker-demo-2`, `probe-only` and `live-verify-wallet` — 100+ NRN of
faucet grants from security testing that would become real transferable tokens.

One correction to the invariant check: the live ledger sums to `999,999,999.99999999999999719`,
about 3e-15 short. That is float dust in SQLite REALs, not a missing coin, and
`models.supply_snapshot` already tolerates it (`abs(total - 1e9) < 1e-6`). The exact residual is
logged rather than hidden. The older local `coordinator/neuron.db` is pre-genesis (4 rows,
sums to 1.000001) and is correctly refused.

### Deploy script

`blockchain/deploy.py` deploys, writes `blockchain/config.json`, and verifies the source via the
Etherscan V2 endpoint Polygonscan runs on. Any network marked public needs **both** `--yes` and
`NEURON_ALLOW_PUBLIC_DEPLOY=1`, so neither a stray flag nor a stray env var is enough alone.
`hardhat.config.js` has no public network at all.

**Mumbai is gone.** The brief named it; Polygon decommissioned it in April 2024 and its RPCs and
faucets no longer exist. The entry resolves and warns; the flow is written against **Amoy**
(chain id 80002).

### Answers

Contract compiled: **yes**. Local tests pass: **yes** (60/60). Migration script works: **yes**,
including against a real ledger snapshot. Ready to deploy when told: **no, and it should not
be** — see `blockchain/MIGRATION_PLAN.md`. Blocking: node payout addresses with proof of
control (nothing collects them), pruning the demo accounts, key custody for an owner key that
can release 600M NRN (multisig/hardware, not a file on one laptop), and the MiCA read
`TOKENOMICS.md` gates on. Also honest: on-chain settlement cannot replace SQLite — one
transaction per node per request is unworkable — so the chain is a settlement and withdrawal
layer over the hot ledger, and this migration is the first run of that same mechanism. Putting
the ledger on-chain removes "the record dies with one Oracle VM"; it does not by itself remove
the coordinator's authority over who earned what.

---

## Session 32 (2026-08-02) — earnings get an owner: payout addresses, proved not claimed

Closes blocker 1 of `blockchain/MIGRATION_PLAN.md`, and it is the piece that had to land
**before** strangers arrive rather than after: today's 16 accounts could have been mapped to
addresses by hand in an afternoon, a thousand cannot.

### A column would have been the wrong fix

`ledger.payout_address` on its own looks like the problem solved while leaving it open —
anyone who can authenticate as a node could write *any* address into it, including someone
else's, and the coordinator would have no way to tell an operator claiming their own wallet
from an attacker pointing a stranger's earnings at their own.

So a binding is a signature. `GET /node/{id}/payout-challenge` issues a single-use nonce;
the node signs a message naming **its own node_id, the address, and that nonce**;
`POST /node/{id}/payout-address` recovers the signer and requires it to equal the address being
claimed. `coordinator/payout.py` holds the logic, deliberately separate from `main.py` so the
verification has somewhere to be tested without HTTP.

Each field in the signed text is load-bearing, and each one is a test:
- the **node_id** means a signature valid for node A cannot bind node B (tested by lifting a
  real signature from one node and replaying it at another);
- the **nonce** is single-use and expiring, consumed *before* the signature is checked — so a
  wrong guess costs the challenge and nobody can grind signatures against one nonce;
- the **address** is what is authorised, and the message says in plain words that signing
  transfers no funds, because this appears in a wallet prompt and "sign this hex blob" is how
  people get robbed.

### The control that survives a stolen token

The interesting case is not the first bind, it is the **re**bind. A `node_token` is a bearer
credential sitting in a config file on a volunteer's disk; if copying it were enough to change
where the money goes, none of the above would matter. So changing an already-bound address
additionally requires `old_signature` — the same message signed by the address currently on
file. An attacker with the token still needs the original operator's key. The register secret
overrides it, which is the recovery path for a genuinely lost key and deliberately a human
decision, since it is also exactly what an attacker would ask for.

### Volunteers should not need a crypto wallet

`agent/payout_key.py` generates an ordinary secp256k1 key on first run, binds it automatically,
and stores it 0600 next to the config — importable into any wallet later. The operator does
nothing and still ends up with an address only they control. Two deliberate refusals: a
**corrupt key file never generates a replacement** (that would strand an address that may
already have been paid), and an operator-supplied `payout_address` is **never auto-bound** —
we hold no key for it, so `python -m agent.bind_payout` prints the exact text to sign in their
own wallet and takes the signature back. Binding is best-effort throughout: an older
coordinator with no payout endpoints, or one briefly unreachable, must never stop a node
serving.

Nice property that falls out: the key is keyed to the *machine*, not the node_id. A node that
regenerates its id (the Session 29 collision fix) rebinds the same address automatically.

### Verified
- `coordinator/test_payout_address.py` **31** — forged signatures, cross-node replay, nonce
  reuse/expiry, rebind with and without the old key, operator recovery, EIP-55 checksums, and
  that the address appears on the node's own dashboard and **nowhere public** (a payout address
  is a persistent pseudonymous identifier; publishing node→address would tie every node's
  earnings together on-chain for anyone watching).
- `agent/test_payout_key.py` **16** — including a real round trip: the agent signs, and
  `coordinator/payout.py` verifies. Both sides build the message independently, and if they
  ever disagree by one character every binding fails while a stubbed test still passes.
- All 14 coordinator suites and all 11 agent suites green. `blockchain/run_local_test.py` now
  **30 + 32** — the test ledger binds two accounts via the column and leaves them out of the
  JSON book, so the run only passes if `migrate_ledger.py` actually reads bindings (it prefers
  them: one was proved, the other typed).

### Not done
User wallets (`w_…`) have no binding path — they authenticate with the wallet-link secret, not
a node token, and 6 of the 16 unmapped live accounts are wallets. `__coordinator__` needs an
operator-chosen address. No policy yet for accounts that never bind. Nothing is bound in
production — this is deployed nowhere, and existing nodes bind on their next agent restart. The
tray still has no payout-address affordance. And the honest limit: an auto-generated key is a
hot key on a volunteer's machine protected by file permissions and nothing else — fine for an
address that only ever *receives*, not for holding value, which is why the agent tells the
operator where the key is instead of hiding it.

New dependency `eth-account` on both the coordinator (imported lazily, so a coordinator without
it still starts and only binding fails) and the agent (optional — a node without it serves and
earns as before). Added to both requirements files and to the PyInstaller spec, where a missing
lazily-imported backend would otherwise ship an exe that silently never binds.

---

## Session 33 (2026-08-02) — the payout work committed, and a prune script that refused to run

### Committed and pushed
`fe105ab` (gitignore: blockchain prep + npm toolchain stay local) and `dd7f4f3` (the whole
payout-address feature: 14 files, 1,234 insertions) are on
`origin/feature/auto-model-tiering`. Nothing from `blockchain/` or `neuronscript_*` went with
them — both are gitignored and the staged file list was checked against them before each
commit. No attribution trailers; author is the founder's own noreply identity.

### `coordinator/prune_test_accounts.py`
Returns development balances to `__ecosystem__` so they cannot become real transferable tokens
on-chain (`blockchain/MIGRATION_PLAN.md` blocker 2). Dry run by default; `--execute` applies;
`--backup` copies the DB first; every transfer is logged to `prune_log.json` with a timestamp
and the rule that selected it.

`__ecosystem__` is not an arbitrary destination — **every** prune target is a ~25 NRN faucet
grant, and the faucet is funded from the 150M ecosystem bucket, so this returns each one to the
bucket it was drawn from. Node earnings came from `__emission_pool__` instead, which is exactly
why node_a/b/c are kept rather than swept along.

Two design choices worth keeping:

**Nothing unnamed is ever swept.** An account matching no rule is KEPT and reported as
unclassified, because a balance nobody classified is a decision, not a default. On the live
snapshot that is **14 accounts** — 10 with zero balance (`nat-probe`, `audit-verify-test`,
`node-a-local`, the three superseded `agent-optinovate-*` ids…) and **4 that hold NRN and need
your call**:

    agent-optinovate    2.025002      node-b-optiplex     0.187746
    stranger-test-win   0.208607      node-c-pavilion     0.187746

All four did real compute on real hardware — `node-b-optiplex`/`node-c-pavilion` are the
OptiPlex and Pavilion under older ids, so by the same logic that keeps node_a/b/c they arguably
belong on the keep list. `--prune-also` / `--keep-also` decide it without a code change.

**The supply check is stricter than a tolerance.** Crediting `__ecosystem__` once per account
(`balance = balance + x`) drifted the total by **3e-8**: each addition into a ~1.5e8 balance
rounds to the nearest double, and doubles are ~3e-8 apart there. Eight of them lost real
precision — invisible under the coordinator's own 1e-6 tolerance, which is how an invariant
starts sliding. Now the amounts are summed in `Decimal` and written once, and the check is that
drift is within **one ulp of the destination balance** — the single unavoidable rounding of one
float addition. On the live snapshot the rehearsal drifted 3e-15.

`coordinator/test_prune_test_accounts.py` **37**: the dry run is byte-for-byte non-mutating (not
merely "intended to be"), a mid-run failure rolls back completely with no partial prune,
`node_a` is not caught by the `node_a-cli-` prefix, a broken invariant or a non-zero escrow is
refused, and the run is idempotent.

### The refusal found a real bug: `__escrow__` has leaked

The dry run against a copy of the live backup **refused to execute**: `__escrow__` holds
**0.056001 NRN** while the `holds` table has **zero rows in state `held`** — all 13 are
`settled` or `released`. Escrow is bookkeeping for in-flight payments and must be 0 when nothing
is in flight, so that NRN is stranded, not reserved.

`coordinator/ledger.py::settle()` moves everything out of escrow in three transfers — node
shares, the fee, the refund — but the node shares are inside `if total_le > 0`. When **no**
planned node is eligible at settlement time, `total_le` is 0, the whole `pool`
(90% of the charge) is never transferred **and never refunded**: it stays in escrow forever.
The arithmetic fits — wallet `w_d35c84dd…` spent 0.705 NRN, so its pool was ~0.6345, and the
three nodes that served it hold 0.584099 between them; the ~0.05 gap is the same order as the
stranded amount. Per-request payout records aren't stored, so that is a strong hypothesis
rather than a proven attribution, and the smaller per-settlement share-rounding residue is a
second, minor contributor.

**This blocks the prune on the live ledger** and wants its own fix — a money-path change
deserves its own tests, not a bolt-on here. Suggested: sweep any post-settlement remainder back
to the paying wallet as part of the refund, and assert escrow returns to 0 after every settle.

### Rehearsed, not run
`--execute` has **not** been run against anything live. The live DB was never opened for
writing: everything above was a copy in a scratch directory, and the copy was only executed
against after simulating a correctly-drained escrow (returning the 0.056001 to the paying
wallet — what a fixed `settle()` would have done). That rehearsal applied 8 transfers across the
real 31-account shape, held the invariant, and brought `__ecosystem__` to 149,999,999.322001 —
0.678 NRN short of its 150M allocation, which is faucet NRN these identities genuinely **spent**
on inference and is now node earnings, correctly staying with the nodes.

`prune_test_accounts.py` and its test were left uncommitted here — committed in Session 34,
along with the escrow fix this session's refusal uncovered.

---

## Session 34 (2026-08-02) — the escrow leak fixed, and the last four accounts classified

### The leak

`ledger.py::settle()` moves money OUT of `__escrow__` in three transfers — node shares, the
fee, the refund. Two of them could silently fail to account for everything:

1. the node payout sat inside `if total_le > 0`, so when **no** planned node was eligible the
   entire 90% pool was neither paid nor refunded — it stayed in escrow permanently;
2. each node's share was rounded independently, so the shares did not add up to the pool and a
   sub-cent residue was left behind on **every single settlement**.

Both are gone. The last eligible node is now paid the remainder rather than its own rounded
share, so the shares sum to the pool exactly; and the refund is derived from what actually
**left** escrow rather than from the amount charged:

```python
refund = round(hold_amount - paid, 6)
```

That one change makes the refund total by construction. It covers the ordinary unused-hold
case, the whole pool when nobody was eligible, and anything a transfer failed to move — the
money the network did not deliver goes back to whoever paid it, which is the only defensible
destination.

Deliberately scoped to this hold's own amount, never to escrow's balance: escrow is a shared
pot, and "sweep whatever is left in escrow" would raid every request still in flight. There is
a test for exactly that (settle one request while another is held, and check the in-flight 4.0
is untouched).

### A second bug, introduced by the first fix and caught by the tests

Computing the refund in rounded decimal while escrow's balance is the result of its own chain
of float operations means the two disagree in the 15th digit — and `models.transfer`'s
`balance >= amount` guard then **refuses the entire refund** over a 1e-15 shortfall. A 7.77 NRN
hold stranded 7.752. The refund is now clamped to `models.escrow_state()`'s balance, which can
only ever shave dust because the refund is by construction no larger than this hold's own
remainder.

### Asserted, not assumed

`settle()` ends by checking `models.escrow_state()`: escrow's balance against the sum of holds
still in state `held`. Not "escrow == 0" — escrow legitimately holds every in-flight request,
and this network runs concurrent requests by design, so zero is only the idle case of the real
invariant. A mismatch logs at ERROR with the request id and the arithmetic; it does not raise,
because the transfers have already committed and turning a bookkeeping discrepancy into a 500
on a request that succeeded would be worse than reporting it.

`coordinator/test_escrow_conservation.py` **40**: normal, one node ineligible, **no** node
eligible, empty plan, plan nodes that no longer exist, six splits that do not divide evenly,
25 consecutive settlements (where a per-settlement residue would only show up in the
accumulation), a failing payout transfer, and the concurrency case. Verified the suite fails
against the old `settle()` before confirming it passes against the new one — 16 failures, in
every case that was never previously exercised. `test_wallet_settlement.py` (76) still green:
it checked escrow drains on the happy path, which is exactly why neither leak was visible.

One honest note: a hold→settle round trip leaves ~1e-15 of float residue in escrow, because
`balance = balance - amount` cannot exactly undo `balance = balance + amount` on a REAL column.
That is rounding, not stranded NRN — eight orders of magnitude below the 1e-6 the supply
invariant tolerates — and the tests assert against a 1e-9 dust threshold rather than pretending
exact zero is achievable.

### The four unclassified accounts, decided
- **`node-b-optiplex`, `node-c-pavilion` → keep.** They are the OptiPlex and the Pavilion under
  earlier node ids — the same machines as `node_b`/`node_c`, so the same rule applies.
- **`stranger-test-win` → prune.** A rehearsal of the stranger join path on the founder's own
  Windows box; never an outside person.
- **`agent-optinovate` → prune.** A dev install of the packaged agent.

Each prune target now carries its own reason into `prune_log.json` rather than sharing one
generic string — the log is an audit trail, and "test identity" would not tell anyone later why
a particular balance moved. The live snapshot now classifies **10 to prune (201.499609 NRN), 11
to keep, and 10 unclassified — all of which hold exactly 0.000000**, so nothing is left
undecided that has any money in it.

### Still blocking the live prune
The fix stops **new** leaks. It does not reconcile the **0.056001 NRN already stranded** in
`__escrow__` on the live ledger, so `prune_test_accounts.py` still refuses to execute there —
correctly. That stranded amount belongs to `w_d35c84dd…`, which is itself a prune target, so it
lands in `__ecosystem__` either way; it just needs a deliberate one-time reconciliation rather
than being folded into a script whose entire safety argument is that it never touches anything
unnamed.

`--execute` has still never been run against anything live, and the live database has never been
opened for writing.

---

## Session 35 (2026-08-02) — stranded escrow reconciled, Sybil signals, merged to trunk

### `coordinator/reconcile_stranded_escrow.py`
The one-time repair the escrow fix could not do for itself: Session 34 stopped **new** leaks,
but 0.056001 NRN was already sitting in `__escrow__` against zero live holds, and
`prune_test_accounts.py` correctly refused to run past it.

Attribution is by weight of evidence, and the script **prints the evidence instead of asserting
it** — per-request payout records were never stored, so which settlement stranded which fraction
cannot be recovered. It infers the payer from settled hold volume rather than hardcoding a
wallet id, which on the live snapshot picks out `w_d35c84dd…` at **98.3%** of settled volume
(a CLI test wallet holds the other 1.7%). `--to` overrides.

Two refusals that matter more than the happy path:
- **escrow holding *less* than its live holds is refused, not patched.** That is a different and
  worse bug — money backing in-flight requests would be missing — and crediting someone would
  paper over it.
- **a live hold is never mistaken for stranded NRN.** Stranded is `balance - sum(held)`, so a
  request in flight is invisible to it.

Also: dry run reads the DB `mode=ro` so it physically cannot write, one transaction, `--backup`,
and `reconcile_log.json` (gitignored — it names a wallet and an amount).
`coordinator/test_reconcile_stranded_escrow.py` **32**.

One deviation from the brief, stated plainly: it does **not** assert `escrow == sum(held)`
*before* — that is exactly the condition being repaired and would refuse every real run. It
asserts escrow ≥ live holds before (less would be the worse bug above), and `==` after.

### Sybil signals — flag and log, block nothing
`platform` now joins `cores`/`ram_gb` at registration (agent sends `platform.platform()`), and
the three together form `hw_fingerprint`. A second node id appearing on the same signature
within 24h records a `fingerprint_reuse` flag.

The fingerprint is **deliberately weak and the code says so**: cores/RAM/OS is low-entropy, two
identical laptops collide, and a VM reports whatever it likes. So false positives are expected
and nothing is ever blocked on it — a flagged node registers, routes and earns exactly like any
other newcomer, which is the property most likely to be "fixed" later by someone who reads the
flag as an accusation. There are four tests asserting the *not blocking*. Enforcement on
evidence this weak would lock out honest volunteers to protect NRN that has no value yet.

**The faucet gap was real.** `claim_faucet` is idempotent per wallet, which stopped a wallet
claiming twice but not a *person*: `wallet_for_oauth` keys on `(provider, external_id)`, so
signing in with Google and then GitHub on the same address minted two wallets and two 25 NRN
grants. Now one grant per **verified** email, with the withheld grant flagged for review.
Verified specifically — an unverified address is a claim, not a fact, and gating on it would let
anyone type someone else's address to deny them a grant. Both cases are tested.

Flags live at `GET /admin/sybil-flags` behind the operator secret and on the `/admin` page,
never in public: these are unproven suspicions about specific people.

A leak the tests caught: `hw_fingerprint` and `platform` were being returned by the **public**
`/node/list`, which would have let anyone group the roster by machine — exactly the correlation
the private-earnings and private-address decisions exist to prevent. Both are now operator-only,
like the addresses. `coordinator/test_sybil_signals.py` **35**.

### Merged to trunk
`feature/auto-model-tiering` → **`main-full`**, fast-forward, pushed.

`main-full` is the repo's default branch and its real history. **Not** `main`: that branch is
the single-commit curated public snapshot and shares **no common ancestor** with development
(`git merge-base` returns nothing), so merging there would have joined two unrelated histories
and conflicted on essentially every file. The literal instruction said "main"; the branch that
*is* main on GitHub is `main-full`. If the curated snapshot is what needs refreshing, that is
its own operation — regenerate it from a `git archive`, as Session 30 did.

### Four tests had been failing since b5b4f22, and my own sweep was hiding it
Verifying the merge by **exit code** instead of by the last printed line surfaced
`coordinator/test_model_tiers.py` failing — and it had been failing since **b5b4f22**
("half-precision weight storage"), which lowered the 7b tier from 6 nodes / 40 GB to 3 / 20 and
left four cases asserting against the old numbers. Confirmed pre-existing by running the suite
in a worktree at `7e9a94d`, before any of this arc.

Two ways it stayed invisible. The suite raises on the first bad assert and prints no summary
line, so its output *ends* with `ok test_…` from the cases that already ran — and every sweep I
had run in these sessions used `tail -1` as the pass signal. That is a bad habit and the reason
it survived: **`| tail -1` is not a test result, the exit code is.**

Worse than the red: two of the four were asserting the *opposite of their names*.
`test_sustained_loss_demotes` and `test_brief_dip_does_not_demote` used `nodes(3, ram=8)` as
"below 7b"; once 7b needed only 3 nodes / 20 GB that population was feasible, so the network
never became infeasible and the demotion path was never exercised at all. The fixtures now
derive from `mt.TIERS`, and a new `_below(tier)` helper **asserts** its population is really
infeasible — so the next threshold change breaks loudly instead of quietly turning a
hysteresis test into a no-op.

### State
All 30 suites green **by exit code**: 18 coordinator, 11 agent, 1 packaging. Working tree clean.
`--execute` has still never been run against the live database, and it has never been opened for
writing. The live ledger therefore still carries its 0.056001 NRN of stranded escrow and its 10
prunable test accounts — both scripts are ready, both need a quiesced coordinator and a
deliberate run.

---

## Session 36 (2026-08-02) — the coordinator deployed, and fixes can finally reach strangers

### The live coordinator was months of work behind
Probed before touching it: `/node/{id}/payout-challenge` → **404**, `/admin/sybil-flags` → **404**.
Everything from Sessions 32-35 existed only on this laptop. Deployed 16 coordinator modules +
`relay_auth.py` to the Oracle VM after a full backup (`~/neuron-backup-20260802-221443`, 56 MB),
installed `eth-account` in the VM venv (payout binding needs it; it is imported lazily so the
coordinator starts fine without it and only binding fails), restarted, verified:

| | before | after |
|---|---|---|
| `/node/{id}/payout-challenge` | 404 | **401** (exists, needs a node token) |
| `/admin/sybil-flags` | 404 | **401** unauth, **200** with the operator key |
| `/supply` invariant | — | `1,000,000,000.0`, ok |
| `/status` | — | 2 nodes online, 21/28 layers |

The DB auto-migrated: payout columns, `sybil_flags`, `platform`/`hw_fingerprint`. The escrow
leak fix is now live, so no further settlement can strand NRN.

### The auto-updater — rewritten, because the old one could never have worked
`agent/updater.py` downloaded `agent.py` and swapped it in. The shipped app is a **frozen
PyInstaller bundle**: the code is inside `neuron-agent.exe` and there is no `agent.py` on disk
to replace. It was also never called from anywhere, and the coordinator advertised
`AGENT_VERSION = "0.3.0"` while the installer said 0.16.5 — fourteen minor versions of drift in
a value nothing read.

Rewritten around what the app actually is. Frozen build: download the published installer,
verify SHA-256, run it `/VERYSILENT` and exit so it can replace the files underneath. Source
checkout: refuse and say `git pull` — a working tree is many modules and possibly local edits,
and silently overwriting it would be hostile.

Three rules, ordered by how badly each ends if broken, and each one tested:

1. **Nothing unverified is ever executed.** No published hash → refuse. Mismatch → refuse *and
   delete the file*, so a rejected executable is not left on disk for something else to find.
   An empty `AGENT_SHA256` therefore means no node installs anything, which is the correct
   failure direction for a mechanism that runs binaries on volunteers' machines unattended.
2. **Never mid-request.** A node vanishing during inference reaches the driver as "socket
   closed mid-message" and kills the answer for everyone on that chain. This needed a real
   signal: `compute_lock` looked like the obvious one and is **wrong** — it only guards a slice
   reload since serving moved to `batching.MicroBatcher`, so it is free almost all the time.
   `node_server.is_busy()` now counts live connections plus a 120s grace, because a driver
   holds a chain across many token round trips and may reconnect between them. A busy-check
   that *raises* is treated as busy, never as idle.
3. **The updater can never take the node down.** Unreachable coordinator, non-JSON response,
   truncated download — logged and skipped, serving continues. `remote_info` returns None for
   "could not ask", never "up to date".

`/agent/version` now returns `{version, download_url, sha256}`, keeping `version` first for
older agents. The download is **not** served from the VM — 1 GB of RAM and a ~200 MB installer
— it points at GitHub Releases and the coordinator only advertises metadata.
`agent/test_updater.py` **28**.

**Honest limit:** the mechanism is wired and safe, but it delivers nothing yet. `AGENT_SHA256`
is empty until a GitHub release is published and hashed, and by rule 1 an empty hash means no
node installs anything. Publishing 0.17.0 and setting that env var on the VM is what switches
it on.

### Installer 0.17.0 — built and verified
`packaging/neuron.iss` 0.16.5 → **0.17.0**, `config.AGENT_VERSION` 0.3.0 → 0.17.0,
`updater.LOCAL_VERSION` 0.17.0. Rebuilt with `--clean` after killing the running exe (a running
copy locks the file; a stale cache once shipped old bytecode for a whole session).

    dist/installer/NEURON-Setup-0.17.0.exe    216.6 MB
    sha256  939925ee8ae7ca343e9e90295e8a40b21c2b61f5cdffe0965834067dec29bfe3

Verified in the bundle: **`_sqlite3.pyd` + `sqlite3.dll` are present** (the Chat UI failure),
and `eth_account`/`eth_keys`/`eth_utils` (payout binding, never shipped before). A smoke run of
the built exe registered, auto-placed on 0-13, and began its slice download — and the log lines
now carry the logger name (`[INFO] neuron.agent ...`), so the logging fix is in the artifact too.

**That smoke run had a side effect I did not intend: it registered a real node
(`agent-optinovate-d4e1d9`) on the LIVE coordinator** and started a 1.78 GB download before the
timeout killed it. Deleted it immediately; node list and the supply invariant verified clean
afterwards. A build smoke test should point at a throwaway coordinator, not production — the
`--headless` run inherits `coordinator` from the default config, and an isolated `LOCALAPPDATA`
isolates *state*, not the network.

### Three version strings, one meaning
`config.AGENT_VERSION` (what the coordinator advertises) · `updater.LOCAL_VERSION` (baked into
the exe) · `neuron.iss AppVersion` (installer filename). The update decision is just the first
being greater than the second. Keeping three in sync by hand is exactly how #1 drifted to 0.3.0
while #3 said 0.16.5 — a single VERSION file they all derive from is the obvious follow-up.

### To switch auto-updates on
Nothing updates yet, by design: `AGENT_SHA256` is empty and an empty hash refuses every
install. Two steps, neither needing a code change:
1. publish `NEURON-Setup-0.17.0.exe` as a GitHub release tagged `v0.17.0` (the advertised
   `download_url` already points there);
2. set `NEURON_AGENT_SHA256=939925ee…bfe3` in the VM's systemd unit and restart.

---

## Session 37 (2026-08-02) — the address stops being frozen at install time

### Why this had to come first
The coordinator's address was written into `config.json` once, at install, and **nothing could
ever revise it**. Changing the public hostname would have stranded every existing node
permanently — and a new installer could not have rescued them, because `ensure_config` only
writes defaults when there is no `config.json` at all. The auto-updater is no help either: it
asks the *coordinator* for updates, so if the coordinator's address is the thing that is wrong,
there is nothing left to ask.

The only mechanism that works is the old host telling nodes where the new one is **while it is
still up**. That has to be in place before a move, not during one. Hence: insurance, shipped
now, used later or never.

### What it does
`config.PUBLIC_URL` (env `NEURON_PUBLIC_URL`) is returned as `coordinator_url` on **every
heartbeat** — the one call a live node makes continuously, so a change reaches the whole
network within 30 seconds instead of never — and on **registration**, so a node that
re-registers (relay ticket refresh, restart, recovery) picks it up immediately.

`Agent.adopt_coordinator_url()` compares, and on a difference **probes the new address before
keeping it**: `GET <new>/node/<id>/ping` with this node's own token. Only then does it write
`config.json`, update `self.base`, and log the move. `coordinator_previous` is kept.

That probe is the design decision worth defending. This is a redirect primitive aimed at every
node simultaneously — adopting blindly would replace one way to strand the network with a
faster one, where a single typo in `PUBLIC_URL` is obeyed everywhere at once and nothing is
left able to correct it. If the probe fails we stay put and get told again on the next beat.
URLs are validated strictly for the same reason: garbage, a relative path or a non-http scheme
is dropped without even being probed.

Worth stating plainly: this is **not** a new trust boundary. A coordinator already tells nodes
which layers to serve and which peers to connect to; telling them its own address is strictly
less than that.

`agent/test_coordinator_migration.py` **36** — the refusals get the most attention (dead
address, HTTP error, malformed URL, no probe attempted), plus the property that actually
matters: a restarted agent reads the new address off disk. **Deployed and verified live**: the
heartbeat now returns `coordinator_url: https://neuronnet.duckdns.org`.

### The plan, confirmed: the name does not move
**`neuronnet.duckdns.org` stays the stable public name.** Cloudflare goes *behind* it, not in
front of a new one. PostgreSQL, a second Oracle VM and the load balancer are then all
server-side work with **zero client churn** — no installer rebuild, no config migration, nothing
for any existing node to do. `PUBLIC_URL` therefore should not need to change; it exists so
that the day it does, it is survivable.

None of those three are built yet: the coordinator still opens `sqlite3.connect` directly
(`models.py`), there is one VM in one region, and there is no Cloudflare anything.

### Shipped as 0.17.1
The agent half ships **inside the .exe**, and `NEURON-Setup-0.17.0.exe` was built before this
change — so 0.17.0 cannot follow a move. **0.17.0 was already published** (I had assumed it
was not; corrected on the founder's word and confirmed against the GitHub API — tag `v0.17.0`,
public, published 10:59Z). Its release notes were therefore restored rather than repurposed,
and 0.17.1 got its own.

    dist/installer/NEURON-Setup-0.17.1.exe    216.7 MB
    sha256  befeddd3c83dbe00cd987310c47e15fe8a7035ed32fe831139ca93c92c87ab71

Four version strings had to move, not the three in the brief: `neuron.iss`,
`config.AGENT_VERSION`, and **`updater.LOCAL_VERSION`**. That last one is what a running agent
compares against; left at 0.17.0 while the coordinator advertised 0.17.1, every fresh 0.17.1
install would have decided it was out of date and tried to update to itself, daily, forever.
The same drift that had `AGENT_VERSION` sitting at 0.3.0 for fourteen versions.

**Step 7 verified behaviourally, after a bogus first attempt.** Grepping the .exe for
`adopt_coordinator_url` found nothing and proved nothing — PyInstaller compresses module
bytecode into the PYZ, so no Python identifier is searchable there. The real check was to run
the built binary against two throwaway coordinators on localhost with `LOCALAPPDATA`
redirected: it registered with A, was told the address is now B, probed `B/node/<id>/ping`,
wrote `coordinator` and `coordinator_previous` to config.json, and sent its **next** call
(slice-info) to B. Whole path, in the artifact that ships, without touching production —
unlike the 0.17.0 smoke test, which registered a real node.

### Live now
`/agent/version` returns 0.17.1 with the v0.17.1 download URL. `sha256` is still empty, so **no
node will install anything** — that is rule 1 of the updater working, not a gap.

### The remaining steps need a browser
`gh` is not installed, so the upload could not be done from here:
1. publish `dist/installer/NEURON-Setup-0.17.1.exe` as a GitHub release tagged **v0.17.1**,
   notes from `RELEASE_NOTES_v0.17.1.md`;
2. set `NEURON_AGENT_SHA256=befeddd3…ab71` in the VM's systemd unit and restart.

After step 2, **0.17.0 installs upgrade themselves to 0.17.1** — 0.17.0 already contains the
updater, so the auto-update path gets its first real use delivering the fix that makes a
coordinator move survivable.

---

## Session 38 (2026-08-03) — auto-update switched on, and the live ledger cleaned

No new code. Three things that had been built and never actually done.

### 0.17.1 published — with a tag mismatch that would have silently broken everything
The founder published the release while this ran. It is **tagged `0.17.1`, not `v0.17.1`** — but
`config.AGENT_DOWNLOAD_URL` derives its URL from the version as `v{version}`, so the address
every node was about to be handed **404'd**. Verified directly rather than assumed:

    advertised  .../releases/download/v0.17.1/NEURON-Setup-0.17.1.exe  -> HTTP 404
    actual      .../releases/download/0.17.1/NEURON-Setup-0.17.1.exe   -> HTTP 200

Left unfixed, switching the hash on would have had every node download nothing, log a warning,
and stay where it was — an auto-update mechanism that appears live and delivers nothing.
Worked around with `NEURON_AGENT_DOWNLOAD_URL` in the systemd unit.

**Then the founder re-tagged it to `v0.17.1`, which inverted the workaround.** The override now
pointed at the old no-`v` URL, which 404s, so the fix had become the bug — a stale override is
worse than none, because it silently overrides a value that has since become correct. Checked
rather than assumed: `v0.17.1` → 200, `0.17.1` → 404. Override removed (the hash stays), so the
URL config.py derives from the version is used again, and the systemd file now carries a comment
explaining why it must not come back.

Verified the way that actually settles it — walked the agent's own path end to end: read
`/agent/version`, fetch the advertised URL, hash the 216,655,277 bytes that arrive, compare to
the advertised hash. Match. **An agent would accept this update.**

**The published asset was verified byte-for-byte**, not assumed: downloaded all 216,655,277
bytes from GitHub and hashed them. `befeddd3…ab71`, identical to the local build and to what the
coordinator now advertises. That check is the whole basis on which every volunteer's machine
will run this binary unattended.

### Auto-update is live
`/agent/version` now returns version `0.17.1`, a non-empty sha256, and a URL that resolves 200.
This is the first moment a fix can reach a stranger without asking them to do anything.

### The live ledger, cleaned — first time either script has touched production
Coordinator stopped, **two** backups (the verified snapshot via `coordinator.backup`, plus a raw
copy at `~/neuron-db-pre-cleanup-20260802-220233.db`), then dry run → read → execute for each,
in order.

*Reconcile:* 0.056001 NRN stranded, 0 requests in flight, attribution 98.3% to
`w_d35c84dd…`. Credited back. `__escrow__` 0.056001 → **0.000000**, matching its live holds.

*Prune:* 10 accounts, **201.555610 NRN** → `__ecosystem__`. That total is 0.056001 higher than
the earlier dry run predicted, and correctly so: the reconciled escrow went to a wallet that is
itself a prune target, so it flowed straight through to the ecosystem bucket. The two scripts
composed exactly as designed.

All **12** unclassified accounts held `0.000000`, so nothing undecided had money in it.

    __emission_pool__   599,999,971.999972
    __founder__         200,000,000.000000
    __ecosystem__       150,000,001.555610
    __liquidity__        50,000,000.000000
    __escrow__                   0.000000
    total_supply     1,000,000,000.0   invariant_ok: True   (exactly 1e9)

What still holds NRN, and nothing else does: `node_a` 9.011876, `node_c` 8.107126, `node_b`
6.082124, `__coordinator__` 2.867800, `node-c-pavilion` and `node-b-optiplex` 0.187746 each.
Real compute on real hardware, and fees actually earned.

`__ecosystem__` sits 1.555610 **above** its 150M allocation — correct, not drift: it is the
faucet NRN that came back plus fees those test wallets paid to nodes and the coordinator along
the way. Supply is conserved; the bucket split moved.

### Blocker 2 of MIGRATION_PLAN.md is closed
Test identities can no longer become real transferable tokens on-chain. Blocker 1 (payout
addresses) was closed in Session 32. Both prerequisites for an on-chain migration are done —
which does not bring it closer, since the decision is still NEURON Chain at 50+ nodes.

---

## Session 39 (2026-08-03) — the whitepaper, written only from what was measured

No code. `whitepaper.md` in the repo root, ~2,900 words, built strictly from `ROADMAP.md`,
this file, `NEURON_COMPLETE_PROJECT.md`, `PROBLEMS.md`, `SCALING.md`, `SECURITY.md`,
`SAFETY.md` and `blockchain/MIGRATION_PLAN.md`. Twelve sections: abstract, problem, solution,
architecture, measured results, security model, NRN economics, decentralisation roadmap,
comparison to Bittensor/Gensyn/io.net/Petals, the environmental argument, limitations,
conclusion. The README's ASCII architecture diagram is carried in verbatim. Nothing from
`blockchain/` or `neuronscript_*` is committed — `MIGRATION_PLAN.md` was read as a source for
the token decisions and stays gitignored where it is.

### Numbers used, and the ones deliberately left out
Every figure traces to a session: 3.2 tok/s single node (S3, 3.18 in the [P2] spike) → 4.61 on
two (S4) → 6.16 at 3.82× overlap on three (S5); wire codec 12,508 → 2,946 B/msg = 4.25× with
6/6 outputs identical (S21); auto-verification 254 ms (S27); reboot-to-serving 13 s (S26);
bit-exact split (S1/S5); slice download 1.40/0.84/0.84 GB vs 3.09 (S8); 22 s clean onboarding
(S29); local engine 27.9 tok/s at 1.5B and 7.8 at 7B (S26/S28); 70B at 0.62 tok/s on one
machine (S25); the relayed run at 0.53 tok/s with 33.6 s of its 45.5 s on the wire (S26); the
post-cleanup ledger to six decimals (S38).

Left out on purpose: NeuronScript's +61% and the per-row-scaling accuracy win. Both are real
and measured (S22/S23), but the kernel sources are gitignored and the paper describing them is
supposed to precede their publication — a whitepaper is exactly the wrong place to describe
them first. The comparison table's competitor column is the project's own reading of those four
projects, not a measurement, and the text says so.

### Two claims in the brief that the sources did not support as written
- **"Halving every 2 years"** is in `ROADMAP.md` and in `NEURON_COMPLETE_PROJECT.md`, and
  `grep -r halv coordinator/` returns **nothing** — the ledger settles a flat 1.0 NRN per
  request. Written as *designed but not implemented*, because "halving every two years" in a
  whitepaper reads as a live emission schedule.
- **"2–5% typical CPU usage"** is `ROADMAP.md`'s framing of the opportunity, not something
  measured here, and the *default* guard yields above a **15%** ceiling rather than capping at
  2%. Both figures are stated for what they are.

Also stated rather than smoothed over: the wire codec is still not deployed to the remote nodes,
so the 74% wire cost is being paid in full; the network is 2 nodes / 21 of 28 layers as of S36;
no outside person has ever run a node; and Phases 2-5 of the decentralisation roadmap are all
labelled planned, since the coordinator still opens `sqlite3.connect` directly in one region.

---

## Session 40–41 (2026-08-03) — GPU capability, grant docs, and one deploy that could not be done

### The live network, read rather than assumed
`/status`: **2 nodes online, 21 of 28 layers, `network_healthy: false`**, 38 lifetime requests,
25.8102 NRN distributed. `/node/list` says which two, and it is not the pair the docs imply:

    agent-optinovate-67e4eb   layers 0-13   16 cores  68 GB   verified
    node-c-pavilion           layers 14-20   4 cores  12 GB   trusted

So the Windows PC and the Pavilion. **The OptiPlex is not registered as a node at all**, which
is exactly why 21-27 is uncovered and the chain is unhealthy. The whitepaper's figures were
already right; what was wrong was the attribution — "as of Session 36" described the reading as
historical when it is current. Fixed, with the uncovered range and the lifetime totals added.

### The Pavilion deploy did NOT happen — no SSH access
`100.79.125.112` pings (14-119 ms) and sshd answers, so the machine is up. But neither key on
this laptop is authorised on it: `id_ed25519` and `oracle_coordinator` were both refused for
`ubuntu`, `homeadmin`, `optin`, `neuron` and `pavilion` — ten attempts, all
`Permission denied (publickey,password)`. `~/.ssh/config` has entries for the OptiPlex and a
NUC, none for the Pavilion. This is Session 5's trap ("a brand-new node needs your SSH pubkey
in its authorized_keys") still unresolved for this machine.

Two consequences, both deliberate:
- **The whitepaper still says the wire codec is not deployed**, in section 11 *and* section 5.
  Removing that line was conditional on the deploy landing. It did not, so the claim stands.
- The brief's `sudo systemctl restart neuron-agent` would not have worked anyway. Session 26
  installed the agent as a **`systemd --user`** service with linger enabled, so the unit is
  `systemctl --user restart neuron-agent`; the `sudo` form reports "Unit not found".

To unblock: add this machine's `~/.ssh/id_ed25519.pub` to the Pavilion's `authorized_keys`, or
run the two commands at the machine.

### GPU support — built, and honest about where it stops
`agent/gpu.py` (new): `detect_gpu()` prefers `torch.cuda` and falls back to `nvidia-smi`, so a
machine with a card but a CPU-only torch build still reports its hardware; `gpu_utilization()`
uses nvidia-smi only, because torch cannot see load from *other* processes and that is precisely
the load a node must yield to. Reported at registration, stored on the node row (migration
included), and `gpu_name` is operator-only in `/node/list` for the same reason `platform` is —
a card model is distinctive enough to correlate a roster on. `has_gpu`/`gpu_vram_gb` stay public,
being no more identifying than the `cores`/`ram_gb` already published.

`resource_guard` gained a `gpu_ceiling` per donation mode: a machine can be CPU-idle while a
game saturates the GPU, and the CPU meter cannot see it. The load-bearing property is the
failure direction — **unreadable utilisation is not busy**. Most machines have no nvidia-smi,
and treating "cannot tell" as "busy" would silently empty the network while looking like calm.

**What was deliberately not done, and why it would have been a regression.** `common.py`
materialises every shard into system RAM and selects no device anywhere in the inference path —
grep confirms no `.cuda()`, no `device_map`. A GPU node therefore computes on its CPU. So:
- VRAM does **not** raise a node's layer cap. The arithmetic is written and tested but gated
  behind `balancer.GPU_EXECUTION = False`. Counting VRAM the pipeline cannot use would hand a
  GPU node more layers than its system RAM holds and OOM-kill a volunteer's machine — the exact
  failure `max_layers_for()` exists to prevent.
- A GPU is **not** a speed multiplier. Speed is the measured `ms_per_layer`, which already
  reflects whatever hardware did the work; multiplying it by a guessed factor double-counts.
  "GPU nodes preferred" is implemented as a tie-break when two equally-fast nodes could receive
  a layer that had to move anyway — a no-op on an all-CPU network, which is every network today.

Making the GPU actually compute is a `common.py` change, and ROADMAP build rule 7 forbids
touching that file without an explicit instruction. Raised rather than quietly done.

**Tested on node_a, which turned out to have no GPU**: no `nvidia-smi`, torch is `2.4.1+cpu`,
`cuda_available: False`. So the real-hardware run exercises the CPU-only path end to end (the
path most volunteers take), and the GPU paths are covered by stubs. No GPU speedup is claimed
anywhere, because none was measured and none is possible yet.

Docs say the same thing in plain words: INSTALL.md and STRANGER_INSTALL.md tell an operator the
card is detected and respected, then state outright that inference still runs on the CPU so it
does not earn faster yet.

### A bug my own test caught
`detect_gpu()`'s docstring promised "never raises", and it did not: the promise rested on the
two probe helpers guarding themselves. A test that made `_from_torch` raise took the whole
function down — and registration depends on that promise, since a node must be able to join as
CPU-only however creatively a GPU stack fails. Both probes are now wrapped at the call site.

### README, CONTRIBUTING, CHANGELOG
README's Status section on `main-full` replaced with the Session 41 text.

**The other README fix had nothing to fix on this branch.** `main-full` does not contain
"Nodes reach each other over Tailscale" — it has said "**No VPN, no port forwarding, no
Tailscale**" since Session 26. That stale line, and "Session 7 complete", are on **`main`**,
which `git log` shows is a strict ancestor **92 commits behind** `main-full`. (Session 35's note
that the two share no common ancestor is now out of date — `git merge-base` returns `76e8e46`,
which is `main`'s own HEAD.) So `main` is a stale pointer, not a divergent branch, and the clean
fix is a fast-forward — a branch operation nobody asked for, so it was left alone and flagged.

`CONTRIBUTING.md` and `CHANGELOG.md` added. The changelog's Unreleased section states the GPU
limitation rather than listing the feature and stopping.

### Tests — 37 green, by exit code
19 coordinator (new `test_gpu_capability` 19 cases) + 17 agent (new `test_gpu` 22 cases) via
`-m`, plus `packaging/test_app_entry.py` **run as a script**. That last one matters: `packaging/`
has no `__init__.py` and `packaging` is also an installed third-party module, so `python -m
packaging.test_app_entry` resolves to the wrong thing and reports `No module named` — a failure
that is not real. My first draft of CONTRIBUTING.md put that exact broken loop in front of
contributors; corrected, with the reason.

Compatibility checked rather than assumed: `RegisterBody` sets no `extra="forbid"`, and Pydantic
ignores unknown fields by default, so a 0.17.1+GPU agent registering against the **currently
deployed** pre-GPU coordinator works — the fields are dropped until the coordinator is updated.
None of this is deployed to the live coordinator.

---

## Session 42 (2026-08-03) — the GPU actually computes, and `main` stops lying

Session 41 left a node able to *report* a GPU while computing on its CPU. This session gives the
pipeline a device.

### `common.py` — one device, resolved once
`_resolve_device()` picks CUDA when `torch.cuda.is_available()`, CPU otherwise, with
`NEURON_DEVICE` overriding both (which is how a GPU machine gets A/B'd back onto CPU). A typo in
that variable and a broken driver both fall through to CPU rather than stopping the node — a
volunteer's machine failing to *start* is worse than it running slowly.

Two invariants keep the change from spreading:

1. **Every public interface stays CPU.** `first_stage`/`mid_stage`/`last_stage`/`apply_lm_head`
   take CPU tensors and return CPU tensors exactly as before; the move happens inside. So
   `wire_codec`, `batching`, `junction_cache`, `relay.py` and `security/proof_of_compute.py` did
   not have to learn what a device is, and none of them can be handed a CUDA tensor.
2. **A CPU-only machine sees no change at all.** `Tensor.to()` returns *self* when device and
   dtype already match, so every added call is a genuine no-op — not a copy.

`move_model_to_device()` exists because `model.to(device)` cannot be used here, and the reason is
specific: `load_model_shard` builds the full architecture on the `meta` device and materializes
only this node's layers, so `.to()` hits `Cannot copy out of meta tensor` on the first foreign
parameter. Walking modules and skipping meta tensors avoids that — but replacing parameters one
at a time **breaks the embed/lm_head tie**, which on a 152k-vocab model silently doubles the
largest single allocation on the node (~0.9 GB of VRAM, on exactly the machines least able to
spare it). `tie_weights()` afterwards restores the shared storage.

TF32 is disabled deliberately. cuBLAS will otherwise run fp32 matmuls at ~10-bit mantissa on
Ampere+, drifting ~1e-3 from the CPU result. That is still inside proof-of-compute's `atol=0.05`,
but it spends a fifth of the honest/cheat budget for nothing — and that separation (honest ~1e-5
vs cheating ~25) is the mechanism that lets strangers earn at all.

### `balancer.py` — VRAM replaces RAM, it does not add to it
`GPU_EXECUTION` flips to `True`, now that the claim behind it is true. The load-bearing detail is
that a GPU node's capacity is its **VRAM instead of** its system RAM. Summing them was the
obvious move and is wrong: the weights live in one place, so 4 GB RAM + 24 GB VRAM would claim
21 layers of room on a machine that has 24 GB in one place — reintroducing precisely the OOM
`max_layers_for()` was written to prevent. A GPU whose VRAM is *smaller* than free RAM still
gets the VRAM figure, for the same reason. `test_gpu_capability` asserts the non-summing
directly rather than only asserting the happy number.

### What is proven, and what is not
**Not verified: the CUDA branch has never executed.** No machine here has an NVIDIA card and the
installed torch is `2.4.1+cpu`, so `torch.cuda.is_available()` is `False` and not one line of the
GPU path has run. The first real GPU node is the test, and no speedup figure is claimed anywhere.

**Verified: it is inert on CPU.** `selftest_shard.py` → `max|delta| = 0.000e+00` and identical
decoded tokens, so the bit-exactness proof the network rests on survives a `common.py` change.
Full suite **37/37 green by exit code**.

One trap worth recording: the first run of the suite showed **29 failures**, all
`ModuleNotFoundError`. Bare `python` on this box is a 3.14 install with none of the deps; the
project venv is `.venv` (3.11.9). The suites were fine. Run them as `.venv\Scripts\python.exe -m
<module>`, never as bare `python`.

### `main` was three sessions stale, and is not any more
Session 41 recorded `main` as a strict ancestor 92 commits behind, with merge-base `76e8e46`.
That was wrong — `git merge-base origin/main origin/main-full` returns **nothing**. `main` was an
**orphan** branch holding one squashed commit (`d912584`, 25 July) with no shared history at all,
which is why it still advertised "Session 7 complete" and "Nodes reach each other over
Tailscale". Force-updated to `main-full` (`d912584 → 7b7c11c`, tagged
`backup/main-before-force-2026-08-03` first). Both stale lines are gone from the live repo.

`main-full`'s README needed no fix: it has carried the Session 41 status and "No VPN, no port
forwarding, no Tailscale" since Session 26. The stale text only ever existed on `main`.

### Four internal planning docs untracked
`NEURON_COMPLETE_PROJECT.md`, `FIRST_STRANGER.md`, `PETALS_NOTES.md` and
`distributed-ai-network-plan.html` removed from the index with `git rm --cached` (they stay on
disk) and added to `.gitignore`, under the existing "private / pre-public docs" heading next to
`TOKENOMICS.md`. Between them they carry the Oracle VM's IP, home LAN addresses
(`192.168.1.10`), founder-allocation pointers, and the legal/budget/risk planning — no
credential, no key, but not the public story either. Verified absent from both branches:
`git ls-tree origin/main-full` and `origin/main` return nothing for all four, and
`/blob/main-full/FIRST_STRANGER.md` is a 404.

**What this does not do, and it matters: the files are still in the history.** `git rm --cached`
removes them from the tip only. All four remain readable in every earlier commit of a 114-commit
public repo — `git log --all -- FIRST_STRANGER.md`, or fetching the blob by SHA, still returns
them to anyone. Actually un-publishing them needs a history rewrite (`git filter-repo`) plus a
force push, and even then GitHub keeps orphaned blobs reachable by SHA until it garbage-collects,
and any fork or clone taken before today keeps its own copy. That was not done here — it was out
of scope for this change — so the correct description of the current state is "no longer shipped
going forward", not "removed from the public repo".

A caching trap, recorded because it nearly produced a false all-clear: the first verification
fetch of the repo root still listed all four files. That was a 15-minute-cached copy of a page
fetched earlier in the same session, not the live state. Ground truth came from `git ls-tree`
against the freshly fetched remote refs. Verify a push against the refs, not against a page that
may be served from cache.

---

## Session 42b (2026-08-03) — verification pass, 0.18.0 built, and a node I knocked offline

Picked up after the GPU work was committed in another window. This entry is the independent
re-verification the founder asked for, plus the installer build, plus one real mistake.

### Verified rather than assumed
- Working tree clean; `origin/main-full` and `origin/main` both at `d962fe4`. The `main` sync
  was done in the other window, so the curated snapshot `d912584` is gone — it had been my
  recommendation NOT to force-push, but that is now settled and there is nothing to undo.
- GPU code intact: `common.py` resolves `DEVICE`, `balancer.GPU_EXECUTION` is `True`,
  `device_name()` returns `cpu` on this machine.
- **All 37 suites green by exit code**, and `selftest_shard.py` reports
  `max|delta| = 0.000e+00` — the CUDA branch is genuinely inert on a CPU box.
- `/agent/version` serves 0.17.1 with the correct sha and a 200 URL. An earlier read came back
  empty and looked like an outage; it was a transient from firing parallel curls at a
  rate-limited endpoint. Re-read before believing a blank response.

### INSTALL.md was left stale by the GPU change
It still told operators "inference itself still runs on your CPU, so a GPU does not make your
node faster yet" — written in Session 41 when that was true, and false the moment `common.py`
learned to move a shard onto CUDA. Rewritten to the honest current claim: CUDA is used
automatically if present, *and* no machine here has an NVIDIA card so the path has never
executed. A doc that overclaims and a doc that under-claims are the same bug.

### Installer 0.18.0 — built, not published
Three version strings bumped (`neuron.iss`, `config.AGENT_VERSION`, `updater.LOCAL_VERSION`);
the download URL derives from `AGENT_VERSION`, so it follows. PyInstaller `--clean`, then Inno
Setup:

    dist/installer/NEURON-Setup-0.18.0.exe    206.6 MB (216,630,112 bytes)
    sha256  dd33d317f92adb2ae5da27f46ac02912ba2106a66c4f21104b27ebd7345fb1a8

**One bullet from the brief's release notes was not written, because it is false.** "Wire codec
deployed to all nodes — activations now 4.25× smaller" — the codec is still not deployed; the
Pavilion deploy has been blocked on SSH for two sessions. The release notes say so explicitly
rather than omitting it, since an earlier draft did make the claim.

Steps 9-11 (publish the release, point the coordinator's `AGENT_SHA256`/`DOWNLOAD_URL` at
0.18.0) are **not done**, and doing 10 before 9 would be the Session 38 trap exactly: nodes
handed a URL that 404s. `gh` is still not installed, so step 9 needs the founder's browser.
There is a second reason to pause: switching auto-update on would push a **never-executed CUDA
branch** to every install unattended.

### The mistake: I knocked node_a off the network
Killing `neuron-agent.exe` is a required build step (a running copy locks the files). The
PowerShell check printed "no neuron-agent running" *after* the stop, which reads as "nothing was
there" but is equally consistent with "killed it, then looked". `agent-optinovate-67e4eb`
(layers 0-13) went offline in that window and the network fell from 21/28 layers to 7/28.

Tried to restore it and made it briefly worse: started the *installed* app, whose
`%LOCALAPPDATA%\NEURON\config.json` turns out to be stale — node_id `agent-optinovate`, pointed
at the old raw-IP coordinator, and that account was one of the ten pruned in Session 38. It
404'd on slice-info and was stopped again. **The config for `67e4eb` is not on this disk**; a
recursive search found only a `test_uninstall_deregister.py` fixture that matched the string by
coincidence (`coordinator: https://c.example`). So whatever runs that node runs it from
somewhere this session could not see, and reviving it is the founder's to do — it is `offline`,
not deregistered, so its identity, reputation and earnings are intact and it returns on restart.

### The smoke-test trap, hit again after being written down
Session 36 recorded: "the `--headless` run inherits `coordinator` from the default config, and
an isolated `LOCALAPPDATA` isolates *state*, not the network." I set `LOCALAPPDATA` to a temp
dir and `NEURON_COORDINATOR` to a dead address, and the run still registered a real node
(`agent-optinovate-1a138a`) against production — `NEURON_COORDINATOR` is not what that path
reads. Deregistered immediately (HTTP 200, `by: node`) and the roster verified clean. Knowing
the trap was not enough; the env var has to actually be the one the code consults. A throwaway
coordinator is the only reliable isolation.

### Still blocked, unchanged
- **Pavilion SSH** — `publickey,password`, no key accepted for any of five usernames. Needs a
  pubkey appended at the machine.
- **`gh` not installed** — Part 4 (repo description) and the 0.18.0 release publish both need it
  or a browser.

---

## Session 42c (2026-08-03) — an opt-out for auto-update, before it is switched on

Small, and deliberately landed *before* `AGENT_SHA256` is set: once the network is auto-updating
and strangers are on it, adding a per-node opt-out means shipping it through the very mechanism
some of them wanted to decline. Cheaper now than retrofitting onto installs already in the field.

`auto_update` (default **true**) in `agent/config.json` and `DEFAULT_CONFIG`. `Agent.update_loop`
passes it to `updater.update_loop(..., enabled=...)`, which logs and returns when false.

Three choices worth recording:
- **The flag is injected, not read from disk.** `updater.py` already takes `busy` as a callable
  "passed in rather than imported so the updater carries no dependency on the node server" — the
  same reasoning applies here, since config lives in `%LOCALAPPDATA%` for a frozen install and
  next to the script for a checkout. The updater should not have to know which.
- **Checked before the initial delay**, so a disabled node never waits 60s and never contacts
  `/agent/version` at all. The test asserts that by making any HTTP call raise.
- **Not applied to `check_once()` or the CLI.** An operator running
  `python -m agent.updater --apply` by hand is asking for an update; a background setting should
  not silently refuse a foreground request.

Read once at startup, so a change takes effect on the next start — stated in INSTALL.md rather
than left for someone to discover.

`agent/test_updater.py` 28 → **34**. One of the six new cases was written wrong first and passed
for the wrong reason: `check(..., calls == [] or True)` — an assertion with `or True` in it can
only pass. Rewritten so the stubbed `check_once` ends the loop itself, which proves a check
really ran rather than that the loop exited early for an unrelated reason. A fake green is worse
than no test, and this one was mine.

All **37** suites green by exit code. No installer rebuild: this reaches existing installs via
the next update and new installs via 0.18.0, which is built but not yet published.

---

## Session 43 (2026-08-03) — auto-update actually switched on, and the grant application

### The release was verified before the switch, not after
`v0.18.0` is published and correctly tagged **with the `v`** — the thing Session 38 got wrong.
Then the check that is the whole basis for running a binary on volunteers' machines unattended:
downloaded all **216,630,112 bytes** from GitHub and hashed them.
`dd33d317…5fb1a8` — identical to the local build and to what the coordinator now advertises.

### The brief's Part 2 would have done nothing, silently
It said to set `NEURON_AGENT_SHA256` and `NEURON_AGENT_DOWNLOAD_URL`. Both would have applied,
and **no node would ever have updated**, because the VM's `config.py` still defaults
`AGENT_VERSION = "0.17.1"` and nothing was going to change it. `/agent/version` would keep
reporting 0.17.1, `is_newer("0.17.1", "0.17.1")` is False, and the 0.18.0 hash would sit there
looking live. Found by reading the deployed file rather than assuming it matched the repo.

So the override sets **`NEURON_AGENT_VERSION=0.18.0`** plus the hash — and deliberately does
**not** set `DOWNLOAD_URL`. With the version correct, `config.py` derives exactly the URL the
brief asked for. Pinning it as well would re-create the Session 38 landmine one version later: a
hardcoded 0.18.0 URL still being served after the version moves to 0.18.1 is the same stale-
override bug in a new hat. The existing override file already carried a comment warning about
precisely this; that comment is preserved and extended. Old file backed up to
`~/override.conf.bak-0.17.1` on the VM.

Verified live, all four independently: version `0.18.0`; sha `dd33d317…`; derived URL is the
`v0.18.0` one; and that URL returns **HTTP 200**. Network unchanged across the restart — 2/2
nodes, 21/28 layers, 38 requests, 25.8102 NRN.

**Auto-update is now genuinely armed.** Every 0.17.1 install takes 0.18.0 within 24h of its next
check. Worth stating plainly: those installs predate the `auto_update` opt-out shipped in
Session 42c, so for them this first update is unconditional — the flag only bites from 0.18.0
onward. Blanking `NEURON_AGENT_SHA256` remains the stop button.

### README
Download link and hash → 0.18.0. Status → Session 43. The GPU line rewritten: it said inference
was still CPU-only, which stopped being true in Session 42. Now says CUDA is used automatically
if present, and still says no machine here has an NVIDIA card so that path has not run on real
GPU hardware. Honest numbers corrected to the live reading — **2 online, 21/28 layers**, and the
chain is incomplete, which the previous text ("1 online, 7 of 28") understated in one direction
and the dashboard's DEGRADED banner explains.

### `grant/nlnet_application.md`
798 words, built only from `whitepaper.md`, `README.md` and the live dashboard. Framed as open
internet infrastructure: the funding items are DHT peer discovery (€15k), coordinator redundancy
and PostgreSQL (€10k), the Android NEON port (€15k) and a code-signing certificate (€5k) —
€45,000, and every one of them is decentralisation or access, not features.

Honest by construction: it leads with "2 machines covering 21 of 28 layers — an incomplete
chain", says no outsider has run a node, and describes distribution as buying capacity and
concurrent users rather than single-answer speed. The accounting layer is described as what it
is — a record of contributed work, in an ordinary database, with no cash value, no exchange and
no sale. Nothing is concealed by that framing: the application links the repository, and
`whitepaper.md` §7 there sets out the on-chain intention in full. If a reviewer asks directly,
the answer is in the same repo the application cites.

**Two edits after review.** Section 2 gained a sentence naming who is actually shut out —
a researcher where these services are restricted, a developer who cannot afford per-token
pricing — which is the concrete version of an otherwise abstract argument about access. The
code-signing line became "certificate **and infrastructure review**": the same €5,000, but it
now says a security review before wider distribution is part of the responsible step, which is
honest about what shipping an unsigned binary to strangers actually requires.

That takes the document to **821 words**, past the 800 it was written to. Left as-is rather than
trimmed, since both additions are substantive and NLnet's form has no hard limit — but the
number is recorded here so nobody assumes the constraint still holds. ("per-token pricing" trips
a naive grep for `token`; it means text tokens, i.e. how the hosted APIs bill, and is the one
sense of the word the framing rules were never about.)

---

## Session 44 (2026-08-03) — a landing page, and the CORS hole it depended on

### The live stats would have shown four dashes
Checked before writing any HTML: `/status` returned **no `Access-Control-Allow-Origin` header**
and `main.py` had no CORS middleware at all. A browser on `github.io` fetches it, gets a 200, and
then refuses to let the page read the body — so the whole "live network" panel would have sat at
`—` with nothing in the UI explaining why. Worth finding by testing rather than by launching.

`CORSMiddleware` added, deliberately narrow, and each choice is a real one:
- **named origins, never `*`.** `/status` is public, but middleware is app-wide, and a wildcard
  invites any page on the internet to use a visitor's browser as a client against endpoints that
  are not public.
- **`allow_credentials=False`.** With named origins it would let a page send a visitor's cookies;
  the privileged endpoints authenticate on headers a web page cannot know, and that stays true
  only while nothing is attached automatically.
- **GET only.** Nothing on the landing page writes.

`config.CORS_ORIGINS` (env `NEURON_CORS_ORIGINS`) defaults to the Pages origin plus localhost.
Deployed to the VM with both files backed up first. Verified both directions, which is the part
that matters: `Origin: https://neuron-network-ai.github.io` → `Access-Control-Allow-Origin`
echoed back; `Origin: https://evil.example` → **no header at all**. `/agent/version` still serves
0.18.0 with the right hash, so the restart cost nothing.

### `docs/index.html`
One file, no framework, no external request of any kind — the favicon is an inline SVG data URI
rather than a fetch, so the page discloses nothing to a third party just by being opened. Dark
theme, CSS grid, renders single-column on a 375px viewport with no horizontal scroll (checked,
not assumed). Live stats every 30s.

Two things the brief did not ask for and the page has anyway, because a public front door is
exactly where this project's honesty either holds or does not:
- **The NRN paragraph says it has no cash value**, no exchange, no sale. Every other document
  says so; a landing page that implied earnings would contradict the README two clicks away.
- **The health line explains an incomplete chain in words.** The panel currently shows 21/28,
  and "21/28" means nothing to a newcomer — so when `network_healthy` is false the page says the
  chain is incomplete and the network cannot serve a full request until it is. A failed fetch
  says the coordinator could not be reached rather than leaving stale numbers looking current.

Rendered against the real endpoint before commit: 2 nodes, 21/28, 38 requests, 25.81 NRN, amber
dot, correct incomplete-chain text.

### The URL in the brief cannot exist
`https://neuron-network-ai.github.io` is an **organisation** site and requires a repository named
literally `neuron-network-ai.github.io`. This repo is `neuron`, so Pages serves it at
`https://neuron-network-ai.github.io/neuron/`. Both currently 404, and the bare root always will
until such a repo exists. CORS is unaffected either way — an Origin is scheme + host and never a
path — so the entry already covers the project path.

**Pages could not be enabled from here** — `gh` is not installed and the API needs auth this
session does not have. The founder enabled it: Settings → Pages → branch `main`, folder `/docs`.

### Live, and verified from the real origin
`https://neuron-network-ai.github.io/neuron/` returns **200**. The bare org root still 404s,
exactly as predicted, and will until a repo named `neuron-network-ai.github.io` exists.

The check that actually settles CORS is not curl with an `Origin` header — it is a browser on the
real origin, because only that exercises the same-origin policy for real. Loaded the published
page and read its DOM back: `origin: https://neuron-network-ai.github.io`, `path: /neuron/`, and
the panel populated with **2 nodes, 21/28, 38 requests, 25.81 NRN**, timestamped and re-fetching.
Zero console errors, so nothing was silently blocked. The dot is amber and the health line is the
incomplete-chain text, which is the correct reading of a 21/28 network rather than a failure.

All four outbound links return 200: the 0.18.0 installer, INSTALL.md, the repository and the
dashboard. (One returned `000` on the first pass — a curl transport blip, not a 404. Worth
re-running a failed probe before believing it; this session saw the same transient DNS failure
take out a `git push` and two `/status` reads.)

Note the Pages **API** still answers 404 anonymously even though the site is up: reading a
repository's Pages configuration needs auth, public repo or not. The site's own response code is
the ground truth, not the API's.

### A logo, from a coincidence that turned out to be structural
`docs/logo.svg` (200×60) and `docs/favicon.svg` (32×32). The brief suggested "incorporate the
letter N subtly within the node connections", and the N is not a pun here: **an N is exactly four
vertices joined by three segments** — bottom-left, top-left, bottom-right, top-right — which is
the shape of a NEURON chain. Each dot is a machine holding a slice; each line is activations
crossing between them. The bright, larger dot is the driver, the one node that holds the
embedding and output projection and the only one that ever sees readable text. It is drawn
differently because it *is* different.

No gradients and no filters — solid fills and opacity only, so it stays crisp at any size and
renders identically everywhere. No brain, no chip, no robot.

The favicon is **redrawn rather than scaled**: proportionally shrinking a 200px mark turns the
strokes to grey mush in a tab. Heavier strokes, tighter padding, and no background rectangle so
it sits correctly on a light or dark tab strip. At 16px the strokes land at 1.75px and the driver
dot at 2.7px radius — comfortably above the point where a shape stops resolving.

`index.html` now loads both, and the old inline data-URI emoji favicon is gone. The `<h1>` still
exists, wrapping the image with `alt="NEURON"`, so the page keeps one heading for screen readers
and search engines rather than losing it to an `<img>`. **The page still makes zero third-party
requests** — both files are same-origin relative paths, verified by listing every absolute URL in
the file: four links and the `/status` fetch, no asset loads.

**Honest limit on the verification.** The logo was confirmed visually — at full size, at reduced
width on the live site, and it reads at both. The **16px favicon was never seen rendered**. The
preview pane failed in three different ways: `navigate` timed out twice on a scratchpad path,
then served a stale tab, then `javascript_tool` reported "no site open" twice immediately after
a navigate that had just succeeded and screenshotted. So legibility at tab size rests on the
geometry above, not on an image anyone looked at. Both files validate as well-formed XML with no
external references, which is the part that could be checked properly from here.

After the Pages rebuild all three assets serve correctly — `index.html` `text/html`, `logo.svg`
and `favicon.svg` both **`image/svg+xml`** — and the published HTML references both. The
remaining check costs the founder one glance at their own browser tab; if the mark reads muddy at
16px, thickening `stroke-width` in `favicon.svg` is a one-line change.

---

## Session 45 (2026-08-03) — the install flow, and a light-theme snippet on a dark page

The four-step flow (Download → Run → Sign in → Contribute), with icons, connectors, numbered
captions and the six capability pills. It is a real improvement: the page previously explained
onboarding in prose, and prose is the wrong shape for "what happens when I click this".

Three things in the brief did not survive contact with the file, all recorded because each would
have shipped a visible defect.

**1. The section it said to replace does not exist.** "Replace the existing 'Join in four steps'
step cards" — `docs/index.html` has no `.steps` grid and no step cards; `grep` returns zero. That
markup is in a separate green mockup, not in the deployed page. Added as a new section between
"How it works" and "What you need" instead, which is where the narrative wants it.

**2. The colours are from a LIGHT mockup and would have been unreadable.** Measured rather than
eyeballed, against this page's `#0d1117`:

| element | supplied | contrast | |
|---|---|---|---|
| `.step-title` | `#1a2e1c` | **1.31:1** | near-black on near-black — invisible |
| `.flow-pill` text | `#166534` | 2.65:1 | fails |
| `.step-desc` | `#6b7280` | 3.91:1 | fails |

WCAG AA wants 4.5:1. Pasted verbatim, all four step headings would have vanished. Six colour
values remapped to the page palette; layout, icons, connectors, copy and the 600px breakpoint are
untouched. After: title 16.02:1, description 6.15:1, number 6.87:1, pill 14.64:1, and the filled
circle's glyph 6.85:1 — every pairing passes.

**3. The Google Fonts `<link>` was dropped.** It would have been this page's only third-party
request, on a site whose pitch is privacy — every visitor's IP handed to Google to fetch a
typeface. That "zero external requests" property is verified and recorded in Session 44, so
breaking it silently for Inter and Space Grotesk was not a trade worth making. The page stack is
used instead. If those typefaces are actually wanted, self-hosting two woff2 files in `docs/`
gets them with the property intact.

Verified rendered at desktop and at 375px, where the media query hides the connectors and the
steps reflow 2×2. External-URL audit re-run: still four links and the `/status` fetch, no asset
loads.

**Left alone deliberately:** the pills duplicate "What you need" almost exactly (No GPU / no
crypto wallet / no payment details). The brief said keep everything else as it is, so both stay —
but one of them is redundant, and "What you need" is the one carrying the GPU nuance the pills
drop. Worth a decision rather than a silent deletion. Also still open: green vs blue for the
whole page, which this snippet does not settle — it is light-green, matching neither the current
blue page nor the dark-green mockup.
on first connect.
```bash
ssh <node-b-host> "cd ~/neuron && ./.venv/bin/python node_b.py --port 50999"
```
```bash
ssh <node-c-host> "cd ~/neuron && ./.venv/bin/python node_c.py --port 50999"
```

**2. Throughput demo — 4 requests in parallel (Windows).**
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe C:\Users\optin\neuron\node_a.py --host-c 100.79.125.112 --host-b 100.114.189.46 --s1 9 --s2 18 --max-new-tokens 80
```
Add `--serial` for the one-at-a-time baseline, `--copies 2` for N=8, or `--prompt
"Hello"` for a single request. `--s1/--s2` set the 3-way layer boundaries (9/9/10
balances this trio).

**3. Verify correctness — 3-stage chain, bit-exact (Windows).**
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe C:\Users\optin\neuron\selftest_shard.py
```

**4. Via the coordinator (S6/S7).** The coordinator runs on the always-on OptiPlex
(`:8001`). Register the nodes, then let node_a discover the chain itself.
```bash
ssh <node-b-host> "cd ~/neuron-coordinator && ~/neuron/.venv/bin/python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8001"
```
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe coordinator\register_nodes.py
```
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe node_a.py --coordinator http://100.114.189.46:8001 --prompt "Why is the sky blue"
```
Dashboard at http://100.114.189.46:8001/dashboard. (When the Pavilion is off, run
node_c locally: `node_c.py --port 51000` + `register_nodes.py --node-c-host 127.0.0.1
--node-c-port 51000`. To run the coordinator locally instead, use `python -m uvicorn
coordinator.main:app --port 8000`.)

**5. Chat UI (Session 10) — talk to the network in a browser.** Runs on the node_a
machine (holds the driver shard). Needs a healthy chain (all 28 layers online).
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe -m uvicorn ui.app:app --host 0.0.0.0 --port 8080
```
Then open http://localhost:8080. `NEURON_COORDINATOR` env var points it at a
different coordinator; defaults to the OptiPlex `:8001`.

**6. OpenAI-compatible API (Session 11).** Already mounted into the Chat UI server
above at `/v1/*` (usage docs at `/api-docs`), or run it standalone:
```bash
C:\Users\optin\neuron\.venv\Scripts\python.exe -m uvicorn api.openai_compat:app --host 0.0.0.0 --port 8081
```
Point any OpenAI SDK at `http://<node_a-host>:8081/v1` with your NRN wallet as the API
key (`pip install openai`). Standalone usage docs at `/docs`.

**Stop the servers** (self-safe pattern — no literal script name in the kill):
```bash
ssh <node-b-host> "pkill -f '[n]ode_b.py'"
```
```bash
ssh <node-c-host> "pkill -f '[n]ode_c.py'"
```

---

## Session 45b (2026-08-03) — the merged page, and stats that would have shown four dashes

The founder supplied a full merged landing page — light green, nav, hero, proof cards,
comparison table, four-step install, dashboard link — which had taken essentially every
correction raised against the earlier draft: the unsigned-installer warning, the "no cash value"
statement, Petals as **partial** rather than ✗, "no single operator can withdraw access" instead
of "uncensorable", and "No new hardware manufactured" instead of the Green AI badge. Better than
what it replaced, so it was adopted whole rather than cherry-picked. Green is now settled.

### The live stats were reading fields that do not exist
The script reached for `d.nodes_online`, `d.layers_covered`, `d.requests_served`,
`d.nrn_distributed`. The real payload nests everything under `network` and `stats` and names it
differently — `online_nodes`, `total_layers_covered`, `total_requests_served`,
`total_nrn_distributed`. Run against the live endpoint, **all four render `—`**.

The dangerous part is not the wrong names, it is the failure mode: the request returns 200 and
the JSON parses, so `.catch()` never fires and the caption underneath still says *"live numbers,
not projections. Updated 15:47:12"* above four dashes. A page that fails loudly is fine; this one
would have looked healthy while showing nothing. Fixed and checked against the real response:
2, 21/28, 38, 25.81.

Second fix in the same block: the status dot keyed off `nodes>0`, which is **green right now**
with two nodes covering 21 of 28 layers — an incomplete chain that cannot serve a request. It
keys off `network_healthy` instead, and the caption spells out what an incomplete chain means
rather than leaving a newcomer to interpret "21/28".

### Contrast, measured again — this time the other way round
Session 45 caught light-on-dark. The same audit on the new light theme caught the mirror image:

| | before | after |
|---|---|---|
| small grey text (`#9ca3af`) — sha, footer, notes | 2.54:1 | 4.83:1 |
| green links, labels, ticks (`#16a34a`) | 3.30:1 | 5.02:1 |
| white on the primary button | 3.30:1 | 5.02:1 |
| the ✗ column in the comparison table (`#d1d5db`) | 1.50:1 | 4.83:1 |

`#15803d` was already in the file as the button hover colour, so no new colour was invented. The
✗ mattered most: it carries meaning in a table making claims about named competitors, and at
`#d1d5db` it was nearly invisible.

### Also this session
- **`favicon.svg` was not linked** in the new head — a regression against Session 44. Restored.
- **Logo assets were still blue** while the site went green. `logo.svg` and `favicon.svg` redrawn
  to the header's mark (driver → hub → two peers) in green, so repo and site stop disagreeing.
  The favicon is again redrawn rather than scaled.
- **"No crypto wallet" removed** on instruction. The page now contains zero occurrences of
  "crypto" or "wallet" — consistent with the compute-credits framing and the NLnet application.
  The install-flow block supplied afterwards still contained that exact pill; it was left out
  rather than silently reintroducing a phrase deleted minutes earlier.
- **The four-step flow** replaced the plain step cards, with `#16a34a` moved to `#15803d`
  wherever it carries small text or an icon, for the same 3.30:1 reason as the rest of the page.
- **Google Fonts shipped**, having been requested twice. Recorded plainly because it is a real
  trade: it is the page's only third-party request, so every visitor's IP and User-Agent reach
  Google to fetch a typeface, on a site whose pitch is that nothing about them is collected.
  Both families fall back to the system stack, so a blocked fetch costs nothing but the styling.
  Self-hosting two woff2 files in `docs/` would keep the typefaces and the property; the offer
  stands.

### The hero pill, cut on a commercial call
"No new hardware manufactured" removed from the hero, asked for as a marketing decision rather
than an accuracy one. Three reasons, and only the third is about honesty:

1. **Every job it could do was already done by something beside it.** The h1 says "idle
   machines", the credits box says "hardware that already existed", the sub-line handles the
   setup objection, and the live stats directly beneath supply the proof. It was the third
   statement of the same idea, placed above the headline, for a reader who did not yet know what
   the product was.
2. **Hero density costs conversion.** Mark → headline → button is the shortest path to the only
   action on the page. Nothing was put in its place; an empty slot beats a decorative one.
3. **It was the one line inviting an argument.** A lone environmental badge on a page making no
   other energy claim provokes "what about the electricity?", and the page cannot answer —
   `whitepaper.md` §10 can (15–45 W above idle, 200–400 W on a GPU node, *less* efficient per
   token than a datacenter GPU; what is avoided is the embodied cost, not the power), but that
   is a sentence with a counterweight, not a chip. This project's competitive asset is that it
   does not overclaim, which makes an unsupported badge a liability wearing a benefit's clothes.

The now-dead `.pill` rule was deleted with it, and the 1.25rem of vertical rhythm it carried was
given back to `.logo-wrap`, so the hero spacing did not silently collapse.

**Two requests this session were already satisfied and were not applied twice:** the install-flow
block (third paste; live and correct — 4 steps, 3 connectors, 1 pill row, no leftover cards) and
the nested `/status` field fix (done in the previous commit; verified no stale flat names remain
in either the working file or the deployed page). Re-running either would have duplicated a
section rather than fixed anything.

---

## Session 46 (2026-08-03) — the Pavilion, and a limitation that had not been true for days

### SSH was never broken — the username was wrong
Five sessions of "the Pavilion is unreachable over SSH" rested on trying `ubuntu`, `homeadmin`,
`optin`, `neuron` and `pavilion`. The account is **`raman`**, which appeared the moment the
founder pasted a shell prompt: `raman@raman-HP-Pavilion-Laptop-15-eh3xxx`. `id_ed25519` was
authorised for it the whole time. The evidence had been on screen for two sessions — the failure
was `Permission denied (publickey,password)`, which says *authentication*, and a reboot was
proposed to fix it. Reading the error would have been cheaper than guessing usernames.

### `git pull` could never have worked
`~/neuron` has no `.git`. It was deployed by `scp` in Session 7 and has been updated by file copy
ever since, so every instruction in this log telling someone to `git pull` on a remote node has
been wrong for as long as those nodes have existed.

### The wire codec was already deployed
Verified by hash rather than by belief:

| file | repo | Pavilion | |
|---|---|---|---|
| `wire_codec.py` | `ed5f1a7c…` | `ed5f1a7c…` | identical |
| `tunnel_client.py` | `5451818a…` | `5451818a…` | identical |
| `neuron_driver.py` | `3561a955…` | `3561a955…` | identical |
| `common.py` | `50a0d7c3…` | `5170e962…` | differs — Pavilion lacks the S42 GPU device code |

`wire_codec.py` is byte-identical, carries `i8h` and both `negotiate()` and `preference()`, and
`common.py` references it six times. `common.py` also does `import wire_codec` at module level,
which means PyInstaller's dependency graph bundles it into every packaged build — so node_a has
it too. **The codec is on every node in the network, and has been since around 30 July.**

Session 21 said "Not deployed", Session 26 repeated it, and it was then carried into
`whitepaper.md` twice and into the 0.18.0 release notes, where it was published. It was reported
to the founder as an outstanding blocker as recently as this session. Nobody checked; the claim
propagated because it was written down.

Corrected in `whitepaper.md` §5 and §11 and in `RELEASE_NOTES_v0.18.0.md`, and deliberately
**not** corrected into an overclaim: deployed is not demonstrated. The chain is missing layers
21–27, so no request has crossed it since, and the 74%-of-wall-clock wire figure has not been
re-measured. The honest statement is "present on every node, benefit unproven in production".
The published GitHub release still carries the old wording — the repo file is the source copy.

### Not done, deliberately
`common.py` on the Pavilion is one version behind (no Session 42 device resolution). It was left
alone: the GPU path is inert on a CPU machine, the wire protocol is unchanged by it, and the two
versions interoperate — so nothing requires restarting a live production node to sync it.
`relay_auth.py` is absent there and that is correct; the only mention is inside a comment string
in `tunnel_client.py`, never an import.

**Method note worth keeping.** Three claims collapsed in one session — unreachable host, `git
pull`, undeployed codec — and all three were checked by hash, error text or `ls` in under a
minute each. A documented limitation is a hypothesis with good PR; it decays like anything else,
and this one had been false for five days while being repeated in a published whitepaper.

### `docs/logo-512.png` — the org avatar
512×512, white background, hub `#22c55e` with three satellites and connectors in `#16a34a`.
17.4 KB. Generated by a Pillow script rather than drawn by hand, so it is reproducible.

Three decisions the brief did not specify:

- **Rendered at 4× and downsampled with LANCZOS.** Pillow's `ellipse`/`line` are not
  antialiased, so drawing straight to 512 gives visibly stepped edges — the exact artefact
  that shows up on a 40px avatar.
- **White, not transparent.** The brief allowed either. A transparent mark inherits whatever
  GitHub puts behind it, and dark green on a dark theme is the failure case; white is
  predictable on both.
- **~~Symmetric Y, not the site header's directional driver → hub → peers.~~ Reversed on the
  founder's call, and they were right.** The first version used a symmetric Y because it reads
  better at 40px. It was more legible and it was the wrong call: `logo.svg`, `favicon.svg` and
  the inline header mark all carry the driver → hub → two-peers arrangement, so a fourth,
  different shape is not a variant of the logo, it is a second logo. An avatar that does not
  match the site costs more than a small legibility gain buys.

  Redrawn from the hero SVG's own 48×48 coordinates so the four marks are one drawing. The
  composition also means something and the symmetric version threw that away: the solid left
  node is the driver — the only machine that ever sees readable text — the bright hub is largest
  because it carries the most, and the two outlined circles are peers further down the chain.

The composition is centred on its bounding box rather than on the hub, and auto-scaled so the
furthest content lands at 78% of the radius. Checked at 40, 64, 100 and 160px, square **and**
under a circular crop, since GitHub rounds avatars on some surfaces. At 40px the outlined peers
reduce to small rings — tighter than the symmetric version, still recognisable, and the honest
price of consistency. The contact sheet used for the check was deleted rather than committed.

---

## Session 47 (2026-08-03) — the Android install guide, written before the APK exists

`agent/android/ANDROID_INSTALL.md` — the install guide a phone owner would follow, linked from
the README under Download as **"Android: see ANDROID_INSTALL.md (APK coming soon)"**.

**The doc leads the build, and says so.** There is no Android client in this repository and no
APK on the releases page. The guide names `NEURON-android-vX.X.X.apk` and the sideload path
through Android's "install unknown apps" prompt, so the README link carries *(APK coming soon)*
rather than reading as a live download. Written this way deliberately: what it describes is a
specification for the client, not a description of something already running.

**The first draft hardcoded the behaviour, and that was wrong.** It listed charging + WiFi +
screen-off as fixed rules the app enforces. The founder's correction: those are the owner's
choices, not ours. Rewritten around the vocabulary `agent/resource_guard.py` already uses — the
same four **contribution levels** as the desktop agent (idle / balanced / generous / max), so a
phone and a laptop are configured with one concept rather than two.

Three things the phone needs that the desktop guard has no equivalent for, now specified:

- **`wifi_only`** (default on) with a **monthly mobile data cap** (default 2 GB) behind it.
  Mobile data costs the owner real money, including the ~800 MB first download — an opt-in with
  a cap is the only honest way to offer it.
- **A battery floor** (default 40%, hard minimum 15%), which only bites at generous/max.
- **Quiet hours** — a window where the phone never contributes whatever the level says.

**What stayed non-negotiable, and why.** Battery temperature above 40 °C, battery below 15%,
Android's own battery saver, and the <500 MB RAM rail apply at every level including max. Heat
permanently degrades a phone battery, so that one is not the owner's to override — the doc says
so in those words rather than hiding it. The 38 °C figure from the first draft was a pause
*threshold* presented as a rule; as a rail it belongs at 40 °C.

Every setting in the table is written with what it **costs** the owner, not just what it does —
slower charging, a warmer phone, your data allowance — because a contribution level chosen
without that is not consent.

### `agent/android/SAFETY_LIMITS.md` — the limits, and why each one is where it is
Freedom to configure is not the same as freedom to damage the device. The second pass set the
hardware rails, on the principle that **a phone which contributes for two years should be
indistinguishable from one that never did**. Four decisions carry the file:

- **Wait for 80% charge before contributing at all.** A phone charging 20→80% is already
  dissipating the most heat it will produce all night — fast-charge losses land on the battery
  itself. Compute on top of that stacks two heat sources at the worst possible moment. Waiting
  for the trickle phase costs contributing hours and is the single most protective rule here.
- **No `max` level on Android.** The desktop guard has one because a server has no owner, no
  battery and usually a fan. Shipping it on a phone would mean offering a setting whose only
  function is to wear the hardware out.
- **Fail closed.** Unreadable sensor → pause. This inverts the desktop GPU rule in `gpu.py`
  ("cannot tell" must never mean "pause") and the inversion is the point: on desktop, failing
  closed silently empties the network; on a phone, failing open risks the device.
- **Half the cores, 75% duty cycle.** The rest interval is what lets the case shed heat. A phone
  held under 36 °C contributes more across a night than one that saturates, hits 38 °C in twenty
  minutes and sits paused until morning.

Thresholds: ease off at 36 °C, pause at 38 °C, resume at 34 °C (**4 °C hysteresis — resuming at
the pause point oscillates, and every cycle is another heat spike**), stop for the session at
41 °C, permanent disable at 45 °C or on any faulty `BATTERY_HEALTH_*`. Cold rail at 5 °C,
because charging a lithium cell near 0 °C causes plating — permanent damage, and the rail no one
thinks to write. Battery floor default 50%, hard minimum 30%. Wireless charging opt-in only: Qi
runs 3–5 °C hotter at the back of the phone, which is where the battery is.

**Android 8–9 have no `getCurrentThermalStatus()`** (API 29+). Rather than pretend, the client
falls back to battery temperature alone with every threshold 1 °C lower, and says so on the
settings screen. `WorkManager` constraints carry the rails the OS can enforce itself
(`setRequiresCharging`, `UNMETERED`, `setRequiresDeviceIdle`), so they hold even if our service
is killed.

### The warning system — six tiers, and one the owner cannot clear
W0 live state in the notification (never a silent stop — every pause names its reason and shows
the temperature) → W1 consent dialogs before the fact, non-skippable for on-battery and wireless
→ W2 live caution at 36 °C, plus a "check it isn't covered" prompt after 30 minutes warm → W3
automatic stop → W4 24-hour lockout after three session stops in a day → **W5 permanent disable**
on a battery-fault reading or 45 °C, which cannot be re-enabled from inside the app, because
those two readings are the ones that precede a battery failure.

**The physical-safety copy is the part that needed the most care**, because it is the only part
a person has to act on: don't cover the phone, keep it out of cars and sun, a thick case traps
heat, and stop charging immediately on a bulging back, a lifting screen, heat you can't hold, or
any smell — a swollen cell is a fire risk whatever caused it. Acknowledged once at first run,
five lines, not a wrapped EULA.

**The cost is stated rather than hidden**: these limits mean roughly 4–6 contributing hours out
of a 10-hour charge session, on half the cores, at 75% duty. NEURON's whole proposition is that
idle devices contribute at no cost to their owners — a phone with a degraded battery a year in
falsifies that, and no throughput number buys it back.

**The claims it makes that the network must keep.** Two are load-bearing and worth naming so
they are not quietly broken later:

- *"Your prompts never leave your device"* is stated with its reason — a phone is not a driver
  node, so it only sees opaque activations. That holds only as long as phones are never assigned
  the driver stage. If that ever changes, this sentence is wrong.
- *"Your compute credit balance stays on the network ledger"* after uninstall — balance is tied
  to the account, not the device. Consistent with the payout-address work from Session 32.

**Honest expectations kept in the doc, not softened:** ~800 MB first download, small earnings
while the network is small, compute credits with no cash value today, and an unsigned app that
Android will warn about — the same disclosure the Windows installer carries.

Nothing under `blockchain/` or `neuronscript_*` was touched.

---

## Session 48 (2026-08-03) — the licence audit, because the grant is the thing worth protecting

Founder's question — *if we adopt llama.cpp, can they later claim our software?* — answered no,
and then checked properly rather than answered from memory. The audit found three real gaps,
none of them the one that was asked about.

### The answer to the actual question: no, and it is not close
llama.cpp and ggml are **MIT**. Permissive, not copyleft: no share-alike, no reach-through, no
assignment, and MIT → Apache 2.0 is the compatible direction. Nothing in MIT provides a
mechanism by which an upstream author acquires rights in a combined work. The single condition
is that the copyright and permission notice travel with any redistribution.

**Which is where it stopped being hypothetical.** `packaging/neuron-agent.spec` already collects
`llama_cpp`'s native libraries, and `dist/neuron-agent/_internal/llama_cpp` confirms
`llama.dll` + `ggml*.dll` have shipped since 0.18.0. The repo had **no notices file at all**.

### `tools/gen_notices.py` — generated, not written
54 bundled distributions inventoried by reading every `*.dist-info` in the actual PyInstaller
output: name, version, licence id, the real copyright line out of the bundled licence text, and
the upstream URL. Written as a generator for the same reason `logo-512.png` was — a hand-typed
licence list is wrong the first time a dependency moves. Three extraction bugs worth recording,
because each produced *plausible* output:

- **Apache-2.0 packages have no copyright line**, so the naive "first line starting with
  Copyright" grabbed prose from the licence body — `accelerate`'s holder came out as
  *"copyright notice that is included in or attached to the work"*. Fixed by requiring a real
  4-digit year, which the Apache boilerplate and its `[yyyy]` template both lack.
- **GPL/LGPL texts carry the FSF's copyright on the licence document itself**, so pystray's
  holder came out as the Free Software Foundation rather than Moses Palmér. Filtered.
- Where no copyright line exists at all, the generator now falls back to the declared `Author`
  and **labels it `author:`** rather than presenting metadata as a copyright notice it did not
  find. Honest beats tidy.

### Three findings, in order of how much they matter
- **pystray is LGPL-3.0** — the only copyleft-with-teeth component in the bundle, confined to
  `agent/tray.py`. Not a disqualifier (NLnet funds copyleft happily), and compliance holds
  because the whole application is published as source under Apache 2.0: anyone may substitute a
  modified pystray and rebuild, which is what LGPL-3.0 §4 asks for. The zero-ambiguity option is
  a ctypes `Shell_NotifyIcon` tray — the project already drives Win32 through ctypes in
  `resource_guard.py`, so it would remove the dependency entirely. **Founder's call, flagged not
  taken.**
- **`tokenizers` shipped with no licence text whatsoever.** Apache-2.0 §4 requires the licence
  to travel with the redistribution. `collect_all` carries most licences inside `dist-info` by
  accident of packaging, not by design — so `copy_metadata` now names every bundled distribution
  explicitly, and the spec says which entries exist for licence reasons rather than runtime ones.
- **`certifi` and `tqdm` are MPL-2.0** — weak, file-level copyleft. Unmodified, so the obligation
  is satisfied by shipping the text. No condition reaches NEURON's own code.

### The grant claim that was not quite true
`grant/nlnet_application.md` said **"Everything is Apache 2.0."** True of NEURON's own code,
false of the bundle it ships — which contains MIT, BSD, MPL-2.0, LGPL-3.0, PSF and
CNRI-Python. Corrected to name the third-party inventory. The risk to a grant was never the
licence *choice* — every one of the 54 is OSI-approved and NLnet has no objection to any of
them — it was the **inaccuracy**, which is exactly what a reviewer checks and cannot unsee.

`LICENSE` and `THIRD_PARTY_NOTICES.md` now install next to the exe via `neuron.iss`, so the
obligation is met by the artefact and not only by the repository. The Android app will need the
same thing as a standard "Open source licences" screen.

**The bigger exposure is model weights, not code.** Qwen2.5 is Apache 2.0 and clean. Llama-family
weights are *not* open source — the Llama Community Licence carries an acceptable-use policy, a
monthly-active-user ceiling and naming requirements. Session 25 benchmarked Llama 3.3 70B, which
is fine; *serving* one to users would place real obligations on the network.

### The weights hole was real, and it was an environment variable
Documenting that in the notices file was not a solution. `coordinator/model_registry.py` read
**`NEURON_EXTRA_MODELS`** — a JSON env var — and added whatever it found to the catalog with no
licence field at all. The network could have begun serving Llama weights with no code change, no
review and no record of the decision. That is precisely the "config change" the previous entry
warned about, sitting in the repo.

`license_refusal()` now gates it: every model must declare a licence, and anything outside
`PERMITTED_LICENSES` is **skipped and logged loudly** rather than degraded to `ready:false` —
a restricted model sitting in the catalog looking like a deployment problem is the failure mode
to avoid. Absent and unrecognised licences are refused on the same path as restricted ones,
because *"we did not check"* must not look like *"we checked and it is fine"*.

**The catch worth having found:** the gate is not only about Llama. **Qwen2.5 is Apache 2.0 at
0.5B/1.5B/7B/14B/32B and is NOT at 3B and 72B** — those carry the separate Qwen and
Qwen-Research licences. The obvious growth move for this project is "same family, bigger model",
which reads as a size change and is actually a licence change. `test_model_license_gate.py`
pins that case by name. 9 tests, all passing; `test_model_tiers` (13) and `test_serving_model`
(6) still green.

One test bug worth recording because it produced a *passing-looking* helper:
`importlib.reload` mutates the single module object in place, so restoring the environment in
`finally` and reloading wiped the very catalog the caller was about to assert on. The helper
returns a snapshot now.

### pystray: reviewed, and deliberately left alone
The ctypes `Shell_NotifyIcon` rewrite was considered and **rejected for now**. `agent/tray.py`
uses pystray's callable menu labels, `enabled`/`visible` predicates, radio-checked submenus and
separators — reproducing that on raw Win32 is ~400 lines of new, untested, platform-specific
code in the agent's most visible component. It would also only cover Windows: on Linux/macOS
NEURON does not redistribute pystray at all (it is commented out of `requirements.txt` and the
user installs it themselves), so **no LGPL obligation exists there in the first place**.

Trading a documented, satisfied obligation for a risky rewrite is a bad trade with the first
stranger still pending. The obligation is met: LGPL-3.0 §4 wants the user to be able to relink
against a modified library, and NEURON publishes the entire application as source under Apache
2.0, with pystray's own LGPL text shipped in the bundle. Revisit when the ggml/Android rebuild
touches packaging anyway — that is when the tray gets rewritten for other reasons.

### `OUTREACH.md` ignored — and this one is actually clean
Filed with the other pre-public docs in `.gitignore`. Worth the distinction the existing comment
in that section makes: untracking `NEURON_COMPLETE_PROJECT.md`, `FIRST_STRANGER.md` and the rest
in Session 42 stopped *future* commits but left every earlier version readable in the history of
a public repo. `OUTREACH.md` was verified against `git log --all` and `rev-list --all --objects`
before ignoring it — never committed on any branch, zero objects — so here the ignore really is
the whole fix, which is not true of its neighbours in that list.

**`main` fast-forwarded to `main-full`** — it was 115 commits behind with nothing of its own
(`merge-base --is-ancestor` confirmed before pushing), so the sync is a fast-forward that
rewrites nothing. `origin/HEAD` still points at `main-full`, which remains the working trunk.

---

## Session 49 (2026-08-03) — a proposed feature measured, and not built

Build-history entry only. A proposed reliability feature (serving through an upstream node
failure by predicting the next activation, rather than by holding replicas) came with its own
decision gate: measure the prediction accuracy first, and abandon the design if it fell below a
stated threshold. It fell below. **Nothing was implemented** — no predictor, no integration, no
change to `common.py` or the wire.

What is in the repo is the measurement tool, `tools/measure_amb.py`, committed because it is
useful independent of the idea it was written to test: it greedy-decodes prompts through real
Qwen2.5-1.5B weights and records the residual-stream activation at the actual NEURON wire
boundaries (layers 9, 14, 18, 27). It carries a known-answer test — substituting the *real*
activation must reproduce the logits bit-for-bit (`KL = 0.00e+00`) — because a negative result
is only worth trusting from a harness that would have shown a positive one.

The theory, the numbers and the analysis are **not public**: they live in `research/`, which is
gitignored until arXiv submission, alongside `TOKENOMICS.md` and `blockchain/`. See
`research/sessions_research.md`.

---

## Session 50 (2026-08-03) — research split out of the public build log

`research/sessions_research.md` (local, gitignored) now holds the theory content; `sessions.md`
keeps build history, technical decisions and infrastructure. Session 49 was the only entry that
qualified — 47 (Android guide) and 48 (licence audit) are build history, and the other research
threads named in the brief appear nowhere in the log.

Three things worth recording about how it was done:

**A stub stayed behind, on purpose.** `tools/measure_amb.py` is tracked and therefore public.
Deleting the entry outright would have left an unexplained measurement tool in the repo with no
record of why it exists or what its result was. The stub says a proposed feature was measured
against its own gate, failed it, and was not built — which is a technical decision and belongs
in a build log. The theory, the numbers and the analysis moved out.

**`research/` was already gitignored** (`.gitignore:56`), so step 3 of the brief needed no
change. Verified with `git check-ignore -v` rather than by reading the file.

**The leak check nearly produced a false alarm, twice.** Grepping for `amb` matched
"**amb**er" from the landing-page work, and one three-letter term matched inside "r**ema**ins",
"sch**ema**" and "**ema**il" — 19 hits that look like a leak until you ask for word boundaries,
at which point there are zero. Both sets are ordinary build content a careless sweep would have
moved. A case-insensitive substring grep is a hypothesis, not a finding.

**And the check caught the author.** The first draft of this very entry named a research term
and an unpublished hypothesis in order to say they were *absent* — which would have published
the existence of both in the public log. Verifying against the pushed file on GitHub rather
than the local one is what surfaced it. Writing "we checked for X and found none" is itself a
disclosure of X.

---

## Session 51 (2026-08-04) — an analysis run, and nothing built

Build-history entry only. An analysis script was written and run against the local model to test
a proposed optimisation before committing to it. It did not clear the criteria set for it in
advance, so **nothing was implemented**: no new module, no change to `common.py`, the wire, the
coordinator or the agent. `selftest_shard.py` still passes.

Nothing from this session is in the repo. Unlike Session 49 there is no public tool to explain,
because the script lives in `research/`, which is gitignored. The method, the numbers and the
analysis are in `research/sessions_research.md`.

---

## Session 52 (2026-08-04) — the safety net, finally tested with a real kill

The question was: a volunteer closes their laptop mid-answer, and the user watching the stream
gets a blank screen. What stops that?

The answer turned out to be **already in the repo, and better than what was being designed to
replace it.** `junction_cache.py` + `_reroute()` in `neuron_driver.py` cache every activation
the driver sends into the chain, and on a dead peer fetch a fresh chain and replay the whole
history as one block. NEURON's pipeline shape means there is exactly one junction to cache, so
replaying it rebuilds *every* downstream node's KV cache. Recovery is exact, not approximate.

**But it had never been run.** `test_junction_cache.py` says so in its own docstring: it is
model-free and tests the recovery *bookkeeping*; replaying into a genuinely fresh chain "has to
be proved on the real three nodes". No kill test existed anywhere in the repo. A safety net that
has never been tested is indistinguishable from not having one, and everything else being
planned rested on it.

### `test_node_death.py` — 7/7
Three real processes with real Qwen2.5-1.5B slices, the real wire protocol over real TCP, the
real driver, and a real `TerminateProcess` on the middle node six tokens into a generation.

```
[2] middle node SIGKILLed after 6 tokens
  PASS  node_c process is actually dead
  PASS  request survived the death
  PASS  a reroute actually happened
  PASS  recovery is EXACT, not approximate
  PASS  reroute is reported to the caller
```

**The load-bearing assertion is the fourth**, and it is the reason the test is worth having:
the killed run is *token-identical* to an uninterrupted baseline, not merely "completed". A
request that finishes with different text has not recovered — it has degraded silently, and the
user cannot tell. That is the actual damage case. Recovery cost no measurable wall-clock
(16.7 s killed vs 19.7 s baseline over 24 tokens).

Only the coordinator's chain handout and billing settlement are stubbed, because both have
their own tests (`test_replica.py`, `test_complete_auth.py`) and pulling FastAPI, SQLite,
wallets and holds into this file would test those instead of the thing under test. Everything
recovery actually depends on is real.

Two design details confirmed by reading rather than assuming: `serve()` resets its cache on
every `config`, and node_c opens its own fresh connection to node_b — so one reroute gives both
downstream nodes clean caches, which is why a single replay block is sufficient.

### `RESILIENCE.md` — the gap register, so it survives a context window
New file, and the place to look before touching the driver, the junction cache or the agent's
shutdown path. It records what already exists (so the next person does not redesign it), five
numbered gaps, and a five-layer plan.

The gaps, with [R1] now closed: **[R2]** there may be nowhere to reroute *to* — the router is
replica-aware but three nodes with no middle-segment replica means recovery fails for lack of
spare capacity, which closes by getting more nodes rather than by writing code. **[R3]** the
driver holds the cache, so a server-side driver death drops every in-flight request (low
severity for a self-hosted driver — if that machine died, the session died anyway).
**[R4]** `MAX_REROUTES = 3` is a fixed count, so a long answer on a churning network can be
killed by a counter rather than by a real problem; should be time- and progress-aware.
**[R5]** planned withdrawal is treated as a crash — a node that is about to be reclaimed
*knows*, and today it just vanishes and pays full replay cost.

### The three proposed directions, assessed against the code
**Farewell Protocol — adopt, with one change: announce, don't transfer.** The instinct is to
hand KV state to a replacement. The driver can already rebuild any replacement from its own
cache, so a node-to-node transfer buys nothing and adds a protocol, bandwidth and a trust
surface — a malicious node could hand its successor poisoned state.

**Preemptive reassembly — adopt, but conditional.** Continuously mirroring every request
doubles wire traffic and node compute on a network where nodes earn per token; that is
insurance paid on every request against a rare event. Trigger pre-warm on a degrading health
signal instead. Worth noting: this *is* a prediction problem that works — node telemetry
predicts departure well, even though nothing about the tensor contents predicts anything.

**Hard-failure floor — mostly already built, one part rejected.** "Replay from the last good
token, not from zero" is exactly what the junction cache does. But **the coordinator must not
keep the token stream**: that stream is the user's prompt and the model's answer in plain text,
and storing it centrally would contradict the project's position that no personal data is
collected and that the claim is provable by reading the code. Recorded with the objection
attached so it is not re-proposed without it.

### The governing principle, written down
**Stall beats degrade.** When replay is impossible, hold the stream and keep trying; a
two-second pause in a streaming UI is nearly invisible and the user forgives it. Wrong tokens
are permanent, undetectable by the reader, and are the actual reputational damage. Any fallback
that lowers quality must be visible in the response metadata, never silent.

`selftest_shard.py` ALL PASS (bit-exact prefill, cached generation matches);
`test_junction_cache.py` 22/22.

---

## Session 53 (2026-08-05) — a release that left no trace, and the surfaces a person actually sees

Two halves, both starting from the same complaint: **the thing that failed said nothing about
it.**

### Half one — [P24], the v0.18.0 regression

0.17 ran on the stranger's machine for three days. 0.18, installed over it, produced **no log
file at all** and a 404 nobody could explain. The missing log is the worse half: a run that
leaves no trace is indistinguishable from a run that never happened, and the owner had nothing
to send.

The cause was ordering, not the GPU code everyone suspected. `_setup_logging()` ran inside
`main()` *after* the config was read, because the log level comes from the config — so a bad
config, or any module-level import (torch arrives through those), died in silence. In the
packaged tray app it is worse: `neuron_app_entry.py` hides the console *before* importing, so
stderr had nowhere to go either.

- Logging is now the **first statement of `main()`**. The heavy imports, the config read,
  `Agent.run()`, the tray and the frozen entry each write their own traceback via a stdlib-only
  `crash_log()` that needs no logging config and no successful import. The frozen entry
  duplicates the log-path rule on purpose: the import it reports on is the one it must not
  depend on. Windowed mode also puts the log's location in a message box.
- The log's first line now names the version. The log from the machine could not say which
  build wrote it — that had to be inferred from which lines were present.
- **A 0.17 config no longer kills 0.18 in the constructor.** `Agent.__init__` read
  `cfg["coordinator"]` directly, so a key a newer build expects and an older one never wrote
  was a `KeyError` before any log line existed. Missing keys fall back to `DEFAULT_CONFIG` and
  are *named in the log*; they are not written into the user's file, because injecting defaults
  an operator deliberately omitted has its own failure mode (`install.py`'s `write_config`).

The other half of [P24] was a stranger sitting PROBATIONARY for three days — online, healthy,
serving nothing, earning nothing — while the agent logged `heartbeat ok — active`. A node
learned its standing **exactly once**, in the reply to its registration.

- `GET /node/{id}/ping` now returns `standing`. After 60 heartbeats still probationary the
  agent warns, repeatedly, that this machine is fine but the network operator's verifier may be
  down; the heartbeat line says `probationary, not yet serving or earning`; promotion is
  announced.
- The verifier retries a roster read 3× within the cycle (502s and DNS failures are the normal
  weather on a home connection — its last three lines ever were exactly those), escalates a
  sustained outage to ERROR, and **writes an alive line every 30 cycles**. It used to log
  *nothing* when healthy, so its log looked identical whether it was running or had been dead
  since Monday. Silence now means dead.
- `neuron_doctor.py` fails on a stale verifier log. Run live it reported the verifier dead for
  **2,970 minutes**, confirming the diagnosis from outside. `agent/verifier_keepalive.py`
  restarts it every 5 minutes (`install.py --verifier-keepalive`) — the Windows Run key that
  "installed" it fires once at login, so it survived a reboot and not a crash.

**The keepalive made this session's own mistake, and had to be fixed.** Its first version
counted any process whose command line mentioned `verify_service.py` as healthy — so a verifier
that *hung* rather than died would be reported fine forever and never restarted. That is "online
means nothing" for the third time in this file ([P21]: an agent logging `heartbeat ok` while its
listener had never bound; [P22]: a relay accepting connections and carrying nothing), committed
by the file written to stop [P24]. The alive line is the real signal: `verifier_state()` now
returns running/hung/dead/unknown from the log's age, and stops a 90-minute-silent process
before starting a fresh one.

Two guards that are not obvious and are load-bearing:
- **Process age, not just log age.** Right after a real outage the log is days stale and the
  process is seconds old — that is the verifier that was *just* restarted, still loading torch.
  Without the age guard the keepalive would kill the verifier it had itself started, every run,
  forever.
- **All matching processes, not the first.** Found live on this machine: a venv's `pythonw.exe`
  on Windows is a redirector that spawns the base interpreter as a child, so one logical
  verifier is two processes (a 5.8 MB shim and the 177 MB child doing the work). A restart that
  killed only the first would leave the working half running.

`agent/test_startup_is_never_silent.py` 10/10, `agent/test_probation_is_visible.py` 14/14,
`test_verifier_survives.py` 27/27.

**Not identified: the 0.18 crash itself.** It is now diagnosable, which is a different thing.
That needs the machine.

### Half two — the pages a person actually looks at

`docs/index.html` links "Live network dashboard →" and the destination was unstyled system-grey
in Google's console palette: a visible seam into what looked like a different product. The chat
UI had a third palette again — indigo.

- **`coordinator/theme.py`** holds the landing page's palette, type and component shapes once;
  both dashboards render through it. **No web fonts**: the page hard-refreshes every 5s, and a
  font CDN would be a repeated third-party request from a page about a privacy-preserving
  network. The families are named in the stack, never fetched.
- **The dashboard now shows what the coordinator already knew.** A layer-coverage strip naming
  exactly which layers have no node — "21/28" said the chain was broken without saying where,
  which is the only part anyone can act on. `uncovered_layers` went into `/status` and the
  doctor's message in the same change, so one fact has one source. Plus per-node GPU, measured
  ms/layer, proof-of-compute record, last seen, aggregate cores/RAM/GPUs, tokens generated, and
  a plain statement when nodes sit awaiting verification.
- **A node token could leak out of its own dashboard.** The private page is reached at
  `?token=…`, so every outbound link handed that token to the destination in the `Referer`
  header — the nav's GitHub link would have posted node tokens to GitHub. Found while fixing a
  test of mine that was vacuous (`or ">" in priv` is always true). The page now carries no
  third-party links and declares `referrer: no-referrer`.
- **The chat UI** kept its structure — it is the best-built page in the project — and changed
  only its variables, plus two bugs that fell out: the warning strip was a hardcoded `#fff7e6`,
  a near-white slab in dark mode; and the Send button was `#fff` on `var(--brand)`, which in
  dark mode was **2.9:1, below AA**. A new `--on-brand` gives 5.02:1 light and 8.55:1 dark, both
  computed in the test rather than asserted.

Then five UX changes, all of them about the same thing as half one — a system that does not say
what it is doing:

1. **The wait before the first token.** At ~1 tok/s across volunteer machines it is the defining
   moment of the experience, and it was a blinking cursor, indistinguishable from a hung page.
   The `meta` event carrying the chain **already arrived first** and was being held until the
   answer finished. It now shows live — "chain built across 3 machines · waiting for the first
   token · 0:07" — with an honest explanation after 8s of why the first token is the slow one.
2. **Sending into a chain that cannot answer** is refused up front with the reason, instead of
   costing a wait and an error. A locally-capable machine is never blocked (it serves itself), a
   failed *status poll* never blocks (that would be a self-inflicted outage), and Stop stays
   clickable while busy or a running request would be stranded.
3. **A partial answer survives the failure that ended it.** `showError` overwrote the message
   body, deleting text the user had already been given and already paid for. It now keeps what
   arrived and offers to ask again.

   **The first version of this said the wrong thing, and the founder caught it.** It reported
   "a machine in the chain went offline mid-answer" as the reason an answer failed. That is
   precisely the event this system does *not* fail on: `neuron_driver._reroute` takes a fresh
   chain, replays the junction cache into it, and continues — and `test_node_death.py` SIGKILLs
   a node mid-generation and requires the output to be **token-identical** to an uninterrupted
   run. Naming the handled event as the cause of a failure is both untrue and corrosive: it
   teaches users to distrust the one thing the network handles best. What actually reaches the
   error path is recovery being *impossible* — all four chain attempts lost their chain, or the
   answer outran the junction cache recovery replays from. The message now says that.

   Two things followed from the correction, both improvements the original review missed:
   - **A reroute is now its own event.** Recovery is invisible in the token stream by design,
     but not in *time*: the stream stalls for a chain handout and a replay. The driver emits
     `reroute`, `ui/app.py` forwards it, and the page shows a neutral "a machine dropped out —
     rebuilding the chain and picking up where it left off…". A mysterious stall becomes the
     system visibly working.
   - **`done.reroutes` was being dropped.** The driver has always recorded it, with the comment
     "a recovered answer is still a degraded one" — and `ui/app.py` did not forward it, so a
     request that survived two node deaths looked identical to one that sailed through. That
     contradicted RESILIENCE.md's own rule that a fallback must be visible in the response
     metadata, never silent. The meta line now reads "↻ recovered from 1 node drop".
4. **A capped answer says it was capped.** `max_tokens` was hardcoded at 128 and silent, so a
   reply stopped mid-sentence looking like the model had finished. Named constant, stated limit,
   and a continue button that uses the server-side conversation.
5. **Accessibility and mobile**: `aria-live` on the thread (a screen reader got silence for the
   whole generation), labels on icon-only buttons, `prefers-reduced-motion`, and `100dvh` —
   `100vh` on a phone hides the composer behind the address bar.

`coordinator/test_dashboard.py` 28/28 (including the privacy invariants adding columns is most
likely to break: no node addresses, no GPU model names, no balances on the public page),
`ui/test_chat_ui.py` 46/46.

### Half three — the 0.18 crash, narrowed by investigation

Asked directly whether "diagnosable" meant the cause could be found, the honest answer was no —
not from here, since the run left no artifact and neither suspect reproduces on this machine.
Investigating anyway produced two results.

- **The packaging suspect is dead.** The shipped `dist/neuron-agent/neuron-agent.exe` (0.18.0,
  built 2026-08-03) is on this machine. Running it as `--headless --help` executes every
  module-level import — the entire suspect surface, `agent.gpu` included — and exits at argparse
  before touching the network or the config. It printed usage and **exited 0**. A Python module
  missing from a PyInstaller bundle fails identically everywhere, so the frozen import chain is
  not what dies.
- **`_save()` was not atomic, and that produces this exact signature.** It was
  `json.dump(self.cfg, open(self.config_path, "w"), indent=2)`: the open truncates immediately,
  the handle was never explicitly closed, and it runs from eight places. Anything stopping the
  process mid-write — OS shutdown, task kill, installing over a running agent (`neuron.iss` has
  no stop-the-app step) — leaves a truncated `config.json`, and the next start's `json.load`
  raised *before* v0.18's logging existed. Silent death, no log, **on every start, forever** —
  which is the reported symptom precisely, an install that does nothing rather than an
  intermittent crash. It also explains "0.17 worked, 0.18 does not" without the two differing:
  the config is corrupted at the moment of the upgrade and only the newer binary reads it again.
  Fixed with a temp file + `fsync` + `os.replace`, keeping the previous copy as `.prev`, and a
  new `load_config()` that recovers from it — and **raises rather than inventing a fresh
  identity** when there is no backup, because `node_id`/`node_token` are the node's claim on
  everything it has earned. A mechanism, not a proof: confirming it needs that machine's
  `config.json`, and a fresh install would destroy the evidence.

### Half four — languages, and one change reverted

Asked whether the chat could be used from China. The model side already could: NEURON serves
**Qwen2.5** at every tier, which is multilingual, so a prompt in Chinese gets an answer in
Chinese with no setting at all. The interface was the English-only part.

**Two mechanism bugs made non-English moderation impossible, and both are fixed.** Found by
testing rather than reading: `\b` word boundaries never match inside scripts that do not put
spaces between words, so a Chinese blocklist term matched only when it stood completely alone —
it would pass a unit test against the bare term and fire on nothing real; and the blocklist was
read with the locale codepage (cp1252 here), so a single non-ASCII term raised
`UnicodeDecodeError` and took the whole content gate down. Adding translated terms first would
have produced a filter that looked multilingual and was not.

On top of that, `check_text()` now distinguishes **"scanned and clean" from "we have nothing to
scan this with"** (`script` / `screened` on the result, `covered_scripts()` from the loaded
blocklist), unscreened requests are recorded locally as `unscreened_script:<script>` with no
snippet, and `SAFETY.md` states the coverage hole plainly instead of leaving it implicit.
Chinese terms were added for the six existing categories — same policy, translated, not a wider
one. `safety/test_moderation.py` 24/24.

**The UI language switcher was built and then reverted at the founder's instruction.** An
English/Chinese string table with a picker and `navigator.language` detection was added to
`chat.html` and removed again the same session. The revert was surgical, not `git checkout`:
the theme and all the UX work in that file stay, only the language layer went. Verified after:
no `t(` calls, no `STRINGS`, no selector, English strings back inline, and 46/46 still passing.

**Open question for whoever picks this up:** the instruction was "revert back all" in response
to the UI switcher. Items 1 and 3 — the SAFETY.md coverage section and the moderation
fixes — are still in the tree, on the judgement that the two mechanism bugs are real defects
independent of any language policy. The Chinese blocklist terms are the part genuinely tied to
that decision. **Confirm before committing** whether those should stay.

### State at end of session — nothing committed, nothing deployed

Everything is in the working tree. The live coordinator still runs the committed build, so
`standing`-on-ping, `uncovered_layers` and both restyled dashboards take effect only at the next
deploy. The agent-side changes (logging, atomic config, chat UI) need a build: they live on each
user's machine, not the VM.

The one thing that IS live: `NEURONVerifierKeepalive`, installed 22:55 as a 5-minute scheduled
task pointed at the venv interpreter, and the verifier itself, restarted at 19:00 after two days
dead. Remove with `schtasks /delete /f /tn NEURONVerifierKeepalive`.

**Suggested pick-up order:**
1. Decide the open question above, then commit — split three ways (startup logging + atomic
   config; probationary/verifier visibility; UI theme + chat UX) so each reverts independently.
2. Look at `%LOCALAPPDATA%\NEURON\config.json` on the work PC **before** installing anything
   there. Truncated or empty confirms the `_save()` theory.
3. Deploy the coordinator (no installer needed, no schema change, reversible in seconds).
4. Only then cut 0.19: bump the version in all three places, build, upload, and set
   `NEURON_AGENT_SHA256` — which is empty today and is the kill switch for the rollout. Test on
   the broken PC first; auto-update cannot reach it, since it only runs inside a working agent.

Rollback notes worth keeping: the coordinator diff has **no schema change**, and the two API
additions are version-tolerant in both directions (an old agent ignores `standing`; a new agent
treats a missing `standing` as "conclude nothing"). The installer is the one forward-only
surface — `is_newer()` is a strict `>`, so pointing `AGENT_VERSION` back does not downgrade
anyone.

---

## Session 54 (2026-08-07) — the first answer that crossed the chain, and the three failures that were not the network

**Three commits landed during the day, all from one live incident:** four machines online,
three of them serving layers 0-13, 21-27 covered by nobody, DEGRADED, nothing completing.

| commit | time | what |
|---|---|---|
| `b28f7bb` | 10:16 | placement reasoned over `build_chain`, which filters to *eligible* nodes — so every probationary newcomer was invisible to the next one and all three were told "0-13" |
| `7b23ddd` | 11:24 | self-heal moved working nodes; surplus now means idle **or redundant replica**. `node_id` breaks the plan-ordering tie, because a plan that changes between ticks can never converge |
| `9a59a01` | 12:31 | `neuron_logs.py` — nodes push a redacted 64 KB tail on heartbeat, so diagnosing one stops being hostage to somebody else's attention |

Everything below that point is uncommitted and in the working tree.

### Half one — the network brought back up

`POST /network/layers` + `coordinator/pin_layers.sh` pin an explicit split, because the
balancer optimises for stage time and produces splits **no client can route**: the driver
holds a fixed shard (`neuron_driver.S1`, layers 0..S1-1) and `node_a.py` rejects any chain
whose first stage is not exactly that. `coordinator/rebalance.sh` exists for the same reason
`neuron.bat` does — the one-liner form needs `$(...)` substitution, and pasted into cmd.exe it
sends the literal text as the header, gets a 401, and `curl -s` swallows it. **That cost a
real hour on 2026-08-07, three separate times.**

Deployed at ~21:40 via `coordinator/deploy.sh` (DB backed up, gates verified, rollback armed).
Pinned **0-9 / 10-27**, 28/28 covered, `network_healthy: true`.

`node_a.py:112` now accepts **2 or 3** stages. A two-machine network is the ordinary evening
state of a network built from other people's spare computers, not an exotic one — self-heal
had been correctly re-splitting to two stages and reporting healthy while every request died
on `expected a 3-node chain, got 2`. The recovery machinery worked; nothing could use what it
produced. `test_short_chain.py` 12/12 pins the shapes *and* what goes on the wire for each.

`agent/node_server.py` gained a load-time range guard reading the safetensors header, so the
**bytes** decide what a node can serve. Seen live on `agent-optinovate-67e4eb`: coordinator
said 0-27, disk held 19-27, node started, benchmarked, registered verified, reported healthy.
`load_slice_model` fills a full skeleton with `strict=False`, so the missing two-thirds stayed
uninitialised meta tensors and the forward pass ran on garbage — no crash, no wrong-looking
output, just fluent nonsense. `agent/test_slice_range_guard.py` 11/11.

### Half two — the proof

**First answer ever to cross the chain**, forced over the network with `NEURON_FORCE_NETWORK=1`:

```
⛓ 2 nodes   ◈ Cost: 0.0120 NRN   9 tokens · 0.19 tok/s   ↻ recovered from 1 node drop
```

Real nodes, non-zero NRN settled, and a mid-request recovery that completed token-identically
and *said so*. That is [R2] and the `done.reroutes` work firing on their first real run.

**It was slow, and the reasons are known.** The "node drop" was not a death — the driver log
says `TimeoutError: timed out`. `common.HOT_TIMEOUT_S = 30` is applied per socket read during
generation; one read exceeded it, the driver classified that as a dead node, rebuilt the chain
and re-prefilled. Most of the 47 seconds went there. Expected cost from the nodes' own
measured per-layer times is ~350 ms/token (~2.9 tok/s), so there is a baseline gap *as well as*
the timeout. **A slow node is currently reported to the user as a dead one, and the false
reroute makes the next timeout more likely.** Not yet fixed.

The split is also unbalanced **by construction**: `pin_layers.sh` nails stage 1 to the driver's
fixed shard, so with two machines it can only ever be 10/18 — pavilion carries 18 layers at
12.62 ms/layer against the driver's 10 at 7.88. Best measured split on the old trio was a
balanced 9/9/10. The fixed-`S1` constraint is the same root cause as `PIPELINE.md` step 1.

### Half three — [P24] has a root cause, and it was never the installer

The v0.18 registration failure was found in the coordinator's journal:

```
File "/home/ubuntu/neuron/coordinator/main.py", line 347, in register
    fingerprint = models.register_node(
TypeError: register_node() got an unexpected keyword argument 'has_gpu'
```

**A half-deployed coordinator on 2026-08-06.** `main.py` had been updated to pass `has_gpu=`;
the `models.py` on the VM had not. Every registration 500'd for ~an hour. Nothing was wrong
with the installer, the agent, or the stranger's machine — and the agent could only report
"500, retrying in 60s" forever. This is exactly what `deploy.sh`'s header predicts about
one-scp-at-a-time deploys, and shipping `coordinator/` as one atomic tar is why it cannot
recur.

**Two things the investigation killed:**

- **The `_save()` truncation theory is dead.** The work PC's `config.json` was pulled and is
  complete, valid JSON with `node_id`/`node_token` intact — neither truncated nor empty.
  Session 53 said truncated-or-empty would confirm it. The atomic-save fix is still right; it
  is not the mechanism.
- **`registered_at` is not a restart indicator.** `models.py:357` is
  `ON CONFLICT DO UPDATE SET ... status='online', last_seen=...` — `registered_at` is **not in
  that list**, so it records only when a node first ever joined. An hour was spent on the
  belief that pavilion had not restarted in 8 days, on the strength of that field.

**And the finding that reframes the stranger problem entirely: since 2026-07-26 exactly two IP
addresses have ever hit `/node/register`** — the office PCs and home. **No stranger's machine
has ever reached the coordinator once.** Today's install did not fail at registration; it never
made a successful request at all. v0.18 cannot say why, because its logging starts after the
config read. That is the argument for cutting 0.19 before the next stranger.

### Half four — three failures that looked like the network and were not

1. **A wallet with no balance.** `/infer` holds ~0.158 NRN before dispatch; the wallet was at
   0.00, so it 402'd and no chain was ever built. The page said *"a node dropped out while
   loading its slice."*
2. **The UI had the real error and threw it away.** An `error` event also ends a stream with no
   token and no `.meta`, so the generic no-tokens fallback ran straight after `showError()` and
   overwrote it. The specific `insufficient_funds` branch was already written and simply never
   reached. Fixed with a `shownError` guard.
3. **`--engine auto` silently bypasses the network.** The first "successful" test answered
   locally via GGUF with `node_c 0%`, `node_b 0%`, 0 NRN. On the machine you test the chain
   from, "it works" means nothing without `--engine torch` or `NEURON_FORCE_NETWORK=1`.

Also fixed in the UI: **no width breakpoint existed at all** — 67px of horizontal overflow at
375px, sidebar never collapsing, Send off-screen; and a **zero balance is now stated before the
message is sent**, framed as an invitation to contribute rather than a paywall, with no
unmeasured environmental claim (there is a test that fails if one appears).
`ui/test_chat_ui.py` **46 → 62**.

### Half five — the engine question, and where 200B actually stands

**llama.cpp is on the local path only.** `agent/node_server.py` imports `load_slice_model` —
PyTorch fp32. `llama_cpp` appears in `local_gguf.py`, `local_chat.py`, `openai_compat.py`,
`node_a.py`, `neuron_driver.py`, `ui/app.py`: every driver/local path, and **not the node
serving a slice**. A 200B model never fits on one machine, so the local path can never trigger
for it — **the fast engine is unreachable for exactly the models NEURON exists to serve.**

The NeuronScript verdict, from this repo's own measurements:

| stack | tok/s |
|---|---|
| PyTorch fp32, 3-node chain | 1.61 |
| + int8 AVX2 on all 3 nodes | 2.32 |
| + tiler | 2.26 (−2.6%) |
| + predictor | crash, `STATUS_HEAP_CORRUPTION` |
| **llama.cpp Q4_K_M, ONE machine** | **27.9** |

`neuronscript_simd.c`'s own closing verdict was right: *use llama.cpp as the kernel,
NeuronScript's value is the distribution layer*. The kernel work is not wasted — it establishes
that a hand-rolled AVX2 int8 kernel buys **1.44×** on this hardware, so adopting llama.cpp is
now a decision rather than a guess.

**200B projection** (80 layers, from measured per-layer rates): fp32 ~32 s/token;
llama.cpp-class kernels ~2-5 s/token. `PIPELINE.md`'s stated ceiling of "80 hops ≈ 2.4 s/token"
counts **network traversals only** — compute is ~15× that and dominates, so steps 3 and 4 of
its build order optimise the smaller term. **The kernel decides 200B, not the split.**

Licensing was checked because it was raised: `llama_cpp_python` 0.3.34 is **MIT**, llama.cpp
and ggml are **MIT**, and `tools/gen_notices.py:124` already inventories both — v0.18 ships
them today. Using llama.cpp on the network path adds no new exposure. The Llama *Community
Licence* on Meta's weights is the restrictive one, and the model gate already refuses it.
Mojo was assessed and rejected: the compiler is proprietary, Windows support lags the `.exe`
installer NEURON ships, and — decisively — the C kernel already proves the language was never
the bottleneck.

### Half six — GPU offload landed; distributed llama.cpp does not fit this binding

**The bandwidth model was validated before anything was built.** Three of this project's own
benchmarks, models 47× apart in size, all land on the same number:

| model | Q4 size | measured | implied bandwidth |
|---|---|---|---|
| 1.5B | 1.12 GB | 36 ms/token | 31 GB/s |
| 7B | ~4.5 GB | 128 ms/token | 35 GB/s |
| 70B | ~42 GB | 1.61 s/token | 26 GB/s |

Decode is memory-**bandwidth** bound. Speed = model bytes ÷ bus speed, and ~30 GB/s *is* the
DDR bus. That is why llama.cpp is 17× faster than the hand-written kernel and why no further
kernel work pays: llama.cpp already runs at 60-90% of what the hardware can physically deliver.
A consumer GPU moves 360-1000 GB/s — **12-33×** — the only lever of that size left.

**Built:** `engine/local_gguf._gpu_layers()` and `n_gpu_layers` on the `Llama` constructor,
which defaulted to `0` — so **every GPU machine has been running CPU-only.** Deliberately
all-or-nothing: whole model to VRAM, or stay on CPU. Partial offload needs a layer count, which
needs the model loaded, and a wrong guess is an OOM on a machine whose owner is looking at a
screen that card is drawing. `GPU_HEADROOM_GB = 1.5` is reserved for the same reason the loader
already leaves a CPU core free. `NEURON_GPU_LAYERS` overrides. A failed GPU load retries on CPU
rather than leaving the node with no engine at all. `engine/test_local_gguf.py` **30/30**, nine
of them new.

**BLOCKED, and this is the finding worth keeping:** serving a *layer range* through llama.cpp
is **not reachable from `llama-cpp-python` 0.3.34**. Checked directly against the installed
package — `rpc_servers` is **not** a parameter of `Llama.__init__`, and there is no layer-range
entry point; `Llama` is a whole-model abstraction. So "swap `load_slice_model` for llama.cpp
inside `node_server.py`" cannot be done with what is installed. See [P30].

One process note: the first version of these tests created real 5 GB and 7 GB files with
`truncate()`, which NTFS actually allocates — 12 GB written per run on a disk with 52 GB free,
and the suite hung. Rewritten to stub `os.path.getsize`; runs in 2 s and writes nothing.

### State at end of session

Nothing committed. **Live and working:** coordinator deployed, split pinned 0-9/10-27, driver
running from source as `agent-optinovate`, one verified end-to-end answer over the chain.

Wallet note: both founder wallets had already **spent** their one-time faucet grants
(`total_earned` 25.0 and 25.962, balance 0.0). 25 NRN was moved to each from `__ecosystem__`
via `models.transfer` — supply invariant still reads exactly 1,000,000,000.0. There is no
top-up path for a user who runs out: the faucet is one-time and node earnings live in a
separate ledger with no route into a wallet. **A stranger hits that wall on their ~158th
message.**

**Landmine for the next session:** `agent-bhpc012101` and `agent-bhpc012104` are offline
holding a stale **0-13**. `_walk` picks `max(layer_end)` at each cursor, so when either powers
on it outbids the driver's 0-9 at cursor 0, drops the driver from the chain entirely, and
reports `missing (14, 27)`. **Run `neuron fix` whenever a machine joins or leaves.** Whether
self-heal makes it worse in that window is unverified — the driver, having lost the tie, is no
longer counted as covering anything and may look like idle surplus.

**Suggested pick-up order:**
1. **`node_server.py` through llama.cpp** — serve a layer *range* via the existing binding
   instead of `load_slice_model`. Check whether `llama_cpp` exposes it, and whether
   `rpc-server` ships with the build. This is the 200B decision.
2. Cut **0.19** before another stranger install — the early logging and the slice range guard
   are the two that matter, and the second is a correctness bug, not a diagnostics gap. Bump
   `updater.LOCAL_VERSION`, `neuron.iss` AppVersion and `AGENT_VERSION` together, and update
   `NEURON_AGENT_VERSION` **and** `NEURON_AGENT_SHA256` on the VM in the same change — the
   updater rejects a hash mismatch, and `is_newer` is a strict `>`, so republishing as 0.18
   would mean no existing node ever updates.
3. A slow node must stop being reported as a dead one (`HOT_TIMEOUT_S`, and the false reroute
   that follows).
4. `neuron.bat` gained `driver` and `faucet`; `coordinator/faucet.sh` is new. `.gitignore` now
   covers `agent/config.*.json`, `agent/*.prev` and `agent/payout_key.json` — the last held a
   wallet private key and was committable.

---

## Session 55 (2026-08-08) — the GPU support that was never once executed, and the OOM it shipped

Session 42 turned on a flag that sized a volunteer's slice from **VRAM**. The code it trusted to
make that safe has **never run**: no NVIDIA card exists in this project and the shipped wheel is
`torch 2.4.1+cpu`. ~500 lines of device code, zero executions, one live hazard.

### Half one — the OOM, and why the guard caused it

`balancer.max_layers_for` sized a GPU node by `gpu_vram_gb`. That was justified in Session 42 by
`common.py` gaining a device resolver. Checked against the code a volunteer's machine actually
runs, the justification does not survive:

- `agent/node_server.py` loads through `slice_downloader.load_slice_model`, and
  `slice_downloader.py:299` returns `common.cast_linears(model)` — **no device move**.
- `common.move_model_to_device` has **one caller repo-wide** (`common.py:256`), on the
  bench/verifier path no agent ever reaches.
- `dist/neuron-agent/_internal/torch/version.py:4` is `2.4.1+cpu`, `cuda = None`.

So the weights sit in **system RAM** while the sizing was done against **VRAM** — and the VRAM
branch had **no OS reserve at all** where the RAM branch reserved 3 GB. Backwards: a card with
no headroom left stutters the desktop it is drawing.

Reproduced without hardware, a 12 GB card in an 8 GB machine on Qwen2.5-7B (0.466 GB/layer):

| | layers | weights, into 8 GB of RAM |
|---|---|---|
| shipped 0.18 (VRAM, no reserve) | **19** | 8.9 GB fp16 / **17.7 GB fp32** |
| this change (system RAM − 3 GB) | **8** | 3.7 GB fp16 / 7.5 GB fp32 |

**`GPU_EXECUTION = False`.** Ships by coordinator restart alone — every installed 0.18 agent
stops being over-assigned with no installer and no agent release. Hardened so re-enabling is
safe rather than a second guess: `sane_vram_gb()` (rejects NaN/inf/non-positive/non-numeric and
anything over 192 GB), `VRAM_OS_RESERVE_GB = 1.5` mirroring `local_gguf.GPU_HEADROOM_GB`, and
the three stale comment blocks rewritten to cite `slice_downloader.py:299` so the flag cannot be
re-flipped on the same misreading.

Two more, both about data outliving the fact it described:

- **`main.py` clamps `gpu_vram_gb` at the edge.** It is the only registration field that becomes
  a *memory budget*, and open join means it arrives with no credential. **Clamped to None, never
  rejected** — a 422 over a cosmetic hardware field is [P24] exactly: a healthy machine whose
  owner sees nothing wrong and which simply never joins.
- **`models.py`: VRAM/name follow `has_gpu`** instead of being COALESCEd like `platform`. A node
  that lost its card kept phantom VRAM **forever**, so the balancer could size a slice from
  memory the machine no longer had. A build that reports the card but omits the figure still
  keeps a good one.

**`coordinator/test_gpu_capability.py` 24 → 38.** Its assertions at `:108-142` **asserted the
harm** — 24 GB of VRAM ⇒ 18 layers. They are inverted, each with a comment saying why, in the
same commit as the fix. **The test change IS the fix, not a way to green a red suite.** The
volunteer is now pinned by name: `{has_gpu: True, gpu_vram_gb: 12.0, ram_gb: 8.0}` must size
from RAM.

**Not fully cleared, and worth stating.** At 8 layers the volunteer still holds 7.5 GB in 8 GB
**at fp32**, because `model_tiers.py:41-50` sizes at fp16 while the runtime defaults to fp32 —
which that file already admits. A tier-table change touching every node, deliberately not folded
into an urgent coordinator-only commit.

**Suite:** 66 modules green (not 64 — `api/`, `security/`, `packaging/` exist beyond the dirs
`neuron.bat:90` sweeps), `selftest_shard.py` ALL PASS. `packaging/test_app_entry.py` cannot run
via `-m` (no package init) and passes 5/5 as a script; `blockchain/test_nrn.py` needs a local
hardhat EVM and is unrunnable here — pre-existing, unrelated.

### Half two — every log line and doc claim made true

The engine logged an offload that **cannot happen**. Verified against the installed package:
`llama_supports_gpu_offload()` → **False**, and `llama_cpp/lib/` holds ggml-base, ggml-cpu,
ggml, llama and mtmd with **no `ggml-cuda`**. `n_gpu_layers=-1` is accepted and silently
ignored, while `local_gguf.py` logged *"offloading every layer"*.

- **The build check now comes first**, ahead of the hardware probe — a card the binary cannot
  address is not a card, so asking about VRAM is asking the wrong question. `NEURON_GPU_LAYERS`
  is gated the same way: **an override that silently does nothing is the same bug with a manual
  trigger, and worse**, because someone set it deliberately and would read the log as
  confirmation. Unknown (symbol missing) still attempts, and says it is unverified.
- **The CPU-retry `except` stays** — correct for a future CUDA wheel — but its test is
  re-commented to say it guards a path this build does not ship, reached only because the stub
  holds the gate open. `engine/test_local_gguf.py` **30 → 35**, including one that asserts the
  installed build reports no offload support, so this flips loudly if a CUDA wheel ever lands.

**`INSTALL.md` asked first-GPU volunteers to send back `device: cuda:0` — a line no machine can
print.** It comes from `common.device_name()`, whose only caller is `common.py:257` on the bench
path. Replaced with a witness the node genuinely emits: `reload()` now reads the device **off
the loaded tensors** and folds it into the `slice ready` line — `weights on cpu`. Deliberately
not `common.DEVICE`: that is configured *intent*, and the gap between intent and where the bytes
actually are is precisely what let this survive a release. Meta tensors are skipped (a slice is
a full skeleton with `strict=False`), and it never raises — a diagnostic that can stop a node
from starting is worse than no diagnostic.

**`test_resource_guard.py` shelled out to the real `nvidia-smi`**, which on a machine with no
card can only ever answer "no card" — so the `gpu_ceiling` branch had **zero** coverage. Stubbed
now, with a busy card, an unreadable one, a probe that raises, and each mode's ceiling checked
through to `gpu_busy`. **18 → 27.**

*Two of this session's own test bugs, both the same shape as the bug being fixed:* an assertion
about `max` mode that was really asserting the stub, and a stub of `rg._gpu.gpu_busy` that
mutated the shared `agent.gpu` module and was never restored, so later checks silently tested
the stub instead of the code. Saved and restored, like `_real_seconds_since_input` already was.

**Docs corrected, published text left intact.** `RELEASE_NOTES_v0.18.0.md` gets a dated
**Correction** block (the notes are published; they are not rewritten), `CHANGELOG.md` a set of
entries under Unreleased with the v0.18.0 claims left below as published, `INSTALL.md` a plain
statement that **this build computes on the CPU whatever card you have**, and the five source
comments that still said "CPU-only pipeline" while `GPU_EXECUTION` was `True`.

**The wording that mattered.** Every Session 42 caveat was honest — "has never executed", "no
speedup is claimed" — and every one blamed the **absent test card**. A hardware gap reads as
*untested*; the real cause was **packaging**, which reads as *impossible in this binary*. Only
the second tells you not to size a volunteer's memory by it. Recorded as [P31].

**Suite after half two:** 66 green, `selftest_shard.py` ALL PASS.

### Half three — the device choice, and the ordering trap that makes it real

The requested feature: a post-install CPU/GPU choice, defaulting to what the machine has, with
contribution defaulting to **balanced**.

**The whole difficulty is one line of import order.** `common.DEVICE` is resolved exactly once,
at import (`common.py:113`), and `agent/agent.py` imports `agent.node_server` → `common` at
**module level**. So a device setting read in `main()` would be read, saved, ticked in the tray
menu, and **do nothing at all** — the device was chosen while the module was still importing.
`_apply_device_preference()` therefore runs at module scope, ahead of that import block, and
reads `--config` out of `sys.argv` itself because argparse has not run yet. There is a test
that fails if anyone moves it, and it is line-anchored: the first version matched the mention of
the import inside its own docstring and reported the ordering backwards.

Rules, in order: an explicit `NEURON_DEVICE` in the environment wins (an operator on the command
line is not overruled by a file); `auto` sets nothing and leaves `common._resolve_device()`
exactly as it was; `cpu` pins the CPU; **`gpu` is honoured only if torch reports a usable CUDA
device.** That last one matters — `torch.device("cuda:0")` is accepted by torch **without
checking anything**, so pinning it on a `+cpu` build would hand `common` a device that fails on
first use. Instead the node logs *"GPU requested, but THIS BUILD COMPUTES ON CPU"*, and the tray
shows *"GPU selected — this build computes on CPU"*. **Said out loud, never silently ignored** —
that silent no-op is exactly what [P31] is about, and repeating it in the fix would be absurd.

- **`install.py`** gains `detect_device()` — `torch.cuda`, not `agent.gpu.detect_gpu()`, because
  a card being *present* says nothing about whether this binary can address it, and conflating
  those two is the v0.18.0 mistake. It writes a concrete `"cpu"`/`"gpu"`, not `"auto"`: a value
  written at install is one a person can read in their own config and change.
- **`tray.py`** gains a **Compute device** submenu mirroring *Donation level*, plus a note line
  that says when a change needs a restart. Its text is a *callable*, like `title`/`status` — a
  plain string freezes at its first value and would still say "applies on restart" afterwards.
- **`donation_mode` defaults to `balanced`** in `agent.py`, `install.py` and the committed
  `agent/config.json` template. `write_config` merges, so **existing configs keep what they
  have** — the same guarantee it already makes for `max_cpu_pct`, now tested for these two keys.

`agent/test_device_choice.py` **34/34** (new), `agent/test_tray.py` **8 → 16**.

**Suite: 67 modules green, `selftest_shard.py` ALL PASS.**

### Half four — the device path, and two lifecycle bugs that were never about GPUs

**Phase 4a: every edit a no-op on CPU, and proved rather than asserted.** `test_batching.py`'s
batched-vs-sequential figure is **`4.530e-06` before the change and `4.530e-06` after** —
measured by restoring `batching.py` from HEAD and re-running, not by trusting that the suite
stayed green.

- `batching.py` — KV left-padding is allocated on the cache's own device (`torch.cat` refuses
  to join a CPU pad to a CUDA cache), and the mask, `position_ids` and `cache_position` follow
  `hidden`, which `run_layers_batched` now moves to the device on entry exactly as
  `common._run_layers` already did.
- **The batched stages returned device tensors while their unbatched twins returned CPU.** That
  divergence *is* the bug: every node in a real chain serves through the batcher, so the path
  that correctly returns to CPU is the one nothing uses. All four now mirror `common`.
- `wire_codec.py` — the Hadamard matrix follows its operand (it is pinned CPU-side on the
  **default** `i8h` codec, so this is the first thing a GPU node would have hit), and every
  `.numpy()` reaches CPU first; `numpy()` raises on a CUDA tensor.
- Every `send_msg` caller passes `common._to_cpu`, fixing `common.py:474-482`'s legacy
  `torch.save` **from the callers** — no `common.py` edit, build rule 7 intact.
- `agent/agent.py` — `torch.cuda.synchronize()` around the self-benchmark. CUDA kernels are
  queued, not run, so without it a GPU node times how fast it can *enqueue* work, reports an
  absurd `ms_per_layer`, and `balancer.solve` hands it nearly every layer: **the same OOM by a
  second route**, through the speed field instead of the memory one.

**Phase 4b stays deferred, and now has a tripwire.** `slice_downloader.py:299` is left alone
with a comment saying why; root **`test_device_path.py`** (18/18) fails if it starts moving
weights, and fails if `GPU_EXECUTION` is switched on while the loader does not — those two facts
are a pair. **Verified the tripwire actually fires** by temporarily adding the device move: 1
failure, then restored.

**Phase 5a — pause meant almost nothing.** `NodeServer.paused` existed and was **never read**;
the agent never passed its own flag in, so Pause skipped the heartbeat and the coordinator went
on routing live requests for up to ~90 s while the owner watched a tray that said "Paused".

The plan said to check what the driver does with a refused connection before writing this, and
that check changed the design: `DEAD_PEER = (ConnectionError, TimeoutError, EOFError, OSError)`
and **`ConnectionRefusedError` is a subclass of `ConnectionError`** — so slamming the socket is
indistinguishable from the machine dying. The driver would rebuild the chain, replay the
junction cache, and tell the user *"a machine dropped out"*: [P28]'s complaint, self-inflicted.

So the refusal is a **typed reply**, checked on `config` (which starts a new request) and not on
`act` (which continues one already in flight). And the reply had to be made *recoverable*: peers
did `assert ack.get("ok")` / `raise RuntimeError`, neither of which any handler catches, so a
polite refusal would have failed the whole answer where a rude one merely rerouted.
`PeerUnavailable(ConnectionError)` fixes that with no change to the error handling — it is
already inside `DEAD_PEER` — while staying distinguishable in the reason string. **Paused and
died are different events.**

**Phase 5b — a migrating node held two slices at once.** `reload()` loaded the new slice while
`self.model` still referenced the old, peaking at ~150% of one slice on machines chosen because
they had room for one. Peak is what OOM-kills a volunteer; steady state is not. Now the old
slice and its batchers are released, `gc.collect()` + `empty_cache()` run in the gap, and only
then does the new one load. The window is guarded: a request arriving in it gets a named
`reloading` refusal instead of an `AttributeError` from inside a batcher.

**The cost is stated rather than discovered later:** a load that fails now leaves the node with
no slice, where before it kept serving the old one. That is the right trade — the node is being
migrated off that slice anyway — but it is a real change, so the failure path is explicit and
tested.

`agent/test_pause_admission.py` **14/14** and `agent/test_reload_lifetime.py` **14/14**, both
new, and **separate commits**: one changes request admission, the other model lifetime.

**Suite: 70 modules green**, `selftest_shard.py` ALL PASS, `test_short_chain.py` 12/12,
`test_node_death.py` 7/7 — still token-identical after a SIGKILL mid-generation.

### Half five — the bug that was never about GPUs

Chasing the fp32 residue left over from half one turned up the largest sizing error in the
repo, and it has nothing to do with graphics cards.

`model_tiers.gb_per_layer` is computed at fp16 (2 bytes/param). Its comment justified that with
**"common.WEIGHT_DTYPE=fp16"**. `common.py:62` reads `NEURON_WEIGHT_DTYPE` and defaults to
**fp32**; nothing in the agent, the installer or the packaged build sets it; and `cast_linears`
is a no-op at fp32. **Every node on every tier was cleared for twice its real footprint** — an
8 GB machine on the 7B tier for 8 layers, 7.46 GB of weights against a 3.75 GB budget. Latent
only because the network serves the 1.5B, where 28 layers is too few for the cap to bind.

Same shape as [P31] itself: an assumption about the runtime, written down as fact, that the
runtime contradicts. Found only by reading `common.py` instead of the comment describing it.

`balancer.effective_gb_per_layer` scales the tier figure by `weight_bytes_for(node)` —
pessimistic (fp32) unless a node says otherwise, applied **exactly once** on every path
(`max_layers_for`, and `layer_caps`/`capacity_shortfall`/`solve` through it). Deliberately not
fixed by doubling the table: a layer's size is a property of the MODEL, the dtype a property of
the NODE, and conflating them makes a genuine fp16 node impossible to size.

**The part worth keeping.** Seven tests failed on the corrected arithmetic — every one with
"fits" or "can hold" in its premise. The first fix was to pin `weight_dtype: "fp16"` in the
fixtures, which passes and is wrong: it preserves the claim that the live 8/8/12 trio can hold
the 7B, when at fp32 it holds **15 of 28 layers**. Testing that directly — removing the pin and
running every test function independently — is what exposed it. The fixtures were resized to
machines that genuinely fit, and **no dtype pin survives anywhere**: `test_migration` 49/49,
`test_model_tiers` 23/23, both honest.

**No live impact from deploying it.** The network is on the 1.5B floor with one online node,
and `/network/model` already reports the 7B as `feasible: false, placeable: false`.

`coordinator/test_weight_dtype_sizing.py` **28/28** (new). **Suite: 71 modules green**,
`selftest_shard.py` ALL PASS.

### Half six — it shipped

Everything above was written while uncommitted. It is now released, so the "nothing committed,
nothing deployed" that stood here is replaced by what actually happened.

**Coordinator, deployed first and alone.** `./coordinator/deploy.sh` — DB backed up, gates
verified (`/status` OK, faucet 401, admin 401, logins live), `GPU_EXECUTION = False` confirmed
on the VM. **Every already-installed 0.18 agent stopped being over-assigned at that restart**,
with no installer and no agent release. That separation was the entire point of Phase 1 and it
held.

**v0.19.0 built and published.** `NEURON-Setup-0.19.0.exe`, 207 MB, SHA-256 `80eb83a2…`. The
hash was verified by **re-downloading GitHub's stored copy and hashing that**, not by trusting
the local build: `updater.py` refuses a mismatch, so a wrong hash there means every node
silently declines the update and the rollout does nothing. Six surfaces moved together —
`updater.LOCAL_VERSION`, `neuron.iss` AppVersion, `config.AGENT_VERSION`, the VM's
`NEURON_AGENT_VERSION` and `NEURON_AGENT_SHA256`, and `docs/index.html`. The landing page was
updated **after** the release existed, so the only route a stranger has never pointed at a 404.

**Seventeen commits pushed** — the repo had been unpushed since before Session 52, so GitHub was
serving a tree older than the last three sessions' work.

**The scrub that had to happen first.** `PROBLEMS.md` [P24] published the office and home IP
addresses. Redacting the file was not enough: the addresses were inside commit `a0dda72`, and
git history would have carried them permanently. Nothing was pushed yet, so the last two commits
were rebuilt with the redaction folded in — `git log -S` now returns nothing for either address.
Also confirmed before pushing: `research/` and the four `neuronscript*.c` files are ignored **and
were never committed**, so `.gitignore` alone was not being relied on for something it cannot do
retroactively.

### Half seven — the tray was telling nobody anything

Prompted by a screenshot: no version, no notifications, "Status: Active" and nothing else.

**The agent already knew all of it.** `Agent.state` has carried a `detail` for every state since
it was written — "earning", "awaiting verification — not yet earning", "coordinator unreachable:
…", the guard's reason list — and the menu rendered one word and discarded the rest. That is most
of why the app reads as basic: not missing information, unspoken information. The version went
into the menu title and the hover tooltip, where "which build are you running?" stops requiring a
log file.

**Notifications, deliberately restrained.** Only states the owner cannot otherwise discover —
promoted to verified, node no longer serving, token superseded — and only on the transition into
them. An app that notifies every poll is muted within the hour, and then the one that mattered
never arrives.

**The heartbeat was defeating remote diagnosis, and this is the one worth remembering.** Every
beat logged at INFO: **2,833 lines and 141 KB a day** of `heartbeat ok — active`, against
`logtail.MAX_BYTES` of **64 KB**. So the tail `neuron_logs.py` collects held **under 11 hours**
and, on an idle node, was ~100% heartbeat — a stranger's slice error or crash scrolled out within
hours. That file exists precisely so diagnosing someone else's machine does not depend on their
attention ([P24]); a window full of "ok" defeats exactly that. Now: transitions log immediately,
unchanged states go to DEBUG, one alive line every 30 beats keeps **silence meaning dead**
(`verify_service.py`'s rule). **A day drops from 2,833 INFO lines to 95**, and the test measures
that against `MAX_BYTES` rather than checking the source changed.

`agent/test_tray.py` 16 → 29, `agent/test_heartbeat_logging.py` 15/15 (new).

### Half eight — two security answers, one of them wrong in our favour

**[P12] described a system that no longer exists.** It still said the ledger mints per request
and `/complete` has no auth. Re-verified against the running code: `ledger.py` settles by
**transfer out of escrow**, `models.credit()` has no production caller left (test fixtures only),
`test_escrow_conservation` is 40/40 including the supply invariant, `/infer` holds and `settle()`
refunds the remainder, and `/complete` requires the issued `complete_token`, settles from the
**coordinator-recorded plan** rather than caller-supplied `node_ids`, 409s a replay and clamps
the token count. Rewritten to 🟢. **A stale security entry is its own hazard** — it either causes
work already done, or gets ignored on the day it matters.

Which answers the question a public repo invites: reading the source grants no ability to move
NRN, because **there is no HTTP route to `models.transfer` at all**.

**The real exposure was operational, and on this machine.** The SSH key to the coordinator VM —
the only key authorised on it, and the VM holds the ledger, every wallet balance and the register
secret — was **unencrypted on disk**. Worse, `chmod 0600` is what the code does and NTFS ignores
POSIX mode bits, so the actual ACL granted Full Control to SYSTEM, Administrators and the user.
Anything running as that user had root-equivalent access to the coordinator with no credential to
steal beyond a file read. `payout_key.json` had the identical blind spot, and its own comment
(`# 0600; no-op on some FSes`) had been quietly admitting it.

Fixed: passphrase on the key, `icacls` ACL restricted to the owner on both files, a persistent
ssh-agent at a fixed socket, `AddKeysToAgent` in `~/.ssh/config` so the passphrase is typed once
per boot and never again, and `deploy.sh` preferring the agent while still falling back to
prompting. Verified from a fresh login shell with `SSH_AUTH_SOCK` unset: reaches the VM with no
`-i` and no prompt.

**Also decided:** rent GPU time online rather than keep deferring the GPU path. Every remaining
GPU item is blocked on the same thing — no machine here has an NVIDIA card, so not one line of
the CUDA path can execute — and that is exactly how [P31] happened. The order that session must
follow is in the decisions log, and it is not arbitrary: 4a verified on hardware before 4b,
`selftest_shard` reconciled before `GPU_EXECUTION` goes back on, packaging last.

### Half nine — the chat UI, ported, and the two bugs running it exposed

The founder's React chat app (`C:\Users\optin\Trust chat`, an AI-Studio build that talked to
Gemini, Ollama, LM Studio and KoboldCPP) is now a NEURON client, living in `ui/web/` and served
at **`/next`** — deliberately NOT at `/`.

**Roughly a third of what was on screen was machinery for a product NEURON is not.** Deleted:
`services/aiProviders.ts`, `SettingsModal` (provider + base-URL config, and the only `recharts`
user), `ModelSelectorModal`, the probe that walked candidate localhost ports, and five sampling
sliders — `/chat` takes `prompt`, `max_tokens`, `use_rag` and `conversation_id`, so temperature
and top-p had nowhere to go, and **a control that moves a number nothing reads is worse than no
control**. No vendor name survives anywhere. The model is displayed, never selected: the tier
controller picks it from how many machines are online.

Added: `services/neuron.ts` (the SSE transport, written against `ui/app.py`'s actual event
shapes rather than a guess), `services/wallet.ts`, and a **wallet panel** where the engine list
used to be — balance and earned are what a contributor wants to see. A failed wallet read shows
`unavailable`, never `0.00`: "you earned nothing" and "we could not ask" are different facts,
and a confident zero on a network that pays people reads as *this does not pay*.

Palette and logo are NEURON's, lifted from `coordinator/theme.py` so the chat, the dashboard and
the site are finally one product. `--accent` is `#15803d`, not the brighter `#16a34a` that
`theme.py` records as 3.30:1 on white — under AA. Contrast computed rather than eyeballed:
**5.02:1 light and 8.55:1 dark**, the same figures `ui/test_chat_ui.py` already asserts.

**`src/services/neuron.test.ts` (15 tests) is the load-bearing part.** `ui/test_chat_ui.py` has
67 assertions and **59 of them grep `chat.html`'s SOURCE TEXT**. A React build compiles to
minified bundles, so all 59 break the day this replaces the old page, and the guarantees — each
bought with a real incident — would vanish silently. They are asserted here instead, over plain
functions: a reroute is not an error, a partial answer survives the error that ended it,
`done.reroutes` is reported, a token is a fragment not the accumulation, a frame split across TCP
chunks still parses, and `errorMessage` never blames a node going offline. **That is the answer
to "the tests all break": not fewer tests, a better place to put them.**

#### Two bugs that only appeared by actually running it

**The cost line was pricing answers in dollars.** It multiplied tokens by *Gemini's* per-token
rates and rendered `< $0.0001`. Wrong twice: arithmetic for a service NEURON does not use, and a
currency symbol asserting a cash value NRN does not have — in the one place a user looks after
every answer, contradicting the installer disclosure, the landing page and this app's own
footer. It now reports the cost the coordinator actually settled.

**`cannot route a 1-stage chain (this driver handles 2 or 3). chain=[[0, 27]]`** — the UI was
right and the network was broken. Both online nodes held **0-27**, because placement had
replicated the single stage earlier the same day (*"copying it removes the current bottleneck"*)
rather than splitting it. Sensible for throughput, unroutable in practice: the replica logic does
not know the driver requires 2 or 3 stages. `pin_layers.sh`'s own header describes this exact
state. Re-split to `[[0,9],[10,27]]` and it routed immediately — **both nodes adopted their new
ranges without a restart.**

That also exposed `neuron.bat` hardcoding `DRIVER=agent-optinovate`, stale since this machine
re-registered today as `agent-optinovate-6ff49d`. Fixed locally (the file is deliberately
gitignored as a personal helper): `neuron fix` now passes nothing, so with one node online the
script infers the driver and with more it **lists them**. Worth being accurate — `pin_layers.sh`
already refused an offline driver and named the online ones, so the stale value produced a clear
error, never a wrong split. What it cost was `neuron fix` failing on the day it is most needed:
right after a machine joins or leaves.

**Proven live, both paths:** locally (`this machine · free`, 4.0 tok/s) and across the real chain
(`2 NODES`, 6.49s, 1.4 tok/s, settled against the wallet — balance 24.962 → 24.408 while earned
rose 50.654 → 51.184, this machine both paying and earning as stage 1).

**Left at `/next` on purpose.** The old page keeps working and its 67 tests stay meaningful until
the new one has earned the swap, which is one line in `ui/app.py`.

### State at end of session

**Shipped and live today:** coordinator sizing fixes deployed; **v0.19.0** published, SHA-verified
against GitHub's stored bytes, and advertised by the coordinator; landing page updated; the repo
pushed after being unpushed since before Session 52. SSH key passphrased with real NTFS ACLs on
it and on `payout_key.json`. Network re-split and routing.

**Committed, NOT released:** the tray work, the heartbeat fix, and the chat UI. None of it
reaches a volunteer until 0.20 is built and published — the lesson this session opened with,
when a source edit could not touch the running 0.18 because it served a bundled copy.

**Suite: 71 Python modules green, `neuron.test.ts` 15/15, `selftest_shard.py` ALL PASS.**

**Open, in priority order:**
1. **Rotate `NEURON_REGISTER_SECRET`** — it was echoed into a session transcript. It grants
   trusted standing AND overrides the payout-rebind check, making it the only path to
   redirecting another node's earnings.
2. **`neuron fix`** when the office PCs wake — [P27]. Run it with no argument first; it lists
   who is online and you name the machine you chat from.
3. **Swap `/next` → `/`** once satisfied, and port whatever of `test_chat_ui.py`'s 67 assertions
   still apply. 59 of them read `chat.html`'s source and cannot survive the swap.
4. **Cut 0.20** — tray, heartbeat and the chat UI, none of which anyone can see yet.
5. **The GPU session** — rent time online; the order it must follow is in the decisions log and
   is not arbitrary.
6. Remaining 0.20 candidates: tray download progress, a chat reachable while downloading, engine
   swap hysteresis (a 7B→1.5B reload churn seen live, trigger unconfirmed), and [P28].

**Two identities on this machine.** It re-registered today as `agent-optinovate-6ff49d` while
`%LOCALAPPDATA%\NEURON\config.json` still names `agent-optinovate`, which is offline and holds
whatever it earned. Worth resolving deliberately rather than discovering later.

---

## Session 56 (2026-08-09) — the register secret rotated, and the three places it was still sitting

Session 55's open item 1. `NEURON_REGISTER_SECRET` had been echoed into a session transcript,
and it is not a cosmetic credential: it grants **trusted standing** on registration, unlocks
every `/admin/*` route, reveals node addresses on `/node/list`, and **overrides the
payout-rebind check** (`main.py:1077`) — which makes it the only path in the system to
redirecting another node's earnings.

**Mapped before touching anything, and the map is why this was safe to do live.** Five holders:
the VM's systemd unit, `.env.coordinator`, and `agent/config.node-{a,b,c}-local.json`. All five
carried the same value (compared by SHA-256 prefix, never by printing it). The installed agent
configs — `agent/config.driver.json` and `%LOCALAPPDATA%\NEURON\config.json` — **do not hold it
at all**, because nodes authenticate with `node_token`. So no volunteer, and no live node, could
be disconnected by this. The coordinator restart was the only exposure, and `deploy.sh` does
that routinely.

**The value never entered a transcript, an argv, or a shell history.** Generated locally into a
scratchpad file, piped to the VM on **ssh stdin**, and verified from the VM against
**127.0.0.1** — so the new secret was never sent across the network to test it. The old one was
read by the remote script out of the unit file it had just backed up, so proving it now fails
needed no transport either.

**The first attempt rolled itself back, and the reason is worth keeping.** `systemctl restart`
returns when the process is **spawned**, not when uvicorn has **bound the port** — so
`is-active` said `active` while every check returned `URLError`, and a fixed 6-second sleep
reported a healthy rotation as a failure. Replaced with a poll on `/status`; the successful run
reported **ready in 4.1 s**. The part that mattered is that the rollback fired: unit restored,
service restarted, network verified healthy (28/28, 2 online) before anything else was tried.
A rotation script whose failure path is untested is a way to lose a coordinator.

**Verified, not assumed:** new secret → 200, **retired secret → 401**, no secret → 401, on
`/admin/logs`. Then the same three from this machine through `.env.coordinator`, loaded exactly
the way `pin_layers.sh` loads it, plus `/node/list` returning addresses (an operator-only field).

**The verifier had to be restarted, and exposed two things.** `verify_service.py` reads the
secret **once at startup**, so it kept 401ing after the rotation — with precisely the right
message (*"the register secret is wrong, so nothing can be verified"*), first line at 14:49:39,
the second the rotation landed. There were also **two logical verifiers** running, one predating
the rotation; deduplicated to one, after which the ERROR line stops. What looked like two
verifier processes and four UI processes is the **venv redirector** — `.venv\Scripts\pythonw.exe`
spawns the base interpreter as a child — not a self-relaunch, and not four servers.

**Retired-secret residue, cleaned.** Three unit backups on the VM still held the old value.
A `cp` from one of those would have reinstated a **compromised** credential *and* 401'd every
client updated today. That one line is now redacted in each; `NEURON_RELAY_SECRET` and
`NEURON_WALLET_LINK_SECRET` in the same files are **still live**, so blanking the files wholesale
would have destroyed a genuine backup.

**Still holding the retired value, deliberately untouched:**
`C:\Users\optin\neuron-stranger\.env.coordinator`. Inert — the value no longer authenticates —
but it is a second checkout whose operator tooling now silently 401s. Left as the founder's
call: if that tree is a *stranger* simulation it should never have carried the operator secret,
and if it is a second operator checkout it wants the live one.

**A claim of mine that was wrong, and is corrected rather than left standing.** The `http://IP:8001`
in `.env.coordinator` looked like it was downgrading every operator call to cleartext. It was
not: **nothing sources that file** — each tool greps it for the one secret key — and
`pin_layers.sh`, `rebalance.sh`, `verify_service.py` and `neuron_logs.py` all default to
`https://neuronnet.duckdns.org` on their own, with `$NEURON_COORDINATOR` unset. The line was
stale documentation from before `setup_https.sh`, not an active hazard. It is corrected to the
https name, and the file now says which line is actually read — a URL a human copies a `curl`
off is worth keeping honest, which is the real (smaller) reason to fix it.

**Suite: 71 Python modules green, 0 failures; `ui/web` vitest 15/15; `selftest_shard.py` ALL
PASS.** Network healthy throughout and after: 28/28 covered, 2 online, 0 flagged.

**Nothing committed** — as instructed. Changes sit in `.env.coordinator` and the three
`agent/config.node-*-local.json` files, all of which are gitignored, plus this entry.

### Item 2 arrived early, and not via the office PCs — [P32]

Checking the roster during the rotation showed **`node-c-pavilion` back on 0-27**, the whole
model, alongside the driver on 0-9. Both start at layer 0, so `router._walk` (`router.py:104`)
takes `max(layer_end)` = 27 from cursor 0, jumps to 28, and produces a **1-stage chain** —
which `node_a.coord_get_chain` refuses. Distributed chat was dead, and had been since some point
after Session 55 half nine.

**`/status` said `network_healthy: true` the entire time**, because 28/28 layers genuinely were
covered. **Coverage is not routability**, and the health check only answers the first question.
Worth stating plainly because it is why this can sit broken without anyone being told:
`pin_layers.sh:11` records the real cost — each request *"dies client-side after the coordinator
has already taken a wallet hold"*, so failed attempts lock escrow for `HOLD_TTL_S` (600 s), and
since nothing completes, **neither machine earns**.

Re-pinned to `[[0,9],[10,27]]`; the walk now yields 2 stages. **But this is a mitigation, not a
fix, and it will revert.** The same command was run and verified ~4 hours earlier in Session 55
half nine — *"both nodes adopted their new ranges without a restart"* — and pavilion had drifted
back by this session.

**The mechanism, recorded as [P32]:** `neuron fix` writes to the **coordinator**, while the
node's own `config.json` is what it re-asserts. `agent.py:413` returns early when config already
holds a range (so the node never re-asks), `agent.py:569` puts that range in the registration
body, and `models.py:361` applies it with **no COALESCE** — while `ms_per_layer`, `head_ms`,
`platform` and `hw_fingerprint` on the four lines below it *are* COALESCEd. The protective
pattern was applied all around that line and not to it. Any restart, reconnect or relay-ticket
refresh closes the loop, silently.

**`layers_pinned` does not prevent it** — it guards `reconsider_placement`, which `agent.py:435`
says is *"Only ever called while PROBATIONARY"*, and pavilion is trusted. What keeps the driver
correct is simply that `agent/config.driver.json` holds the *right* range (0-9), so its
re-assertion is harmless. That asymmetry is the whole difference between the two machines.

[P27] and [P32] are the offline and online halves of one question: the operator tooling treats
the coordinator as authoritative for placement, and the agent treats it as advisory.

**The automatic repair shares the blind spot — which is why nothing caught this.** The
coordinator is not passive here: a node joining with no range gets `suggest_placement`, which IS
stage-aware (`main.py:898`), and a node leaving that opens a hole is handled by `self_heal` every
sweep. But `self_heal` (`migration.py:297`) opens with `if not missing: return` — it keys
entirely on **uncovered layers**. Pavilion on 0-27 plus the driver on 0-9 leaves nothing
uncovered, so self-heal inspected the network, found it healthy, and did nothing.
`PIPELINE_STAGES` is enforced when handing out a NEW placement and never re-checked against the
roster afterwards. The coverage/routability confusion is not only in `/status`; it is in the
repair path too.

### Option C, built: the network now says when it cannot route

Three options were on the table — **A** correct pavilion's local config (contained, needs access
to that machine, fixes this instance); **B** COALESCE the range on re-registration (fixes the
class, ships by restart alone, but inverts placement ownership for every node); **C** check stage
count, not just coverage. **C was chosen and built**, because it is the one that would have
caught this automatically and it is correct whichever of A or B follows.

**Detection only — it moves nothing, and that restraint is the design.** Repairing means
rewriting ranges the nodes themselves re-assert, so until ownership is settled a repairing sweep
and a re-registering node would overwrite each other every 60 seconds. Making a silent failure
loud is the half that holds under either outcome.

- **`router.chain_shape(nodes, total)`** — pure description of a roster: `stages`, `ranges`,
  `routable`, deterministic chooser. It checks `missing` **as well as** the stage count, because
  a chain that stops at a gap still has a plausible-looking stage count and calling that routable
  would repeat the exact half-answer being fixed.
- **`/status` gains `stages`, `chain_ranges`, `routable`**, and **`network_healthy` now means "a
  request can complete"** — what the dashboard dot, `neuron_doctor`, the landing page and
  `ui/app.py` all already treated it as meaning.
- **The sweep logs the TRANSITION**, not the state: an unroutable network never self-corrects, so
  a per-sweep line would be 1,440/day and read as wallpaper (the heartbeat lesson from half
  seven). It names the cost and the remedy.
- **The dashboard and doctor now distinguish the two failures.** "Chain incomplete" for both sent
  an operator hunting a missing node while every layer was present.
- **`config.MIN_PIPELINE_STAGES = 2`** sits next to `PIPELINE_STAGES`. The ceiling was enforced
  when handing out a placement; the floor was enforced nowhere. One rule, and splitting it is why
  half went unchecked.

`coordinator/test_routability.py` **14/14** (new) pins the DISTINCTION, not the incident. **The
tripwire was verified to fire** — reverting `network_healthy` to coverage-only fails
`test_summary_is_not_healthy_when_covered_but_unroutable`, then restored. One test bug of my own,
the same shape as always: `_offline()` wrote the `status` column, which `_node_dict` **derives**
from `last_seen` — so it asserted nothing until it backdated the heartbeat instead.

Both dashboard branches were rendered against a real unroutable roster rather than trusted to the
suite: banner red with the accurate reason at 1 stage, green at 2.

**Suite: 72 Python modules green (71 + the new one), 0 failures; vitest 15/15; `selftest_shard.py`
ALL PASS.** C was then **deployed** — `deploy.sh`, gates verified — and the coordinator logged its
own first transition at 16:44: `[health] chain is routable again: 2 stage(s) [[0, 9], [10, 27]]`.

### The question that changed the answer

Asked whether this was work for a thousand-machine network or for one PC. It was the right
question, and it inverted the recommendation.

**A was rejected.** Editing pavilion's `config.json` only works because pavilion is in this
house. At a thousand nodes there is no access to a volunteer's config — so A is not narrow, it is
*structurally unavailable* the moment the network is real. Any fix requiring a specific machine
to be reachable has a ceiling of "what one person can babysit". It was very nearly done, because
it is cheap and it works tonight, which is exactly what makes it the wrong habit.

The same lens condemns `neuron fix` itself: a manual step, run by an awake operator holding a
secret who knows which machine they chat from, **once per join or leave** — when churn is the
steady state of a real network, not an event.

### B, built: the coordinator owns placement

A node **proposes**; the coordinator **disposes**. The naive `COALESCE` is unsafe —
`reconsider_placement` moves a probationary node by rewriting config and re-registering, so
ignoring the claim would leave node and coordinator disagreeing about what is being served: a
correctness hazard (wrong activations → failed proof-of-compute → flagged, with no clue why),
worse than the bug being fixed.

What made the safe version small is a fact already in the code: **the node already follows the
coordinator.** `slice-info` returns the coordinator's range and `setup()` serves it. The
registration echo was never how a node *learned* its range — only how a *stale* one got written
back. Removing it closes the one path by which the two could diverge; it does not open one.

- `register_node` no longer updates the range on conflict; INSERT still establishes it, so new
  nodes and wiped-and-rejoined machines are unaffected. `update_layers()` — `/network/layers`,
  migration cutover, self-heal — is untouched.
- The claim is **recorded**, not discarded (`reported_layer_*`, `placement_drift`). Ignoring the
  node silently would trade one invisible fact for another. NULL means "no claim seen yet", so
  live nodes do not all light up on the first deploy.
- The registration reply reads `assigned_layers` **back from the DB**, exactly as `standing`
  above it already did — it used to echo the caller, so the agent's startup line would have
  printed the stale range it had just failed to impose.
- Agent half (needs 0.20): `setup()` persists the served range, making config a cache of the
  coordinator's answer rather than a rival opinion; and a `--layers` that contradicts the
  assignment now **says so**, because an override that silently does nothing is [P31] with a
  manual trigger.

**It ships by coordinator restart alone** and fixes every installed 0.18/0.19 agent — it stops
*listening* to the stale claim rather than requiring agents to stop making it. Same lever as
`GPU_EXECUTION` in Session 55.

`coordinator/test_placement_ownership.py` **9/9** (new), including the live sequence end to end.
**Tripwire verified:** restoring the overwrite fails with "the pin must survive" and "the pin
reverted — [P32] is back", then restored.

### Deployed — and it caught a recurrence on the way in

B went out the same evening. Schema migration verified against the live DB: both columns added,
5 node rows and 39 ledger rows intact, **supply still exactly 1,000,000,000**.

Between the afternoon pin and that deploy the split had **already been lost again** — this time
`agent-optinovate-6ff49d`, the driver itself, sitting on **0-27**. One stage. Chat dead. The
coordinator said so without being asked:

```
19:47:29 [health] NETWORK NOT ROUTABLE: the chain walks to 1 stage(s) [[0, 27]], and a driver
         accepts 2-3. ... Chats will fail AFTER a wallet hold is taken. Fix: pin_layers.sh ...
19:48:32 [health] chain is routable again: 2 stage(s) [[0, 9], [10, 27]]
```

Twice in one day the pin was lost in silence; the third time the network reported it itself, in
one line, with the remedy. That is C justified on evidence rather than argument — and B means
this is the first pin that will actually hold.

*One check of mine was wrong and worth keeping:* verifying the deployed code, `grep -c
"layer_start=excluded.layer_start"` returned 1 and I read it as a failure — but
`reported_layer_start=excluded.layer_start` **contains** that string. Anchored properly, the bare
assignment count is 0. A grep whose pattern is a substring of the thing it is meant to exclude
will confirm whatever you already believe.

**Open:** cut 0.20 (tray, heartbeat, chat UI, agent placement-cache); `/next` → `/`;
the GPU session. [P27] still live for the two office PCs (offline, stale 0-13). Longer-horizon,
and now named: generalise the driver past 2–3 stages and the hardcoded `S1=10` (`config.py:83`
already says raising `PIPELINE_STAGES` needs it), and a user who is not also a node — today the
machine you chat from must hold layers 0–9.

**Re-verified after the doc changes:** 71 Python modules green, vitest 15/15, `selftest_shard.py`
ALL PASS. No source file was modified in this half — only `PROBLEMS.md` and this log.

### Half eleven — the founder's answer: pay for availability, not just for work

The layer/download thread was being chased from the wrong end. The founder reframed it: NEURON is
a passenger train. People board and leave; the train does not stop, and the fare does not rise
because the carriage is half empty. **Then: assign nodes time slots, and reward them for staying
awake in one.**

That is the piece the whole architecture was missing, and it is not new — it is
`TOKENOMICS.md` §11.4, written and never built. §11.4 already says emission **must** be decoupled
from traffic and paid **per device-hour** (~1 NRN/hour from the 600M `__emission_pool__`),
decoupled precisely because anything paying above the spend for metered work can be farmed by
generating your own traffic. It even names a planned `coordinator/emission.py` on the health
sweep. Only `last_seen` existed.

What the slot idea adds over §11.4 is **direction**: hours are paid for *where and when coverage
is short*, so availability is rostered instead of accidental. Phones are the reason it matters —
`SAFETY_LIMITS.md` has them contribute only at battery ≥80% and plugged in, so they arrive and
leave in correlated nightly waves, which is exactly when coverage is thinnest. The cheapest
supply on the network is available precisely at the hours nobody covers.

**Built (`coordinator/emission.py`, new).** A slot-hour pays only if **all three** hold: the node
was present for at least `SLOT_MIN_ATTENDANCE_FRAC` of the hour, it held a block of the serving
model, and **a proof-of-compute challenge landed inside that slot**. The third is the load-bearing
one: paying for presence invents a reason to fake presence, and a script that heartbeats while
computing nothing would otherwise be the most profitable node on the network. Nothing about
per-request earning had that hole, because serving is self-verifying.

- **Rate = base × scarcity multiplier**, bounded and monotonic in replica depth — a recruiting
  signal, not an auction. Uncapped, a slot with one eligible node prices itself arbitrarily high
  at the moment the network can least afford it.
- **Unverified nodes do not count toward replica depth.** If they did, a block held only by
  machines that cannot prove they compute would price itself as healthy and never attract a real
  one.
- **Nothing is minted** — payment is `models.transfer` out of `__emission_pool__`, so the fixed
  1,000,000,000 supply is untouched. A test asserts the supply is identical before and after, and
  that the pool fell by exactly what was paid.
- **At most once per slot.** `attendance` is keyed `(node_id, slot_start)` and settlement claims
  the row *before* money moves. Claim-then-pay can at worst pay nothing for a claimed slot, which
  is visible; pay-then-claim can pay twice, which is not.
- **Open slots are never settled**, a **daily cap** bounds a scarcity spike, and pool exhaustion
  is logged as the tokenomics event it is rather than retried forever.
- **`GET /network/slots`** publishes where cover is needed and what that hour pays — deliberately
  public, because it is a recruiting signal, and it says where the chain is *about to* stop
  working rather than whether it works now.
- Attendance accrues from the **existing heartbeat**, with the credited delta capped at two ping
  intervals: without that, a node absent for six hours banks the whole gap as presence on its
  first beat — paid precisely for being away.

*One bug of my own, caught by its own test:* the accrual read `row["last_seen"] or now`, and
`0.0` is falsy — a zero timestamp silently became "just beat" and accrued nothing. `is None`, not
truthiness.

`coordinator/test_emission_slots.py` **15/15** (new).

**The fare does not move.** Users still pay per token out of escrow; emission is a separate stream
from a separate bucket. Scarcity shows up as a night-shift premium paid by the railway, never as
a ticket price — which is the half of the train analogy I missed the first time.

**This makes [P29] blocking, and that is recorded.** Emission credits the **node's** ledger row,
and `nodes` has no owner column at all — so the network is now actively accruing balances no
human can spend. Paying volunteers in something they cannot reach is worse than not paying them.
The node→wallet binding can mirror `/node/{id}/payout-address` exactly.

**Not done:** deploying it, and deriving a phone's declared window from observed charge history
(the plumbing is in; Android reads it from nothing yet).

### Half twelve — the chain repairs itself, because a manual step was the actual bug

All four PCs came online overnight and the network was found broken:
`[[0,9],[10,13],[14,20],[21,27]]` — **four stages**, with the driver stranded in the middle on
14-20 while an 8 GB office PC held stage 1. Chat dead twice over.

The journal shows what happened, and two of the day's fixes working while it did. Machines
flapped from 04:56 onward; `[gap-heal]` re-split on each flap, accreting segments; the routability
watch called every bad state with its remedy; and `[placement] ... keeping the assignment` fired
twice against nodes re-asserting stale ranges. Then at 06:26 `[migration] steady -> preparing` —
four nodes satisfied the 7B tier's `min_nodes: 3`, so the network started **moving itself onto
Qwen2.5-7B**, stuck at 1-of-3 ready while the machines it needed kept dropping.

**Auto-promotion is now off** (`NEURON_TIER_PROMOTE_MARGIN=100`, a drop-in on the VM). The
migration state is in-memory, so the restart cleared the in-flight move. `PROMOTE_MARGIN` is the
right lever because `_meets` scales by `(1 + margin)` for promotion only — demotion re-checks at
margin 0.0, so the floor still works and a genuinely shrunken network can still fall back. Which
model the network serves is now a decision, not a function of who is awake.

**The founder's objection was the correct one:** *"every time a new PC is added we have to fix
this again — that's terrible."* Right. `pin_layers.sh` being a MANUAL step was the defect, not
the shapes it fixed.

`router.canonical_assignment` is that script expressed server-side, applied from the health sweep
whenever the chain is unroutable: stage 1 pinned to the driver's shard, at most
`PIPELINE_STAGES` stages, every remaining machine replicating rather than deepening the pipeline.
Stability is deliberate — whoever already holds stage 1 keeps it (moving the driver breaks chat
even when the chain looks legal), and everyone else is ordered by current `layer_start` so as few
slices move as possible. A routable chain is never touched, and a second pass is a no-op.

**It is only safe because placement ownership shipped first.** Before [P32], a re-registering node
would overwrite whatever the repair decided, and the sweep would have fought the roster every 60
seconds instead of fixing it. That is why the watch shipped as detection-only in the morning and
can repair by evening.

**A blind spot in my own check, found in production.** `chain_shape` counted stages and coverage
but never verified stage 1's WIDTH — so it reported `routable: true` for
`[[0,16],[17,23],[24,27]]`: three stages, 28/28 covered, and refused by every driver.
`DRIVER_STAGE1_LAYERS` now makes that explicit and `stage1_ok` is reported separately, because it
is a different failure with a different remedy.

**Proven live rather than asserted:** the chain was deliberately pushed back into
`[[0,16],[17,23],[24,27]]` and left alone. `[repair] chain was unroutable — reassigned 4 node(s)
to [[0, 9], [10, 18], [19, 27]] (3 stage(s), routable=True)` — **under 15 seconds, no human.**

`coordinator/test_auto_repair.py` **9/9** (new), `test_routability.py` 14 → 16.
**Suite: 75 modules green, 0 failures.**

### Half thirteen — the coordinator forgot which model it served, and "go restart it" is not a fix

Chat failed with `socket closed mid-message`, four retries, `RuntimeError`. The node log gave it
away: `agent-bhpc012104` was serving **Qwen2.5-7B layers 24-27** while the coordinator believed
the network ran 1.5B.

**Cause, and it was mine to trigger.** The 7B migration cut over at 07:06 UTC. I restarted the
coordinator minutes later to disable auto-promotion — and `_serving` lived **only** in a
module-level dict initialised to the config floor. It came back believing 1.5B. Both Qwen2.5
tiers have 28 layers, so every range validated, coverage read 28/28, and the routability check
said healthy, while nodes fed each other activations from different models. **Every signal green,
product broken** — the same shape as the morning's failure, one level deeper. Recorded as [P33].

Fixed: a `settings` table, `set_serving_model` writes through, `_load_serving()` runs at startup
**before** the health loop (the sweep assigns ranges against the serving model, so a coordinator
that has not yet remembered which model it serves hands out ranges for the wrong one), and
startup now logs what it restored. Auto-repair is gated on `migration.phase == "steady"` so it
cannot rewrite ranges mid-cutover and strand half the network on each partition.

**Then the founder asked the question that mattered: "the machine is 100 km away — do I have to
go there?"** No, and telling him to restart four agents was a bad answer. A volunteer's PC is
never reachable; "restart the agent" is not an instruction this product can give at any scale.

The remote-reload machinery already existed and was proven — the migration handshake is how the
network reached 7B that morning without anyone touching a machine. What was missing was any way
to **aim** it. `POST /network/model` (operator) now pins the model and moves every node onto it:
prepare, download, report ready, cut over together. `model_id: null` restores capacity-driven
tiering.

Ordering matters and is worth remembering: the coordinator must first be told the **truth** about
what it serves. Pinning 1.5B while it wrongly believed it already served 1.5B was a no-op —
`update()` only plans when target != serving. Setting serving to 7B (reality), then pinning 1.5B,
produced a real migration.

Used live to walk the network back from 7B to 1.5B **with no physical access to any machine**.

**Also flagged in [P33]:** `canonical_assignment` splits evenly and never consults
`balancer.max_layers_for`. Harmless on the 1.5B floor; on 7B it would have handed an 8 GB office
PC ~8.4 GB of weights — the [P26] hazard, in code I wrote today. Must be memory-aware before any
tier above the floor is served again.

**The gap that made this unfixable remotely, still open:** a node never reports which model it is
serving. Registration carries the layer range but no `model_id`, so the coordinator cannot
detect the mismatch, repair it, or display it. Same class as [P32]'s placement drift, same remedy.

**Suite: 75 modules green, 0 failures.**

### Half fourteen — 0.25 tok/s, and the check that was running but checking nothing

Chat came back at **0.25 tok/s**. The first diagnosis was a replica reporting 4150 ms/layer
against 8-22 for its peers; `REPLICA_SLOWDOWN_LIMIT` now drops outliers before weighting, and
`ms_per_layer` gained a TTL plus hourly re-measurement, because a figure taken once at startup
while a machine thrashed was believed forever — and worse, a node excluded by the new guard
never serves, so nothing could ever revise it. The guard would have made the stale reading a
life sentence.

**That was not the cause.** Real request records: 10 tokens in 15.7s, then 35.0s, then 37.0s —
wild variance for identical work, plus requests completing with **zero tokens**. Not slowness;
failure-and-retry, each retry re-prefilling the whole chain. Relay RTT measured 16 ms, so the
Amsterdam hop was never the problem either.

**The real finding: proof-of-compute was a one-time gate.** The verifier logged
`5 node(s), 0 awaiting verification` on every sweep for hours — running perfectly, checking
nothing, because every node had already passed once. Verification proved each node honest on the
day it joined and never again.

Turning on re-verification (one already-verified node per cycle, oldest first, passes at DEBUG)
found two bad nodes within minutes:

- **`node-c-pavilion`** — deterministic wrong answers, `max_err 33.79`, identical across
  attempts. Wrong weights for its assigned range.
- **`agent-bhpc012101`** — `socket closed mid-message` on every challenge: the exact error the
  driver reports when a chat dies.

The second exposed a further hole: "could not challenge" was inconclusive **forever**. Right for
one hiccup, wrong as a permanent amnesty — a node on the wrong weights fails by *hanging up*, not
by answering wrong, so it accumulated nothing and kept a perfect reputation while breaking every
request it touched. `UNREACHABLE_STRIKES = 5` closes it, and fired live at 13:02.

Also: the verifier could be **killed by its own success message** — `UnicodeEncodeError` on the
`→` in the VERIFIED line via Windows cp1252. It had already died once. A monitor that dies on the
shape of its own output is worse than no monitor.

Recorded as [P35]. The point worth keeping: the network already had the mechanism to exclude a
bad node automatically. It was never asked to run, so the only remedy left was deleting nodes by
hand — which does not scale past machines you can name.

**On speed, honestly.** 0.25 was contaminated by these two nodes. The clean figure while they
were excluded was 0.64 tok/s, and even that is not the product's speed: `ui/app.py` routes any
capable machine to its own CPU, measured this morning at **6.69 tok/s local**. Splitting a 1.5B
model across three PCs buys a network hop and a bottleneck stage, which the code's own comment
says outright. The pipeline exists for models a single machine cannot hold, and the lever there
is **quantization** (fp32 → Q4 is ~3.5x less compute and ~4x less wire), which `SCALING.md:186`
already sketches. Everything else is worth percent.

**Suite: 75 modules green, 0 failures.**

**Open:** run the pipeline quantized (the only multiple-sized speed win); node reports its
`model_id` reaching agents (needs 0.20); deploy emission (built, tested, undeployed); close
[P29]'s ownership gap; `/next` → `/`; cut 0.20; the GPU session.

---

## Session 57 (2026-08-11) — the flags were ours, and the check that could not pass

Opened on a dashboard the founder pasted as the state of the network, and the objection that
came with it: *these flagged PCs are your work, in the name of a fix.* That was correct, and
three separate defects sat behind it.

**First, a number nobody should have believed.** `agent-bhpc012101-18f1da` showed `ms/layer
4146.6` beside peers at 8–20, and it was read as current evidence. `router.stage_ms` had aged
that figure out hours earlier ([P34]) and was scoring the node at the default prior — but the
node table and the node's own dashboard read the raw column with no age check. The operator page
was the worse half: it told that volunteer *"4146.6 ms/layer measured"*, accusing their PC of
being 500× slower than its peers with a number the coordinator itself had discarded. Freshness
now has ONE definition, `router.ms_per_layer_fresh`, called by `stage_ms` and both dashboards.
The table keeps the number and marks it `· stale`; the node's page omits it rather than accuse
the hardware. A NULL `ms_per_layer_at` is legacy, **not** expired — reading it as stale would
reset every pre-column node to the prior, network-wide. This is [P37]'s own lesson, a stale
diagnostic field read as a live one, recurring in the surface a person actually looks at.

**Second, the flags were manufactured, and the founder was right about it.** `RangeMismatch`
already refuses to punish a node that *tells* us it holds something else — but [P37] explains why
that is the minority path: asked for layers it never downloaded, a node raises on uninitialized
meta tensors, `_handle` catches only three exception types, the thread dies and `conn.close()`
slams the socket. So drift reaches the verifier as `socket closed mid-message` — the generic
branch — where [P35]'s `UNREACHABLE_STRIKES` attests it as a real failure every five cycles.
Two correct fixes composed into a machine that generated evidence against honest volunteers.
Live proof: 1/18 → 1/22 in twenty minutes. `placement_drift` predicted all of this on 2026-08-10
and was *"read by nothing"*; it is read by the verifier now, and while it is set neither failure
path records anything. Asymmetric on purpose — passes are still recorded, because a pass proves
the node holds the assigned range and the counters only ever grow.

**Third, and it explains the other half of the table:** `82cbee` sat at 2/4, one pass from
clearing its flag, never re-challenged. The rotation sorts by `last_checked` with a default of
`0.0` and takes one node per cycle — and the unreachable path `continue`d without stamping. A
node that never answers therefore parks itself at the front forever and starves everyone else.
[P35] put flagged nodes on that rotation precisely so they could recover; an unstamped failure
silently denied it.

### The fix that nearly flagged the driver

With the verifier restarted, a new line appeared every sweep: `agent-optinovate-6ff49d:
PLACEMENT MISMATCH — node's actual range (None, 10) does not match the expected (0, 10)`. The
driver. 26/26. `_check_range` tolerates an agent too old to report what it holds; the middle
probe compared the whole tuple for equality instead, and the driver acks `s2` while omitting
`s1`. So proof-of-compute had quietly stopped checking the most important node on the network,
landing in the *"nothing recorded"* branch where it looked harmless — [P35]'s disease returning
through its own cure. **A check that cannot pass is not a strict check, it is a dead one.**

Fixed by comparing only what the node actually claims. That unblocked a path which had **never
once executed** — and pointed it at the driver, which came back `max_err 28.6, strike 1 of 3`.
The verifier was stopped at strike 1. Three would have attested a failure against the only
machine holding stage 1, and a flagged driver is not a degraded network, it is no network at all.

The cause is a real incompatibility: `make_middle_challenge` computes `layers[s1:s2]` on a raw
hidden state with no embedding, while a first-stage node embeds token ids first. `28.6` carries
[P37]'s signature — the right answer to a different question — but that is a hypothesis, and a
guess is not grounds for scoring somebody's machine. Stage-1 nodes are now skipped and **said**
to be skipped, once per process at WARNING: *"Proof-of-compute does not currently cover the
driver."* Deliberately the inverse of [P31]. A node that is stage 1 *and* last stage still goes
down the fully-verified path.

### The probe, and why it settled nothing about 18f1da

Every online node answers `{"ok": true, "s2": <the value we asked for>}` — no `s1`, no `holds`.
`18f1da` cheerfully acked `s1=19, s2=28` and then hung up mid-`act`, which proves it does *not*
hold 19–27. On 0.19 the ack **echoes the request**, so it is not a claim about anything. Its
`reported_layer_*` of `10-13` remains a stale registration field, and re-pinning placement on a
stale field is how this started — so nothing was re-pinned. `holds` (0.20) is the answer.

### Results, measured live

| | before | after |
|---|---|---|
| `18f1da` | 1/18 → 1/22, climbing | **1/23, frozen** — no further false failures |
| `82cbee` | 2/4, never re-checked | **12/14**, recovered to `verified` unaided |
| `node-c-pavilion` | 243/245 | 252/254, checked every rotation |
| driver | silently unverified | unverified **and says so** |

Network `routable: true`, 2 stages, throughout. **Suite: 69 modules green, 0 failures**, up from
67 — `coordinator/test_stale_speed_display.py` (13) and `test_drift_is_not_evidence.py` (9), plus
7 cases in `security/test_proof_of_compute.py`, whose header had recorded this protocol as
manually-verified-only. The ack comparison is pure logic and needs no weights, which is exactly
why it went unpinned long enough to break.

One test written this session was thrown away and rewritten: it asserted `... or True`, which
passes forever and checks nothing — the same failure [P35] logged about the verifier itself.

### Shipped — and the regression that came with it

Everything from sessions 56+ went to the live coordinator: emission, auto-repair, serving-model
persistence, `POST /network/model`, plus the day's four fixes. DB backed up each time, auth gates
re-verified at 401, `/status` OK. `/agent/version` still reads 0.19.0 — the version is pinned by
the VM's environment, so the bump did not leak out and nothing auto-updates until a hash is
published deliberately.

**Then chat fell to 0.05 tok/s, and it was mine.** `DEFAULT_MS_PER_LAYER` is 40.0, and the
`ms_per_layer` TTL reached the live coordinator for the first time in this deploy. It scored
`node-c-pavilion` — 4 cores, 99% reliable, measured at **11.0** — at the 40.0 prior because its
reading was 12.9 h old, while the 8 GB `82cbee` kept its fresher **20.4**. That inverts the
preference between them: `fastest_pick` moved traffic off the reliable machine and onto the box
whose hardware twin now measures 5308 ms/layer. **20 seconds per token.**

The TTL's justification ([P34]: an outlier is excluded, so it never serves, so nothing revises
it) rests on `agent.remeasure_loop` — which ships in **0.20**. Enabling its consumer on a 0.19
fleet means expiry has no second measurement to fall back to: it does not age a bad number back
in, it throws a good one away and substitutes a guess. Routing now uses the measurement whatever
its age; only the public display treats an old figure as unknown. Recovery measured live:
**0.05 → 0.9 tok/s**, against 0.64 as the last clean distributed figure.

Two tests failed on that revert, both asserting the old rule, and both were right to fail. They
now state the new one — including a case reproducing the exact inversion, that `11.0 stale` must
still beat `20.4 fresh`. Revisit when 0.20 is on every node.

**The lesson worth keeping:** the TTL was correct, tested, and reviewed. What made it a
regression was a dependency nobody wrote down — it is a **0.20 feature's consumer**, shipped to a
0.19 network. "Tested" says nothing about which fleet the assumptions hold on.

### Also fixed: an age that measured the wrong thing

`ms_per_layer_at` records the last REGISTRATION, not the last measurement, and the agent re-sends
its cached figure every time — so a node re-asserting a stale number restamped it as current. The
upsert now restamps only when the value actually CHANGES. A re-assertion is not a measurement:
the same shape as [P32], where a node re-states something stale and the coordinator records it as
new. (An earlier claim of mine that this made the TTL universally dead was wrong, and is
corrected in [P34]: the driver's figure was 25.0 h old and pavilion's 12.9 h, so expiry did work
— for nodes that go long enough without re-registering.)

### Not verified

`28.6` is diagnosed by reading the code, not by building a stage-1 challenge and watching it
pass. Until that exists the driver is uncovered. And every fix above is verifier-side and
coordinator-side; the node half has never run on a node.

**Emission is now live and has never been watched.** It settles slots on every health sweep and
writes to the ledger. No `[emission] sweep failed` appeared during the deploys, which is not the
same as confirming the NRN it distributes is correct.

**0.9 tok/s is one measurement of one prompt.** [P34]'s best-path estimate was 2.6; the gap is
that stage 2 runs on 8 GB machines, one of which now self-measures 5308 ms/layer.

## Session 58 (2026-08-16) — the growth tooling, and three deploys that reported success while changing nothing

Built from a written spec (`GROWTH_PLAN.md`, gitignored) rather than from the code, which is
worth recording because most of what the spec asserted about this repo turned out to be wrong.
The tool lives in `~/neuron-growth`, outside this repository: its own venv, no torch, and it
only ever issues unauthenticated `GET` at the coordinator. It is now a systemd unit on the
Oracle VM beside the coordinator, public at **https://status.neuronnet.duckdns.org**.

Three parts: a status page and Atom feed, an announcer that posts to NEURON's own Discord, and
a scout that finds public threads where NEURON is arguably an answer and drafts a reply. **A
human posts. Always.** No third-party platform is ever written to — `tests/test_guard_no_posting.py`
walks the AST of every module and fails the build if a network write appears outside a two-file
allowlist, if a submission endpoint URL appears in any string, or if `scout/sources/reddit.py`
exists at all. Verified the only way that means anything: a Reddit auto-poster was planted and
three separate tests fired on it.

### What the coordinator actually exposes, having read it instead of guessing

- **`GET /node/list` needs no credential.** It returns the roster to an authenticated operator
  and strips `{node_token, tailscale_ip, port, hw_fingerprint, platform, gpu_name}` for anyone
  else. So the growth tool holds no secret at all — not as an oversight, but because a tool that
  cannot obtain the private fields cannot leak them. Its own privacy filter is the second line.
- **There is no `label` column.** `node_id` *is* the public identity, and it is operator-chosen
  free text, so the "hash anything that looks like an address" rule has to be applied to it. A
  node registered as `192.168.1.55` renders as `node-de883800`.
- **No public `os`.** `platform` is withheld from anonymous callers for the same correlation
  reason as the addresses. The spec asked for it in the snapshot; reality won.
- **`total_layers` follows the serving model**, so pinning it at 28 would go stale silently.
- **The OpenAI-compatible `/v1` API is served by `ui/app.py` — the agent's UI on :8080 — not by
  the coordinator.** `neuronnet.duckdns.org/v1/*` is a 404. Its bearer token is an NRN wallet id
  with no secret beside it, verified only for existence and ban status. That means a wallet id
  is a credential, not an identifier: anything holding one can spend that wallet on inference.
  Worth deciding deliberately at 3 nodes rather than at 300.

### A privacy filter that corrupted the thing it was protecting

The scrubber rewrote any string matching a loose IPv6 pattern. An ISO timestamp's clock —
`08:15:23` — matches `(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}` perfectly, so `fetched_at` came back
as `node-680d4327` and the page rendered *Invalid Date*. Caught only because the value was
printed during a live check. The lesson is not the regex: **a privacy filter that mangles
ordinary fields is a filter someone switches off**, and that is when the real leak happens. It
now uses a strict check — every candidate must survive `ipaddress` parsing — with the broader
hostname heuristic reserved for node ids, where it belongs.

### The announcer published something false about the founder's own machines

First live run, straight to Discord: *"A node joined NEURON that the operator does not run."*
The rule fired on any node whose standing was not `trusted`. But the founder's own machines
report `verified` — they joined through open registration and were promoted by proof-of-compute
rather than the register secret. Nothing in the public projection says who owns a node, by
design, so **"a stranger joined" is not a claim this data can support at all.** Narrowed to
`probationary` and reworded to state only what is observable: *a node joined through open
registration*. Same family as the recurring fault in this log — a field read as evidence for
something it does not actually attest.

### Three deploys that said "done" and changed nothing

`systemctl enable --now` does not restart a running unit. Two redeploys therefore copied new
files onto the VM and left the previous build serving them. The consequence was not cosmetic:
the fix that closes `/queue` to the internet was "deployed" twice while the old, open build kept
running, and the scout queue stayed world-readable — the same application serves the public
status page and the operator tools, and nothing had gated them apart.

The deploy script made it worse by *printing* the gate's status codes and then saying `done`
regardless, so a broken deploy was visually identical to a good one. It now `restart`s, and the
gate is an assertion that exits non-zero with `DEPLOY FAILED`. A third run died before reaching
the server entirely: an unquoted heredoc delimiter let the local shell try to expand the remote
script's variables, and `set -u` killed it. **A check that reports instead of failing is not a
check.** That is [P35]'s lesson arriving from a different direction.

### Smaller things, each found only by running it

- HN's Algolia treats a multi-word query as *optional* words: the unquoted search returned the
  firehose, including a story about home batteries and one titled "Asus Bike Booster". Fixed
  with quoted phrases and `advancedSyntax`.
- GitHub search is 10 req/min unauthenticated. It 403'd after three queries and the code
  reported that as **0 results** — "nobody is talking about this" and "I was blocked" are
  different facts and only one of them is news. It now paces and says which happened.
- The keyword `petals` matched a GitHub issue about *cherry-blossom petals* in a UI animation.
  Ambiguous terms now need a companion word.
- The README told the operator to create a `.env`, and nothing read it. Adding a reader then
  broke four tests, because the suite silently inherited the developer's own `.env`.
- The scout queue reported "LLM configured" from the base URL alone, while every call 401'd for
  a missing token and fell back to templates. A banner that lies about draft quality is worse
  than no banner.

### Noted, not fixed — they are in this repo

- The public coordinator reports `coordinator_version: 0.1.0` while this tree is at 0.20.2.
  Either the VM runs an old build or that field is not wired to the real version. 0.20.2 exists
  specifically so a rollback can be confirmed by reading it, which makes this worth checking.
- Three different installer versions are advertised at once: README offers 0.18.0, the landing
  page 0.19.0, the repo is 0.20.2. The download link is what turns a reader into a node, and it
  currently points two releases back — with a pinned SHA-256 for that old build beneath it.

### What it does not do

It does not bring anyone. The status page and the Discord announcements only convert someone
already looking; the scout is the sole part that reaches a stranger, and its last step is a
human deciding the reply genuinely helps and pasting it. There is no lawful automation of that
step, and the deferred work is listed in `neuron-growth/README.md` rather than here.

## Session 59 (2026-08-16) — the Pavilion on Ethernet, and a measurement that had to be taken twice

Session 23 concluded *"the real bottleneck is now the network, not the kernel"* — three
machines at 15%/14%/11% utilisation, idle ~85% of the time waiting on the wire — and made
one recommendation: **put node_c on Ethernet before any further engine work.** Session 24
repeated it. It sat undone for two weeks. It is done now, and the number is larger than the
one that motivated it.

### Plugging the cable in was not what fixed it

`enp2s0` came up at 1000 Mb/s full duplex and then sat in `connecting (getting IP
configuration)` indefinitely. Not a driver fault: DHCP requests went out and the router
answered — with `192.168.1.10`, which the OptiPlex already holds **statically**, because it
is the household DNS server the router itself advertises to every client. NetworkManager's
conflict detection refused the lease, correctly, every 40 seconds.

The router's Static DHCP table had a reservation for `.10` against MAC
`00:11:22:33:44:55`, which reads exactly like a placeholder someone typed once. It is not:
that is the Pavilion's own wired MAC. So the reservation was doing precisely what it was
told, and the mistake was older — `.10` was handed to the Pavilion while the OptiPlex was
already sitting on it. **The router was advertising an address as the DNS server while
simultaneously treating it as free to lease.** Any new wired device would have hit this;
the Pavilion was just the first to ask.

Resolved by giving the Pavilion `192.168.1.11` (`nmcli ... ipv4.method manual`), which the
wired profile takes at route metric 100 against Wi-Fi's 600, so it wins the default route
without disturbing Wi-Fi as a fallback. **The router still does not know about `.11`** — the
reservation row should be moved from `.10` to `.11`, or this recurs the day the pool reaches
it. Removing `.10` from the OptiPlex was considered and rejected: it would have taken
household DNS down for everyone, on a box also running ~60 containers.

### The measurement, taken twice, because the first one was of the wrong thing

ICMP first: Tailscale RTT to the Pavilion fell from a logged 44–148 ms to 0.68–9.6 ms. That
looked like a 30–60× win, so it was checked with the real payload — 12,508 bytes, the fp32
activation size from `bench_wire.py` — round-tripping between the OptiPlex and the Pavilion:

    300 round trips, back to back:
      wired   p50 0.94 ms   stdev 0.14   max 1.89
      wi-fi   p50 1.90 ms   stdev 0.45   max 5.09

Two times, not sixty. That was written down as a correction to Session 23 — **and the
correction was wrong.** 300 back-to-back round trips keep a Wi-Fi radio awake. NEURON never
produces that pattern: each link carries one activation, then waits 100–300 ms while the
node downstream computes. Per link, the traffic is *sparse*, which is the case power-save
punishes. Re-run with a gap between sends:

    60 round trips, paced:
                  gap=0ms    gap=150ms   gap=300ms
      wired        0.81 ms     2.40 ms      4.00 ms
      wi-fi        1.88 ms    54.54 ms    108.71 ms

**At the duty cycle the pipeline actually runs at, Wi-Fi cost 54–109 ms per hop and Ethernet
costs 2.4–4.0 ms.** Session 23 was right; the intermediate "correction" was an artefact of
benchmarking a pattern the system does not have. Two hops per token puts **110–220 ms per
token of pure wire wait** into the old numbers, against a token that cost 300–1100 ms — which
is the same story as "idle ~85% waiting on the wire", arrived at independently.

The lesson is narrower than "measure, don't guess", because both measurements were real: a
link's latency is not a property of the link alone, it is a property of the link **and the
duty cycle you drive it at**. Back-to-back and paced differed by 29× on the same hardware in
the same minute. Same family as [P34] — a number that was true when captured and false when
applied to a different question.

### What this unblocks

Every throughput figure on record was measured across that penalty, including the
`3.2 → 4.6 → 6.2 tok/s` scaling curve and the sub-linearity attributed to heterogeneity and
the head cost. Those should be re-run. Jitter is also down from 23.9 ms to 0.44 ms (mdev,
LAN ping), which is the noise that forced config-by-config interleaving and manufactured the
fake +73% — so for the first time a block-vs-block A/B on this network may be trustworthy.

Not verified: end-to-end tok/s. `/infer` requires an OAuth-linked wallet and holds NRN, so no
generation was run; everything above is the transport measured directly. The engine-side
claim — that GEMM work is now the dominant cost — remains inferred, not observed.

## Session 60 (2026-08-16) — the capacity case, and the precision nobody could declare

The goal was a ~4B model in fp32 across the 12 GB Pavilion and the 8 GB node, excluding the
big 16-core Windows PC: neither can hold it, together they can. The download side was never in doubt —
`slice_downloader.py` has fetched per-tensor byte ranges since Session 8, which is exactly what
Sergio's `spikingbrain-cpu-cluster` shows working on 2012-era hardware. The coordinator's
arithmetic was the blocker, in two places, and the first thing measurement did was move the
target.

### fp32 does not fit, and that is arithmetic rather than conservatism

`tools/measure_model.py` is new and reads a model's published safetensors header — ~40 KB, no
auth — instead of deriving figures by hand in a comment, which is how every number in the tier
table got there. Qwen3-4B-Instruct-2507: 36 layers × 100,930,816 params, embedding 388,956,160
and tied, Apache-2.0, ungated. **16.09 GB at fp32.**

The two machines have 20 GB between them. After the 3 GB OS reserve and the 25% headroom the
coordinator budgets 10.5 GB, and even at zero reserve it is 16.09 GB into 20 GB with nothing
left for two OSes, two Python processes, the KV cache or CastLinear's transient fp32 copy. The
refusal is correct.

**At fp16 storage it fits with room**: 8.04 GB, per-node caps of 29 + 18 against 36 layers
needed. Not a new idea and not a guess — `common.WEIGHT_DTYPE` implements it, every GEMM still
runs in fp32 because these CPUs have no half-precision GEMM ([P2]), and `test_weight_dtype.py`
measured it two sessions' worth of work ago: 2 B/param resident, checksum drift under 1e-2,
~2.9× slower at batch 1 falling to ~1.6× at batch 8. So the capacity claim is true, and the
pitch should say fp16 rather than fp32. Neither machine can hold the model alone either way.

### The head, and a field that was "accepted" nowhere

Two holes, both in the sizing model, both worst in exactly the two-machine case:

**The driver was never charged for the embedding and `lm_head`.** `model_tiers` said so in its
own comment and left the column out for want of a measurement. It is a FIXED cost, so its share
grows as the network shrinks — noise across ten nodes, 41% of an 8 GB machine's budget here.

**A node had no way to say what precision it stores at.** `balancer.weight_bytes_for` has read
`weight_dtype` since the dtype correction shipped and its docstring said the field was
"accepted so the agent-side change is additive". It was accepted nowhere: no `RegisterBody`
field, no column, so it could never reach `_node_dict`. Every node was sized at the pessimistic
4 bytes/param whichever precision it actually ran — safe, and precisely what made this case
impossible to express. A comment describing an integration that does not exist reads exactly
like one describing an integration that does.

Both closed. `head_gb` is charged to one node, on the same fp16 basis as `gb_per_layer` and
through the same `effective_gb`, so the two figures cannot drift onto different bases — which
is the fault one level up that the correction exists to fix.

### Two things the tests found that the code did not

`common.py` lowercased `NEURON_WEIGHT_DTYPE` without stripping it. A trailing space — a systemd
unit, a `.bat`, a copied README line — raised `KeyError: 'fp16 '` at import, before the node
server's logging existed, so an operator got a traceback naming a dict literal. Meanwhile
`agent.weight_dtype()` stripped, reported `fp16`, and the coordinator sized that machine for
half the footprint of a process that was not running. Found only because the agent cannot
import `common` (torch at module scope, and the agent is the ARM-compatible half), so the
mapping is written twice and the test resolves every value through both.

Inserting the 4b tier between `1.5b` and `7b` silently redefined `TIERS[1]` and `TIERS[2]` under
all 26 positional references in `test_model_tiers.py`. Three of them kept passing while
asserting about the wrong tier — the failure the change should have surfaced, concealed by the
same mechanism that caused it. Tiers are addressed by name now. A test that passes for the
wrong reason is the same family as the flags in Session 57 and the `fetched_at` scrub in
Session 58: a value read as evidence for something it does not attest.

### What was deliberately not built

**`ram_free_gb`.** `max_layers_for` prefers it and a comment wished for it, and the wish is a
trap: that branch skips the OS reserve entirely, so a node reporting a free figure is sized
*more generously* than one reporting a total, and the figure is a registration-time snapshot
with no age that nothing re-reads. A machine that registers at 3am idle keeps a 3am-idle budget
all day, with the layers it was handed on that basis still resident when its owner opens a
browser. That is [P34] a third time. The reasoning now sits in the comment that would otherwise
invite it.

**A promotable 4b tier.** At `min_nodes` 2 the live 3-node network clears the 15% promote
margin, and the big Windows PC makes it placeable even at fp32 — so shipping the row plainly
would have migrated production onto a 4B model on the next health sweep. That is the
2026-08-07 auto-promotion arriving from a new direction. `manual_only` makes a tier invisible to
the ladder in both directions: never promoted to, and never the answer a demotion falls back
to, so a shrinking network cannot land on an experiment either.

### It does not run yet, and the reason is a third function

`router.canonical_assignment` caps every stage except the two that most need it — the driver's
capacity is never consulted at all, and the LAST stage takes the whole tail with its cap
deliberately bypassed. On this roster that means the driver takes layers 0–9 (a fixed
`DRIVER_STAGE1_LAYERS`) and the 8 GB node takes the remaining 26, which is 5.25 GB against a
3.75 GB budget. `balancer.solve` proposes 18/18 for the same two machines and `plan_migration`
respects the caps; auto-repair runs last, writes directly, and overrides both. Filed as [P44],
🔴 because it is the automatic path and nothing downstream re-checks what it wrote.

### 68 GB of RAM does not exist, and the roster says it anyway

The founder read the machine sizes back and stopped on one: the OptiPlex is a 64 GB box, and
nothing about "68 GB" is a real specification. It is also not bad data. `agent.py` reports
`int(psutil.virtual_memory().total // 10**9)` — **decimal** GB — while RAM is installed in
**binary** GiB, and 64 GiB is 68.7 decimal GB. The dashboard has been printing 68 since the
first roster (there is a line of it in Session 43's output). Same box, different unit.

Worth writing down for two reasons. The number is load-bearing — every capacity decision in
this session came out of it — and the truncation always rounds down, so every machine is
credited less than it has: the 12 GiB Pavilion is 12.88 GB and gets 12, the 8 GiB node is 8.59
and gets 8. About 3 and 2 layers of Qwen3-4B at fp16, thrown away. The direction is the safe
one and the units are at least self-consistent (the tier table is decimal too), so nothing
here is wrong — but a figure that reads as a typo is a figure people stop trusting, and this
one is the input to every OOM decision the coordinator makes. Recorded as [P43] item 4.

Every capacity figure above is on the reported basis, so the true margins are slightly better.
The fp32 refusal does not move: on true decimal GB the pair holds 24 of 36 layers, not 21.

### The env var that had to be identical on two machines

`node_a.coord_get_chain` refuses any chain whose stage 1 is not `[0, expected_s1 - 1]`, and
that number was read from `NEURON_S1` **at import, in two processes on two different
machines** — `neuron_driver` and `coordinator/config` — each carrying a comment that it had to
match the other. So changing the width of stage 1 meant a coordinated restart of the
coordinator and every driver, including volunteers' PCs nobody can reach. Layer ranges have
the whole prepare→ready→cutover handshake for exactly this problem; `s1` had an environment
variable and a comment.

The constant was deleted rather than distributed. `coord_get_chain` never checked a fixed 10 —
it checks against `expected_s1`, which the caller supplies. So the coordinator publishes the
width on slice-info, the agent fetches a driver shard of that width, and the driver reads the
width back off the shard it loaded. **The shard decides**, and that direction is the load-
bearing one: a driver must assert only what it can serve, because claiming the coordinator's
newer number while holding the old weights would run 10 layers where the chain expects 18 and
hand the next node an activation from the wrong depth — a wrong answer instead of a clean
refusal.

Which then unlocked the thing that was actually wanted: **`s1` is per MODEL now.** 10 is right
for 28 layers over three machines and wrong for 36 over two, where the second node would be
handed 26 layers it cannot hold. A global constant was always the wrong shape for that; it
simply could not vary while two machines had to agree on it by hand. The 4b tier declares 18,
the floor keeps 10, and the capacity case places with **no environment variables set
anywhere**: `pavilion 0-17 / node-b 18-35`, no overflow, routable, and slice-info hands a
driver the same 18 the chain will assert.

Checked rather than assumed, and it does not hold: the migration handshake does **not** cover
the driver shard. It covers a node's compute slice; the driver shard is a separate download
loaded once per process, and `start_local_chat()` runs once at startup with nothing
re-invoking it. A driver whose shard predates a width change refuses every chain until the
agent restarts. Refusing is the safe direction, so for now `coord_get_chain` says *the shard
is stale, restart the agent* rather than printing two ranges at a person. Recorded in [P44]
instead of papered over.

So the honest state: the arithmetic is right, tested, and says yes at fp16. **No forward pass
of a 4B model has been run.** Everything here is a published header plus a dtype measurement
taken on the 1.5B. What remains is a deploy, `NEURON_WEIGHT_DTYPE=fp16` on both nodes, the
OptiPlex out of the roster, a model pin, and 4.41 GB and 3.63 GB downloaded to machines that
have never held a 4B slice. `CAPACITY_CASE.md` is the runbook.

## Session 61 (2026-08-17) — a donation cap, and the claim panel nobody had clicked

Short session, two things, both of them consequences of reading what is actually there rather
than what the plan assumed.

### The capacity case died and came back as a feature

`/node/list`, read instead of assumed: **two machines online.** The 16-core Windows PC (68 GB)
and the 4-core Pavilion (12 GB). There is no 8 GB node. The OptiPlex is not registered and is
not going to be — it holds household DNS and ~60 containers. Session 40-41 found exactly this
and it is still true.

So "run a 4B model across the 12 GB Pavilion and the 8 GB node" could not be run at all:
exclude the big machine and one node is left, and `MIN_PIPELINE_STAGES` is 2. Worse, with a
64 GiB machine in a two-node network **nothing is a capacity case until it is nearly 50 GB** —
the band is 48.8-55.5 GB, a ~13B model at fp32, meaning a 49 GB download onto a 4-core laptop
with the big machine at 100% of its budget.

The way out was not a bigger model, it was a smaller donation. `donation_mode` has always
governed WHEN a node serves; nothing governed HOW MUCH of the machine it commits. Someone with
a 64 GB workstation happy to lend 8 GB had to choose between the whole machine and nothing,
which is the worst trade to put in front of the person most able to help. `donate_ram_gb` is
that setting, and the capacity case falls out of it: the Windows PC capped at 8 GB is an 8 GB
node, and Qwen3-4B at fp16 places 18/18 across it and the Pavilion with neither able to hold 36
layers alone.

It is a DECLARATION the coordinator enforces, not a runtime limiter — the agent reports the
capped figure as `ram_gb`, `max_layers_for` sizes from it, and the node is never handed a slice
bigger than the cap, so there is nothing to police while it runs. Four lines of agent and no
coordinator change at all: the sizing path was already asking the right question and was only
ever being answered with the hardware's number.

### 68 GB is not a machine anyone owns

The founder read the roster back and stopped on it. `agent.py` sends
`psutil.virtual_memory().total // 10**9` — decimal GB — while RAM is installed in binary GiB,
so a 64 GiB machine reports 68.7 truncated to 68. Not bad data, a unit. The truncation always
rounds down, so every node is credited slightly less than it has: the Pavilion is 12.88 GB and
gets 12. About 3 layers of Qwen3-4B at fp16, discarded, in the one number every OOM decision
rests on. [P43] item 4.

Also corrected: this log said the 68 GB machine was the OptiPlex. Session 40-41's own roster
says otherwise — 16 cores is the Windows PC, the OptiPlex is 6.

### The claim panel, finally clicked

[P39] phase 2 shipped a claim panel in chat.html. The React rewrite at `/next` had none, so
swapping the routes would have silently dropped it — invisibly, since it only appears for a
contributor who has not claimed yet.

Ported as behaviour, not DOM. The load-bearing choice: `claimNodeEarnings` takes its provider
as an argument instead of reading `window.ethereum`. The old tests stubbed `window.ethereum`
wholesale and asserted on chat.html's source strings, so connect → challenge → sign → bind had
**never executed anywhere**. Driven in a browser with a wallet that signs and one that refuses:
declining reports "nothing changed", leaves the button live, and posts nothing; signing posts
only the bind, with a body of exactly `{address, nonce, signature}` — checked live, because
"the page cannot name somebody else as the owner" is the property the phase exists for.

`npm run dev` served a signed-out machine with no node, so neither the wallet rows nor this
panel could be looked at without an agent, a coordinator and an OAuth round trip. That is a
large part of why it had never been clicked, and a dev-only mock middleware now fixes it.

Not done: a real MetaMask signature, and the desktop rebuild.

### Deployed, 0.20.3 released, and the first node updated

The coordinator is live on the new code and the chain never moved: `driver_stage1_layers`
appears on slice-info (it exists nowhere else), `[[0,9],[10,27]]` unchanged, routable, healthy.
`tools/predeploy_check.py` predicted exactly that beforehand by running the new
`canonical_assignment` against the live roster, which is the check worth having — a deploy that
re-splits the chain costs every node a re-download and takes chat down while they finish.

Four things went wrong on the way, none of them in the code being deployed:

- **`bash` from cmd.exe is WSL on this machine**, not Git Bash. `$HOME=/home/user1`, the key is
  at `/mnt/c/...`, `cygpath` does not exist and `%USERPROFILE%` is not inherited. `deploy.sh`
  resolved the key from `$HOME` and reported `Permission denied (publickey)` — a message that
  sends you to the VM's `authorized_keys` for a fault entirely on the local side. Two attempts
  to fix it failed, the second because `/[a-z]/Users/...` never globs in Git Bash either: `/c`
  is a virtual mount the root does not enumerate. It now checks every layout and prints
  `uname -s`, `$HOME` and every path tried when it still cannot find a key.
- **`--dry-run` passed both times.** It never opens an ssh connection, so the one step that
  would catch a key problem is the one the rehearsal skips.
- **`zz-agent-release.conf`** won on lexical order over the `agent-sha.conf` drop-in, so the
  coordinator kept publishing 0.20.2 and its old hash after the deploy. Harmless — it meant the
  fleet never entered the daily decline loop an empty hash would have caused — but it took a
  `systemctl show` to see, because a shadowed drop-in looks identical to one that did not apply.
- **`agent/config.json` was tracked in git** ([P45]), and updating the Pavilion meant turning a
  copied directory into a checkout. One `git checkout -f` from wiping the node_token for ~213
  NRN. Found by asking what the command would overwrite rather than running it.

The Pavilion is now on 0.20.3, and its first log is the best evidence of the session:
`cpu x86_64: avx2 ok` — [P41]'s probe on real Linux hardware, correct, and not blocking a
working node — followed by a config migration that picked up `donate_ram_gb`, and
`weight_dtype=fp32` arriving at the coordinator for the first time. That field has been read by
the balancer for weeks with nothing able to reach it.

It also re-downloaded its 1.69 GB slice, because the slice predated the marker file and [P36]
treats provenance that cannot be established as a mismatch. Working as intended, and it will
not recur.

The Windows PC is still on 0.20.2 and is the frozen build with `auto_update` on, so it should
install 0.20.3 by itself. Nothing has ever exercised that path against a real release.

## Session 62 (2026-08-17) — the emission reconciliation, and the walk-back that could never fire

One thing: [P40]. 262.89 NRN distributed, never once checked against the rows that authorised
it. `coordinator/reconcile_emission.py` is that check — read-only, exits 1 on a discrepancy.

### Writing it found the bug before it was pointed at anything

`close_slots` claims the attendance row before it moves the money. That ordering is deliberate
and right: claim-then-pay can at worst pay nothing for a claimed slot, pay-then-claim can pay
twice. The walk-back for that "at worst" was `settle_attendance(entry, slot, 0.0)` — and it
**could never have fired even once**, because that UPDATE carries `WHERE paid_at IS NULL` and
the successful claim four lines above has just falsified it. A payment the pool could not make
left the row saying it had been paid in full, forever, and `emitted_since` — which is what the
daily cap is read against — counted NRN that never moved.

The comment above the line said *"the row stays settled at 0"*. It stayed settled at the full
reward. Nobody had reached the branch, because the pool holds ~600M against 262 spent.

`models.void_settlement` is the fix, and it is its own function rather than a `force` flag on
`settle_attendance` on purpose: that function's entire guard is the `paid_at IS NULL` a flag
would switch off, and the guard is what makes a replayed sweep a no-op instead of a second
payment. Voiding can only ever move a reward DOWN on a row that is already final.

Found by writing the reconciliation and asking what would make its two legs disagree. That is
the argument for the whole exercise: the check paid for itself before it ran.

### Two of the three numbers were weaker than the plan assumed

[P40] said sum the settled rewards, compare against the pool's drop and the rise in
`total_earned`, and the three must agree. Reading the code first ([P43]'s lesson, again):

- **The pool's seed is not stored anywhere.** `genesis.seed_genesis` computes
  `600,000,000 − already_minted` and keeps only the result as a balance, so "the drop" is not
  computable from the live database at all. `genesis.py` now records it — which helps every
  future database and not this one. For the live one the repository already held the answer:
  `backups-offbox/neuron-20260802-115534.db` has no `attendance` table, so it predates every
  emission payment, so its pool balance **599999971.999972** *is* the seed. A test asserts both
  halves of that against the file. It also implies `already_minted = 28.000028` at genesis
  against 28.678027 of node earnings five days later — consistent, and the first independent
  corroboration that figure has had.
- **`total_earned` cannot be decomposed.** No transactions table, and it mixes emission,
  per-request earnings, the faucet and operator sweeps — and `claim_node_earnings.py` credits
  the destination without decrementing the source, so summing it across the ledger
  double-counts every swept NRN, the 251 included. Leg C is a per-payee floor, not an equality,
  and it says so. Two attribution hazards are reported rather than guessed: [P39] phase 3 pays
  `get_node_owner(node) or node` **decided at settle time**, and nothing records when a link
  was made; and `delete_node` removes a node's row while leaving its attendance and ledger, so
  attendance can outlive the machine it belonged to.

Where a leg is inferred, the script says INFERRED. Without a seed, leg B equals leg A by
construction, and reporting that as agreement would be the check lying about its own strength.

### The parts that matter more than the sums

A settled row is frozen — `touch_node` and `mark_slot_poc` both carry `WHERE paid_at IS NULL` —
so `--replay` can re-price every slot from its own rows and diff. Asserted directly rather than
assumed, with a test that heartbeats a settled row and checks nothing moved. The one input
settlement does *not* freeze is `total_layers`, so rows whose verdict depends on the serving
model are named instead of silently priced against today's.

Plus the checks a sum cannot see: rewards above the `base × scarcity_max` ceiling, rows paid
against their own failed qualification gates, a payee with no ledger row, and closed slots
nobody settled — which is invisible in the logs, since an idle sweep returns early and stays
quiet.

The A↔B tolerance scales with the arithmetic done. One ULP of a 6e8 float is ~1.2e-7, so a few
hundred payments can honestly disagree by ~3e-5; a fixed 1e-6 epsilon would have reported a
discrepancy on the first clean run and cost an afternoon.

### Tested against the real code, not around it

37 tests. The load-bearing one builds a 201-row, three-day, three-node ledger by driving
`models` and `close_slots` exactly as production does, then reconciles the resulting file: 585
NRN, all three legs agreeing, replay clean. Every discrepancy the script claims to detect is
then written into a hand-built database and fired — a detector nobody has seen fire is not a
detector, which is [P40]'s own thesis applied to [P40]'s own fix. The pricing copy the script
carries (so it runs against a snapshot without importing the live config) is pinned equal to
`emission.plan_slot` over 200 randomised slots.

### Then it was run, and 330.43 NRN checks out

The founder ran it on the VM. `/home/ubuntu/neuron/coordinator/neuron.db`, 318 attendance rows,
316 settled, 118 paying, 2 unsettled — the slot that had just closed.

    A  recorded   330.427894 NRN
    B  left pool  330.4278938 NRN
    C  arrived    321.741023 + 8.686871 orphaned = 330.427894 NRN

The A↔B gap is 1.4e-5, against a tolerance of the same order — float residue from ~118
decrements of a 6e8 balance. Deriving that tolerance from the arithmetic actually performed
rather than fixing it at 1e-6 is what stopped the first honest run reporting a discrepancy.

**All 316 settled rows re-priced correctly under `--replay`.** Not just that the totals add up:
each individual payment, re-derived from its own frozen inputs including the replica depth
within its slot, matched what was paid. No mispricing anywhere in the history.

And the seed stopped being an inference. 599,999,971.999972 from the 2026-08-02 backup minus
the settled rewards lands on the live pool balance to five decimals. If that backup had not
predated emission, leg B would have been wrong by the difference. Production confirmed a file
date.

Two things fell out that are not accounting errors:

- **8.686871 NRN paid to a machine that no longer exists.** `agent-bhpc012104-82cbee`, one of
  [P37]'s three, was deleted — and `delete_node` removes only the `nodes` row, leaving the
  attendance and the ledger. The reconciler declined to attribute it rather than guessing,
  which is that branch firing on real data for the first time.

  I first wrote this up as stranded NRN. It isn't: the second run printed the ledger row, which
  reads `balance 0.000000, total_earned 8.787881`, and since `total_earned` never decrements
  that pairing means the balance was moved out — sweepable because `claim_node_earnings.py`
  reads the ledger rather than `nodes`. The finding said *"payee unknowable"*, which is a claim
  about attribution, and I read it as a claim about the money. Corrected in [P39] item 5, where
  the real residue is the mechanism: it worked out only because that node was the founder's.
- **199 of 318 settled node-hours paid nothing, and the output could not say why.** Every one
  failed a gate rather than being zeroed by the cap (`zeroed-though-qualified` never fired), so
  it is emission working. Added `why-hours-earned-nothing`. The reason strings interpolate a
  percentage and so cannot be counted; `qualifies` now returns a stable code alongside the
  wording the equivalence test pins against `plan_slot`.

The breakdown was 168 no-proof-of-compute to 31 below-attendance-floor, and `node-c-pavilion`
carries 64 of the former with **zero** of the latter — reliably online, reliably unpaid. That
should be impossible: `verify_service` re-checks one node per 60-second cycle, oldest first, so
on a two-node roster every machine is challenged 15–30 times an hour. It fits [P37] instead —
flagged nodes were excluded from both verifier lists, so they could never be re-challenged and
therefore could never earn — fixed 2026-08-10.

Which makes the question *when*, and a count cannot answer it. Each code now carries the window
its hours fall in.

### The dates said "now", and the cause is structural — [P47]

`agent-optinovate-6ff49d`: 81 missed hours running to **2026-08-17 13:00**, the most recent
closed slot. Not [P37], not a verifier outage. `verify_service.py:305` skips stage-1 nodes
outright, because `make_middle_challenge` computes layers on a raw hidden state while a
first-stage node embeds token ids first — so the driver would answer "wrong" every time. That
skip is right, and it is logged loudly.

What nobody joined up: emission pays only on a passing challenge (`mark_slot_poc`), so a node
that is never challenged can never earn an availability hour. **The driver has been unable to
earn emission for its entire existence** — ~243 NRN at the current multiplier, against 333 NRN
distributed in total. And [P43] already established the driver carries the largest fixed cost on
the network. The machine asked to give the most is the one the reward cannot reach.

Second cause, which likely explains the Pavilion's 64: every branch where the verifier
deliberately declines to conclude anything — `RangeMismatch` (*"THE NODE IS FINE; THE PLACEMENT
IS STALE"*), `ChallengeRefused` (*"healthy, nothing recorded"*), pre-strike unreachability —
leaves `poc_ok` at 0, and emission cannot tell that from a failure. Those branches were written
to protect a node's reputation after [P37]. None of them protects its earnings.

Third: `/node/{id}/peer-attest` never calls `mark_slot_poc`, so peer verification firing for the
first time would not fix the driver either.

None of this was visible until the zeros were counted per node and dated. The reconciliation was
built to check arithmetic and found a product bug instead.

Also caught while adding it: `d.setdefault(k, {})[c] = d[k].get(c, 0) + 1` evaluates its
right-hand side first, so the lookup runs before the key exists. Python's assignment order,
and it only shows on the first row for each node.

39 tests now.

## Known limits / next steps
- **The 3.2 / 4.6 / 6.2 tok/s scaling curve predates Ethernet** and was measured with
  54–109 ms of Wi-Fi power-save latency on every node_c hop (Session 59). The sub-linearity
  attributed to heterogeneity and node_a's head cost may be substantially that instead.
  Re-run before optimising against it.
- ~~**Emission has never been observed, and it is distributing NRN now**~~ — [P40]
  **reconciled on the live ledger 2026-08-17**: 330.427894 NRN, three legs agreeing, all
  316 settled rows re-pricing correctly. Re-run it after any change to emission or the
  ledger; it takes seconds and exits non-zero on a discrepancy:

      python3 /tmp/reconcile_emission.py --db /home/ubuntu/neuron/coordinator/neuron.db --seed-from-backup --replay

  What remains is [P40] item 2 — running it as a standing assertion rather than by hand.
- **Throughput scales with nodes (single 3.2 → 2-node 4.6 → 3-node 6.2 tok/s), but
  sub-linearly** because the nodes are heterogeneous and node_a carries the fixed
  head/orchestration cost. Next wins:
  - **Heterogeneity-aware auto-balance** — measure each node's per-layer ms at
    startup and solve for the split that equalises stage time (accounting for the
    head on node_a). On this trio that's ~9/9/10; it will differ per hardware set.
  - **Offload the head** — the `lm_head` GEMM pins node_a. A dedicated head node, or
    sharding the vocab projection, would free the driver to carry more layers.
  - **A model too big for one node** (the capacity case) — sized and tiered as of
    Session 60, NOT yet run. Qwen3-4B is 16.09 GB at fp32 and does not fit across
    the 12 GB + 8 GB pair; at fp16 storage it is 8.04 GB and does, 29 + 18 layer
    slots against 36. Placement is done and needs no env var — the 4b tier
    declares `stage1_layers: 18` and the driver follows it ([P44]). To run it:
    deploy, `NEURON_WEIGHT_DTYPE=fp16` on both nodes, the OptiPlex out of the
    roster, pin the model. `CAPACITY_CASE.md` is the runbook.
  - **Dynamic layer assignment** as nodes join/leave.
  - **More concurrency** keeps lifting throughput until every node is ~100% (N 4→8
    took 5.86→6.16); a bigger connection backlog lets more clients queue.
- CPU-only, fp32. bf16 is slower here (no CPU bf16 GEMM). True int4/int8 speedup =
  the llama.cpp/GGUF path, which can't do this hand-written layer split.
- Provisioning a new node needs a one-time manual hand-off (SSH pubkey + a sudo
  `apt install pythonX.Y-venv`); everything after that is automated.
