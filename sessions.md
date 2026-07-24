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
| Reach (Tailscale) | — | `ssh raman@100.79.125.112` | `ssh homeadmin@100.114.189.46` |
| Port | connects out | listens **50999** | listens **50999** |

Stack on all three: `torch==2.4.1` (CPU), `transformers==4.44.2`, `accelerate`. The
Python minor version need NOT match across nodes — only the torch/transformers
versions must, since that's what makes the pickled tensors compatible over TCP.

> Note: Claude Code runs **natively on Windows** here, not in WSL2. The OptiPlex's
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

## How to run

**1. Start the last stage (OptiPlex) and the middle stage (Pavilion).** Shards load
on first connect.
```bash
ssh homeadmin@100.114.189.46 "cd ~/neuron && ./.venv/bin/python node_b.py --port 50999"
```
```bash
ssh raman@100.79.125.112 "cd ~/neuron && ./.venv/bin/python node_c.py --port 50999"
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
ssh homeadmin@100.114.189.46 "cd ~/neuron-coordinator && ~/neuron/.venv/bin/python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8001"
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
ssh homeadmin@100.114.189.46 "pkill -f '[n]ode_b.py'"
```
```bash
ssh raman@100.79.125.112 "pkill -f '[n]ode_c.py'"
```

---

## Known limits / next steps
- **Throughput scales with nodes (single 3.2 → 2-node 4.6 → 3-node 6.2 tok/s), but
  sub-linearly** because the nodes are heterogeneous and node_a carries the fixed
  head/orchestration cost. Next wins:
  - **Heterogeneity-aware auto-balance** — measure each node's per-layer ms at
    startup and solve for the split that equalises stage time (accounting for the
    head on node_a). On this trio that's ~9/9/10; it will differ per hardware set.
  - **Offload the head** — the `lm_head` GEMM pins node_a. A dedicated head node, or
    sharding the vocab projection, would free the driver to carry more layers.
  - **A model too big for one node** (the capacity case), and dynamic layer
    assignment as nodes join/leave.
  - **More concurrency** keeps lifting throughput until every node is ~100% (N 4→8
    took 5.86→6.16); a bigger connection backlog lets more clients queue.
- CPU-only, fp32. bf16 is slower here (no CPU bf16 GEMM). True int4/int8 speedup =
  the llama.cpp/GGUF path, which can't do this hand-written layer split.
- Provisioning a new node needs a one-time manual hand-off (SSH pubkey + a sudo
  `apt install pythonX.Y-venv`); everything after that is automated.
