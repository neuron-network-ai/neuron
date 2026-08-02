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
