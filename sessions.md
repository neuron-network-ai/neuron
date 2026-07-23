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
