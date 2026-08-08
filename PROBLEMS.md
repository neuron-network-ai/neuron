# NEURON — Problems, Risks & Decisions

Living log of open problems, risks, and the decisions taken about them. Complements
`ROADMAP.md` (the plan) and `sessions.md` (what was built). Update as things change.

Status keys: 🔴 open/unaddressed · 🟡 mitigation known, not done · 🟢 resolved · 🔵 design note

---

## Decisions log

- **2026-07-24 — Speed spike before scaling the roadmap.** The founder is counting on
  per-user speed for the Green AI thesis. Rather than complete Sessions 12–20 on an
  unverified speed assumption, OR prematurely productionise quantization before there
  are real users, run a **cheap measurement spike** (int8 vs fp32 on real hardware) to
  find the true speed ceiling with data. Measurement only — not shipped. Then resume
  Session 12 (first stranger node). See [P1], [P2]. Result recorded under [P2].
- **2026-07-24 — Spike outcome & go-forward.** int8 = **3.46× faster** (speed ceiling is
  fine) but naive int8 **broke quality** [P9]. Decision: **thesis de-risked → resume the
  roadmap at Session 12** as planned; do NOT integrate quantization now. Log a dedicated
  *quality-preserving quantization* session (GPTQ/AWQ or llama.cpp GGUF+RPC) as prioritized
  future work — evaluate the llama.cpp engine pivot deliberately, with real users in view.
- **2026-08-08 — Rent a cloud GPU rather than keep deferring the GPU path.** Every remaining
  GPU item is blocked on the same thing: **no machine in this project has an NVIDIA card**, so
  not one line of the CUDA path can be executed, let alone measured. That is exactly how
  [P31] happened — a capability built, never run, then documented as though it had been — and
  it is not a gap that more code review closes. Decision: **hire GPU time online** and do the
  work on real hardware.
  What that session must cover, in order, because the order is what keeps it safe:
  1. Verify **Phase 4a** on CUDA. It is written and bit-identical on CPU (`test_batching`'s
     figure is unchanged to the digit), but "passes on CPU" is not "works on a GPU".
  2. Then **Phase 4b** — the one-line device move at `slice_downloader.py:299`.
     `test_device_path.py` fails until it is done deliberately, and says what to verify first.
  3. Reconcile **`selftest_shard.py`**, which requires `diff == 0` between a CPU `load_model()`
     and a CUDA `load_model_shard()` — as written, **build rule 6 is unsatisfiable on a GPU
     box**, so it fails on the first machine that could actually prove the feature works.
  4. Only then **`balancer.GPU_EXECUTION`** back on, with the VRAM reserve and bound that are
     already in place and dormant.
  5. **CUDA packaging last**, and it is its own problem: ~2.4 GB of torch against today's
     207 MB installer, `updater.DOWNLOAD_TIMEOUT = 600` (needing a sustained ~32 Mbit/s), no
     resume, no disk-space check, no rollback (`updater.py:173` is `os._exit(0)`), a
     `/agent/version` that cannot express a per-capability build, and NVIDIA redistributables
     that `tools/gen_notices.py` does not cover. **With one installer, every volunteer pays
     that download, including the ones with no GPU.** Fix the updater before shipping it.
  Rented time is cheap next to the lever: decode is bandwidth-bound at ~30 GB/s on DDR against
  360–1000 GB/s on a consumer card, and 200B needs ~30–80 CPU machines or ~10 GPU ones.
- **2026-07-28 — Pre-launch audit before the first real external stranger.** No real
  stranger has ever run a node — everything "live" so far is the founder's own machines.
  Before handing the installer to an actual friend, ran three parallel audits (security,
  economics, reliability) across the current coordinator/agent/relay code, independently
  verified every finding against the real code (not just the audit's claims), and fixed
  everything ranked "must-fix before a real stranger joins." See [P14]. Also added an
  honest first-run disclosure (earnings have no cash value; why Windows may flag the
  unsigned installer) to `INSTALL.md` and to the installer itself (`packaging/DISCLOSURE.txt`,
  shown as an Inno Setup `InfoBeforeFile` page) — a friend who only runs the exe and never
  reads `INSTALL.md` still sees it.

---

## Problems & risks

### [P31] 🟡 GPU support was shipped, documented, and had never once executed — partly fixed (2026-08-08)

**The capability announced in v0.18.0 does not exist in the binary that announced it.** Verified
2026-08-08 against the installed package, not inferred:

| claim | reality |
|---|---|
| `common.py` moves a shard onto CUDA | true, and **unreached**: `slice_downloader.py:299` returns `cast_linears(model)` with no device move, and `agent/node_server.py` uses that loader. `common.move_model_to_device` has **one caller repo-wide** (`common.py:256`), on the bench/verifier path. |
| torch can use the card | `dist/neuron-agent/_internal/torch/version.py:4` is `2.4.1+cpu`, `cuda = None`. |
| `local_gguf` offloads every layer | `llama_supports_gpu_offload()` → **False**; the package ships ggml-base/ggml-cpu/ggml/llama/mtmd and **no `ggml-cuda`**. `n_gpu_layers=-1` was accepted and silently ignored while the log said "offloading every layer". |
| `agent.log` says `device: cuda:0` | comes from `common.device_name()`, whose only caller is `common.py:257` — **a line no agent can print**. `INSTALL.md` asked first-GPU volunteers to send it. |

**The one that was a live hazard, not just a false claim.** `balancer.GPU_EXECUTION = True` sized
a GPU node by VRAM, with **no OS reserve** on that branch while the RAM branch reserved 3 GB. The
weights were in system RAM the whole time. A volunteer with a **12 GB card and 8 GB of RAM** was
eligible for **19 layers** of Qwen2.5-7B — 8.9 GB at the tier's fp16 basis, ~17.7 GB at the fp32
the runtime actually defaults to. That is the OOM `max_layers_for` exists to prevent, produced by
the function that prevents it. `coordinator/test_gpu_capability.py:108-142` **asserted the harm**
(24 GB ⇒ 18 layers), so the suite was green throughout.

**Fixed (2026-08-08):**

1. **`GPU_EXECUTION = False`.** Ships by **coordinator restart alone** — every already-installed
   0.18 agent stops being over-assigned with no installer and no agent release.
2. **`sane_vram_gb()` + `VRAM_OS_RESERVE_GB = 1.5`**, so re-enabling later is safe rather than a
   second guess. `main.py` clamps the field at the registration edge — **to `None`, never a
   rejection**, since a 422 over a cosmetic field is [P24]'s failure class.
3. **VRAM/name follow `has_gpu`** in `models.py` instead of being COALESCEd, so a node that loses
   its card stops carrying phantom VRAM forever.
4. **The offload gate**, checked before the hardware probe and applied to `NEURON_GPU_LAYERS`
   too — an override that silently does nothing is the same bug with a manual trigger.
5. **`weights on <device>`** on every slice load, read off the loaded tensors, never from
   `common.DEVICE`. The gap between configured intent and where the bytes actually are is what
   let this survive a whole release, so the log now reports the bytes.
6. **Corrections** in `RELEASE_NOTES_v0.18.0.md` (dated block, published text left intact),
   `CHANGELOG.md`, `INSTALL.md`, and the five source comments that still said "CPU-only pipeline"
   while `GPU_EXECUTION` was `True`.

**Also fixed (2026-08-08, second pass):**

7. **The device path now follows the activations**, so fixing the loader would no longer mean
   crash-on-first-token. `batching.py`'s auxiliary tensors (KV padding, mask, position_ids,
   cache_position) derive their device from `hidden`/`k`; the batched stages return CPU like
   their unbatched twins in `common.py` already did — that divergence was the bug, since a real
   chain serves through the batcher and never touches the unbatched path; `wire_codec.py`'s
   Hadamard matrix follows its operand and every `.numpy()` reaches CPU first; and every
   `send_msg` caller passes `common._to_cpu`, which fixes `common.py:474-482`'s legacy
   `torch.save` **from the callers** without editing `common.py` (build rule 7).
   **Provably inert on CPU:** `test_batching.py`'s batched-vs-sequential figure is `4.530e-06`
   before the change and `4.530e-06` after, to the digit.
8. **`torch.cuda.synchronize()` in the self-benchmark.** CUDA kernels are queued, not run, so
   the old timing measured how fast a machine can *enqueue* work. A GPU node would have reported
   an absurd `ms_per_layer`, and `balancer.solve` would have handed the fastest-looking machine
   in the network nearly every layer — the same OOM by a second route, through the speed field
   instead of the memory one.
9. **A tripwire instead of a fix for the loader.** `slice_downloader.py:299` is deliberately
   left alone and now carries a comment saying why; root `test_device_path.py` fails if it
   starts moving weights, and says what must be verified on real hardware first. It also fails
   if `GPU_EXECUTION` is switched on while the loader still leaves weights in system RAM —
   those two facts are a pair, and splitting them is what caused this entry.

10. **The fp32 sizing gap — every node on every tier was cleared for twice its real footprint.**
    Not a GPU problem at all, and the largest of the lot. `model_tiers.gb_per_layer` is computed
    at fp16 (2 bytes/param) and its comment justified that with "common.WEIGHT_DTYPE=fp16".
    **That is false**: `common.py:62` reads `NEURON_WEIGHT_DTYPE` and defaults to **fp32**,
    nothing in the agent or installer sets it, and `cast_linears` is a no-op at fp32. So an
    8 GB machine on the 7B tier was cleared for 8 layers = **7.46 GB of weights against a
    3.75 GB budget**. Latent only because the network serves the 1.5B, where 28 layers is too
    few for the cap to bind. Fixed by `balancer.effective_gb_per_layer`, which scales the
    tier's figure by `weight_bytes_for(node)` — pessimistic (fp32) by default, and honest for a
    node once it reports `weight_dtype`. Deliberately NOT fixed by doubling the table: a
    layer's size is a property of the MODEL, the dtype is a property of the NODE.
    `coordinator/test_weight_dtype_sizing.py` 28/28.
    - **What it revealed about the fleet:** the live trio (8/8/12 GB) holds **15 of the 28
      layers** a 7B needs at fp32 and therefore cannot serve it. Two test fixtures asserted
      that it could. They were resized to machines that genuinely fit rather than pinned to
      fp16, because pinning would have preserved exactly the kind of false premise that caused
      this entry. `test_migration` 49/49 and `test_model_tiers` 23/23 with no dtype pin.
    - **No live impact from deploying it:** the network is on the 1.5B floor with one online
      node, and the 7B already reads `feasible: false, placeable: false`. The fix changes what
      the ladder will clear as machines join, not what is served today.

**Still open:**
- **Phase 4b — the loader itself.** One line, guarded by the tripwire above. It needs real GPU
  hardware to verify, because "passes on CPU" is not "works on a GPU" and that distinction is
  the whole subject of this entry. **Decided 2026-08-08: rent GPU time online rather than wait
  for a card** — see the decisions log for the order that session must follow, which is not
  arbitrary: 4a verified before 4b, `selftest_shard` reconciled before `GPU_EXECUTION` goes
  back on, packaging last.
- **`selftest_shard.py`** compares a CPU `load_model()` against a CUDA `load_model_shard()` and
  requires `diff == 0`, so **on a GPU box build rule 6 is unsatisfiable** as written.
- **CUDA packaging.** With one installer, bundling a CUDA torch means every volunteer downloads
  ~2.5 GB against today's 206.6 MB, and `updater.py` has a 600 s timeout, no resume, no
  disk-space check and no rollback. `/agent/version` cannot express a per-capability build, and
  NVIDIA redistributables are not covered by `tools/gen_notices.py` — a real licensing gap.
- **The real lever is still unbuilt.** Decode is bandwidth-bound at ~30 GB/s on DDR against
  360–1000 GB/s on a consumer card; 200B needs ~30–80 CPU machines or ~10 GPU ones. See [P30].

**The process finding worth keeping.** Every individual caveat in Session 42 was honest — "the
CUDA path has never run", "no speedup is claimed" — and the conclusion drawn from them was still
wrong, because they all blamed the **absent test card** (a hardware gap, which reads as
"untested") when the cause was **packaging** (which reads as "impossible"). "Written but
unverified" and "cannot execute in this binary" are different claims, and only the second one
tells you not to size a volunteer's memory by it.

### [P30] 🔴 The fast engine cannot reach the models NEURON exists to serve

`agent/node_server.py` runs **PyTorch fp32** (`load_slice_model`). `llama_cpp` appears only on
driver/local paths — `local_gguf.py`, `local_chat.py`, `openai_compat.py`, `node_a.py`,
`neuron_driver.py`, `ui/app.py`. The local engine only triggers when a machine can hold the
**whole** model, and a 200B model never fits on anyone's machine by definition. **So the ~17×
engine is structurally unreachable for exactly the models the network exists for.**

Decode is memory-bandwidth bound; this project's own three benchmarks (1.5B, 7B, 70B — 47×
apart) all imply ~30 GB/s, which is the DDR bus. Projected 200B across 80 machines: **~32
s/token** on fp32, **~2-5 s/token** with llama.cpp-class kernels. `PIPELINE.md`'s stated
ceiling of "80 hops ≈ 2.4 s/token" counts network traversals **only** — compute is ~15× that
and dominates, so its steps 3 and 4 optimise the smaller term. The kernel decides 200B, not
the split.

**Blocked on a missing capability, verified 2026-08-07 against the installed package:**
`llama-cpp-python` **0.3.34** does **not** expose `rpc_servers` on `Llama.__init__`, and offers
no layer-range entry point — `Llama` is a whole-model abstraction. The routes are therefore:

1. a llama-cpp-python build that exposes the RPC backend, or the standalone `rpc-server` binary
   driven out-of-process;
2. C++ against `ggml` directly, embedding llama.cpp and speaking NEURON's existing wire
   protocol — which would also remove Python and PyTorch from volunteer machines entirely;
3. stay on PyTorch for the network and accept ~32 s/token at 200B.

Route 2 is the one worth doing: it owns the layer that is NEURON's (the wire protocol,
placement, economics) and borrows the layer that is not (the kernel). **Do not write another
matmul** — this repo has measured a hand-written AVX2 int8 kernel at **1.44×** against
llama.cpp's ~17×, on the same CPU, in the same language.

**GPU is the larger multiplier and is half-done.** `has_gpu`/`gpu_vram_gb`/`gpu_name` are in
the coordinator schema, `agent/gpu.py` detects, and as of 2026-08-07 `local_gguf` offloads to
VRAM. But the *network* path still cannot use a GPU, for the same reason above. With GPUs the
arithmetic changes completely — 200B on ten RTX 3060s is ~3 tok/s on **8× fewer machines** than
the CPU fleet — and network latency then becomes the bottleneck, which **flips `PIPELINE.md`'s
build order**: the direct-reply and single-codec fixes stop being premature the moment the
first GPU joins.

### [P29] 🔴 A user who runs out of NRN has no way to get more

`/infer` holds ~0.158 NRN before it dispatches, so a wallet below that is refused before a
chain is built. The faucet is **one-time per wallet** (409 on re-claim) and gated on
`is_oauth_wallet`. Node earnings land in a *separate* ledger row with no route into a user
wallet. So the sequence for any real user is: sign in, get 25 NRN, spend it over ~158
messages, and then the product stops permanently with no action available.

Hit on 2026-08-07 by the founder's own two wallets — both at `balance 0.0` with `total_earned`
25.0 and 25.962. Unblocked by hand: `models.transfer('__ecosystem__', <wallet>, 25.0)`, which
moves rather than mints (supply invariant still exactly 1,000,000,000.0). That is an operator
action over SSH, not a product.

The UI now at least *says* so before the message is sent, and invites the user to contribute a
machine — but earning by donating hardware credits the **node**, not the wallet, so the
invitation does not yet actually solve the problem it points at. **Either node earnings need a
path into the owner's wallet, or the faucet needs a recurring allowance.** Decide before
strangers arrive, not after.

### [P28] 🟡 A node that is merely slow is reported to the user as a node that died

`common.HOT_TIMEOUT_S = 30` applies per socket read during generation. On 2026-08-07 the first
real distributed answer tripped it — driver log `TimeoutError: timed out` — and the driver
classified the timeout as a dead node, tore down the chain and re-prefilled. The reply carried
`↻ recovered from 1 node drop`. **Nothing dropped.**

Three costs. The user is told a machine failed when none did, which is the same class of
misdirection as the no-tokens bug fixed the same day. The reroute re-prefills the whole chain,
so the request gets *slower*, which makes the next timeout more likely — a node that is merely
slow can be driven into a loop of self-inflicted "deaths". And a genuinely slow-but-working
volunteer looks unreliable in the logs.

A read timeout and a closed socket are different events and should be reported differently;
the timeout also wants to scale with the stage's measured `ms_per_layer` rather than being one
constant for every node on every network.

### [P27] 🟡 An offline node's stale range outbids a correctly pinned one when it returns

`router._walk` advances by `max(layer_end)` among the nodes starting at each cursor. Two office
PCs are offline holding a stale **0-13** from before the 2026-08-07 placement fix.
`POST /network/layers` only rewrites nodes that are **online**, so those ranges were not
corrected when the split was pinned to 0-9 / 10-27.

When either powers on: at cursor 0 the candidates are the driver (0-9) and a stale node (0-13),
`max` picks 13, the **driver is dropped from the chain entirely**, the cursor jumps to 14, and
nothing starts at 14 — `missing (14, 27)`, 503, chat dead. Same mechanism as the morning's
`missing layers 21-27`, different numbers.

Preferring the widest claim is right for genuine replicas and wrong for stale ranges, and
nothing currently distinguishes them. Workaround: **run `neuron fix` whenever a machine joins
or leaves.** Unverified: whether self-heal makes the window worse — the driver, having lost the
tie, is no longer in `covering_ids` and may read as idle surplus to the reassignment sweep.

### [P26] 🟢 The network promoted itself onto a model no single node could hold — fixed (2026-08-07)

**Observed live.** The network auto-upgraded from Qwen2.5-1.5B to **Qwen2.5-7B-Instruct** the
moment three nodes were eligible, and the even split handed **9–10 layers** of it — about
4.5 GB of weights at fp16 — to machines with **8 GB and 12 GB of TOTAL RAM**, on top of Windows,
a browser and whatever else the volunteer was doing. That is an OOM kill, not a slow stage.

**Cause — two gates that each assumed the other was asking.**

1. `model_tiers` qualified the promotion on **aggregate** capacity: `min_nodes` and
   `min_ram_gb`, checked against the sum across the network. "3 nodes · 20 GB" is satisfied by
   8+8+12, and *none of those machines can hold a third of a 7B model*. A pipeline stage lives on
   one machine; summing RAM answers a question nobody asked.
2. `migration.plan_migration` — used by both tier migrations and self-heal — partitioned layers
   **evenly**, with no memory awareness at all. It never looked at a node's RAM, so it could not
   have refused.

`balancer.py` had solved exactly this in Session 14 (`max_layers_for`, `solve(gb_per_layer=…)`,
prompted by the identical arithmetic: Llama-3.1-8B at fp32 is 0.87 GB/layer, and an equal 3-way
split assigned 9.3 GB to a machine with 5–6 GB free). Two things kept that from helping:
the migration path never called it, **and it was dead code in production anyway** — the cap reads
`ram_free_gb`, which no node has ever reported (`agent.py` sends `ram_gb`, psutil *total*) and
`_balanced_plan` never passed `gb_per_layer` either. The guard existed, passed its unit tests, and
had never once applied to a real plan.

**Fixed, four parts:**

1. **`max_layers_for` falls back to total RAM** minus `RAM_OS_RESERVE_GB` (3 GB — Windows 11 idling
   with a browser, on the machines here) when no free figure is reported, so the cap is live against
   the data real nodes actually send. VRAM still wins for a GPU node, unchanged.
2. **`plan_migration` fits the split to the machines.** It still starts even, then shifts layers
   off any node over its cap onto nodes with room (`balancer.fit_to_capacity`, now shared with
   `solve`). On the trio above: 10/9/9 becomes 8/8/12.
3. **A migration whose target no partition can hold is refused**, not attempted —
   `partition_shortfall > 0` leaves the network serving what it has, with the reason in
   `/network/migration` and on the dashboard. Deliberately *not* a new phase: `self_heal` runs only
   while `phase == "steady"`, so parking a blocked migration elsewhere would have left the network
   unable to grow **and** unable to repair itself, for as long as the unservable tier stayed
   qualified.
4. **The tier gate asks the per-node question too.** `placeable(nodes, tier)` gates promotion and
   demotion alongside the aggregate check, so the ladder stops advertising an upgrade that can
   never happen. Tiers carry a measured `gb_per_layer`; a tier without one (env-injected via
   `NEURON_MODEL_TIERS`) is unconstrained, because an unknown footprint is not grounds to refuse
   to serve.

Self-heal is deliberately **not** gated on capacity: a coverage gap means no request can complete
at all, so it re-splits on the best fit available and reports `capacity_shortfall` instead. The
honest response to "the survivors cannot hold this model" is to serve a smaller one, and that is
the tier controller's demotion — which now happens, because placement feeds it.

**Known gap, recorded not guessed:** the driver additionally holds the embedding and lm_head
(~2.2 GB for the 7B) and no cap accounts for it, so the first node in a plan is under-charged by
that much. Wants a measured `head_gb` column.

**Files:** `coordinator/balancer.py`, `coordinator/migration.py`, `coordinator/model_tiers.py`,
`coordinator/main.py`. **Tests:** `coordinator/test_migration.py`, `coordinator/test_model_tiers.py`.

### [P25] 🟢 Every stranger was sent to the same layers, so the chain could never close — fixed (2026-08-07)

**Observed live.** The dashboard read **21/28 layers, DEGRADED — chain incomplete** with four
nodes online and three of them serving the *identical* range, layers 0–13:

| node | layers | standing |
|---|---|---|
| agent-optinovate-67e4eb | 0–13 | verified |
| agent-bhpc012104-5452f9 | 0–13 | probationary |
| agent-bhpc012101-80c3fc | 0–13 | probationary |
| node-c-pavilion | 14–20 | trusted |

Layers 21–27 belonged to nobody. Four machines, three copies of one slice, and not a single
request able to complete.

**Cause — placement reasoned over the ELIGIBLE nodes only.** `suggest_placement` walked the
chain from `build_chain`, which filters to `eligible` (trusted or proof-of-compute verified,
`models.py`). That filter is correct for **routing** — an unverified node must not receive live
traffic — and wrong for **placement**. A probationary node has already been handed a range and
has already downloaded that slice; it is a real claim on that segment. Filtering it out made
every newcomer invisible to the next one:

1. Only the trusted node (14–20) was eligible, so the first gap read **0–13**.
2. Stranger #1 asked, was told 0–13, registered — probationary, therefore invisible.
3. Stranger #2 asked. The coordinator's view was *unchanged*. Told **0–13** again.
4. Stranger #3, same. Nobody was ever offered 21–27.

Then it froze: `ensure_placement` returns early whenever config already holds a range, so no
node re-asked, and the answer written on first run was permanent.

**Nothing self-corrected**, either. Self-heal only reassigns nodes that are eligible, online AND
*true idle surplus*; here every eligible node was covering something, so surplus was empty and
the sweep did nothing on every tick, forever — [R6] in `RESILIENCE.md`. The balancer meanwhile
had a perfectly good split sitting on `/network/plan` (0–22 / 23–27, 2.8× better than an equal
split) that only a human calling `POST /network/rebalance` would ever apply. Nobody was looking.

**Fixed, three parts:**

1. **Placement now reasons over every ONLINE node** (`router.placement_roster`), routing still
   over eligible ones only. An unverified node's range counts as taken, so the next joiner is
   sent somewhere useful. Regression test: three strangers joining a network whose only eligible
   node holds 14–20 must close both gaps before anyone duplicates a slice.
2. **`GET /node/placement?exclude=<node_id>`** answers "where would I go if I weren't already
   here?", and a **probationary** agent asks it right after registering — moving if the answer
   differs, then re-registering on the new range (once; `_replaced` bounds it). Free at exactly
   that moment and no other: a probationary node serves no live traffic, so nothing is lost.
   Self-stabilising, because a node whose range is genuinely needed is handed the same range
   back. `--layers` sets `layers_pinned` and is never second-guessed.
3. **[R6]** — a coverage gap with no idle surplus now triggers a full re-split instead of a
   no-op. See `RESILIENCE.md`.

Also fixed on the way: `_walk` did not clamp a gap to the model's layer count, so a node holding
a range beyond `total` (what a migration onto a *smaller* model leaves behind until each node
reloads) stretched the gap past the end of the model and self-heal planned a slice tens of layers
too long.

**Files:** `coordinator/router.py`, `coordinator/migration.py`, `coordinator/main.py`,
`agent/agent.py`. **Tests:** `coordinator/test_placement.py`, `coordinator/test_migration.py`,
`agent/test_replacement.py`.

### [P24] 🟡 Strangers register fine, then sit PROBATIONARY forever — BLOCKS S12

**ROOT CAUSE FOUND 2026-08-07, and it was never the installer.** The coordinator's journal has
the registration failure in full:

```
Aug 06 06:25:55 … "POST /node/register HTTP/1.1" 500 Internal Server Error
  File "/home/ubuntu/neuron/coordinator/main.py", line 347, in register
      fingerprint = models.register_node(
  TypeError: register_node() got an unexpected keyword argument 'has_gpu'
```

**A half-deployed coordinator.** `main.py` had been updated to pass `has_gpu=`; the `models.py`
on the VM had not. Every registration 500'd for about an hour. Nothing was wrong with the
agent, the installer, or the volunteer's machine — and the agent could only log
`500, retrying in 60s`, forever, at WARNING. This is precisely what `coordinator/deploy.sh`'s
header warns about; shipping `coordinator/` as one atomic tar is why it cannot recur. Verified
fixed: every `/node/register` in the 36h to 2026-08-07 returned 200, and the driver registered
200 OK at 21:38.

**Two theories this killed:**

- **`_save()` truncation is NOT the mechanism.** The work PC's `config.json` was finally pulled
  and is complete, valid JSON with `node_id`/`node_token` intact. Session 53 said
  truncated-or-empty would confirm it; it is neither. The atomic-save fix stays — it is correct
  work — but it does not explain 0.18.
- **`registered_at` is not a restart indicator.** `coordinator/models.py:357` is
  `ON CONFLICT(node_id) DO UPDATE SET … status='online', last_seen=excluded.last_seen`, and
  `registered_at` is **not in that list**. It records only a node's first-ever join. An hour was
  spent restarting a machine on the strength of that field reading "8 days".

**The finding that reframes the whole entry: no stranger's machine has ever reached the
coordinator.** Since 2026-07-26 exactly **two** IP addresses have hit `/node/register`, and both
are ours — the office and the home connection. (The addresses themselves are deliberately not
recorded here: this repository is public, and they identify a residence. The count is the
finding; the numbers add nothing.) The 2026-08-07 install did not fail *at* registration; it
never made a successful request at all. v0.18 cannot say which, because its logging starts
after the config read. **Do not put 0.18 in front of another stranger** — cut 0.19 first, and
if that machine is still reachable, take `%LOCALAPPDATA%\NEURON\` off it before a reinstall
destroys the evidence. The absence of a log is itself the finding.

**Fixed on 2026-08-05 (everything except the 0.18 crash itself, which needs the machine).**
The four numbered fixes below are done, plus the two the entry did not name — the silent
startup and the config-shaped crash suspect. What is still open is stated at the end.

- **Logging now exists before anything can fail.** `_setup_logging()` is the first statement of
  `main()`, ahead of the config read that used to precede it. The module-level imports, the
  config read, `Agent.run()`, the tray, and `packaging/neuron_app_entry.py` each write their own
  traceback to `agent.log` via a stdlib-only `crash_log()` that needs no logging config and no
  successful import. The frozen entry duplicates the log-path rule deliberately: the import it
  is reporting on is the one it must not depend on. Windowed tray mode also puts the log's
  location in a message box, because there is no console for a traceback to reach. The first
  line of the log now names the version, which the log from the machine could not.
  `agent/test_startup_is_never_silent.py` 10/10.
- **A 0.17 config no longer kills 0.18 in the constructor.** `Agent.__init__` read
  `cfg["coordinator"]` directly, so any key a newer build expects and an older one never wrote
  was a `KeyError` before a single log line existed — [P24]'s own second suspect. Missing keys
  now fall back to `DEFAULT_CONFIG` and are *named in the log*; they are not written into the
  user's file (injecting defaults an operator deliberately omitted has its own failure mode —
  see `install.py`'s `write_config`).
- **The agent says when it is online and useless (fix 3).** `GET /node/{id}/ping` now returns
  `standing`, so a node learns it more than once — it used to learn it exactly once, in the
  reply to its registration, which is why the fact scrolled away on day one. After 60
  heartbeats still probationary the agent WARNs, repeatedly, that this machine is healthy but
  serves nothing and earns nothing and that the operator's verifier may be down; the heartbeat
  line itself says `probationary, not yet serving or earning` instead of `active`; and
  promotion is announced. `agent/test_probation_is_visible.py` 14/14.
- **The verifier retries, escalates, and proves it is alive (fix 4).** A roster read is retried
  3× with backoff inside the cycle (502s and DNS failures are the normal weather on a home
  connection — its last three lines ever were exactly those), a sustained outage escalates
  WARNING → ERROR naming the consequence, and recovery is logged. **The important one:** a
  healthy verifier used to log *nothing* (`nothing to verify` is debug-level), so its log looked
  identical whether it was running or had been dead since Monday. It now writes an alive line
  every 30 cycles, so silence means dead. `test_verifier_survives.py` 15/15.
- **Something watches it now (fix 1 and 2).** `neuron_doctor.py` fails when `verify_service.log`
  stops growing for 90 minutes — run against the live network on 2026-08-05 it reported the
  verifier dead for **2,970 minutes**, confirming this entry's diagnosis from the outside. The
  probationary check the entry called for is also in the doctor. And `agent/verifier_keepalive.py`
  restarts the verifier if it is not running: the Windows Run key that "installed" it fires once
  at login and never again, so it survives a reboot and not a crash. Install the 5-minute check
  with `python agent/install.py --verifier-keepalive` (a scheduled task on Windows, a cron line
  elsewhere); `--with-verifier` now installs it too.
- **Supervision is now installed and proven (2026-08-05 22:55).** `NEURONVerifierKeepalive` runs
  every 5 minutes as a scheduled task, pointed at the **venv** interpreter — not PATH's, which
  has no torch and would start a verifier that dies on import every five minutes, the trap
  `install.py` already documents for the Run key. Both branches have live evidence: the *start*
  branch is what brought the verifier back at 19:00 after two days dead, and the *detect-and-
  skip* branch was exercised at 22:55 (task exit 0, no second process spawned). Remove with
  `schtasks /delete /f /tn NEURONVerifierKeepalive`.
  - **Its own liveness check had to be fixed first.** The first version counted any process
    whose command line mentioned `verify_service.py` as healthy, so a *hung* verifier would be
    reported fine forever — "online means nothing" for the third time in this file, after [P21]
    and [P22], committed by the file written to prevent [P24]. It now reads the alive line's age:
    a process silent for 90 minutes is stopped and replaced. Two guards that are load-bearing —
    **process age** (right after an outage the log is days stale and the process seconds old:
    that is the verifier just restarted, still loading torch, and without this guard the
    keepalive would kill what it had itself started, forever) and **all matching PIDs, not the
    first** (a venv's `pythonw.exe` on Windows is a redirector that spawns the base interpreter
    as a child, so one verifier is two processes — 5.8 MB shim plus the 177 MB child doing the
    work — and killing only the first leaves the working half behind).
  - **Known limit, stated rather than assumed:** the task is `Logon Mode: Interactive only`, so
    it covers a crash — the failure that actually happened — but not a reboot with nobody
    signed in. That needs `/ru SYSTEM` and admin rights.
- **Investigated further (2026-08-05, same day). One suspect eliminated, a new mechanism found.**
  - **The packaging suspect is DEAD.** The shipped `dist/neuron-agent/neuron-agent.exe` (0.18.0,
    built 2026-08-03) was run here as `--headless --help`, which executes every module-level
    import — the whole suspect surface, `agent.gpu` included — and then exits at argparse before
    touching the network or the config. It printed usage and **exited 0**. A Python module
    missing from a PyInstaller bundle fails identically on every machine, so the frozen import
    chain is not what dies.
  - **`_save()` was not atomic, and that produces this exact signature.** It was
    `json.dump(self.cfg, open(self.config_path, "w"), indent=2)`: the open **truncates
    immediately**, the handle was never explicitly closed, and it runs from eight places
    including registration and migration cutover. Anything stopping the process mid-write — an
    OS shutdown, a task kill, installing a new version over a running agent (`neuron.iss` has no
    stop-the-app step) — leaves a truncated `config.json`. The next start's `json.load` then
    raises *before* v0.18's logging existed: silent death, no log file, **on every start,
    forever**, which is the reported symptom precisely (not an intermittent crash — an install
    that simply does nothing). It also explains "0.17 worked, 0.18 does not": the config is
    corrupted at the moment of the upgrade, and only the newer binary ever reads it again.
  - **Fixed:** `_save()` writes a temp file in the same directory, `flush()` + `fsync()`, then
    `os.replace()` (atomic on Windows and POSIX), keeping the previous good copy as `.prev`.
    New `load_config()` falls back to `.prev` when the live file is unreadable, preserves the
    broken one as `.corrupt` for evidence, and **raises rather than inventing a fresh identity**
    when there is no backup — `node_id`/`node_token` are the node's claim on everything it has
    earned, so silently regenerating them would orphan the owner's balance and rejoin as a
    stranger. `agent/test_startup_is_never_silent.py` 19/19.
  - **This is a mechanism, not a proof.** Confirming it needs that machine's `config.json`; if
    it is truncated or empty, that is the answer. Worth checking before anything else.
- **Still open:** the v0.18.0 crash has not been *identified*, only made diagnosable — that
  needs the machine. Run the installed 0.18 exe from a terminal, or just run it again once a
  build with these changes exists: it writes `agent.log` no matter where it dies. The two
  suspects the entry names (the GPU path, the in-place config upgrade) are unchanged, though the
  second is no longer fatal. Note that machine cannot auto-update to the fix — auto-update only
  runs inside a working agent, and that one crashes — so it needs a manual install.

**Original entry follows.**

### [P24-orig] 🔴 Strangers register fine, then sit PROBATIONARY forever

**This entry replaces an earlier, wrong diagnosis.** The first version blamed a failed
registration producing a null node_id and a `/node/None/slice-info` 404. The agent log from the
machine disproves that: registration worked perfectly.

```
13:03:33 auto-placed on layers 10-18 (fill-gap: chain is missing layers 10-18)
13:03:33 registered as agent-bhpc012104 [probationary], assigned layers [10, 18] (8 cores, 8 GB)
13:03:35 downloading slice: layers 10-18 (~0.84 GB) ...
13:05:50 node server started on port 50999
13:05:50 relay tunnel started - reachable via 150.230.22.250:9001 (NAT-friendly)
```

Node id assigned, slice downloaded, server up, NAT relay up, then `heartbeat ok - active`
continuously across three separate days. The join path works end to end. The 404 was a
different, unlogged event; the null-node_id mechanism is real but was not what happened here,
and the earlier entry asserted it without the log.

- **What actually went wrong**, from the same log's third line:

  ```
  PROBATIONARY: serving challenges only - a verifier must confirm this node
  (proof-of-compute) before it receives live requests or earns NRN
  ```

  It was never promoted. Not once in the whole log. A probationary node serves no live
  requests and earns no NRN, so the owner's experience is: installed it, it says it is running,
  nothing ever happens, zero balance, forever.

- **Root cause: `verify_service.py` is not running.** It is the operator-side service that
  promotes probationary nodes, and `verify_service.log` stops dead at **2026-08-03 17:27**,
  two days before this was investigated. Its final entries are all failures:

  ```
  2026-08-03 00:01:02 coordinator unreachable: 502 Bad Gateway .../node/list
  2026-08-03 14:28:21 coordinator unreachable: Failed to resolve 'neuronnet.duckdns.org'
  2026-08-03 17:27:47 coordinator unreachable: Failed to resolve 'neuronnet.duckdns.org'
  ```

  Every node it ever verified is `agent-optinovate-*` — the founder's own machines. It never
  reached the outside one.

- **This is the exact failure the verifier was written to prevent**, in its own words: *"a
  stranger who joined at 3am sat at zero NRN until somebody noticed them. A network whose
  onboarding requires the operator to be awake is a demo."* The service exists, it works, and
  it died silently — so the network is back to being a demo and nothing said so. Same class as
  [R6]: the information existed, nothing was watching it.

- **Fixes, in order:**
  1. **Restart the verifier and keep it up.** It has a systemd unit on Linux and is installed by
     `agent/install.py`; on the founder's Windows box it is evidently not supervised. A service
     whose death is invisible will die again.
  2. **The doctor must check it.** `/status` already returns `probationary_nodes`. Any node
     stuck probationary across more than a couple of verifier cycles means promotion is broken.
     Added to `neuron_doctor.py` in the same change as this entry.
  3. **The agent should say so.** After N heartbeats still probationary, log a WARNING naming
     the situation ("still awaiting verification after 30 minutes — the network operator's
     verifier may be down") rather than an indefinite calm `heartbeat ok - active`.
  4. **Retry harder on transient coordinator errors.** 502 and DNS failures are expected on a
     home connection; the verifier should back off and keep trying, not stop.

- **Secondary finding from the same log:** the work PC ran at 76-100% CPU and repeatedly logged
  `paused (cpu 91% > donation ceiling 85%) - not advertising availability`. Even fully verified,
  a busy work machine would contribute rarely. Not a bug — the resource guard behaving exactly
  as designed — but it means "install it on a work PC" is not a path to useful capacity, and the
  install guide should set that expectation.

- **v0.18.0 is a REGRESSION: 0.17 worked on this machine, 0.18 does not.** The log above is
  0.17 -- it registered, downloaded, served heartbeats for days. 0.18 produced **no log at all**
  and reportedly a 404. Two separate faults, and the first is what makes the second
  undiagnosable.

  **Why there is no log: `_setup_logging()` is called at `agent/agent.py:994`, inside `main()`,
  and only after the config has been read** (`cfg.get("log_level")`). Every module-level import
  (`from agent import gpu`, line 46, among them) and the whole config load run *before* any log
  file exists. Any failure in that window is completely silent -- no file, no message, nothing
  the owner can send. **This is the highest-value fix in this entry:** move file logging to the
  first statement of `main()`, ahead of the config read, so a bad config gets logged instead of
  swallowing itself. A run that produces no log is indistinguishable from a run that never
  happened.

  **Prime suspect for the crash: the GPU path, the substantive change in 0.18.** Its own release
  notes say *"No machine in this project has an NVIDIA GPU ... it has never actually executed on
  a GPU."* A work PC is exactly the machine likely to have one. `agent/gpu.py` guards its
  `nvidia-smi` and torch calls and the import itself looks safe, so this is a suspect, not a
  conclusion -- but it is the only substantive difference between the version that worked and
  the version that does not.

  **Second suspect: the in-place upgrade.** The release notes tell users to install over the top
  and keep `%LOCALAPPDATA%\NEURON`, so 0.18 reads 0.17's config. A field 0.18 expects and 0.17
  never wrote would throw during the config load -- before logging exists.

  **The one action that identifies it:** run the installed 0.18 executable from a terminal
  rather than the tray or service, so the traceback reaches stderr. Until logging moves earlier,
  there is no other way to see this class of failure.

  Windows blocking is real and separate (code-signing, ROADMAP S16), but it does not explain a
  404 -- a blocked binary does not make HTTP requests.

### [P23] 🔴 `prune_test_accounts.py` will sweep the first stranger's wallet — BLOCKS S12

- **Symptom:** the founder's two OAuth wallets show `balance: 0.0` on the live coordinator while
  `total_earned` survives intact (25.0 and 25.962). Not a bug and not a database loss — they
  were deliberately swept by the Session 33–35 dev-account cleanup, when they genuinely were
  dev wallets. `coordinator/test_prune_test_accounts.py` even fixtures their real balances
  (24.295, 25.0). The money went to `__ecosystem__` as designed.
- **The actual problem is forward-looking.** `classify()` prunes by prefix:

  ```python
  PRUNE_PREFIXES = (
      ("node_a-cli-", "CLI test wallet from wallet-settlement development"),
      ("w_",          "faucet-funded test wallet (OAuth wallet development)"),
  )
  ```

  and `wallet_for_oauth()` mints **every** user wallet as `"w_" + secrets.token_hex(16)`. There
  is no other format. So the rule that means "dev wallet" today means "every real user who ever
  signs in" tomorrow. Run the script once after a stranger joins and it takes their 25 NRN
  welcome grant, silently, and files it as a test account in the audit log.
- **Why it was right when written and wrong now:** when the rule was added, the only `w_`
  wallets in existence *were* the founder's. Open join (S12) changes that, and nothing in the
  script notices the change.
- **Mitigating for the moment:** the script is a manual `--execute` one-time repair, not a
  routine, and it refuses to run while `__escrow__` is non-empty. So this is not firing today.
- **Fix (decide before the first stranger, not after):** drop the blanket `w_` prefix so real
  wallets fall through to `unclassified` — which the script already refuses to execute on — and
  name the handful of genuine dev wallets with `--prune-also`. Alternative: gate the prefix
  behind an explicit `--include-oauth-wallets` flag defaulting off. Either way
  `test_prune_test_accounts.py` needs its expectations updated in the same change: three `w_`
  fixtures move from `PRUNED` to `unclassified`, and the "already-empty prune target" case
  needs a non-`w_` target.
- **Restoring the founder's 49.3 NRN is a separate, optional decision.** The pre-sweep balances
  are recoverable from `backups-offbox/neuron-20260802-115534.db`
  (`w_ef7ca467…` 25.0, `w_d35c84dd…` 24.295). Not done automatically: moving balances on a live
  ledger is the founder's call, not a repair to be applied silently.

### [P1] 🔴 Single-user speed vs. the Green AI thesis — HIGHEST
- **Symptom:** the chat UI shows ~0.7–1.4 tok/s for one user; the headline 6.16 tok/s
  is *aggregate throughput* under concurrent load, not single-request speed.
- **Reality:** a serial CPU pipeline over the internet **cannot** beat a GPU datacenter
  on single-user latency. That is physics, not a bug.
- **Why it may be OK:** Green AI competes on **energy, cost, capacity, and access**, not
  latency. "Fast enough" + free + green + runs-huge-models is the real moat. Petals runs
  70B across volunteers at ~1–2 tok/s and is respected.
- **Risk if ignored:** pitching NEURON as "fast" sets up a credibility collapse; adoption
  stalls if users expect ChatGPT speed.
- **Mitigations (see P2–P3, P8):** quantization, better networking, speculative decoding,
  right-sizing splits, and a *positioning* that leads with green/cost/capacity.

### [P2] 🟡 Still running fp32 — biggest untapped speedup
- **Symptom:** all inference is fp32. bf16 was rejected (these CPUs lack a bf16 GEMM).
- **Opportunity:** int8/int4 could give 2–4×. Prior note (sessions): "true int4/int8
  speedup = the llama.cpp/GGUF path" — i.e. PyTorch int8 on these CPUs was *assumed*
  weak but never measured on the full model.
- **Action:** MEASURE int8 dynamic-quant vs fp32 on the real model/CPU. → done, below.
- **Measurement (2026-07-24, this Windows CPU, full Qwen2.5-1.5B, greedy, 40 tok):**
  - fp32: **3.18 tok/s** (matches the known ~3.2 single-machine baseline ✅)
  - int8 dynamic (all `nn.Linear` → qint8): **10.99 tok/s = 3.46× faster** ✅
  - **CONCLUSION — speed is achievable; the thesis is not dead on speed.** ~7–11 tok/s
    (comfortably readable) is within reach → see new caveat [P9].
  - qengines available on this box: `onednn, x86, fbgemm` (fbgemm used).

### [P19] 🟢 The pipeline wire ran arbitrary code from any peer — fixed (2026-07-29)
- **Symptom:** `common.recv_msg` did `torch.load(io.BytesIO(data), weights_only=False)` on
  whatever arrived on the socket. A `torch.save` payload is a **pickle**, and unpickling with
  `weights_only=False` calls whatever the sender's `__reduce__` names. Demonstrated locally:
  a crafted `act` message executes the sender's code in the receiver's process.
- **Why it mattered here specifically:** this is not a "don't accept files from strangers"
  theoretical. Every node deserialises messages from the node before it in the chain, and the
  driver deserialises the reply — so the reach was **both directions**, driver ↔ node. Since
  Session 12 each node's port is published on a **public relay**, so the sender need not even
  be in the chain. `router.build_chain` will happily put an open-join stranger's machine in
  the pipeline of a request originating on the founder's PC. It was the single most direct
  path from "a stranger installed the agent" to "a stranger runs code on your desktop", and
  no document (`SECURITY.md`, the S14 audit, [P14]) had ever named it.
- **Fixed:** `weights_only=True` on the legacy path — verified against every message shape the
  protocol actually sends (config, config-ack, act, act-reply, bye all carry only
  dict/str/int/float/bool/Tensor). The new `wire_codec` frames are JSON + raw tensor bytes and
  contain nothing executable at all. Also capped the 8-byte length prefix at 512 MB: node
  ports face the open internet, and an unchecked length let a stray scanner make a 1 GB relay
  VM allocate an arbitrary buffer (same class of bug as the `relay.recv_json` hardening).
- **Regression tests:** `test_wire_codec.py` — a hostile pickle is refused, an absurd length
  prefix is refused before allocating, and a legacy `torch.save` sender still round-trips.

### [P20] 🟡 The wire ships raw fp32 activations, once per token per hop — measured, now 4.3× smaller
- **Symptom:** `common.send_msg` `torch.save`d full-precision tensors. Measured on the real
  3-stage chain (Qwen2.5-1.5B, H=1536, 6 prompts × 48 tokens): **12,508 bytes per message**,
  of which 1,153 is pure pickle framing. Paid at every junction, every token — and since
  Session 12 a relayed hop crosses the public VM **twice**, so relay egress pays it twice.
- **What the number means at the size NEURON actually exists for.** A 70B model is H=8192
  over ~20 stages. One decode token then costs 33.6 KB per hop, **0.69 MB across the chain**,
  before TCP and relay overhead. On a 10 Mbit/s home upload that is ~27 ms of pure
  serialisation per hop — **~0.55 s/token spent on the wire**, latency the compute never
  sees. At `i8h` it is 8.2 KB/hop, 0.17 MB/token, ~0.13 s. [P3] already observed the network
  dominates per-token cost; this is one of the reasons why, and it was never measured
  until now.
- **Measured (2026-07-29, `bench_wire.py`, 6 prompts × 48 tokens, codec at all three
  junctions so error compounds exactly as on the wire):**

  | codec | B/msg | vs before | text identical to fp32 | max abs Δlogit |
  |---|---|---|---|---|
  | `torch.save` fp32 (was) | 12508 | 1.00× | 6/6 | 0.0000 |
  | `f32` (new framing, no pickle) | 11355 | 1.10× | 6/6 | 0.0000 |
  | `f16` | 5723 | 2.19× | 6/6 | 0.0069 |
  | **`i8h`** (Hadamard + blockwise int8) | **2946** | **4.25×** | **6/6** | **0.2054** |

  and the schemes measured and **rejected** (exploratory sweep, 3 prompts × 48 tokens, so
  the identity column is out of 3):

  | codec | B/msg | vs before | identical | max abs Δlogit |
  |---|---|---|---|---|
  | fp8 e4m3 | 2786 | 4.43× | 0/3 | **nan** |
  | int8 per-tensor | 2792 | 4.42× | 0/3 | 30.46 |
  | int8 blockwise-256 (*Petals' scheme*) | 2812 | 4.39× | 2/3 | 1.13 |
  | int8 blockwise-64 | 3014 | 4.09× | 1/3 | 0.59 |
  | int4 blockwise-32 | 1577 | 7.82× | 0/3 | 4.52 |

- **[P9] again, on the wire.** Real junction activations measured at **absmax 6620, std 42,
  worst channel ≈ 750× the median**. An absmax quantizer takes its scale from that one
  channel, so everything else collapses — which is why plain int8 lands at Δlogit 30. fp8
  e4m3 cannot even represent 6620 (its max is 448), overflows to inf, and the generation goes
  NaN. **Note that Petals' own scheme — blockwise int8, no rotation — diverged on 1 of 3
  prompts here.** Copying the paper's mechanism verbatim was not sufficient.
- **What fixed it:** QuaRot's insight (arXiv:2404.00456), applied at the transport layer
  instead of the model. A Hadamard rotation is orthogonal, so it spreads the outlier evenly
  across the block without changing the vector; the sender rotates before quantizing and the
  receiver rotates back. Because it is pure transport there is **no weight surgery and no
  calibration** — the model never sees it. Same bytes as unrotated int8, ~7× less error
  (rel_l2 0.0037 vs 0.0276).
- **The rotation has to be cheap or it is not worth doing.** The textbook log-n butterfly
  cost 1.26 ms per call at H=8192 — a fifth of the wire time it was saving. Done instead as
  a single matmul against a cached Hadamard matrix: **0.045 ms, 28× faster**. Whole-codec
  cost is now 0.54 ms encode+decode per hop against ~6.4 ms of transmission saved on a
  10 Mbit/s upload. **On a fast link that trade reverses** (0.54 ms to save 0.06 ms), so
  `NEURON_WIRE_CODEC` pins the codec for LAN/datacenter deployments; the default assumes
  volunteers' home connections, which is what NEURON is.
- **int4 was measured and deliberately NOT shipped:** ~0.53 B/elem at ~9% relative error per
  hop. Survivable over a 3-node chain, not over the 20-node chain a 70B model implies, and
  the wire is the one place where being wrong is silent.
- **i8h is gated on model size, because the same benchmark run against 0.5B disagrees with
  the 1.5B result:**

  | model | `i8h` identical | `i8h` max Δlogit | `f16` identical | `f16` max Δlogit |
  |---|---|---|---|---|
  | Qwen2.5-1.5B (H=1536) | 6/6 | 0.2054 | 6/6 | 0.0069 |
  | Qwen2.5-0.5B (H=896) | **3/6** | 0.5034 | 6/6 | 0.0075 |

  The diverging 0.5B answers stay correct and on-topic — they re-word, typically 100+
  characters in — so this is drift, not the [P9]-style collapse unrotated int8 causes. But
  it is drift the larger model does not show, in the expected direction: fewer parameters,
  less redundancy to absorb the noise. `wire_codec.preference()` therefore offers `i8h` only
  at H ≥ 1536 and `f16` (still 2.3×, 6/6) below it. Small models are both the fragile ones
  and the cheap ones to ship uncompressed, so nothing is given up. **Two data points, not a
  curve** — measure a third model before trusting the threshold away from 896/1536.
- **A false lead worth recording.** An end-to-end socket run appeared to show i8h giving a
  factually worse answer on 0.5B ("the sky is blue because it reflects sunlight" vs
  "because of the scattering of sunlight by tiny…"). That was a test-rig artifact: a stray
  earlier `node_b.py` was still bound to the port, and `SO_REUSEADDR` let a second one bind
  alongside it, so connections landed on either process. With one listener, all four codecs
  return the identical answer. The size gate rests on the in-process benchmark above, which
  has no sockets and no such failure mode.
- **Rollout:** codec is negotiated per hop in the config handshake, and a peer that offers
  nothing recognised keeps the legacy format — so a half-upgraded fleet keeps working.
  **Not yet deployed to the live nodes** (Pavilion/OptiPlex still run the old build; they
  will negotiate down to legacy until updated).

### [P21] 🟢 Nothing restarts a node — fixed and installed on both remote nodes (2026-08-01)
- **Fixed by:** `agent/install.py --startup` on the Pavilion and the OptiPlex. Both now run the
  agent as a `systemd --user` service with **`Restart=always` / `RestartSec=10`** (was
  `Restart=on-failure`, which leaves a node dead after any clean-looking exit).
- **The flag in this file's "what to do" did not exist.** `install.py` had `--no-startup`, not
  `--startup`, so the documented command failed with an argparse error — and running plain
  `install.py` instead would have been worse: `write_config()` wrote DEFAULT_CONFIG straight
  over the existing file, discarding `node_id`, `node_token`, the layer range the machine was
  actually serving and its `register_secret`, and re-pinning every node to layers 10-18. The
  documented repair for [P21] would have de-identified both live nodes. `write_config()` now
  merges and `--startup` exists as a non-interactive repair path for an already-working node.
- **A unit file alone was never going to be enough.** A `systemd --user` service does **not**
  start at boot unless the user has *lingering* enabled — without it the unit is bound to a
  login session, so `WantedBy=default.target` fires when somebody logs in and never after an
  unattended reboot, while looking correctly installed and `enabled` the whole time. Both
  machines had `Linger=no`. The installer now enables it (unprivileged first, then `sudo -n`)
  and, when neither is permitted — a machine with no sudo, i.e. exactly a stranger's laptop —
  falls back to **cron**, which needs no privileges: an `@reboot` line plus a two-minute
  keepalive (`agent/neuron-keepalive.sh`) that restarts the agent if it is not running. That
  fallback also bounds a crash at two minutes instead of "until somebody notices".
  `agent/uninstall.py` removes both, or an uninstalled agent would be resurrected every two
  minutes by a cron job nobody remembered.
- **The keepalive greps with the `[a]gent[.]py` bracket trick and lives in its own FILE** for
  the reason `sessions.md` already documents one level up: inline in the crontab, the guard's
  own command line contains the string it greps for, so it matches itself, concludes the agent
  is already running, and never starts anything.
- **"Online" now has to mean "serving".** `NodeServer.run()` reports a failed bind
  (`self.listening` / `self.bind_error`) instead of dying silently in its daemon thread;
  `agent.setup()` waits for the listener before advertising and retries the whole setup
  otherwise, and `heartbeat_loop()` refuses to ping while the listener is down, so the
  coordinator marks the node offline and routes around it. **This caught a live instance the
  moment it shipped:** both remote machines still had hand-started `node_ns.py` servers from
  Session 23 squatting port 50999 (17h 57m uptime), so the agent's bind failed — before this
  change it would have gone on logging `heartbeat ok — active` forever against a node that
  never bound. The stale processes were stopped; the agents own the port now.
- **VERIFIED BY A REAL REBOOT (2026-08-01).** The Pavilion was power-cycled: booted 14:03:14,
  `neuron-agent` active at **14:03:27 — 13 s after boot**, nobody logged in, no SSH, listening
  on :50999 and back in `/node/list` far inside the 2-minute bar. Linger is the part that
  earned its keep — a user-level unit starting with no login is exactly what a reboot tests,
  and it is what was missing. A `SIGKILL` of the agent on the OptiPlex separately recovered in
  **20 s**, covering the crash case. (The Pavilion cannot be rebooted remotely: no passwordless
  sudo, and polkit refuses `systemctl reboot` over SSH — it needs someone at the machine.)
- **Original entry, kept for the record:**

### [P21-orig] 🔴 Nothing restarts a node — the network dies on any reboot, and the fix already exists
- **Symptom:** the Pavilion and OptiPlex run the agent as a bare foreground process. A reboot,
  a crash, a closed SSH session or an OOM kill takes that node off the network permanently
  until somebody SSHes in and starts it by hand. Hit repeatedly on 2026-07-30: `nohup … &`
  over SSH does **not** survive the session ending (`setsid`, with a delay before the SSH
  command returns, does) — and even then nothing brings it back after a reboot.
- **Why it looked worse than it was:** a dead agent is invisible. The coordinator just shows
  the node `offline`, identical to "the owner closed their laptop", so the network silently
  runs degraded. Combined with [P4] this is the single most common way NEURON appears broken.
- **The fix is already written and was simply never used here.** `agent/install.py --startup`
  installs a real service on all three platforms — Windows registry Run key, `systemd --user`
  on Linux, and a `launchd` LaunchAgent on macOS ([P15]) — and `agent/uninstall.py` mirrors
  it. The packaged installer (`packaging/neuron.iss`) is the intended path. The two remote
  machines bypassed all of it: they were set up by hand-copying a few files in an early
  session, so they have no service, no auto-start and no restart-on-failure.
- **What to do:** (a) run the installer path (or `install.py --startup`) on the Pavilion and
  OptiPlex instead of launching the agent by hand; (b) make the systemd unit
  `Restart=always` with a `RestartSec` backoff, matching what the coordinator's own unit
  already does; (c) surface "node was auto-restarted N times" so a flapping machine is
  visible rather than silently degrading the network.
- **Ship this before handing anyone an installer.** A stranger will never SSH in to restart
  anything — if the agent does not come back by itself, that machine leaves the network on
  its first reboot and never returns, and the earn-rate they were promised quietly becomes
  zero.
- **Worse than dying: a node can report itself healthy while serving nothing.** `agent.py`
  starts `NodeServer.run()` in a daemon thread and never checks that it bound. Restart an
  agent while the previous process still holds port 50999 and the bind raises inside that
  thread, the thread dies silently, and the main loop carries on logging `heartbeat ok —
  active` forever. Observed live on the Pavilion: the coordinator showed the network
  `healthy: True, 28/28 layers` while the node refused every connection, so `/infer` handed
  drivers a chain that could not run. **The heartbeat must assert the listener is actually
  accepting**, not merely that the process is alive — otherwise "online" means nothing and
  routing sends real requests into a black hole.
- **Deploy note learned the hard way:** `nohup … &` over SSH dies with the session, and even
  `setsid … &` needs the SSH command to stay alive a few seconds before returning or the
  process never detaches. `pkill -f 'agent.agent'` also matches the `bash -c` running it and
  kills its own shell — use the bracket form (`'[a]gent.agent'`), the same trick `sessions.md`
  already documents for `[n]ode_b.py`.

### [P22] 🟢 A relay tunnel died silently while the node reported healthy — fixed (2026-08-01)
- **Symptom:** after ~2 hours the Windows node's public relay port accepted TCP and then
  answered nothing (20 s timeout), while both Linux nodes completed the same handshake in
  0.10 s. The agent logged `heartbeat ok — active` throughout, so `/node/list` showed the node
  **online and verified**, routing sent it real requests, and it served none of them. Restarting
  the agent fixed it instantly.
- **Root cause, measured not guessed:** `tunnel_client.run_tunnel` holds an outbound control
  connection and blocks in `recv_json` waiting for the relay to push `new_conn`. That socket is
  idle by design. When the relay's end goes away the socket stays ESTABLISHED until TCP
  keepalive gives up — and `SO_KEEPALIVE` alone uses the OS default idle timer, which on this
  machine is the Windows default **7,200,000 ms = 2 hours** (confirmed: no `KeepAliveTime`
  override in the registry). The observed ~2 h dead window is that timer, exactly.
- **Why it mattered more than it looks:** every stranger is relayed (`behind_nat` is the
  default since [P10]), and it reproduced on Windows, which is what most of them will run. It is
  [P21]'s "online means nothing" in a second place — Session 26 made the heartbeat assert the
  *local listener* had bound, and nothing asserted the *tunnel* still carried traffic.
- **Fixed, two layers:**
  1. `tunnel_client.set_keepalive()` — keepalive at **60 s idle / 10 s interval** via
     `SIO_KEEPALIVE_VALS` on Windows and `TCP_KEEPIDLE`/`INTVL`/`CNT` elsewhere. Detection drops
     from hours to about a minute, and the existing reconnect loop then does its job.
  2. `agent.relay_reachable()` — every 4th heartbeat the agent dials **its own public relay
     endpoint** and completes a real config handshake. A plain TCP connect proves nothing here:
     the relay accepts on the public port whether or not it can still reach the node, so a dead
     tunnel is indistinguishable from a healthy one until bytes come back. On failure the agent
     restarts the tunnel (its own stop flag, separate from the agent's) and **withholds the
     heartbeat**, so the coordinator marks it offline and routes around it instead of feeding a
     black hole.
- **Tests:** `agent/test_relay_liveness.py` — 5/5. The load-bearing case is a fake endpoint that
  *accepts and never speaks*, which a connect-only check would wrongly pass; plus a healthy
  handshake, a non-relayed node (no false alarm), and the assertion that a dead tunnel restarts
  the tunnel and sends no heartbeat.
- **Verified live:** deployed to all three nodes; all three public endpoints answer in
  0.03-0.10 s, and **zero false alarms** across 4+ probe cycles on every node.

### [P3] 🟡 Network latency dominates per-token cost
- **Symptom:** in the 8-request run, ~0.4 s **per token** was network. The Pavilion was on
  an Amsterdam **relay** (`relay "ams"`), not a direct link.
- **Mitigations:** prefer direct Tailscale links (avoid relays), co-locate nodes / LAN
  clusters, send fewer/batched hops, and use fewer stages for small models (P8).

### [P4] 🔴 Node availability / laptop suspend
- **Symptom:** the Pavilion (node_c) suspends and drops off Tailscale when idle, degrading
  the network to 2 nodes → incomplete 28-layer chain → no service. The S9 resource guard
  also pauses nodes on load/battery/user-activity.
- **Reality:** node churn is *expected* in a volunteer network. A single fixed 3-node chain
  has no redundancy.
- **Mitigation:** replication (multiple independent pipelines, P8) + coordinator routing
  around offline nodes; keep dev nodes awake for now.

### [P5] 🟡 No per-wallet NRN balance / debit (from S11)
- **Symptom:** the OpenAI API records the wallet and reports 1.0 NRN/request cost, but does
  not persist a user-side balance or debit it. Coordinator ledger only credits nodes.
- **Fix:** coordinator-side user-balance table + debit on `/complete`. Ties to S17 (on-chain).

### [P6] 🔵 Quality vs. size vs. speed tension
- Qwen2.5-1.5B is small; output is not production-grade. Bigger models (S15) improve quality
  but need more/bigger nodes and are slower. Every quality gain costs speed and vice-versa —
  the core three-way trade to manage deliberately.

### [P7] 🟢 Heterogeneous nodes + manual layer split — RESOLVED (S14)
- Pipeline runs at the speed of its slowest stage; layer split WAS hand-tuned (`--s1/--s2`).
- **Fixed (2026-07-25, Session 14):** `coordinator/balancer.py` solves for the time-equalizing
  split from each node's self-measured `ms_per_layer` + the driver's `head_ms` (`benchmark.py`).
  `GET /network/plan` computes it, `POST /network/rebalance` applies it. Reproduces the hand-tuned
  9/9/10 automatically; live on the cloud coordinator. Remaining nicety: dynamic re-balance +
  auto-reload when nodes join/leave mid-flight (today a range change needs the driver to reload
  with the new S1).

### [P9] 🔴 Naive int8 quantization destroys output quality (found in the P2 spike)
- **Symptom:** with 3.46× speed, the int8 model answered "Explain how a rainbow forms"
  with *"I'm sorry, but I can't provide an answer…"* while fp32 answered correctly.
  Naive dynamic int8 (all Linear incl. attention + the 150k-vocab head, no calibration)
  is too aggressive for a small (1.5B) model.
- **The real path (speed AND quality):** quality-preserving quantization —
  - **GPTQ / AWQ** (int4, per-channel scales + calibration; ~1% quality loss typical), or
  - **llama.cpp GGUF** (`Q4_K_M` / `Q8_0`) — proper k-quant schemes; also has an **RPC
    backend that distributes layers across machines**, i.e. quantized *and* distributed.
    This is a possible production-engine pivot from the hand-rolled fp32 PyTorch split
    (relates to [P8]); big change, big payoff — evaluate deliberately, not now.
  - cheaper interim: weight-only int8, or quantize MLP-only and keep attention/head in fp.
- **Status:** speed is proven reachable; the *method* is the open work. Schedule a
  dedicated "quality-preserving quantization" session (candidate: alongside/after S14),
  NOT a rushed integration now. No real users yet (per ROADMAP's One Rule).

### [P10] 🔴 No stranger can actually join yet — everything assumes ONE Tailscale net (BLOCKS S12)
- **Symptom:** the coordinator is Tailscale-only (`100.114.189.46:8001`, ufw scoped to
  `tailscale0`) and nodes reach each other over Tailscale IPs (`register_nodes` stores
  `tailscale_ip`; node_a dials node_c's IP; node_c dials node_b's). A stranger is **not**
  on the tailnet, so they can neither reach the coordinator nor be reached by/reach peers.
- **Two sub-problems:**
  1. **Public coordinator** — needs a public address: Cloudflare Tunnel, **Tailscale Funnel**,
     ngrok, or a cloud VPS (Oracle free tier was noted but never set up). Easy-ish.
  2. **Node ↔ node connectivity across NAT** — the hard one. Home nodes are behind NAT and
     can't accept inbound TCP. Middle/last stages currently MUST accept inbound. Options:
     (a) stranger installs Tailscale + joins via auth key — works today, but not "one click"
         and doesn't scale/secure to thousands; good enough to *prove* a first stranger;
     (b) relay all pipeline traffic through the public coordinator (re-architecture: nodes
         hold an outbound connection, coordinator brokers) — the real scalable fix;
     (c) NAT hole-punching (libp2p/WebRTC/holepunch) — most work.
- **Consequence for S12:** the "5-step, no-tech, one-click" install in the ROADMAP is not
  achievable on the current architecture. The realistic first-stranger proof is path (a);
  the true product needs (b). Decide the path before writing the install guide.
- **DECISION (2026-07-24):** move the coordinator to a **free-tier cloud VM** (Oracle Always
  Free / GCP e2-micro) so it has a public address WITHOUT exposing the founder's personal
  OptiPlex homeserver or Pavilion — those stay private, node-only. Solves sub-problem 1
  cleanly. Sub-problem 2 (stranger node↔node NAT) is still open; the cloud VM is the natural
  future home for the relay/broker. Deploy guide written: `coordinator/DEPLOY.md`; the
  coordinator is dependency-light (`coordinator/requirements.txt`, no torch). Repo stays
  PRIVATE for now. Blocked on the founder creating the VM (account/card = their action);
  deployment is turnkey once SSH exists.
- **DONE (2026-07-24) — coordinator DEPLOYED & PUBLICLY LIVE.** Oracle Always Free VM,
  Amsterdam, **x86 `VM.Standard.E2.1.Micro`** (ARM A1.Flex was out of capacity — known Oracle
  issue), Ubuntu 22.04, 1 GB RAM (ample; coordinator ~100 MB). Public IP **150.230.22.250:8001**.
  systemd service `neuron-coordinator` (auto-restart), **strong `NEURON_REGISTER_SECRET`** set
  (saved locally in gitignored `.env.coordinator`), iptables opened for 8001 (Oracle Ubuntu
  REJECTs non-22 by default) + Oracle VCN security-list ingress rule for 8001. `/status`,
  `/dashboard`, `/agent/version` all return 200 from the public internet. SSH:
  `ssh -i ~/.ssh/oracle_coordinator ubuntu@150.230.22.250`. **[P10] sub-problem 1 (public
  coordinator) = RESOLVED.** Remaining: sub-problem 2 (stranger node↔node NAT).
- **DONE (2026-07-24) — live network MIGRATED to the cloud coordinator + inference PROVEN.**
  `register_nodes.py` now reads `NEURON_REGISTER_SECRET` from env; registered node_a/node_c/node_b
  against `150.230.22.250:8001` (healthy 28/28), ran a prompt via `node_a.py --coordinator <cloud>`
  → correct answer, NRN credited on the CLOUD ledger (a 0.3214 / c 0.2893 / b 0.2893, fee 0.10).
  Nodes keep Tailscale for pipeline traffic; only the coordinator moved. Old OptiPlex coordinator
  left running as a harmless fallback (its nodes time out offline).
- **2026-07-24 — [P10] sub-problem 2 (stranger node↔NAT) — relay BUILT, PROVEN, DEPLOYED.**
  New `relay.py` (public host) + `tunnel_client.py` (node), **pure stdlib, ARM-safe,
  protocol-agnostic byte-splice → ZERO changes to node_*/common**. A node makes only OUTBOUND
  connections to the relay; the relay exposes a public port and reverse-tunnels to it (like a tiny
  self-hosted ngrok). Local selftest PASS (50 KB binary + 8 concurrent, byte-exact) against a node
  reachable ONLY via outbound. Deployed to the cloud VM as systemd `neuron-relay` (control 8010,
  data 8011, public 9000-9100 opened in iptables). **REMAINING for a live relayed/stranger node:**
  (a) open 8010/8011 + the node's public port (e.g. 9002) in the Oracle **security list** (console);
  (b) run `tunnel_client.py` on the node; (c) register it in the coordinator with the relay endpoint
  (`150.230.22.250:900X`) instead of a Tailscale IP; (d) prove an inference. The coordinator handing
  out a relay endpoint for NAT'd nodes is what finally makes a real first stranger possible.
- **2026-07-24 — LIVE PROOF PASSED (real node through the cloud relay).** Opened `8010-9100` in the
  Oracle security list; ran `tunnel_client` on node_b (OptiPlex, OUTBOUND-only to the cloud relay,
  public :9002); ran a real inference with the node_c→node_b hop FORCED through the relay
  (`node_a.py --host-b 150.230.22.250 --port-b 9002`) — over the public internet, NO Tailscale for
  that hop → correct answer (19 tok). node_b behaved exactly as a stranger reachable only via the
  relay. Test tunnel torn down after; node_b back to normal. **NAT-traversal mechanism proven
  end-to-end.**
- **2026-07-24 — relay onboarding AUTOMATED + deployed (one-click for a NAT'd node).** Coordinator
  `/node/register` now accepts `behind_nat` → assigns a free port from the pool (config `RELAY_*`),
  stores the node at the relay endpoint, returns a `relay` block. `tunnel_client.run_tunnel()` is a
  callable; `agent.py` auto-starts it from that block on registration (persists it in config for
  re-runs) → **a NAT'd node self-configures, zero manual steps**. Isolated test PASS (registered
  behind_nat → got port 9000 → reachable via the cloud relay byte-exact using only the coordinator's
  response). Deployed to the cloud coordinator (DB persisted, 3 nodes intact; `behind_nat` register
  live-verified). **Open-join DONE (2026-07-25, Session 17):** registration no longer needs the
  shared secret — a secret-less node joins *probationary* (excluded from routing/earning) and is
  promoted to *verified* by a proof-of-compute pass; the secret now just marks a node *trusted*
  (fast-path). `coordinator/test_open_join.py` 17/17; not yet deployed to the live coordinator.
  **What's genuinely left for a first stranger:** an actual outside person installs the agent
  (package + install guide + 4th-node placement), and `/complete` still needs auth ([P12]).
  Beyond ~100 relayed nodes: `SCALING.md`.
- **2026-08-01 — the relay is now the DEFAULT for every node, and the live nodes were moved
  onto it.** The machinery from 2026-07-24 was all present and none of it was in use: both
  remote nodes had `behind_nat: false` in their configs, so the coordinator stored their
  **Tailscale** addresses and `router.chain_public` handed those out to every peer that asked
  for a chain. A stranger placed anywhere except the final segment must dial the next hop, and
  `100.114.189.46` is not a routable address for anyone outside the founder's tailnet — so the
  relay existed, ran, and was bypassed by every real request. Changes:
  - `agent.use_relay()` defaults `behind_nat` to **True** (it was `.get("behind_nat", False)`,
    which contradicted `DEFAULT_CONFIG`'s `true` for any config that simply omitted the key),
    with `--relay/--no-relay` to override. `--no-relay` is now the exception a LAN cluster opts
    into, rather than the accidental default.
  - `setup()` re-registers when a node is in relay mode but holds no relay endpoint, not just
    when its ticket is stale. Without that, a node switched to relay mode after its first
    registration would keep advertising its old direct address forever, since `register()` is
    skipped once credentials exist.
  - Registering with `behind_nat` and getting **no** relay block back is now logged as a
    warning naming the address peers were given instead. It used to pass silently, which is
    the exact failure this entry exists to prevent.
  - Both live nodes re-registered onto relay endpoints (`150.230.22.250:9002` / `:9003`) and
    both were confirmed reachable on those ports from a machine with Tailscale **stopped**.
- **2026-08-01 — two bugs found by actually running the stranger path, either one fatal:**
  1. **`agent/__init__.py` did not exist**, so INSTALL.md's step 4 (`python agent/agent.py`)
     died on `ImportError: cannot import name 'local_chat' from partially initialized module
     'agent'`. Python puts `<repo>/agent` on `sys.path` ahead of anything the script adds, and
     a regular module named `agent` beats a namespace-package directory of the same name, so
     agent.py imported *itself*. Both live nodes had an `__init__.py` created by hand during
     setup, which is why this was invisible for eleven sessions — every stranger would have hit
     it on the first command in the install guide.
  2. **Proof-of-compute could not promote a node on any segment except the last.**
     `node_server.py`'s probe role set `s2 = self.hi` (the inclusive last layer) where `s2` is
     used as a Python slice bound (`layers[s1:s2]`), so it ran one layer too few and advertised
     a range `security/proof_of_compute.attest_middle` rejects outright. Auto-placement puts a
     joining stranger wherever the coverage GAP is — which was layers 0-9 here — so the
     realistic case was the broken one. Fixed to `self.hi + 1`.

### [P11] 🟡 Public-launch hygiene (before the repo goes public)
- Hardcoded private Tailscale/LAN IPs across README/ROADMAP/agent/coordinator — genericize to
  a placeholder + a real public coordinator URL (low security risk — CGNAT, unreachable — but
  leaks infra and is useless to strangers).
- `neuron-dev-secret` default registration secret — the PUBLIC coordinator must set
  `NEURON_REGISTER_SECRET` to a real value, else anyone can register nodes.
- Otherwise clean: no tokens/keys/passwords/DB in tracked files (`.gitignore` verified).

### [P8] 🔵 Scaling model: pipeline depth vs. replication (key architecture decision)
- **More nodes ≠ faster single request.** A deeper single pipeline is *slower* per request
  (more hops). The right use of a huge group is **replication** (many independent 3-node
  pipelines → ~linear aggregate throughput) and **bigger models** (split a model no single
  machine can hold). The coordinator currently assembles ONE chain; scaling throughput needs
  it to build and load-balance across **many** pipelines. This is the big future design step.
  **Partial (2026-07-25, Session 18): REPLICATION landed at the segment level** — `router.build_chain`
  now load-balances across nodes that share a segment (a 4th node registers with an existing node's
  layer range; each request picks a replica at random, so both earn under concurrent load). This is
  the throughput-via-replication shape for ONE pipeline slot; full many-pipeline / cross-driver
  load-balancing (multiple driver hosts) is still the bigger open step.
  **→ Full prototype→worldwide scaling plan now written up in `SCALING.md`** (connectivity: P2P +
  relay fabric; coordination: regional → DHT; topology: many small pipelines; phased plan; Petals
  as the proven reference model; and the rule: don't build the scale layer before the first stranger).

### [P18] 🟢 Sign-in was unshippable, and the network ran on plain HTTP — both fixed live (2026-07-29)
- **Login could not ship.** It ran on the driver, so an OAuth *client secret* had to exist in
  that process — i.e. on every stranger's PC once each installed agent began serving its own
  Chat UI. A secret inside a distributed binary is extractable, and the alternative (each user
  registering their own Google Cloud project before they can send a message) is not a product.
  Moved to the coordinator: the founder registers ONE app, every install just gets a button.
  Agent holds no secret at all; the browser hand-back is a single-use 120s code, never the
  wallet_id (which spends real NRN and must not sit in a URL or history).
- **Google forced the HTTPS work.** Google rejects redirect URIs that are plain HTTP or a raw
  IP, so `http://150.230.22.250:8001/...` could not be registered at all — GitHub accepted it,
  but GitHub is a developer platform, so GitHub-only silently capped the audience at people who
  already write code. `coordinator/setup_https.sh` (Caddy + Let's Encrypt, auto-renewing) took
  the coordinator to `https://neuronnet.duckdns.org`. Node tokens no longer cross the internet
  in cleartext either, which mattered independently ([P11]).
- **Verified live, both providers, in the ledger** — `google … email_verified: 1` and `github`,
  distinct identities and wallets. A normal person can now install, click once, and chat.
- **Gotchas worth keeping:** (1) systemd **drop-ins override the main unit**, so
  setup_https.sh's edit to `.service` was silently beaten by an older `PUBLIC_BASE_URL` in
  `.service.d/*.conf` — the coordinator advertised a redirect_uri that no longer matched, with
  no error until a login was attempted. The script now writes the drop-in and prints the
  effective value. (2) DuckDNS defaults a new subdomain to the IP of whoever created it, not
  the server. (3) Oracle has a SECOND firewall in the cloud console; without ingress rules for
  80/443 the certificate request just times out. (4) Changing PUBLIC_BASE_URL invalidates every
  registered redirect URI — each provider's callback must be updated in the same breath.
- **Open:** `RELAY_HOST` stays a bare IP on purpose (raw TCP, no TLS name). The consent screen
  must be PUBLISHED, not left in Testing, or only hand-added test users can sign in (cap 100);
  NEURON asks only for openid/email/profile, all non-sensitive, so publishing needs no review.

### [P17] 🟢 Anyone could mint a funded wallet with no login — the whole ban system was decorative — fixed (2026-07-29)
- **`POST /wallet/faucet` was completely ungated**, unlike its two sibling endpoints
  (`/wallet/oauth`, `/wallet/{id}/violation`), which both verify `X-Wallet-Link-Secret`. And
  `models.claim_faucet` opens with `INSERT OR IGNORE INTO ledger … VALUES (?, 'wallet')` — it
  *creates* the row for whatever string it is handed. Chained with `api/openai_compat.py`'s
  `_auth()`, which accepted any non-empty bearer string as a wallet with zero validation:

      POST /wallet/faucet {"wallet_id":"abuse-1"}  -> funded wallet, no account, no cost
      use "abuse-1" as the API key                 -> full anonymous model access
      banned -> mint "abuse-2"                     -> unlimited, instant ban reset

  So every ban was one HTTP call away from being reset, and there was no real identity behind
  any request to act on. `SAFETY.md`'s "repeat violations escalate against your wallet
  identity" was, in practice, unenforceable.
- **Fix (three gates):** the faucet now requires the operator secret **and** a wallet already
  linked to a real Google/GitHub login; **`/infer` refuses any wallet with no login behind it**
  — the load-bearing one, since every driver must call `/infer` to get a node chain, so it is
  the one check a user cannot patch out of their own client; and the API verifies bearer keys
  against the coordinator (60 s TTL cache, so it is not a per-request round-trip).
- **Plus the operator lever that was missing:** bans previously only fired via the automatic
  `MODERATION_BAN_THRESHOLD`, which counts violations the **driver self-reports** — and for a
  self-hosted install the driver is the user's own machine, so a stripped client never reports
  itself. Added `/admin` (identity console: provider, provider-verified-email flag, violation
  and request counts, last seen, per-identity history, ban/unban) backed by secret-gated
  endpoints. 23 regression tests in `coordinator/test_identity_gate.py`.
- **Still true:** a determined abuser can make a fresh Google account (`SAFETY.md` says so
  honestly), and **content policy remains unenforceable on a driver NEURON does not run** —
  that user holds the plaintext and the moderation code. Against them the control is not
  content-based at all: they need a real account to get a chain, and that account can be banned.
- **Also added:** `requests`-table retention (`NEURON_REQUEST_RETENTION_DAYS`, 90d default,
  pruned on the health sweep). It is the only table that grows with *traffic* rather than users
  — ~1.25 GB/day at 1M users × 5 requests, which no single-file SQLite on the 1 GB coordinator
  VM survives. Identities, ledger rows and `moderation_events` are never pruned: bans depend on
  them. **Open:** SQLite on a 1-core/1 GB VM will hit write-concurrency limits in the low
  hundreds of concurrent users, well before disk — Postgres is the migration when that nears.

### [P16] 🟢 Auto-placement piled every new machine on the tail, capping network throughput at ONE request — fixed (2026-07-29)
- **Found by simulating the founder's question "what happens when 10 users hit send at once?"**
  `router.suggest_placement` advised a joining node to *"replicate the LAST segment"* once the
  chain was complete. With layers split 0-9 / 10-18 / 19-27, seven machines joining a 3-node
  network **all** landed on 19-27. Every request still funnelled through the single node holding
  0-9 and the single node holding 10-18 — and `agent/node_server.py`'s module-level
  `compute_lock` serialises each machine's forward pass, so those two ran strictly one request
  at a time. Simulated: **node_a served 10/10 concurrent requests, node_b 10/10.** Ten machines
  delivered one machine's throughput; the seven added zero.
- **Why it hid:** the policy's own comment claimed replicating the tail "adds throughput", and
  `[P8]`'s segment-level replication (S18) genuinely does work — `build_chain` picks among
  replicas correctly at *every* cursor position. The routing engine was never the problem; it
  was being handed a lopsided topology. Nothing tested the resulting layout, only that a single
  request could be routed.
- **Fix:** placement now replicates the **least-replicated stage**, ties breaking toward the
  earliest (every request traverses the front first, so a shortfall there throttles everything
  behind it). Same 10-machine simulation now spreads 4/3/3 across stages, busiest node 5/10
  instead of 10/10. Regression tests in `coordinator/test_placement.py` assert the layout is
  balanced and that many distinct parallel chains exist.
- **Still true after the fix (physics, not bugs):** concurrent users ≈ machines ÷ machines-per-
  chain, so ~3 machines per concurrent user at 28 layers — 100k concurrent needs ~300k machines.
  And `compute_lock` means **no batching**: a datacenter GPU amortises one forward pass across
  many users, this cannot, which is a genuine structural ceiling. Related: `[P13]` — latency,
  not concurrency, is the nearer blocker (a session is ~40 min wall-clock today).
- **Also corrected the same day:** `TOKENOMICS.md` §0/§8's "green AI / no new power draw" claim,
  which §11.5 already contradicted (*"the ~0.1 W green figure describes idle, not inference"*).
  Consumer-CPU inference costs *more* energy per token than a datacenter GPU; the defensible
  claims are no-new-hardware, sovereignty, and privacy-by-architecture.

### [P12] 🟡 Ledger MINTS per request + payout path is unauthenticated (economics integrity)
> **PAYOUT-AUTH HALF RESOLVED (2026-07-25, Session 19).** `/infer` now issues a per-request
> `complete_token` and records the chain it chose; `/complete` requires that token (wrong/missing →
> 401) and **settles from the coordinator-recorded plan, never caller-supplied `node_ids`** — so a
> completion can no longer mint NRN to an arbitrary node or to the unchosen replica, and third parties
> can't forge/replay completions. `tokens_generated` is clamped to `max_tokens`. Test:
> `coordinator/test_complete_auth.py` 13/13. **STILL OPEN:** the ledger still MINTS per request (no
> user debit, no fixed-supply enforcement) — that is the economics rewrite in TOKENOMICS.md §11
> (genesis buckets + wallets + debit + settle), deliberately deferred until after the first stranger.

- `coordinator/ledger.py` creates 1.0 NRN out of nothing per completed request (node credit
  at :35, and the 0.10 fee at :38-39 mints **unconditionally**, even for a 0-node chain) —
  directly contradicts TOKENOMICS.md's fixed 1B supply ("all issuance from the emission
  schedule, not open-ended minting"). No debit function exists anywhere ([P5] is the
  user-side half of this).
- `POST /infer/{id}/complete` (`coordinator/main.py:222-231`) has **no auth** and trusts
  caller-supplied `node_ids` + `tokens_generated` — anyone who can reach the public cloud
  coordinator can mint NRN to arbitrary registered nodes.
- Price constant is duplicated (`coordinator/config.py:19` vs `api/openai_compat.py:45`) —
  already drifted once; needs a single `GET /pricing` source.
- **Fix designed (2026-07-25): TOKENOMICS.md §11** — genesis buckets + transfer-only
  settlement + sum==1e9 invariant + authenticated /complete settling from the
  coordinator-recorded pipeline plan. ~2-4 sessions.

### [P13] 🟡 Prefill path is UNMEASURED — blocks token pricing AND may be a UX killer
- Per-chat-turn compute spans **3.2×** (28 vs 91 node-seconds) depending on whether prefill
  is batched or token-sequential; if sequential, TTFT on a 200-token prompt is ~152 s (fatal
  for chat) and any discounted input-token price undercharges real compute ~4×.
- Also a farming surface: under any work-metered subsidy, cheap batched prefill + expensive
  metering = prompt-stuffing exploit (see TOKENOMICS.md §11.4/§11.8).
- **Action: measure prefill (one selftest run with a long prompt, time the prefill pass)
  before publishing any per-token price sheet; charge input at full weight until then.**

### [P14] 🟢 Pre-launch audit (2026-07-28) — 4 real bugs found and fixed before the first real stranger
Three parallel audits (security / economics / reliability) against the current code, each
finding independently spot-verified by reading the actual source before acting on it.

- **RESOLVED — Relay had zero authentication on tunnel registration.** Any stranger who could
  reach the public relay control port (`150.230.22.250:8010`) could register `{node_id,
  public_port}` with **no credential at all**, and `relay.py`'s `self.controls[pub]`
  unconditionally overwrote whatever was already registered on that port — a stranger could
  hijack (traffic interception) or blackhole any NAT'd node's public port, or squat an
  unclaimed one. This is exactly the path a real behind-NAT stranger (`behind_nat: true` is
  the agent default) depends on. **Fix:** new `relay_auth.py` — the coordinator is the only
  party that knows the `node_id -> public_port` binding, so it mints an HMAC ticket
  (`HMAC(shared secret, node_id + ":" + public_port)`) at registration time and hands it to
  the node; the relay (still a DB-less, dependency-free stdlib process) independently
  recomputes the same HMAC to verify, no callback to the coordinator needed. Binding node_id
  + port together blocks replay onto a different port; without the secret an attacker can't
  mint a ticket for ANY node/port, closing squatting too. `coordinator/config.py`
  `RELAY_SECRET` (env `NEURON_RELAY_SECRET`) shared with `relay.py --secret`. Threaded through
  `tunnel_client.run_tunnel(ticket=...)` and `agent/agent.py`. Test: `test_relay_auth.py` 9/9
  + `coordinator/test_open_join.py` (ticket issued/verified on `behind_nat` register).
- **RESOLVED — Duplicate-payout race in `/infer/{id}/complete`.** The endpoint read
  `req["status"]=="completed"` as a pre-check, then called `ledger.distribute(plan)`
  **without checking** whether its own `models.complete_request()` call actually won the
  atomic `UPDATE ... WHERE status='pending'`. N concurrent completion calls with the SAME
  valid `complete_token` (trivial with `asyncio.gather` — anyone holding a token from their
  own `/infer` call, which open join lets any stranger make, could do this to themselves)
  would all pass the pre-check before any of them committed, and every one of them minted a
  fresh payout. **Fix:** `coordinator/main.py`'s `complete()` now gates `ledger.distribute()`
  on `complete_request()`'s own return value — only the caller that actually flips the row to
  `completed` gets paid; every racer after that gets a clean 409, not a second mint. Verified
  the fix by first reproducing the multi-payout with 8 concurrent racers against the
  pre-fix code (reliably failed), then confirming the fix closes it (reliably passes, run 3×).
  Test: `coordinator/test_complete_auth.py` (`8 concurrent racers -> exactly 1 paid`).
- **RESOLVED — Verified (non-trusted) nodes could be identity-hijacked on re-registration.**
  `POST /node/register`'s hijack guard only blocked a secret-less re-registration when
  `existing["trusted"]` was true — a node that open-joined and passed proof-of-compute
  (`standing="verified"`, `trusted=False` in the DB) was **unprotected**: anyone could
  re-register that exact `node_id` with no secret, the `ON CONFLICT` update would hand them
  a **fresh `node_token`**, and they'd inherit the victim's verified standing while silently
  locking the real owner out of their own dashboard/balance. **Fix:** the guard now checks
  `existing["standing"] in ("trusted", "verified")`; the one way around the secret is
  presenting that exact node's *current* `X-Node-Token` (proves you already control it —
  legitimate self-recovery, e.g. after losing local config, stays possible). Test:
  `coordinator/test_open_join.py` (`hijack of verified id blocked (409)`, wrong-token
  rejected, correct-token re-register succeeds).
- **RESOLVED — Model migration ([P-tiering], `coordinator/migration.py`) had no
  offline-eviction; one flaky node could wedge it forever, invisibly.** A node's plan was
  only recomputed when entering "preparing" fresh or the target model changed — never on
  later ticks. A planned node going offline mid-preparing (very plausible for a real
  stranger's laptop under the `idle` donation mode, which pauses on any owner activity)
  meant cutover required `planned <= ready` forever against a node that would never report
  ready again — silently stuck, since the dashboard didn't surface migration phase at all
  (only the raw `/network/migration` JSON did). **Fix:** `update()` now replans (against
  whoever is currently online+eligible) whenever a planned node drops out, not just on a
  target change; the dashboard now shows an in-progress migration inline. Test:
  `coordinator/test_migration.py` (`test_replan_when_a_planned_node_drops_offline`,
  `test_no_replan_when_nothing_changed` — a plain tick must NOT reset ready progress).
- **Found live while deploying this fix, also shipped — the relay's public control/data
  ports were ALREADY receiving malformed data from the open internet** (background
  scanners, most likely — not necessarily a targeted attack) that crashed handler threads:
  the journal on the live VM showed repeated `MemoryError` in `relay.py`'s `_recvn`, because
  an unvalidated 4-byte length prefix let garbage bytes be read as a ~4GB length and reach
  `sock.recv(huge_number)` — a real resource-exhaustion risk on a 1GB-RAM VM, not just log
  noise. **Fix:** `recv_json` now caps the declared length at `MAX_MSG_BYTES` (64KB, generous
  for a handshake message) and catches decode/parse errors, treating any of it as "not a
  real client" (clean `None`, connection dropped) rather than an unhandled exception. Test:
  `test_relay_auth.py` (huge length / undecodable bytes / malformed JSON all handled; a real
  message still round-trips).
- **Cheap fix, also shipped — mid-request node drops failed cleanly but logged nothing
  server-side** (`ui/app.py`'s SSE error branch). Now logged via `logging.getLogger
  ("neuron.ui")` so a real stranger's dropped connection shows up in the server log instead
  of only ever being visible if they report it themselves.
- **Accepted, not fixed now — zero monitoring/alerting beyond stdout/systemd logs, and no
  database backup mechanism** beyond the one manual pre-deploy snapshot noted in
  `neuron-machines` history. `Restart=always` in the systemd unit covers process crashes;
  SQLite corruption/data loss has no recovery path. Reasonable to leave for a single-friend
  test; revisit before any wider rollout.
- **Accepted, inherent to open-join — Sybil replica-stacking.** `router.build_chain` picks
  uniformly among nodes tied for a segment; nothing stops one machine registering many
  `node_id`s to capture a disproportionate share of the random draws. Proof-of-compute
  verifies correctness, not identity uniqueness. A fundamental open-join tradeoff, not a bug
  — no hardware fingerprinting exists to fix it, and it doesn't inflate total mint (see [P12]).
- **Confirmed solid, no action needed:** `/complete` still correctly settles from the
  coordinator-recorded plan (never caller-supplied `node_ids`) — [P12]'s fix holds up under
  concurrency too; SQL is parameterized everywhere (no attacker-influenced column names);
  node tokens are never logged; no CORS/debug/reload misconfiguration; `/node/list` and
  per-node dashboards stay correctly privacy-gated ([P11] holds); `tunnel_client.py`'s
  earlier timeout-churn bug stays fixed; reputation/proof-of-compute is manual-only today so
  a flapping stranger node cannot get auto-flagged just for churning.

---

### [P15] 🟢 macOS idle-detection silently always-idle — fixed (2026-07-28)
The founder's friend (the planned first real stranger) has a Mac. `INSTALL.md` lists macOS as
supported for the source-install path, but `agent/resource_guard.py`'s idle detection only
handled Windows (`GetLastInputInfo`) and Linux (`xprintidle`) — on macOS neither branch
matched, so it silently fell through to the headless-server default (`return 1e9`, "always
idle"). Under the default `idle` donation mode (which is supposed to yield the moment the
owner touches the machine), a Mac node would never actually detect real user activity and
would keep donating compute while the owner was using it — defeating the mode's whole point.
Not a crash, a silent correctness gap, found by directly re-checking the code rather than
assuming "macOS: supported" in `INSTALL.md` was accurate. **Fix:** `_macos_idle_seconds()` —
pure `ctypes` call to `CGEventSourceSecondsSinceLastEventType` (CoreGraphics), no `pyobjc`
dependency, matching the module's existing minimal-deps style; needs no special OS permission.
Wired into `seconds_since_input()`'s dispatch, fails safe to always-idle if the framework
somehow isn't loadable. Never tested on real macOS hardware (this dev environment is
Windows) — verified by flipping the platform-detection flags and mocking the CoreGraphics
call (`agent/test_resource_guard.py`, 2 new cases, 18/18 total). **Still open:** no macOS
packaging exists at all (no `.dmg`/PyInstaller build — only Windows has a real installer);
the friend would need to run the source-install path (`INSTALL.md`).

**Update (2026-07-28, same day) — macOS `launchd` auto-start added.** `agent/install.py`'s
optional `--startup` helper only handled Windows (registry) / Linux (`systemd --user`) —
found by the same "verify, don't trust the platform list" check: on macOS it took the Linux
branch and would have crashed outright (`FileNotFoundError: systemctl`, no systemd on
macOS). Added `add_to_startup_macos()` — writes a per-user LaunchAgent plist
(`~/Library/LaunchAgents/com.neuron.agent.plist`, `RunAtLoad`+`KeepAlive`) and loads it via
`launchctl load -w`; `agent/uninstall.py` mirrors it (`launchctl unload -w` to stop, then
removes the plist file). `start_background()`'s macOS path spawns `agent.py` directly
(like Windows) rather than assuming a LaunchAgent already exists, so `--no-startup` still
works. New `agent/test_install_macos.py` (13/13, mocks `launchctl`/file paths — no real Mac
available to test against). This is still the OPTIONAL auto-start helper, not the base
`INSTALL.md` flow (`python agent/agent.py` run directly) — the base flow already worked on
macOS without this.

---

## Not-doing (deliberately, for now)
- Chasing single-user latency parity with GPU clouds — unwinnable, wrong hill.
- Productionising quantization before there are real users (measure first, integrate later).
- On-chain NRN before Session 12 (first stranger node) — per ROADMAP.
