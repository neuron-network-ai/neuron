# Changelog

Versions are the **installer/agent** version — the number in `NEURON-Setup-<x>.exe`, in
`config.AGENT_VERSION` and in `updater.LOCAL_VERSION`, which must all agree or a node either
never updates or tries to update to itself forever.

This project is early alpha. NRN has no cash value, and the network is a handful of machines.

## Unreleased

Coordinator-side only — nothing here changes the agent, so no installer version is needed.

- **The emission ledger can now be checked against its own attendance rows** —
  `coordinator/reconcile_emission.py` ([P40]). 262.89 NRN has been distributed and nothing had
  ever compared what the ledger *holds* against what the attendance rows *say it should hold*.
  Read-only by construction: one `mode=ro` connection, no `--execute`, and no SQL that is not a
  `SELECT` — each asserted from the module's syntax tree, plus a test that hashes a database
  before and after a full run. Exits 1 on a discrepancy, so it can be scheduled.
  **Run on the live ledger 2026-08-17 and it reconciles**: 330.427894 NRN recorded against a
  330.4278938 NRN drop in the pool, every one of 316 settled rows re-pricing correctly from its
  own frozen inputs, supply invariant intact. First independent confirmation that the NRN
  distributed so far is correct.
- **It says why a node-hour earned nothing, and when.** 199 of 318 settled hours on the live
  ledger paid zero — all refused by a gate rather than zeroed by the daily cap. Now counted per
  reason, per node, with the window each run of them falls in. That last part is what turned a
  statistic into [P47]: the misses run to the current slot, and the worst-hit machine is the
  driver, which `verify_service` skips by design and which therefore **cannot earn availability
  emission at all**. Roughly 243 NRN of unearnable hours against 333 NRN distributed.
- **Fixed: a payment the pool could not make was recorded as though it had been made.**
  `close_slots` claims the attendance row before it moves the money — correct, since the
  alternative can pay twice — and walked the reward back with
  `settle_attendance(..., 0.0)` when the transfer failed. That UPDATE carries
  `WHERE paid_at IS NULL`, which the claim four lines earlier has just falsified, so it could
  never fire: the row kept its full reward, and `emitted_since` — what the daily cap is read
  against — counted NRN that never moved. `models.void_settlement` replaces it. Latent (the
  pool holds ~600M against 262 spent) and found by asking what would make the new
  reconciliation's two legs disagree.
- **Genesis records what the emission pool was seeded with** (`settings['emission_pool_seed']`).
  The seed is `600,000,000` minus a backdate computed from balances that have since moved, so
  after the fact it is not recoverable — meaning "how much has left the pool" could not be
  answered from the database at all. New databases only; the live one predates the line and
  needs `--seed`.

## v0.20.4

Same release as 0.20.3 plus the wallet work that landed after it was built. Cut as a new
version rather than rebuilt as 0.20.3, because different bytes under a published version is
exactly what `agent_version` exists to make impossible — the coordinator advertises a SHA, and
a second build of the same source does not match it.

- **You can see how much of the free grant is left** — `≈ N network answers left` in the
  sidebar, with a nudge to contribute below 30. Hidden entirely when the balance could not be
  read, because "we could not ask" must never render as a confident number.
- **Running out explains itself.** The grant is one-time by design: NRN pays other volunteers
  for their electricity and CPU, and an automatic top-up would commit their hardware to
  unlimited free use. The message now says that, and names the part nobody would guess —
  contributing this machine needs *less* memory than running the model on it does.

## v0.20.3

The release that makes a model too big for one machine actually placeable, and the one that
finally had somebody click the claim button.

- **A volunteer can cap how much of their machine they lend** — `donate_ram_gb` in the agent
  config. `donation_mode` has always governed WHEN a node serves; nothing governed HOW MUCH of
  the machine it commits, so someone with a 64 GB workstation happy to lend 8 GB had to choose
  between the whole thing and nothing. It is a declaration the coordinator enforces rather than
  a runtime limiter: the agent reports the capped figure as `ram_gb`, the sizing path uses it,
  and the node is never handed a slice bigger than the cap.
- **The driver is charged for the weights it actually holds.** Every node was sized by
  `gb_per_layer * layers`; the first one also holds the embedding and `lm_head` and nobody
  counted them. A fixed cost, so it hurts most on the fewest machines — 41% of an 8 GB node's
  budget on Qwen3-4B.
- **A node reports the precision it stores weights at.** `weight_dtype` was read by the
  balancer and could never arrive — no field, no column — so every node was sized at 4
  bytes/param whether or not it ran fp16. This is what decides whether a 4B model fits across
  two machines.
- **Stage-1 width follows placement instead of an environment variable on two machines.**
  `NEURON_S1` had to be identical in the coordinator and in every driver, both reading it at
  import, so changing it meant a coordinated restart across machines nobody can reach. The
  coordinator owns it now and publishes it; the driver derives its width from the shard it
  downloaded. Stage 1 is therefore per-MODEL, which is what a 36-layer model over two machines
  needs.
- **Auto-repair checks the driver, and stops covering the tail silently.** It capped every
  stage except the two that most need it. The tail is still assigned when it does not fit — a
  gap means not one request completes — but the repair log now names the node and how far over.
- **An old CPU is told why it cannot run this**, instead of dying at `0xC000001D` with no
  message. Refuses only a positive determination of x86-without-AVX2, names an override, and
  says plainly that the risk is documented rather than reproduced.
- **`NEURON_WEIGHT_DTYPE` with a trailing space** raised `KeyError` at import, before the node
  server's logging existed, while the agent reported the stripped value to the coordinator — so
  the machine was sized for half the footprint of a process that was not running.
- **The node-earnings claim panel exists in the React UI** and has been exercised end to end
  for the first time: declining a signature reports that nothing changed and leaves the button
  usable, and the bind request carries no wallet id. `/next` had no panel at all, so swapping
  the routes would have silently dropped the feature.
- **`tools/measure_model.py`** reads a model's published safetensors header and emits a tier
  row, replacing hand arithmetic in a comment. **`tools/capacity_dryrun.py`** says what the
  coordinator will decide before it is deployed, through the coordinator's own functions.
- Version now checked in lockstep across `updater.LOCAL_VERSION`, `config.AGENT_VERSION` and
  `neuron.iss` — the coupling this file's own header describes, previously enforced by a
  comment addressed to a human.

- **Every node on every tier was cleared to hold twice what it can.** The tier table's
  per-layer figure is computed at fp16, and its comment justified that with a claim about the
  runtime that the runtime contradicts: weights are stored at **fp32** by default. So an 8 GB
  machine on the 7B tier was cleared for 8 layers — 7.46 GB of weights against a 3.75 GB
  budget. It stayed latent only because the network serves the 1.5B, where 28 layers is too few
  for the cap to bind. The coordinator now scales the tier figure by the dtype a node actually
  stores at, pessimistically by default and exactly once on every path. Nothing about what the
  network serves today changes; what it will be cleared to serve as machines join does.
- **Pausing a node now actually stops it taking work.** `NodeServer.paused` existed and was
  never read: the agent never passed its own flag in, so the tray's Pause only skipped the
  heartbeat and the coordinator kept routing live requests to the machine for up to ~90 seconds
  while its owner believed it had stopped. A paused node now refuses **new** requests and
  **finishes ones already in flight** — the user who asked first is not punished for someone
  else's pause. The refusal is a named reply rather than a closed socket, because
  `ConnectionRefusedError` is a `ConnectionError` and the driver cannot tell a slammed socket
  apart from a dead machine; it would rebuild the chain, replay the junction cache and report
  "a machine dropped out". A refusal by name reroutes cleanly and says what really happened.
- **A migrating node no longer holds two model slices at once.** `reload()` loaded the new
  slice while the old one was still referenced, peaking at ~150% of one slice on machines
  picked because they had room for one. It now releases the old slice — and its batchers, which
  close over it — before loading, with `empty_cache()` in the gap for a GPU node. Requests
  arriving in that window get a named "reloading" refusal instead of an `AttributeError` from
  inside a batcher. Stated plainly because it is a real trade: a load that fails now leaves the
  node with no slice, where before it kept serving the old one.
- **The pipeline follows its own device instead of assuming CPU.** Batching's auxiliary tensors
  (KV padding, attention mask, position ids, cache positions) derive their device from the
  activations; the batched stages return CPU tensors for the wire exactly as their unbatched
  twins always did; the wire codec's Hadamard matrix follows its operand and every `.numpy()`
  reaches CPU first. **Bit-identical on a CPU machine** — the batched-vs-sequential figure is
  `4.530e-06` before and after, unchanged to the digit. Also: the node self-benchmark now calls
  `torch.cuda.synchronize()`, without which a GPU node would have timed how fast it can queue
  work and been handed nearly every layer in the network as a result.
- **A tripwire on the one line that would turn all of this on.** `slice_downloader` still does
  not move weights to a device, on purpose, and `test_device_path.py` fails if that changes —
  naming what has to be verified on real hardware first, and failing too if VRAM is counted as
  capacity while the loader leaves weights in system RAM. Those two facts are a pair; splitting
  them is what produced the v0.18.0 OOM.
- **A compute-device choice, decided at install and changeable from the tray.** The installer
  now detects what the machine can actually use and writes `device: "cpu"` or `"gpu"` into the
  config; the tray gains a **Compute device** dial (Automatic / CPU only / GPU) beside the
  donation one. Because `common.DEVICE` is resolved once at import — and the agent imports
  `node_server` → `common` at module level — the setting is applied *before* those imports, not
  in `main()`, where it would have been read, saved, ticked in the menu and done nothing. A
  change applies on restart, and the menu says so. **Choosing GPU on this build reports "GPU
  selected — this build computes on CPU" rather than silently doing nothing.**
- **Fresh installs now donate `balanced` instead of `idle`** — the node contributes while you
  work, yielding above 50% CPU and on battery, rather than only when the machine is untouched.
  Existing configs are not changed.
- **Correction: v0.18.0's "Inference can now run on the GPU" and "VRAM is now capacity" were
  wrong, and the second was a live hazard.** Those two entries are left below as published; this
  is the correction. The build ships `torch 2.4.1+cpu` and a `llama-cpp-python` with no GPU
  backend (`llama_supports_gpu_offload()` → `False`, no `ggml-cuda` in the package), and the
  loader every volunteer node uses returns its slice **without moving it to a device**. So no
  inference has ever run on a GPU. Because the weights were in system RAM while the coordinator
  sized nodes by VRAM — with no OS reserve on that branch — a 12 GB card in an 8 GB machine was
  eligible for **19 layers** of a 7B model. `balancer.GPU_EXECUTION` is now **off**, which
  restores the pre-Session-42 sizing for every already-installed agent **on coordinator restart
  alone, with no update required**. The v0.18.0 caveat blamed the absent test card; the cause
  was packaging, and the capability had never executed once. See [P31].
- **A reported VRAM figure is bounded, and a stale one expires.** `gpu_vram_gb` is the only
  registration field that becomes a memory budget, and open join means it arrives with no
  credential: it is now clamped to a believable range (and to `None` when it is not) — never
  rejected, because a 422 over a cosmetic hardware field locks a volunteer out for a reason
  nobody can see. A node that loses its card no longer keeps phantom VRAM forever.
- **The local engine no longer claims an offload that cannot happen.** `n_gpu_layers` is
  accepted and silently ignored by a llama.cpp build with no GPU backend, so the log said
  "offloading every layer" while every layer ran on the CPU. The build capability is now checked
  **before** the hardware, and `NEURON_GPU_LAYERS` is refused the same way — an override that
  silently does nothing is the same bug with a manual trigger.
- **A node prints where its weights actually are.** Every slice load logs `weights on cpu`
  (or `cuda:0`), read off the loaded tensors rather than from the configured device. `INSTALL.md`
  asked first-GPU volunteers to report a `device: cuda:0` line that comes from a code path the
  agent never reaches, so it could not have appeared on any machine.
- **The GPU-yield branch is tested.** `test_resource_guard.py` shelled out to the real
  `nvidia-smi`, which on a machine with no card can only answer "no card" — so the `gpu_ceiling`
  branch had zero coverage. It is stubbed now, with cases for a busy card, an unreadable one and
  a probe that raises.
- **A model upgrade can no longer hand a machine more layers than it can hold.** The network
  promoted itself onto Qwen2.5-7B and split it evenly across nodes with 8 and 12 GB of total RAM
  — ~4.5 GB of weights each, on machines someone is also using. The tier gate qualified that on
  *aggregate* capacity ("3 nodes · 20 GB"), which says nothing about whether one machine can hold
  one slice, and the partitioner had no memory awareness to catch it. Now: the split is fitted to
  each node's reported RAM (or VRAM), a target no per-node partition can hold is refused outright
  with the reason on the dashboard, and the tier ladder marks a model the network can afford but
  cannot place as "won't fit" rather than "ready". The balancer's memory cap — written in Session
  14 and never once applied to a real plan, because it read a field no node reports — is live.
  See [P26] in `PROBLEMS.md`.
- **The dashboards look like the rest of NEURON.** `/dashboard` and the private per-node page
  were unstyled system-grey in Google's console palette, so following "Live network dashboard →"
  from the site crossed a visible seam into what looked like a different product. Both now render
  through `coordinator/theme.py`, which carries the landing page's palette, type and component
  shapes. No web fonts are fetched: the page hard-refreshes every 5 seconds, and a font CDN
  would make that a repeated third-party request from a page about a privacy-preserving network.
- **The dashboard shows what the coordinator already knew.** A **layer coverage strip** naming
  exactly which layers have no node — "21/28" said the chain was broken without saying where,
  which is the one thing needed to fix it; `uncovered_layers` is now in `/status` too, and
  `neuron_doctor.py` names the gap. Plus per-node GPU, measured ms/layer and proof-of-compute
  record, aggregate cores/RAM/GPUs, tokens generated, the agent version, and a plain statement
  when nodes are stuck awaiting verification. Node addresses, GPU model names and per-node
  balances remain off the public page.
- **The chat UI says what it is doing.** The wait before the first token — the defining moment
  on a ~1 tok/s volunteer network — was a blinking cursor, indistinguishable from a hung page,
  while the event carrying the assembled chain had already arrived and was being held until the
  answer finished. It now shows live ("chain built across 3 machines · waiting for the first
  token · 0:07") and explains why the first token is the slow one. Sending into a chain that
  cannot answer is refused up front with the reason instead of costing a wait and an error.
- **Recovery from a node dying is now visible as recovery, not as a failure.** A machine
  dropping out mid-answer is rebuilt around and the answer continues token-identically, but the
  stream pauses while a new chain is built and the conversation so far is replayed into it. That
  pause used to look like a hang; the page now says "a machine dropped out — rebuilding the
  chain and picking up where it left off". A completed answer that survived one records it
  ("↻ recovered from 1 node drop") — the driver always tracked this and the UI was dropping it,
  so surviving two node deaths looked identical to an uneventful request.
- **A partial answer now survives the failure that ended it.** The error handler overwrote the
  message body, deleting text the user had already received and paid for. What arrived is kept
  and can be retried. The message no longer blames a node going offline — that case is
  recovered; reaching the error path means recovery itself was exhausted, and it says so. An
  answer that hits the 128-token cap says so instead of appearing to stop mid-sentence, and
  offers to continue.
- **Chat UI accessibility.** `aria-live` on the message thread (a screen reader previously got
  silence for an entire generation), labels on icon-only controls, `prefers-reduced-motion`
  support, and `100dvh` so the composer is not hidden behind a phone's address bar. Its palette
  also moved off indigo onto the brand green, which fixed a Send button that was 2.9:1 in dark
  mode — below AA — and a warning strip hardcoded to a near-white that ignored the dark theme.
- **A node token can no longer leak out of its own dashboard.** The private page is reached at
  `?token=<node token>`, so any outbound link handed that token to the destination in the
  `Referer` header. It now carries no third-party links and declares `referrer: no-referrer`.
- **The agent's config can no longer be left half-written.** `_save()` truncated the file and
  then wrote it, with no explicit close — so anything that stopped the process mid-write left a
  truncated `config.json`, and the next start died parsing it before logging existed. It now
  writes a temp file, fsyncs, and renames atomically, keeping the previous copy; an unreadable
  config is recovered from that copy rather than replaced with defaults, because `node_id` and
  `node_token` are the node's claim on everything it has earned.
- **Moderation no longer treats "we have no patterns for this language" as "clean".** The
  blocklist was English-only against a multilingual model, and two bugs made fixing that
  impossible: `\b` word boundaries never match inside scripts without spaces between words, and
  the blocklist was read with the locale codepage so any non-ASCII term crashed the gate. Both
  fixed, Chinese terms added for the existing categories, and unscreened requests are now
  recorded so the coverage gap is measurable. See SAFETY.md for the honest limits.
- **A startup failure can no longer be silent.** v0.18.0 was installed on a real machine and
  produced no log file at all, which made the failure undiagnosable — a run that leaves no
  trace is indistinguishable from one that never happened. Logging is now configured as the
  first statement of `main()`, ahead of the config read that used to precede it, and the
  module-level imports (torch among them), the config read and the frozen tray entry point
  each record their own death to `agent.log` with the standard library alone. In windowed tray
  mode, where there is no console, a crash also puts the log's location on screen. The log's
  first line now names the version that wrote it.
- **A config written by an older build no longer takes the agent down.** An in-place upgrade
  reads the previous version's `config.json`; a key this build expects and that one never
  wrote was a `KeyError` in the constructor, before anything was logged. Missing keys fall back
  to the built-in defaults and are named in the log, not silently injected into the user's file.
- **A node says when it is online and useless.** Every heartbeat now carries the node's
  standing, so an agent that is still probationary — excluded from routing, earning nothing —
  says so every 30 minutes instead of logging `heartbeat ok — active` indefinitely, and
  announces its promotion when it comes. PROBLEMS.md [P24]: a stranger's node ran that way for
  three days because the operator's verifier had died.
- **The verifier survives the weather, and proves it is alive.** `verify_service.py` retries a
  failed roster read within the cycle (a home connection produces 502s and DNS failures
  routinely), escalates a sustained outage from WARNING to ERROR, and writes a periodic alive
  line so that silence in its log means *dead* rather than *idle*. `neuron_doctor.py` fails
  when that log stops growing, and `agent/verifier_keepalive.py` (installed via
  `install.py --verifier-keepalive`) restarts it every five minutes if it is not running.
- **NVIDIA GPU capability is detected and reported.** A node reports `has_gpu`, `gpu_vram_gb`
  and `gpu_name` at registration; the coordinator stores them, and the balancer prefers a GPU
  node when two candidates tie on measured speed. `gpu_name` is operator-only in `/node/list`,
  like `platform` and node addresses — a card model is distinctive enough to correlate a roster
  on.
- **A node yields while its GPU is busy.** Each donation mode gained a `gpu_ceiling`, so a
  machine whose CPU looks idle while a game or a render saturates the GPU now pauses. Machines
  with no `nvidia-smi` are unaffected: unreadable utilisation never counts as busy.
- **Inference can now run on the GPU.** `common.py` resolves an execution device once at import
  (CUDA when present, CPU otherwise, `NEURON_DEVICE` overrides both) and moves a node's shard
  onto it. Every stage still takes and returns CPU tensors, so the wire codec, the relay,
  batching and proof-of-compute are untouched. TF32 is left off on purpose: its ~1e-3 drift from
  CPU arithmetic would spend a fifth of the tolerance proof-of-compute uses to tell honest work
  from cheating.
- **VRAM is now capacity.** `balancer.GPU_EXECUTION` is on, and a GPU node is sized by its VRAM
  *instead of* its system RAM — not the sum of the two, which would claim roughly double the
  memory that exists and cause the exact OOM the cap was written to prevent.
- **Unverified, and stated plainly:** no machine in this project has an NVIDIA GPU and the
  installed torch is a `+cpu` build, so **the CUDA branch has never executed**. What is proven
  is that it is inert on a CPU machine — `selftest_shard.py` still reports
  `max|delta| = 0.000e+00` against the unsharded model. No speedup figure is claimed.
- `CONTRIBUTING.md` and this file added; README status refreshed.

## v0.17.1

- **A node can follow the coordinator if it moves.** The coordinator's public address is
  returned on every heartbeat and on registration. An agent adopts a new one only after
  probing it successfully (`GET <new>/node/<id>/ping` with its own token) and keeps the
  previous value — so a typo in one config value cannot strand the whole network at once.
- Insurance shipped before it is needed: `neuronnet.duckdns.org` is intended to stay the
  stable public name, with Cloudflare going behind it rather than replacing it.

## v0.17.0

- **The app updates itself.** It asks the coordinator once a day whether a newer version
  exists, verifies a published SHA-256 before running anything, refuses and deletes the file
  on a mismatch, and never updates mid-request. An empty published hash means no node installs
  anything — the correct failure direction for a mechanism that runs binaries unattended.
- **The local chat UI works again.** 0.16.5 shipped without `_sqlite3`, so the Chat UI and API
  Docs tray entries stayed greyed out with no explanation.
- **Payout keys ship.** `eth_account`/`eth_keys`/`eth_utils` are bundled, so a node can
  generate a secp256k1 key on first run and bind an address it proves it controls.
- Log lines carry the logger name, so `agent.log` says which component spoke.
- `config.AGENT_VERSION` corrected from `0.3.0`, where it had drifted for fourteen versions.

## v0.16.5

- **First public installer.** Published as a GitHub release, with the project moved to the
  `neuron-network-ai` organisation and personal identity stripped from the repository.
- Autostart is ticked by default in the installer, and the support/disclosure URLs are correct.
- Windows still calls the installer "unrecognized" — it is not code-signed, and that needs a
  certificate rather than a code change.

## Earlier (0.1.0 – 0.16.3)

Internal development only; none of these were distributed to anyone outside the project. The
work is logged session by session in [sessions.md](sessions.md) — layer-split inference proven
bit-exact, the KV cache, the coordinator and NRN ledger, the byte-range slice downloader, the
NAT relay, open join with proof-of-compute, peer quorum verification, the non-executable wire
codec, and the fixed-supply ledger.
