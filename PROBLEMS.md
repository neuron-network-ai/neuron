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

## Not-doing (deliberately, for now)
- Chasing single-user latency parity with GPU clouds — unwinnable, wrong hill.
- Productionising quantization before there are real users (measure first, integrate later).
- On-chain NRN before Session 12 (first stranger node) — per ROADMAP.
