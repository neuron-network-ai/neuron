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

### [P21] 🔴 Nothing restarts a node — the network dies on any reboot, and the fix already exists
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
