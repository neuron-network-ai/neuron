# NEURON — Complete Project Roadmap
**Network of Existing Utilised Resources — Open Nodes**

This file is the single source of truth for Claude Code.
Read this + sessions.md before every session.
Never modify this file during a session — it is a reference, not a log.
Session results go in sessions.md only.

---

## What NEURON Is

A distributed AI inference network. A 1MB agent installed on any laptop, desktop,
or Android phone contributes 1–2% of idle CPU to run LLM inference collectively.
Each machine holds a slice of a transformer model's layers. Together they think
as one model — larger than anything a single machine could run.

**The core insight:** The world already has billions of processors running 24/7 at
2–5% capacity. NEURON harvests that idle capacity. No new hardware. No new data
centers. No new power draw. Green AI by design.

**The gap nobody has filled:**
Every existing project (Bittensor, Gensyn, io.net, Petals) requires GPU hardware,
technical setup, staking crypto upfront, or all three. NEURON requires none of
these. One click. Silent. Earns crypto while idle. Works on any device.

---

## Architecture Overview

```
USER
  ↓ types prompt at neuron.network (future UI)
COORDINATOR (always-on server)
  ↓ finds available node chain
  ↓ routes request
NODE CHAIN (strangers' machines worldwide)
  node_a → node_c → node_b → node_a (lm_head)
  each holds assigned model layer slice only
  ↓ inference runs across machines
COORDINATOR
  ↓ collects result, pays NRN coin rewards
USER
  ↓ reads response
```

**Three-layer model:**
- Infrastructure: coordinator + node network
- Intelligence: any open-source LLM (swappable)
- Interface: chat UI + developer API + agent tray

---

## Current Machines

| Node | Machine | IP (Tailscale) | Role |
|------|---------|----------------|------|
| node_a | Windows PC, 63GB RAM | find with ipconfig | Driver + lm_head + layers 0-9 |
| node_b | Dell OptiPlex, Ubuntu, 15GB RAM | 100.114.189.46 | Layers 19-27, coordinator host |
| node_c | HP Pavilion, Ubuntu, 11GB RAM | 100.79.125.112 | Layers 10-18 |

**Coordinator:** running on OptiPlex at http://100.114.189.46:8001
**Code location:** C:\Users\optin\neuron\
**Model:** Qwen2.5-1.5B-Instruct (safetensors, 28 layers)
**GitHub:** https://github.com/raman011sharma-code/neuron (private)

---

## Proven Results (do not re-prove these)

```
Split inference:     bit-exact across 3 nodes ✅
KV cache:            working, autoregressive ✅
Layer rebalancing:   split 9/9/10 optimal for this trio ✅
Parallel throughput: 6.16 tok/s, 3.82× pipeline overlap ✅
Coordinator:         node registry, routing, ledger, health ✅
NRN economics:       1.0 NRN/request, 10% fee, 90% by layer share ✅
Deployment:          coordinator 24/7 on OptiPlex :8001 ✅
GitHub:              code pushed, README stranger-ready ✅
```

---

## Coin Economics

```
Token name:     NEURON (NRN)
Total supply:   1,000,000,000 NRN
Distribution:
  60% → node rewards (earned over 10 years, halving every 2 years)
  20% → founder (Raman Kumar Sharma)
  15% → ecosystem grants and partnerships
   5% → liquidity pool at public launch

Per request:
  1.0 NRN total cost to user
  0.1 NRN → coordinator fee (10%)
  0.9 NRN → nodes, split by layer share (layer_count/28 × 0.9)

Ledger: SQLite in coordinator (Phase 0)
        ERC-20 on Polygon (Phase 1, after 1,000 users)
        Own chain (Phase 2, if needed)
```

---

## Complete Session Plan

### ✅ COMPLETED SESSIONS

**Session 1** — Split inference, 2 nodes, bit-exact
**Session 2** — KV cache, autoregressive, light nodes (sharding)
**Session 3** — lm_head moved to node_a, layer rebalancing
**Session 4** — Parallel throughput proven, 4.61 tok/s on 2 nodes
**Session 5** — 3-node pipeline, 6.16 tok/s, 3.82× overlap
**Session 6** — Coordinator: registry, routing, ledger, health, token auth
**Session 7** — Coordinator deployed to OptiPlex, GitHub push, README

---

### 🔄 ACTIVE SESSION

**Session 8 — Slice Downloader**

Goal: nodes download ONLY their assigned layer shards, not the full model.
This is the foundation of the 1MB agent.

Deliverables:
- slice_downloader.py
  - Maps layers → safetensor shards via model.safetensors.index.json
  - Downloads only required shards + tokenizer (node_a) + lm_head (node_a)
  - CLI: python slice_downloader.py --coordinator URL --node-id X --output-dir Y
- coordinator/main.py gains: GET /node/{node_id}/slice-info
  Returns model_id, layer_start, layer_end, shards_needed, estimated_gb
- Verified: slice loads correctly, selftest bit-exact
- Measured: download size per node vs full model

Success metric: node_a downloads <2.5GB instead of 6GB, inference still correct.

---

### 📋 UPCOMING SESSIONS

---

**Session 9 — Agent Installer (Windows + Linux)**

Goal: package everything into a single installer that a non-technical person
can run. This is the 1MB agent.

Deliverables:
- agent/agent.py — main loop
  - Phones home to coordinator on startup
  - Receives node_id and layer assignment
  - Calls slice_downloader.py to fetch only needed shards
  - Starts node server (node_b/node_c style listener)
  - Sends heartbeat ping every 30 seconds
  - Caps CPU at 2% using psutil
  - Activates only when machine idle > 5 minutes
  - Runs silently in background
- agent/tray.py — system tray icon
  - Green = earning, Grey = idle, Red = error
  - Shows NRN balance
  - Pause/Resume button
  - Exit button (clean shutdown)
- agent/updater.py — auto-update from coordinator
- agent/resource_guard.py — CPU/RAM caps enforcer
- Packaged as:
  - Windows: PyInstaller → neuron-agent-setup.exe
  - Linux: PyInstaller → neuron-agent.AppImage

Success metric: fresh Windows machine installs agent in under 2 minutes,
node appears in coordinator /node/list, starts earning NRN automatically.

Notes:
- Agent code must be ARM-compatible (phones later)
- All comms encrypted TLS to coordinator
- Zero personal data collected — provable by reading code
- One-command uninstall removes everything

---

**Session 10 — Chat UI**

Goal: a web interface where users can send prompts and get responses
powered by the NEURON network. No local model needed.

Deliverables:
- ui/app.py — FastAPI web server
  - GET / → serves chat.html
  - POST /chat → accepts prompt, calls coordinator /infer,
    runs inference through node chain, streams response
  - GET /network → shows live node count and health
- ui/static/chat.html — simple clean chat interface
  - Text input + send button
  - Response streams token by token (SSE or websocket)
  - Shows: "Powered by X nodes worldwide"
  - Shows: "Cost: 1.0 NRN" per response
  - No login required for demo
- Deployable on OptiPlex alongside coordinator

Success metric: open browser, type prompt, get streamed response
from the 3-node network, see token-by-token output.

---

**Session 11 — Developer API (OpenAI-compatible)**

Goal: any app that uses OpenAI's API can switch to NEURON by changing
one URL. Zero code changes for the developer.

Deliverables:
- api/openai_compat.py — FastAPI router
  - POST /v1/chat/completions (OpenAI format)
  - POST /v1/completions
  - GET /v1/models
  - Streaming support (stream: true)
  - API key = NRN wallet address
  - Per-token NRN deduction
- Documentation page at /docs
- Example: curl call that works identically to OpenAI

Success metric: existing OpenAI SDK code works against NEURON
by changing base_url only.

---

**Session 12 — First Stranger Node**

Goal: someone who is not you installs the agent and earns NRN.
This is the most important milestone in the entire project.

Deliverables:
- Make GitHub repo public
- Write installation guide (5 steps, no technical knowledge needed)
- Test agent installer on a machine you have never touched
- First external node appears in coordinator /node/list
- First NRN earned by someone else

Success metric: one node from one stranger, online, earning.
Everything before this was building. This is the product.

---

**Session 13 — Android Agent (Termux)**

Goal: Android phone as a NEURON node.

Deliverables:
- agent/android/agent_termux.py — Termux-compatible agent
  - Detects phone is charging before activating
  - Only runs on WiFi (never mobile data)
  - Stricter CPU cap (1% on phone)
  - ARM-optimised (llama.cpp GGUF path for phone)
  - Termux:Boot integration for auto-start
- Installation guide for Android

Note: iOS deferred — Apple restricts background execution.

Success metric: Android phone appears as node, earns NRN while charging.

---

**Session 14 — Heterogeneity-Aware Auto-Balancing**

Goal: coordinator automatically assigns optimal layer split based on
each node's measured per-layer speed. No manual --s1 --s2 needed.

Deliverables:
- coordinator/balancer.py
  - At node registration: runs benchmark (time 10 forward passes)
  - Records ms/layer for each node
  - Solves for split that equalises stage time
  - Re-balances when nodes join or leave
  - Accounts for lm_head fixed cost on node_a
- coordinator/main.py updated: auto-assign layers on register
- Remove manual layer flags from node scripts

Success metric: new node joins, coordinator auto-assigns optimal layers,
throughput equals or exceeds manual tuning.

---

**Session 15 — Model Versioning + RAG**

Goal: NEURON is not locked to 2026 knowledge. Models can be swapped.
Real-time information retrieved before inference.

Deliverables:
- coordinator/model_registry.py
  - Tracks available models (model_id, layers, shard map)
  - Nodes can serve multiple models (if RAM allows)
  - Coordinator selects model per request
- rag/retriever.py
  - Before inference: search DuckDuckGo for relevant context
  - Inject retrieved text into prompt
  - User gets current information despite model cutoff
- /v1/models endpoint returns available models

Success metric: user asks "what happened in the news today" and
gets a real answer using retrieved current information.

---

**Session 16 — Security Hardening**

Goal: NEURON is safe for strangers to install and safe for users to trust.

Deliverables:
- Proof of Compute: challenge-response verification
  - Coordinator sends test input with known output
  - Node must return correct output to earn NRN
  - Wrong output = no payment + reputation flag
- Node reputation system
  - Track accuracy rate per node
  - Low-reputation nodes get fewer requests
  - High-reputation nodes earn bonus NRN
- Agent code signing (Windows + Linux)
  - Requires code signing certificate
  - Prevents antivirus flagging
- Rate limiting on coordinator endpoints
- DDoS basic protection

---

**Session 17 — NRN Token on Polygon**

Goal: real blockchain coin. Not a database ledger.

Deliverables:
- Deploy ERC-20 contract on Polygon Mumbai testnet first
- Test earn/spend loop with real blockchain transactions
- Migrate coordinator ledger to on-chain
- Simple wallet UI in agent tray (balance, withdraw)
- Legal: get Dutch lawyer opinion on utility token classification
  before mainnet deployment

Note: do NOT do this before Session 12 (first stranger node).
The coin only matters after the network has real users.

---

**Session 18 — Public Launch Preparation**

Goal: NEURON ready for public announcement.

Deliverables:
- Landing page at neuron.network (or github.io temporary)
  - One paragraph what it is
  - One-click download button (Windows + Linux + Android)
  - Live node map (public dashboard)
  - NRN earned today counter
- Press kit: 3 screenshots, 1 architecture diagram, founder bio
- Discord server for node operators
- Referral system: install agent, get referral link,
  earn 5% of what referrals earn forever

---

**Session 19 — Scale Testing**

Goal: stress test coordinator and network at simulated scale.

Deliverables:
- load_test.py: simulate 100 simultaneous users
- Identify coordinator bottlenecks
- Identify node bottlenecks
- Fix top 3 performance issues found
- Document max concurrent users on current 3-node setup

---

**Session 20 — Grant Application Package**

Goal: NLnet + SIDN Fund applications ready to submit.

Deliverables:
- grant/nlnet_application.md — answers to 6 NLnet questions
- grant/sidn_application.md — SIDN Fund application
- grant/technical_summary.md — 1-page technical proof for reviewers
- grant/budget.md — itemised budget (hardware, legal, hosting)
- All sessions.md entries formatted as evidence

---

## File Structure (target after all sessions)

```
neuron/
├── common.py              # shared: model loading, KV cache, TCP framing
├── node_a.py              # driver node: embedding + first layers + lm_head
├── node_b.py              # last layers server
├── node_c.py              # middle layers relay
├── selftest.py            # bit-exact correctness test
├── selftest_shard.py      # 3-node chain correctness test
├── slice_downloader.py    # download only assigned layer shards (S8)
├── coordinator/
│   ├── main.py            # FastAPI app + all endpoints
│   ├── models.py          # SQLite database models
│   ├── router.py          # chain assembly + routing
│   ├── ledger.py          # NRN earn/spend tracking
│   ├── balancer.py        # auto layer assignment (S14)
│   ├── model_registry.py  # multi-model support (S15)
│   ├── config.py          # settings
│   └── register_nodes.py  # register the 3 dev nodes
├── agent/
│   ├── agent.py           # main agent loop (S9)
│   ├── tray.py            # system tray UI (S9)
│   ├── updater.py         # auto-updater (S9)
│   ├── resource_guard.py  # CPU/RAM caps (S9)
│   └── android/
│       └── agent_termux.py # Android agent (S13)
├── ui/
│   ├── app.py             # chat UI server (S10)
│   └── static/
│       └── chat.html      # chat interface (S10)
├── api/
│   └── openai_compat.py   # OpenAI-compatible API (S11)
├── rag/
│   └── retriever.py       # real-time info retrieval (S15)
├── grant/
│   ├── nlnet_application.md
│   ├── sidn_application.md
│   └── technical_summary.md
├── sessions.md            # session log — append only
├── ROADMAP.md             # this file — reference only
├── README.md              # stranger-ready documentation
└── LICENSE                # Apache 2.0
```

---

## Rules for Claude Code

1. Read sessions.md AND ROADMAP.md before every session
2. Never modify ROADMAP.md — it is a reference document
3. All session results go in sessions.md only
4. Never push .db files, node_tokens.json, .venv, 
   *.safetensors, or models--* to GitHub
5. Never break existing passing tests
6. selftest_shard.py must pass after every session
7. Never modify common.py without explicit instruction
8. ARM-compatible code in all agent files
9. Zero personal data collected anywhere
10. Update sessions.md at the end of every session

---

## What NEURON Is Not

- Not a replacement for GPU clusters (for now)
- Not faster than a single machine for one user
- Not a training platform — inference only
- Not a get-rich-quick scheme — coin value follows network value
- Not finished until Session 12 (first stranger node)

---

## The One Rule

The network has no value without nodes.
Nodes have no value without users.
Users have no value without the network.

Build the agent. Get the first stranger. Everything else follows.

---

*Last updated: Session 11 complete. Session 12 active.*
*Founder: Raman Kumar Sharma, Rotterdam, Netherlands*
*Contact: via GitHub*
