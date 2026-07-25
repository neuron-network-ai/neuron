# NEURON — First Stranger: where we are & what's left

The single map to the milestone that matters most: **one person who isn't the founder runs
NEURON and earns NRN.** This is both a status snapshot ("what and where we are") and the
concrete checklist to get there.

---

## Where we are (built & committed: v0.1 → v0.10)

| Component | Status | Where it runs |
|---|---|---|
| 3-node distributed inference (bit-exact, KV cache) | ✅ built, proven | node_a (Windows) · node_c (Pavilion) · node_b (OptiPlex) |
| **Coordinator** (registry, routing, ledger, health) | ✅ **LIVE** | cloud VM `150.230.22.250:8001` |
| **NAT relay** + auto-onboarding | ✅ **LIVE** | cloud VM `:8010/8011`, public `:9000-9100` |
| Slice downloader (byte-range, only your layers) | ✅ built | agent |
| Agent (register → slice → serve → heartbeat → tray) | ✅ built | not yet run by a stranger |
| Chat UI (SSE streaming) | ✅ built, running | **your PC** `localhost:8080` (not public) |
| OpenAI-compatible API (`/v1/*`) | ✅ built | mounted in the UI server |
| Auto-balancing (per-node speed) | ✅ live | coordinator |
| RAG + model registry | ✅ live | driver + coordinator |
| Proof-of-compute + reputation + rate limit | ✅ live | coordinator + verifier |
| **GitHub repo** | 🔒 **PRIVATE** | — |

**The plumbing is done:** a stranger's machine can already reach the cloud coordinator from
anywhere, get a slice, and serve through the relay — all outbound, no ports. What's missing is
the *front door* (where they download, a friendly runnable thing, and open join).

### Live endpoints
- Coordinator: `http://150.230.22.250:8001` · dashboard `/dashboard` · models `/models`
- Chat UI (local only): `http://localhost:8080`  (needs public hosting for strangers — see gaps)

### The join flow (already works)
```
Stranger's machine (anywhere, behind NAT)  ── all OUTBOUND ──▶  coordinator :8001
   register → slice-info → download slice → serve → auto-relay-tunnel → heartbeat → earn NRN
```

---

## Path A — first stranger (minimal, fastest to the milestone)

| # | Step | Who | Status |
|---|---|---|---|
| 1 | Tidy code for public — genericize leftover private IPs / secret defaults (env-driven) | me | ⬜ todo (repo audit done, clean) |
| 2 | Make the GitHub repo public | **you** | ⬜ after step 1 |
| 3 | Open join — drop the shared secret; register anyone at *probationary* reputation, earn only after passing proof-of-compute | me | ✅ done (open registration; probationary→verified via proof-of-compute; trusted fast-path keeps the secret; `coordinator/test_open_join.py` 17/17) |
| 4 | Place a 4th node — chose REPLICATION over a 4-stage re-split: the 4th node mirrors an existing segment and the router load-balances across replicas (coordinator-only; each chain stays 3-stage so drivers/nodes are untouched; deeper pipeline rejected per [P8]) | me | ✅ coordinator done (`router.build_chain` replica-aware; `coordinator/test_replica.py` 9/9). Live demo with a real stranger's machine = steps 5–7 |
| 5 | Package the agent — PyInstaller → one-file exe/AppImage | me | ⬜ best-effort (heavy ~GB; no code-signing here) |
| 6 | Dead-simple 5-step install guide | me | ⬜ |
| 7 | **A friend on a different network runs it → appears in /node/list → earns NRN** | **you** | 🎯 **the milestone** |

**Fastest honest route:** steps 1–4 (mine) + 2 & 7 (yours). A semi-technical friend could join
via "install Python + run" even before the packaged exe (step 5).

---

## Path B — real consumer product (bigger, later)

- **llama.cpp light app** — one-click, phones, GPU (the cross-cutting engine track; see `SCALING.md`).
- **Landing page / `neuron.network`** — download button + live node map + "NRN earned today".
- **Public chat hosting** so strangers can *use* it. ⚠️ The chat UI **is** the driver (needs the
  model + torch), so it can't run on the 1 GB cloud VM — it needs a driver host (your PC via a
  tunnel, or a bigger instance).
- **Open join at scale** + **agent code signing** (needs a cert, ~$100–400/yr).

---

## Honest tricky bits (don't gloss over)
- **4th-node placement** — adding a node to a full 3-node pipeline means either re-splitting
  (all nodes reload) or running it as redundancy (needs load-balancing to actually earn). Real work.
- **Public chat** — needs a driver host with the model; the tiny cloud VM can't do it.
- **Packaged agent** — heavy (bundles torch), Windows/Linux only, and unsigned exes trip antivirus
  until code-signed. The truly light agent is the llama.cpp path (Path B).
- **Repo goes public** — do the P11 tidy first (genericize IPs, ensure the real register secret is
  env-only). Decide whether `TOKENOMICS.md` (founder allocation) belongs in a public repo.

---

## Bottom line
- **Network: a stranger can already join from anywhere.** ✅
- **Experience: no front door yet** — download + runnable app + open join. That's Path A.
- **First stranger is close** — it needs a tidy + public repo + open-join + one friend, not the
  whole consumer app. The polished "anyone/phones/one-click" version is Path B (llama.cpp).
