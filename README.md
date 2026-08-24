# NEURON

A private AI chat that runs on your own computer — and while you're not using the machine, it
joins a network of ordinary computers that together run AI models too big for any one of them.

## Download

**[⬇ NEURON-Setup-0.20.25.exe](https://github.com/neuron-network-ai/neuron/releases/download/v0.20.25/NEURON-Setup-0.20.25.exe)** (217 MB)

SHA-256 `039e60655cad6969b5ab21c83e63ca8491675ef4f023617f97b9360d63c72740` — the same hash every
installed agent checks before it will run an update, read back from `/agent/version` and from
the published asset itself.

Windows installer — no technical knowledge needed. Double-click, then open
**http://localhost:8080** and sign in with Google or GitHub.

Two things to expect: Windows will say the installer is "unrecognized" (it isn't code-signed yet
— the full source is in this repository), and the first start downloads **1.4–1.8 GB**, which is
your share of the AI model. After that it just runs, and updates itself.

### If your browser or Windows blocks it

The installer is not code-signed, so a browser may refuse the **download itself** — not just
warn when you run it. Chrome says *"blocked"* or *"Failed – virus detected"*, Edge says it
*"isn't commonly downloaded"*. Nothing is wrong with the file; an unsigned 200 MB executable
that few people have fetched yet has no reputation to check, and that is what those messages
mean. Getting past it, in the order the blocks appear:

1. **The browser blocks the download.** Chrome: open `chrome://downloads`, find the entry, and
   click **Keep**. Edge: **Downloads** (`Ctrl+J`) → the **…** beside the file → **Keep** →
   **Show more** → **Keep anyway**. Firefox: the download panel → right-click → **Allow
   download**.
2. **Check you got the real file before you run it** — this is the step that makes the rest
   safe, and it takes one command. In PowerShell, from your Downloads folder:

   ```
   certutil -hashfile NEURON-Setup-0.20.25.exe SHA256
   ```

   It must print `039e60655cad6969b5ab21c83e63ca8491675ef4f023617f97b9360d63c72740` — the hash
   above, which is also what `/agent/version` serves and what every installed agent checks
   before it will run an update. The file is **217,140,114 bytes**, which Windows displays as
   **207 MB**. If the hash matches, you have the published build exactly; if it does not, delete
   it and download again, and please open an issue.
3. **SmartScreen: "Windows protected your PC".** → **More info** → **Run anyway**. This one
   has a way through.
4. **Smart App Control: "blocked an app that may be unsafe".** **This one does not, and we will
   not pretend otherwise.** The dialog offers only *Okay* and *Get apps from the Store* —
   there is no "run anyway", no per-app allow, and no exclusion that reaches it. Smart App
   Control is a different mechanism from SmartScreen: it demands a valid Authenticode signature
   from a CA in the Microsoft Trusted Root Program **plus** standing in Microsoft's Intelligent
   Security Graph, and an unsigned build cannot acquire either. Unblocking, re-downloading and
   antivirus exclusions all leave it exactly where it was.

   Your options today are honest ones:
   - **Install from source instead** (below) — Python is signed, so nothing is blocked. This is
     the recommended route while NEURON is unsigned.
   - **Turn Smart App Control off** — Windows Security → **App & browser control** → **Smart App
     Control settings** → **Off**. ⚠ On most builds this is **one-way**: it cannot be switched
     back on without resetting or reinstalling Windows. Microsoft announced in December 2025
     that a later update would make it reversible, so newer builds may differ — check before
     you decide. Do not turn off a system-wide protection just for us; option one costs you
     nothing.

   If Smart App Control blocked you, we would genuinely like to know — it tells us how many
   people this wall is silently turning away.
5. **Still nothing when you double-click.** Windows marks files from the internet and can block
   them silently: right-click the exe → **Properties** → on the **General** tab, tick
   **Unblock** at the bottom → **Apply**. (That is the same dialog whose **Details** tab shows
   the version — a build before 0.20.26 reports `File version 0.0.0.0` there, which is a missing
   field in our installer script and not a sign of a bad download.)
6. **Antivirus deleted it.** Some engines flag any large unsigned PyInstaller bundle. Restore it
   from quarantine, or add the Downloads folder as an exclusion for the install, and please tell
   us which engine — false positives can be reported and fixed.

**The real fix is a code-signing certificate, and Smart App Control is what turns that from a
courtesy into a blocker.** A browser warning costs a click; Smart App Control means a Windows 11
machine with it enabled **cannot install NEURON at all** by the normal route. The project does
not have a certificate yet — see [PACKAGING.md](PACKAGING.md). Until it does, the hash proves
the download, and the source install is the way in.

**On Linux or macOS none of this applies** — install from source, below.

Android: see [ANDROID_INSTALL.md](agent/android/ANDROID_INSTALL.md) (APK coming soon)

On Linux or macOS, see [For developers](#for-developers-source-install) below.

## What you get

- **A private AI chat** at `localhost:8080`. By default nothing you type leaves your computer —
  the answer is computed on your own machine. Two exceptions, both named in
  [PRIVACY.md](PRIVACY.md): switching on web search sends that query to a search engine, and a
  prompt blocked by the safety filter reports its category (never the text).
- **A 7B model at ~7.8 tokens/second** on your own CPU, if your machine has the RAM for it
  (~14 GB free). Smaller machines run a 1.5B model at ~28 tok/s instead.
- **Your machine earns NRN** while it sits idle, for helping run the shared network.
- **No GPU required.** No crypto wallet, no payment details, nothing collected about you.
- It **pauses the moment you touch the keyboard**, and on battery. You won't notice it running.

## What the network is

One computer can only run an AI model that fits in its memory. A big model doesn't fit anywhere
ordinary — that's why the good ones live in data centres.

NEURON splits a model into **layer slices** and gives each machine one slice to hold. Your
computer loads only its own layers, runs them on the data coming in, and passes the result to the
next machine. Nobody holds the whole model; together they run it end to end.

So the more people who join, the bigger the model everyone can use. The network serves the largest
model its current members can back, and moves up on its own as machines join:

| Model | Needs |
|---|---|
| Qwen2.5-1.5B | 2 machines, 6 GB between them |
| Qwen2.5-7B | 3 machines, 20 GB |
| Qwen2.5-72B | 20 machines, 180 GB |

When the model already fits on your machine, it just runs there — faster, private, and free. The
network is for the case where it doesn't.

## Honest numbers

This is an early alpha, and these are the real figures, not projections:

- **Local engine:** 7.8 tok/s on the 7B model, 27.9 tok/s on 1.5B (measured, 16-core CPU, Q4_K_M).
- **The network is tiny.** As of 3 August 2026: **2 machines registered, 2 online**, covering 21
  of 28 layers — so the chain is still incomplete and the dashboard reads DEGRADED. Live count on
  the [dashboard](https://neuronnet.duckdns.org/dashboard).
- **Lifetime requests served: 38.** Total NRN paid out to nodes: 25.81.
- **NRN has no cash value.** It is a record of compute contributed, nothing more. There is no
  exchange, no sale, and no promise that there will be one.
- Supply is fixed at 1,000,000,000 NRN and the ledger is transfer-only — nothing mints.
- **One coordinator, one relay, one SQLite ledger**, all on a single cloud VM. Backed up hourly,
  but it is still one machine. Removing that is the current work.

Earnings today are small because the network is small. That's the honest position.

## For developers (source install)

Works on Windows, Linux and macOS. Python 3.11+.

```bash
git clone https://github.com/neuron-network-ai/neuron.git
cd neuron
python -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate
pip install -r agent/requirements.txt
python agent/agent.py
```

Then open http://localhost:8080, sign in, and start chatting. Your node registers itself, picks
its own layer range, downloads only that slice, and is **verified by other nodes within about a
minute** — no operator, no approval queue, no shared secret.

Remove everything: `python agent/uninstall.py`.

Running the pipeline by hand, without the agent — each stage serves a contiguous layer range:

```bash
python node_c.py --port 50999      # middle stage — layers 10–18
python node_b.py --port 50999      # last stage — layers 19–27 + final norm
python node_a.py --coordinator https://neuronnet.duckdns.org --prompt "Why is the sky blue"
```

`python selftest_shard.py` proves the split is bit-exact against running the whole model on one
machine. Your own coordinator: `python -m uvicorn coordinator.main:app --port 8001`.

## Architecture

```
   User
    │  1. POST /infer {prompt}
    ▼
 ┌─────────────┐   returns the chain (which nodes / which layers / where) + request_id
 │ Coordinator │   registry · health-checks · routing · NRN ledger · dashboard
 │ neuronnet   │   neuronnet.duckdns.org
 │ .duckdns.org│
 └─────────────┘
    │  2. the client runs the returned pipeline, activations over TCP:
    ▼
  node_a ─────────► node_c ─────────► node_b
  layers 0–9        layers 10–18      layers 19–27
  embed + lm_head   (middle relay)    + final norm
    ▲                                     │
    └────────── final hidden state ◄──────┘   node_a applies lm_head, picks the
    │                                          next token, and loops until done
    │  3. POST /infer/{id}/complete  → coordinator credits NRN to each node
    ▼
   User  (receives the generated text)
```

One line: **User → Coordinator → [ node_a → node_c → node_b ] → Coordinator → User.**

**No VPN, no port forwarding, no Tailscale.** Your node makes only outbound connections; a public
relay hands your peers an address for you, which is what lets a machine behind a home router take
part.

Every completed request settles **1.0 NRN**: the coordinator keeps 10%, and the remaining 0.9 is
split across the chain in proportion to layers held — a node holding `L` of 28 layers earns
`0.9 · L/28`. Your balance is visible only to you; the public dashboard never shows it.

## Status

**Session 43 complete** — the network is live and open to strangers. Coordinator at
[neuronnet.duckdns.org](https://neuronnet.duckdns.org) (dashboard at
[/dashboard](https://neuronnet.duckdns.org/dashboard)). Install the agent and your machine
registers itself, picks its own layer range, downloads only that slice, and is auto-verified
within about a minute — no operator, no approval queue, no shared secret. NAT traversal is
built in via the relay — no VPN, no port forwarding, no Tailscale. Just install and run.

- **Windows installer shipped**, and **auto-update is live**: the app checks daily, verifies the
  published SHA-256 before running anything, and never updates mid-request. Set
  `auto_update: false` in your config if you would rather update by hand.
- **Earnings have an owner**: each node registers an Ethereum address it proves it controls.
- **NVIDIA GPUs with CUDA are used automatically if present — no configuration needed.** The
  node also steps aside while the GPU is busy. No machine in this project has an NVIDIA card, so
  that path is written and tested but has not yet run on real GPU hardware.
- **First stranger: pending.** Nobody outside the project has run it yet.

Next: several stateless coordinators over a shared database (so it isn't one process), then
peer-to-peer discovery so a chain can form without asking any central service.

## Repository layout

```
agent/               the node agent: register, place, download slice, serve, heartbeat, tray
coordinator/         FastAPI + SQLite: registry, placement, routing, model tiers, ledger, dashboard
engine/              local quantized execution (llama.cpp / GGUF) when the model fits your machine
ui/                  local chat UI, OAuth sign-in, conversation history
api/                 OpenAI-compatible endpoint (/v1/*)
safety/              prompt and response moderation
security/            proof-of-compute challenge used to verify nodes
packaging/           PyInstaller spec and the Windows installer
rag/                 retrieval, so answers can use current information

common.py            model sharding, manual layer driver, KV cache, TCP framing
node_a.py            driver stage: embed + first layers + lm_head, parallel request driver
node_b.py            last stage: layers + final norm
node_c.py            middle relay stage
neuron_driver.py     streaming generation across the chain
batching.py          batching several requests through one pipeline pass
junction_cache.py    caches activations at stage boundaries
slice_downloader.py  fetches only the byte ranges of the weights a node owns
wire_codec.py        activation wire format (4.3× smaller, no pickle)
relay.py             public relay that lets NAT'd nodes reach each other
tunnel_client.py     the node side of that relay
verify_service.py    runs verification challenges against joining nodes
selftest_shard.py    proves the sharded split is bit-exact vs. one machine
sessions.md          full engineering log, session by session
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). © 2026 NEURON Labs.

The installer bundles third-party components that keep their own licences — all OSI-approved
open source. Every one is inventoried in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md),
generated from the shipped bundle by `tools/gen_notices.py`.
