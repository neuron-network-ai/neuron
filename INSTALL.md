# Join NEURON — run a node in ~5 steps

NEURON is a distributed AI network: instead of one big data center, many ordinary computers
each run a **slice** of an AI model and work together. By running a node, your machine
contributes a little spare CPU while it's idle and **earns NRN**, the network's contribution
credit.

> **What NRN is (honestly):** NRN has **no cash value**. It's how the network records who
> contributed compute, and — per the roadmap — what you'll spend to run your own AI queries.
> Think "credit for compute you gave, redeemable for compute you use," not money.

**Before you start — honest expectations:**
- **CPU-only today.** Your node uses ~1–2% CPU while your machine is idle and **pauses
  automatically** when you start using the computer. No GPU required.
- **~1 GB download.** Your node fetches only *its* slice of the model (not the whole thing),
  plus the Python libraries. First start takes a few minutes.
- **No account, no crypto wallet, no personal data.** The node only ever sends the coordinator
  your node id, the layer range you serve, and your CPU/RAM counts. You can read every line —
  it's open source.
- **No inbound ports / no Tailscale needed.** Your node makes only *outbound* connections and is
  reached back through the network's relay, so it works behind a normal home router.

---

## Two ways to join

**Windows: use the installer (easiest).** Download `NEURON-Setup-<version>.exe` from
[Releases](https://github.com/raman011sharma-code/neuron-network/releases), run it, click
through — it installs a system-tray app that starts your node automatically. No Python needed.

> **A quick note on the security warning:** the installer isn't code-signed yet (that needs a
> paid certificate the project doesn't have), so Windows SmartScreen or your antivirus will
> likely flag it as "unrecognized." That's normal for small open-source projects, not a sign
> something's wrong — the installer's first screen explains this and shows you where to read
> the source before you trust it. If SmartScreen blocks it outright: "More info" → "Run anyway."

**Linux / macOS, or if you'd rather run from source:** follow the 5 steps below.

---

## Requirements (source install)

- **Python 3.10 or newer** — check with `python --version` (on some systems it's `python3`).
- **Git** — to download the code (or download the repo as a ZIP).
- **~3 GB free disk** (your model slice + Python libraries).
- Windows, Linux, or macOS.

---

## The 5 steps (source install)

### 1. Get the code
```bash
git clone https://github.com/raman011sharma-code/neuron-network.git
cd neuron-network
```

### 2. Create a clean Python environment
```bash
python -m venv .venv
```
Activate it — **Windows (PowerShell):**
```bash
.venv\Scripts\Activate.ps1
```
**Linux / macOS:**
```bash
source .venv/bin/activate
```

### 3. Install the node's dependencies
```bash
pip install -r agent/requirements.txt
```
(This pulls in PyTorch, so it's the biggest step — give it a few minutes.)

### 4. Start your node
```bash
python agent/agent.py
```
That's it. The node will **register itself, pick which slice to serve automatically, download
that slice, and start contributing.** Leave the window open; watch `agent/agent.log` for progress.

### 5. Confirm you're in
Open the live network dashboard in a browser:

**http://150.230.22.250:8001/dashboard**

You'll see your node appear in the list. New nodes start as **`probationary`** — you're
connected and downloading, but you don't serve live traffic or earn yet until the network
operator verifies your node (next section). Once verified you flip to **`verified`** and begin
earning NRN as requests flow through you.

---

## What happens after you start

```
run agent  →  registered (probationary)  →  slice downloaded  →  operator verifies you
           →  verified  →  you serve requests and earn NRN (private to you: tray → My Dashboard)
```

- **Pause is automatic.** Touch your keyboard/mouse or load your CPU and the node stops
  advertising itself within seconds; it resumes when you're idle again. It also pauses on
  battery and when free memory is low.
- **Stop anytime:** press `Ctrl-C` in the window.
- **Remove completely:** `python agent/uninstall.py` (deregisters your node and deletes the
  downloaded slice).
- **Optional desktop tray icon** (green = earning, pause button, NRN balance): `pip install
  pystray Pillow`, then run `python agent/tray.py` alongside the agent.

---

## For the network operator — verifying a new node

A probationary node only starts earning after it passes a **proof-of-compute** challenge (this
is what keeps cheaters from farming NRN without doing real work). From a machine that has the
model and the registration secret:

```bash
python -m security.proof_of_compute \
  --coordinator http://150.230.22.250:8001 \
  --node-id <the-new-node-id>
```
(Set `NEURON_REGISTER_SECRET` in your environment first, or pass `--register-secret`.) A pass
records the attestation and promotes the node to `verified`. Proof-of-compute currently verifies
**last-segment** nodes, which is exactly where new nodes are auto-placed.

---

## Troubleshooting

- **`python` not found** → try `python3`, or install Python 3.10+ from python.org and re-open the
  terminal.
- **The install step is slow / large** → that's PyTorch downloading; it's a one-time cost.
- **My node shows `probationary` forever** → it needs the operator to run the verification command
  above. Ping them.
- **`coordinator unreachable`** → check your internet connection; the node retries automatically
  every 60 seconds.
- **It says it paused** → that's normal — you're using the machine, or it's on battery / low on
  memory. It resumes when idle.

---

*The Windows installer above is the one-click path — no Python setup. The source steps still work
everywhere and are the only path on Linux/macOS today.*
