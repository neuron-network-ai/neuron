# Handoff — start of Session 65

Paste the block at the bottom into a new window.

---

## What Session 64 did

**Security was the session's aim and it was met.** [P52]: the pipeline wire is now encrypted and
authenticated end to end. The coordinator mints a per-hop grant sealed to each node's own token,
the two peers do an ephemeral X25519 exchange, and every frame after that is AES-GCM. Proved
against a real byte-recording relay: the prompt is not in the bytes. Proved per-hop across four
nodes: different key each hop, and a hostile node cannot reuse its grant against its neighbour.
Forward secrecy is deliberate — **the coordinator itself cannot decrypt a recording.**

**[P30] measured, routed, and phase 2 built.** llama.cpp is 8.7x PyTorch on this machine
(26.93 vs 3.09 tok/s). A real relayed hop to the Pavilion is **22.6 ms**, so the network is 7% of
a token's cost today and the ENGINE is the bottleneck — phase 1's gate passed. `agent/rpc_engine.py`
supervises a localhost-only `ggml-rpc-server` reachable only through the encrypted channel;
measured 28.07 tok/s through it, ~19x the live chain.

**Route reversed twice in one day, honestly both times.** ggml-rpc was rejected because llama.cpp
says never put it on an open network and NEURON had none — then [P52] built the private channel
that objection required, and the route came back. If the channel is ever removed, the objection
returns intact.

**[P51] found and closed.** The agent thread ran unsupervised, so an exception went to a `stderr`
that does not exist in a windowed app — alive, listening, unregistered, silent for 81 minutes.
Now supervised, with a heartbeat watchdog for the case no exception handler can catch.

**[P50], [P46], [P39], [P48], [P49] all closed or guarded.** Install verification, the update
notice, the reachable claim strip, React parity, the node identity that nearly orphaned 5.26 NRN.

**PRIVACY.md written, and a claim of mine corrected.** Two things leave even on the local path:
optional web search (off by default) and a blocked prompt's category (never the text). Both now
disclosed in PRIVACY.md, DISCLOSURE.txt and README.

**0.20.5 built** — `dist/installer/NEURON-Setup-0.20.5.exe`, its packaged bundle verified
coherent by [P46]'s own guard.

## Read this before planning

**ROADMAP.md is ~47 sessions stale** (last updated Session 16; it still lists Session 8 as
active). Its *rules* and its One Rule are not stale and are still binding — build rule 1 says
read it every session, and Session 64 did not until asked.

**It disagrees with PROBLEMS.md and the disagreement matters.** ROADMAP lists *"not faster than a
single machine for one user"* under What NEURON Is Not; PROBLEMS ranks single-user speed
**HIGHEST** ([P1]). Both were written honestly and they diverged. **Reconcile them, or every
session works from whichever file it happens to open.**

**And the One Rule is unmet after 63 sessions:** *"Build the agent. Get the first stranger.
Everything else follows."* The live roster is two machines, both the founder's. No stranger has
ever run a node. `FIRST_STRANGER.md` says the plumbing is done and what is missing is the front
door — which is now largely built, so what remains is **outreach**, and `OUTREACH.md` shows
nothing was ever sent.

**The founder's own framing, which corrects a misreading:** those two machines are a **testbed**,
not the network. Judging them as production inflates the severity of "no replica" and
"single point of failure".

## [P30] speed — where the phases actually stand

The whole point of the engine work is making NEURON fast. Status, so a fresh window does not
have to reconstruct it:

| phase | state | what it produced / what it needs |
|---|---|---|
| **1 — measure a real hop** | ✅ **DONE** | median **22.6 ms** to the Pavilion through the real relay. Compute is 324 ms/token today, so the network is **7%** and the ENGINE is the bottleneck. The gate that could have cancelled everything, passed. `tools/bench_hop.py` |
| **2 — one node, fast** | ✅ **BUILT** | `agent/rpc_engine.py` — a localhost-only `ggml-rpc-server` reachable only through the [P52] channel. Measured **28.07 tok/s** through it, against 1.44 live. 14 tests. |
| **3 — two nodes** | ❌ **NOT STARTED** | needs a SECOND machine running the engine, and a decision (below). Everything so far is loopback on one PC plus arithmetic. |
| **4 — packaging** | ❌ **NOT STARTED** | ship the binary with the installer; [P41]'s CPU floor becomes a dispatch table. |

**Phase 3 is blocked on two things, and only one of them is code:**

  1. **A second machine to run the engine on.** The Pavilion is reachable only through the relay
     from here. Putting `ggml-rpc-server` on any machine is the founder's call and must be asked
     for, not assumed.
  2. **A design decision that changes what a driver IS.** In ggml's RPC design the CLIENT reads
     the model file and uploads tensors; the server holds no model (`-c` caches what it is sent).
     With mmap the driver does not need it in RAM, but it does need **the whole model on disk** —
     where today it downloads only its own slice. For a 1.5B that is nothing; for the big models
     this project exists for it is ~100 GB on whoever drives. Decide before building.

**Projected end state, and it is a projection, not a measurement:** ~37 ms compute + 23 ms hop
≈ **16.7 tok/s**, against 1.44 today. That clears TOKENOMICS §11.6's "answers under 30 s" gate,
which is what currently blocks any purchase path. Treat 16.7 as the direction — when the same
arithmetic was checked against the live chain it was ~2x optimistic, because it omits per-token
framing, the driver's own embed/`lm_head`, batching and Python overhead.

## Do these first

1. **Install 0.20.5.** Built and sitting in `dist/installer/`. Everything user-facing from
   Sessions 63-64 is invisible until it runs — the claim strip, install verification, the update
   notice, [P51]'s watchdog, [P49]'s `node_server` fix.
2. **Then claim this machine's node through the UI.** Sign in and press the button. It is the
   first real execution of connect → sign → bind, and the driver's balance is still tied to a
   file on one disk until it happens.
3. **Push, or decide not to.** 47 commits. Do NOT point the download links at 0.20.5 until a
   release actually exists — `test_download_links.py` enforces that.
4. **Re-run the reconciliation** after anything touches the ledger. It is clean with ZERO
   warnings, which is what makes [P40] item 2 (a cron) worth doing.

## Open, roughly in order

- **The first stranger.** The roadmap's One Rule, unmet. Outreach is a human act; nothing in the
  repo can do it.
- **The capacity case has NEVER run.** [P43]: no forward pass has ever executed on the 4B. The
  network serves a 1.5B, which fits on one machine — so *"models too large for any single
  machine"*, the site's headline, has never been demonstrated. Run it once.
- **[P30] phase 3** — two nodes on the new engine, and proof-of-compute must still pass. **Open
  design question:** in ggml's RPC design the CLIENT reads the model file and uploads tensors, so
  the driver needs the whole model on DISK (not RAM). Today it downloads only its slice. Decide
  before this replaces anything.
- **[P52] residual** — the channel authenticates the CALLER, not the PEER. A chain member is
  coordinator-selected, not trusted, and ggml-rpc is a memory protocol. Belongs with the
  open-join question in SECURITY.md.
- **`NEURON_REQUIRE_SECURE=1`** — the end state for the encryption rollout. Flip it once the
  fleet has moved; until then plaintext is accepted so an upgrade does not partition the network.
- **[P1] / [P30] speed** — 1.44 tok/s live. Phases 3-4 are the fix.
- **TERMS.md** — deliberately NOT written. Legal document; TOKENOMICS §12.7's reasoning applies.
- **[P37]'s open item** — auto-repair moves a node's range and never tells it. Recurred twice;
  a person fixed it by hand both times.
- **No replica** — one machine per stage. Real, and a testbed property rather than a defect.

## Environment gotchas

- **Bash from cmd.exe is WSL, not Git Bash.** Use `"C:\Program Files\Git\bin\bash.exe"` for
  anything touching the SSH keys.
- **The VM deploy key carries a passphrase.** Load it into the agent socket first:
  `SSH_AUTH_SOCK=/c/Users/optin/.ssh/neuron-agent.sock ssh-add /c/Users/optin/.ssh/oracle_coordinator`
  It drops when the PC sleeps.
- **Python is `.venv\Scripts\python.exe`.** PATH python has no torch.
- **`packaging/test_app_entry.py` runs by PATH, not `-m`** — a local `packaging/__init__.py`
  would shadow the PyPI package for the whole repo.
- **The desktop app serves its own packaged copy.** Repo edits to the UI are invisible until a
  rebuild — this has caught three sessions running.
- **`good-state-0.20.5`** is a tag. `git reset --hard good-state-0.20.5` returns to a known-good
  point.

## Prompt for the next window

```
Continue NEURON. Read PROBLEMS.md [P30] [P52] [P51] [P43], ROADMAP.md, and
NEXT_SESSION.md — that has the handoff. Read ROADMAP.md properly: build rule 1
says every session, and Session 64 skipped it until asked.

STATE: the wire is encrypted and authenticated end to end ([P52]) — per hop,
different key each hop, and the coordinator itself cannot decrypt a recording.
[P30] phase 1 passed (a real relayed hop is 22.6 ms, so the ENGINE is 93% of a
token's cost) and phase 2 is built (agent/rpc_engine.py, 28 tok/s through the
encrypted channel vs 1.44 live). NEURON-Setup-0.20.5.exe is built and unpushed
along with 47 commits.

FIRST: install 0.20.5, then sign in and claim this machine's node through the
UI. Everything user-facing from two sessions is invisible until that installer
runs, and the claim is the first real execution of connect → sign → bind.

THEN [P30] PHASES 3 AND 4, which is the speed work and the reason the engine
plan exists. Phase 1 is DONE (22.6 ms real hop; the engine is 93% of a token's
cost) and phase 2 is BUILT (agent/rpc_engine.py, 28 tok/s through the encrypted
channel). Phase 3 needs a SECOND machine running ggml-rpc-server — ASK which
machine, do not pick one — and a decision first: in ggml's RPC design the driver
needs the whole model file on DISK, where today it downloads only its slice.
That changes what a driver is, so settle it before writing code. Phase 4 is
packaging the binary with the installer, which also retires [P41]'s CPU floor.
Projected ~16.7 tok/s against 1.44 today; treat that as a direction, since the
same arithmetic ran ~2x optimistic against the live chain.

AND DECIDE, because this is a direction question and not mine to settle:
ROADMAP.md's One Rule is "build the agent, get the first stranger, everything
else follows" and after 63 sessions no stranger has ever run a node — the roster
is two of the founder's own machines, which are a TESTBED, not the network.
Meanwhile ROADMAP says "not faster than a single machine for one user" is what
NEURON is NOT, while PROBLEMS ranks single-user speed HIGHEST. Those two
documents disagree. Reconcile them before building, or each session works from
whichever one it opens.

ALSO NEVER DEMONSTRATED: [P43] says no forward pass has ever run on the 4B. The
site's headline is "models too large for any single machine" and the network
serves a 1.5B that fits on one. Running that once is worth more than another
optimisation.

WATCH OUT: [P50] before touching an agent config (one PC, two identities, one
file). The desktop app serves its own packaged UI, so repo edits are invisible
until a rebuild. And [P30] phase 3 has an open design question — in ggml's RPC
design the DRIVER needs the whole model file on disk, where today it downloads
only its slice.

Nothing is pushed. Pushing also publishes the corrected download links and
PRIVACY.md to the GitHub Pages site.

Environment gotchas and the rest of the open list are in NEXT_SESSION.md.
```
