# NEURON — Security

Built in Session 16 (+ ongoing). This is the trust model that lets strangers install the
agent and users trust the answers.

> This document covers node trust — is a node computing honestly? For what's prohibited to
> generate *through* NEURON (a different, content-focused question), see `SAFETY.md`.

## Proof of Compute — a node must prove it did the work
A lazy/malicious node could return garbage to farm NRN without computing. To catch that:
- A verifier sends the node a **challenge** (a known input for its layer range); the node
  runs its slice; the verifier checks the output against the **locally-computed expected**.
- **Honest work matches to ~1e-5; garbage or lazy (echo-the-input) cheating is off by ~25+.**
  A tolerance of `atol=0.05` cleanly separates them (verified live against a real node).
- Code: `security/proof_of_compute.py` (verifier side; needs torch). It challenges a
  LAST-stage node (`layers[s2:n]` + final norm). Middle-node challenges need a no-relay mode
  on node_c (extension). The coordinator stays torch-free — it only records pass/fail.

## Reputation — cheaters get flagged and cut off
- The coordinator tracks `challenges_passed` / `challenges_failed` per node;
  reputation = pass-rate.
- A node with ≥ `REPUTATION_MIN_SAMPLES` (3) samples and pass-rate < `REPUTATION_THRESHOLD`
  (0.6) is **flagged**. Flagged nodes are excluded from **routing AND coverage** — they get
  no requests and earn nothing (verified: flagging a node dropped its layers from the chain).
- Endpoint: `POST /node/{id}/attest {passed}` (trusted verifier; register-secret gated).

## Rate limiting — basic DDoS guard
- Per-IP: `RATE_LIMIT_MAX` (120) requests per `RATE_LIMIT_WINDOW_S` (60 s) → HTTP 429.
  Middleware in the coordinator (verified: a 60-request burst was throttled).

## Zero personal data collected
- A node sends only: `node_id`, layer range, core/RAM counts, and its IP. Never user files,
  network traffic, or screen. Provable by reading the open code.

---

## Manual / pre-launch (NOT code — needs certs / ops)
- **Code signing the agent binaries** (Windows Authenticode + Linux) — needs a code-signing
  certificate (~$100–400/yr). Prevents antivirus flagging of the installer. **Do before public
  distribution.** Not buildable in-repo (requires the cert + signing infra).
- **TLS on the coordinator** (Caddy auto-HTTPS) so node tokens don't cross the internet in
  cleartext — steps in `coordinator/DEPLOY.md`.
- **Open join** — today registration needs a shared secret; a truly open network should drop
  the shared secret and gate join on **proof-of-compute + reputation** instead (the primitives
  built here are the foundation). Tie NRN payout to sustained good reputation.

## Open / future
- **Verifier automation** — a periodic challenger (like the heartbeat) that attests each node
  and reports results; currently attestations are issued manually / by the driver.
- **Middle-node challenges** (no-relay mode on node_c) and tolerance vs bit-exactness across
  heterogeneous hardware (a stranger's different CPU may need a looser `atol`).
- **Sybil resistance** — one attacker spinning up many nodes. Proof-of-compute + reputation
  raise the cost, but stake/identity may be needed at real scale (ties to on-chain NRN, S17).
