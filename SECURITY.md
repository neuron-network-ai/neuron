# NEURON — Security

Built in Session 16 (+ ongoing). This is the trust model that lets strangers install the
agent and users trust the answers.

> This document covers node trust — is a node computing honestly? For what's prohibited to
> generate *through* NEURON (a different, content-focused question), see `SAFETY.md`.

## Who is protected from whom

**NEURON has TWO kinds of user, and this document used to say "user" as though it had one.**
That ambiguity is not cosmetic: the two classes have different things to lose, and the hardest
open question in the system is one class being exposed to the other.

| | who they are | what they have to lose |
|---|---|---|
| **Chat users** | send prompts, read answers, spend NRN | the content of their conversation |
| **Node operators** (donors) | donate a machine's idle time, earn NRN | their machine, and their fair earnings |

A node operator is also a chat user whenever they type into their own app, so the same person is
often both — but the exposures are different and must be reasoned about separately.

### The matrix

| threat | protected? | by what |
|---|---|---|
| chat user ← an eavesdropper on the internet | **yes** | the encrypted hop ([P52]) |
| chat user ← whoever runs the relay | **yes** | the relay splices ciphertext it has no key for |
| chat user ← **the coordinator itself** | **yes** | ephemeral X25519: it introduces the two parties and cannot decrypt what follows, even from a recording |
| chat user ← **a node operator in their own chain** | **NO** | *nothing. See below — this is the open frontier.* |
| node operator ← a malicious chat user | partial | the wire carries no executable content ([P19]), message size is capped, `resource_guard` caps CPU/RAM |
| node operator ← another node operator | partial | one key per hop, and a grant cannot be retargeted at a different node ([P52]) |
| node operator ← the coordinator | **no, by design** | it assigns work, issues tokens and can flag — the network trusts it for placement and payment |
| chat user ← another chat user | partial | separate sessions and keys; the same node still serves many people in turn |

### The open frontier, stated plainly

**A node in the chain computes on the activations it receives.** That is what a node is for. So
the machine belonging to a stranger, in a chain the coordinator chose, sees data derived from
another person's prompt in the clear — inside its own process, after decryption.

Encryption fixed the network, the relay and the coordinator. It does not and cannot fix this,
and no amount of transport security will: the work requires the plaintext.

What genuinely mitigates it today, none of which is a solution:

  * **no single node holds the whole model**, so no node can decode what it is passed into text
    on its own — it has one contiguous slice of the layers and neither the embedding nor the
    `lm_head` unless it is the driver;
  * **no single node holds the whole conversation** — replicas mean consecutive requests from
    one person may traverse different machines;
  * **activations are not text**, but they must not be treated as opaque either: recovering
    input from embeddings and intermediate states is an active research area, and the safe
    assumption is that anything derived from a person's words still carries them;
  * **reputation and flagging** give a deterrent against a node that misbehaves in ways the
    verifier can see — which does not include quietly logging what it computes.

**This is the honest limit of a decentralised inference network**, and it is the reason
`docs/index.html` says *private on your own machine* rather than claiming privacy outright: when
the model fits locally nothing leaves the machine at all, and that IS complete privacy. When the
network is used, the guarantee is confidentiality in transit — not confidentiality from the
volunteers doing the work.

Do not let a future document round this up. Overstating it is [P31]'s mistake pointed at the one
property users cannot verify for themselves.

---

## Proof of Compute — a node must prove it did the work
A lazy/malicious node could return garbage to farm NRN without computing. To catch that:
- A verifier sends the node a **challenge** (a known input for its layer range); the node
  runs its slice; the verifier checks the output against the **locally-computed expected**.
- **Honest work matches to ~1e-5; garbage or lazy (echo-the-input) cheating is off by ~25+.**
  A tolerance of `atol=0.05` cleanly separates them (verified live against a real node).
- Code: `security/proof_of_compute.py` (verifier side; needs torch). It challenges a
  LAST-stage node (`layers[s2:n]` + final norm) via `attest()`, or a MIDDLE node via a
  **probe mode**: a config message with no `host_b` tells `node_c.py` / `agent/node_server.py`
  to run its own layers in isolation and return the raw result instead of relaying to a next
  hop, so it can be challenged without a real downstream chain (`attest_middle()`). The probe
  ack echoes the node's OWN actual range, checked by the verifier — a node whose real range
  doesn't match what the coordinator thinks it registered fails loudly instead of silently
  passing. The coordinator stays torch-free — it only records pass/fail.
- **Automatic, no manual command needed**: `coordinator/register_nodes.py --auto-verify` runs
  `verify_loop()` continuously in the background alongside the existing heartbeat — it finds
  every probationary, non-flagged node and challenges it every `--verify-interval` seconds
  (default 60), so a new arrival gets promoted (or correctly stays probationary if it fails)
  without anyone running the CLI by hand. This has to live in a founder-run, secret-holding
  process (not the coordinator itself, which is deliberately torch-free and can't compute the
  comparison) and not in every stranger's own agent (that would let anyone self-verify).

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

## The pipeline wire carries no executable content
- Activations move over a raw TCP frame that is a **JSON header plus raw tensor bytes**
  (`wire_codec.py`). Nothing in that format can run code.
- This replaced `torch.save`/`torch.load`, which is **pickle**. `common.recv_msg` used to
  call `torch.load(..., weights_only=False)` on whatever arrived, so any peer could execute
  arbitrary code in the receiving process — in both directions, driver ↔ node, and reachable
  from the public relay ports rather than only from the chain. Fixed 2026-07-29; see
  `PROBLEMS.md` [P19]. Legacy senders are still accepted for a rolling upgrade, but now via
  `weights_only=True`, which admits only plain tensors and primitives.
- The declared message length is capped (`common.MAX_MSG_BYTES`, 512 MB) so a stray scanner
  on a public node port cannot make a small VM allocate an arbitrary buffer.
- Regression tests: `test_wire_codec.py`.

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
- ~~Verifier automation~~ / ~~Middle-node challenges~~ — **done**, see above.
- Tolerance vs bit-exactness across heterogeneous hardware (a stranger's different CPU may
  need a looser `atol` than `0.05`) — not yet needed live, no real stranger hardware tested.
- **Sybil resistance** — one attacker spinning up many nodes. Proof-of-compute + reputation
  raise the cost, but stake/identity may be needed at real scale (ties to on-chain NRN, S17).
- **Self-healing coverage gaps with NO existing replica** (deliberately NOT built this
  session — see below): today, verification lets an ALREADY-REGISTERED replica become
  usable automatically the instant it passes (routing/coverage are computed live from
  current online+eligible status on every read, so nothing else has to happen). But if a
  segment has ZERO nodes registered for it at all, nothing can conjure capacity from
  nothing — closing that gap needs actively reassigning an already-assigned node's range
  (robbing one segment to patch another) or recruiting genuine spare capacity, which means
  extending the tested `MigrationController` state machine (currently hard-coded to trigger
  only on a MODEL-tier change, not a same-model roster change) or building a parallel
  mechanism. That's real, separate, riskier surgery on economically-critical, already-tested
  code — not something to bolt on hastily alongside this session's changes.
