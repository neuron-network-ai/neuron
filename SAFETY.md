# NEURON — Content Safety & Acceptable Use

This is the policy for what NEURON's chat and API layer will and won't generate, and how
that's enforced today. Different document from `SECURITY.md` (which is the *node* trust model
— proof-of-compute, reputation, node-to-node security) — this one is about *content*, a
different audience and a different legal category.

---

## Why this exists, and why it works differently here

NEURON splits a model's layers across machines. Only the **driver** — the node holding the
embedding + `lm_head` (today the machine running the Chat UI / OpenAI-compatible API) — ever
handles plaintext prompts or generated text. Every other machine in the pipeline (middle/last
compute nodes, run by volunteers) only ever processes opaque floating-point tensors — they
cannot read what they're computing (`common.py`'s `mid_stage`/`last_stage` never touch a
tokenizer). So enforcement lives **only at the driver**, at two points: before a prompt is
dispatched to the network, and on the generated text as it streams back.

This does **not** mean electricity/hardware use by volunteer nodes is irrelevant just because
the content is unseen to them — see `INSTALL.md`'s node-operator disclosure.

**"The driver" is no longer just one machine.** Every installed agent can now also run its
own personal Chat UI (`agent/local_chat.py`) — a small, fixed driver shard downloaded
separately from whatever compute range that machine happens to serve for the network. That
means each person's own installation moderates their own requests locally, on their own
plaintext, before anything reaches the shared network — instead of a single centralized
website acting as the one choke point for everyone. Compute nodes stay exactly as blind as
before either way; this only multiplies *where* the driver role can run, not what it's
allowed to see.

---

## Prohibited uses

Through NEURON's chat interface or API, you may not generate, request, or attempt to elicit:

- Child sexual abuse material (CSAM), or any sexual content involving minors.
- Instructions or material facilitating chemical, biological, radiological, or nuclear (CBRN)
  weapons, or other weapons intended to cause mass casualties.
- Malware, ransomware, or other code whose primary purpose is to attack, damage, or gain
  unauthorized access to systems.
- Content that instructs, encourages, or facilitates self-harm or suicide.
- Non-consensual sexual content, including deepfakes of real people.
- Material that plans or facilitates serious violence, mass-casualty attacks, or terrorism.

This list is not exhaustive, and it will change as the project and its user base grow. If
you're unsure whether something is in scope, ask before you build against it.

---

## How enforcement works today — honest limits

**v1 is a keyword/phrase blocklist** (`safety/moderation.py`, `safety/blocklist.json`),
case-insensitive, checked at two points:
1. **Input** — before your prompt is ever sent to the node network (chat UI and the
   OpenAI-compatible API both check here).
2. **Output** — on the generated text as it streams back, aborted mid-stream on a match.

**This is deliberately simple and is not a promise of robustness.** A keyword list is
trivially evaded by paraphrasing, other languages, encoding tricks, or a determined jailbreak.
It is a first line of defense appropriate for a small, pre-launch network — not a production
trust & safety system. The `check_text()` function is written so a future classifier-based
backend can slot in without changing any call site.

---

## Repeat violations escalate against your wallet identity

A single block is a one-off — it stops that one request and forgets who sent it the moment
the response is sent. Since a wallet requires a real Google/GitHub login, blocks now also
count against that identity: the driver reports **only a category label** (e.g.
`weapons_cbrn`) to the coordinator's `POST /wallet/{id}/violation` — never the flagged text or
a snippet, so this still doesn't cost NEURON the "coordinator never sees the moderation
content" property. At `MODERATION_BAN_THRESHOLD` violations (3 by default,
`coordinator/config.py`) the wallet is banned: `/infer` refuses it with a 403, before any
funds are even held.

**A login is now genuinely required, not just offered.** This used to be bypassable, which
made everything in this section decorative: `POST /wallet/faucet` was ungated (unlike the two
sibling endpoints that check `X-Wallet-Link-Secret`), and `claim_faucet` *creates* the ledger
row for whatever `wallet_id` string it receives. So anyone could mint a funded wallet with no
account, use it as an API bearer key, and mint a fresh one the moment it was banned — free,
instant, unlimited. Closed on three fronts:

- `/wallet/faucet` requires the operator secret **and** a wallet already linked to a real
  Google/GitHub login.
- **`/infer` refuses any wallet not backed by a login.** This is the load-bearing one: every
  driver must call `/infer` to get a node chain, so it's the one check a user cannot patch out
  of their own copy of the client.
- The OpenAI-compatible API verifies the bearer key against the coordinator instead of
  accepting any non-empty string as a wallet.

**Operator review and manual bans.** The automatic threshold only counts violations the driver
*self-reports*, and for a self-hosted install the driver runs on the user's own machine — so a
stripped client never reports itself and never trips it. `/admin` is a console listing every
identity (provider, whether the provider verified their email, violation count, request count,
last seen) with per-identity history and a ban button, backed by secret-gated endpoints. That
is the lever for everything the keyword filter misses: a jailbreak, a paraphrase, an abuse
report. Bans are enforced at `/infer`, server-side, so they bite a modified client too.

**Honest limits:** (1) a determined bad actor can create a new OAuth identity (different
Google/GitHub account, or the same person under a different email) and get a fresh wallet with
a clean record — this raises the cost of repeat abuse, it does not make it impossible; a real
identity-verification layer is a much bigger, separate project. (2) the ban threshold is
low enough on purpose that a wallet isn't locked out from one blocklist false-positive, but
that also means genuine repeat bad-faith use gets exactly 2 free attempts before consequences.
(3) this only escalates INSIDE NEURON (wallet banned from `/infer`) — it is not, and does not
claim to be, a report to any external authority. (4) **content policy itself is only
enforceable on drivers NEURON runs.** Someone running their own agent holds the plaintext and
the moderation code on their own machine and can simply delete the check. Against that user
the controls are not content-based at all: they need a real account to get a chain from the
coordinator, and that account can be banned.

A blocked request never reaches the node network (no volunteer compute is spent on it), and a
blocked generation is never billed or reported as completed.

---

## What's logged, and where

A blocked request/response logs: timestamp, direction (input/output), category, request id, a
hashed identity (not raw), and a snippet truncated to 40 characters — **on the driver machine
only, never sent to the coordinator.**

**Update:** the coordinator used to also store the full raw prompt text of every request in
its own database (`coordinator/models.py`'s `requests` table) — for an operational reason
unrelated to moderation (bounding a driver's self-reported token count against something real
at settlement time), but it was full plaintext, sitting there indefinitely, readable by
whoever has DB access. That only ever needed the prompt's *length*, not its content, so the
coordinator now stores `prompt_len` (a character count) instead and never writes prompt text
for new requests. Historical rows from before this change still have their old text — this
was a stop-the-leak fix, not a retroactive scrub, and that's a separate decision. `SECURITY.md`'s
"zero personal data collected" claim is scoped to *node* telemetry (node id, layer range,
core/RAM counts, IP) and is accurate as scoped; for chat/API users, the prompt text itself
still passes through the coordinator process transiently on every `/infer` call (it's part of
the request body) even though it's no longer persisted — stated here plainly rather than left
implicit.

---

## Reporting

If you believe NEURON generated something that violates this policy, or you have a safety
concern about the network, open an issue on the GitHub repo (see `README.md`) or contact the
maintainer directly.

---

## Roadmap

- Upgrade from keyword matching to a real classifier once there's budget/infrastructure for it.
- ~~A durable per-user identity (tied to the wallet system) to handle repeat violations~~ —
  **done**: see "Repeat violations escalate against your wallet identity" above. Still open:
  real identity verification to make a ban actually costly to evade with a fresh login.
- ~~Formal log retention policy once real user volume exists.~~ — **partly done**: the
  coordinator's `requests` table (the only table that grows with traffic rather than with
  users — roughly 1.25 GB/day at 1M users × 5 requests) is pruned to
  `NEURON_REQUEST_RETENTION_DAYS` (90 by default) on the existing health sweep. Identities,
  ledger rows and `moderation_events` are never pruned: bans depend on them and they grow
  slowly. The driver-side moderation log still has no retention policy.

---

**This is not legal advice.** This policy has not been reviewed by a lawyer. Before any public
launch at meaningful scale, get one — the same posture `TOKENOMICS.md` already takes on the
economics side.
