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

A blocked request never reaches the node network (no volunteer compute is spent on it), and a
blocked generation is never billed or reported as completed.

---

## What's logged, and where

A blocked request/response logs: timestamp, direction (input/output), category, request id, a
hashed identity (not raw), and a snippet truncated to 40 characters — **on the driver machine
only, never sent to the coordinator.**

**Separately, and independent of this policy:** the coordinator's database already stores the
full raw prompt text of every request (`coordinator/models.py`'s `requests` table), for
operational reasons unrelated to moderation. `SECURITY.md`'s "zero personal data collected"
claim is scoped to *node* telemetry (node id, layer range, core/RAM counts, IP) and is accurate
as scoped — but if you're a chat/API user, your prompts are not private from the coordinator
operator today. Stated here plainly rather than left implicit.

---

## Reporting

If you believe NEURON generated something that violates this policy, or you have a safety
concern about the network, open an issue on the GitHub repo (see `README.md`) or contact the
maintainer directly.

---

## Roadmap

- Upgrade from keyword matching to a real classifier once there's budget/infrastructure for it.
- A durable per-user identity (tied to the wallet system) to handle repeat violations —
  today there's no way to attach a ban to an anonymous chat session.
- Formal log retention policy once real user volume exists.

---

**This is not legal advice.** This policy has not been reviewed by a lawyer. Before any public
launch at meaningful scale, get one — the same posture `TOKENOMICS.md` already takes on the
economics side.
