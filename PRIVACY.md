# NEURON — Privacy

Written to be checkable against the code, not to reassure. Every claim below names the file that
implements it, so you can verify it rather than trust it. Where something is a limitation, it is
stated as one.

**NEURON has two kinds of user and they are exposed to different things.** This document is split
that way, because writing "user" for both is how a privacy promise ends up true of one and false
of the other:

- **[If you chat](#if-you-chat)** — you type prompts and read answers.
- **[If you donate a machine](#if-you-donate-a-machine)** — your PC serves part of the network.

Most people who run NEURON are both.

---

## If you chat

### Your words never leave this computer, even though the network answers

**The network answers every request (changed 2026-08-22).** NEURON used to run the model on your
own machine whenever it fitted, and use the network only for models one machine could not hold.
That is no longer the default: requests go to the node chain, and if the chain cannot serve one
it fails and says so rather than quietly answering here. Local execution still exists and is
opt-in — set `NEURON_LOCAL_FIRST=1` — and this document describes the default.

**What that does and does not change for you.** Your prompt is turned into numbers *on this
computer*: it holds the tokenizer, the embedding and the model's first layers, so what crosses
the wire is activations, never your text. That was already true of the network path and is why
the table below reads the way it does. What HAS changed is that this now happens on every
message rather than only on the large ones, and each one costs NRN.

A model running here answered for free and sent nothing at all. If that is what you want, the
env var above restores it.

The app tells you which is happening: the header reads **"Answers run here"** when your machine
is serving itself, and **"Powered by N nodes"** when the network is.

### When the network is used, your prompt is encrypted in transit

For a model too large for one machine, your text is turned into numbers on **your** machine and
those numbers travel to other volunteers' machines.

| who | can they read it? |
|---|---|
| anyone on the internet between the machines | **no** |
| whoever runs the relay | **no** |
| **the NEURON coordinator itself** | **no** — from v0.20.23; see the correction below |
| the volunteer machines computing your answer | they hold numbers, not your words — see below |

Each link is separately encrypted, with a one-time key, using an ephemeral key exchange
(`security/wire_crypto.py`). "Ephemeral" is the part that matters: even the coordinator — which
introduces the two machines and knows their credentials — cannot decrypt a recorded session
afterwards.

### A correction to this document, and when it takes effect

**Up to and including v0.20.22, the row above was wrong.** Before your machine could use the
network it asked the coordinator which machines to use, and that request carried your prompt —
`node_a.coord_get_chain` sent `{"prompt": ...}` to `/infer`. The coordinator never read it: both
uses were its LENGTH, one for a cost estimate and one for the character count kept against the
request, and the text was never stored (`coordinator/models.py` keeps `prompt_len` only) and
never logged. But it was sent, and "nothing reads it" is a weaker promise than the one this
table made. Found and fixed on 2026-08-22; recorded as [P65] in
[PROBLEMS.md](PROBLEMS.md), which is where this project keeps its mistakes.

**What changed:** your machine now sends the number of characters instead of the characters
(`prompt_chars`). The coordinator still accepts the old shape, because agents already installed
still send it — so the fix reaches you when you update, and the row above is true of v0.20.23
onward. If you are running an earlier build, it is not yet true of yours; the app updates itself
daily, and you can force it from the tray.

### What "privacy by architecture" claims, and what it does not

The landing page marks NEURON with a tick on this. It is worth being exact about what that tick
is claiming, because a privacy claim that is vaguer than the code is how the correction above
became necessary.

**It claims these, and each is checkable:**

- your prompt TEXT never leaves this computer at all: the tokenizer, the embedding and the
  model's first layers run here, so what is sent is activations (`neuron_driver.py`, `node_a.py`).
  With `NEURON_LOCAL_FIRST=1` nothing leaves at all, because the whole model runs here
  (`engine/local_gguf.py`, and the header says which mode you are in);
- when the network is used, every hop is separately encrypted with a one-time ephemeral key, so
  neither the relay operator nor the coordinator can read the wire, then or from a recording
  afterwards (`security/wire_crypto.py`);
- the coordinator is sent the LENGTH of your prompt and never the prompt (`node_a.py`);
- your conversations are stored only on your machine (`ui/conversations.py`);
- nothing else is collected — see the list below, which is exhaustive.

**It does not claim** that a volunteer whose machine is in your chain is mathematically unable to
learn anything. That machine holds the numbers it computes on, in plaintext, because that is the
work it is doing. What limits it is structural rather than cryptographic: it holds a middle slice
of the layers and neither the part that reads text in nor the part that writes text out, so it
cannot turn those numbers back into your words on its own; and different requests may take
different routes, so no one machine holds a whole conversation.

**No decentralised inference network can currently offer the mathematical version of that
guarantee.** The techniques that would — homomorphic evaluation, or secure multi-party
computation of a transformer — are orders of magnitude too slow to serve a chat, and the
hardware-enclave route would mean only certain CPUs could join, which is the opposite of the
point. If that changes, this document changes with it. Until then the tick means the
architecture, not a promise, is what protects you — and the sentence above is the honest edge
of it.

### The limitation, stated rather than glossed

A machine computing part of your answer necessarily has the numbers in plaintext inside its own
process. That is the work it is doing. What limits it:

- **no single machine holds the whole model**, so it cannot turn those numbers back into your
  text on its own — it has a middle slice of the layers, and neither the part that reads text in
  nor the part that writes text out;
- **no single machine holds your whole conversation** — different requests may take different
  routes.

That is a real structural protection and it is **not** a mathematical guarantee against a
determined volunteer whose machine is in your chain. No decentralised inference network can
currently offer that one, and NEURON does not claim it.

### Two things that do leave your machine, and both are named

1. **Web search — off unless you turn it on.** If you enable it for a message, that query is
   sent to DuckDuckGo so the answer can use current information (`rag/retriever.py`). This goes
   to a third party, not to NEURON. It is a per-message choice and the default is off.
2. **A blocked prompt reports its category.** If a prompt is refused under
   [SAFETY.md](SAFETY.md), the coordinator is told your account and the *category*
   (e.g. `weapons_cbrn`) so repeated attempts can be acted on
   (`safety/moderation.py: report_violation`). **The text is never sent** — the snippet stays in
   a log on your own machine.

### What is stored, and where

- **Your conversations stay on your machine**, in a SQLite file beside the app
  (`ui/conversations.py`). They are not uploaded and NEURON has no copy.
- **Your account** is a wallet id created from a Google or GitHub login. NEURON stores the email
  that login provides, so an account can be recovered and abuse can be acted on.
- **Ordinary traffic to the coordinator** — your balance, network status, the update check —
  reveals that you are using NEURON and when. It carries no prompt content.

---

## If you donate a machine

### What your machine sends

A node ID, the layer range it is serving, CPU and RAM counts, and a heartbeat. No personal data,
no file contents, no browsing, no screen.

### What your machine receives and computes on

Numbers derived from other people's prompts. **You cannot read them** — your machine holds a
slice of the layers and not the parts that convert text to numbers or back. This is the mirror
image of the limitation above, and it is worth understanding in both directions: you are trusted
with something you cannot decode, and you are relying on the same being true of others.

### Your earnings, and the one thing that can lose them

Earnings accrue to your node until you **claim** them to the account you sign into. Until you do,
they are held against a credential stored in a single file on that disk — lose the file, lose the
balance. The app offers the claim above the message box; it takes one signature.

### What is logged on your machine

`agent.log`, locally, containing node IDs, layer ranges and errors. It is not uploaded. If you
send it to get help, read it first — it names paths on your disk.

---

## Not collected, anywhere

No files, no browsing history, no keystrokes, no screen contents, no location, no advertising
identifiers, and no telemetry beyond what is listed above. There is no analytics SDK in this
repository.

---

## How to verify any of this

Every claim names a file. The repository is public, and the network's own code is the only
authority worth trusting here:

- local-vs-network decision — `engine/local_gguf.py`, `ui/app.py`
- encryption of each hop — `security/wire_crypto.py`, `security/test_wire_crypto.py`
- what moderation sends — `safety/moderation.py`
- what a node reports — `agent/agent.py`
- who is protected from whom — [SECURITY.md](SECURITY.md)

If you find a claim here that the code does not support, that is a bug in this document and
worth reporting as one.
