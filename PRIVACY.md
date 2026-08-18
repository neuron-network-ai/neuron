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

### When the model fits on your machine, your prompt never leaves it

This is the ordinary case on a reasonably modern PC. NEURON checks whether your machine can hold
the serving model and, if it can, runs it locally (`engine/local_gguf.py`). Your text is not sent
anywhere, the answer costs 0 NRN, and there is nothing for anyone else to see.

The app tells you which is happening: the header reads **"Answers run here"** when your machine
is serving itself, and **"Powered by N nodes"** when the network is.

### When the network is used, your prompt is encrypted in transit

For a model too large for one machine, your text is turned into numbers on **your** machine and
those numbers travel to other volunteers' machines.

| who | can they read it? |
|---|---|
| anyone on the internet between the machines | **no** |
| whoever runs the relay | **no** |
| **the NEURON coordinator itself** | **no** |
| the volunteer machines computing your answer | they hold numbers, not your words — see below |

Each link is separately encrypted, with a one-time key, using an ephemeral key exchange
(`security/wire_crypto.py`). "Ephemeral" is the part that matters: even the coordinator — which
introduces the two machines and knows their credentials — cannot decrypt a recorded session
afterwards.

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
