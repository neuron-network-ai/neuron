# NEURON v0.16.4

**Early alpha.** A small network (a handful of machines). It works, and it is honest about
what it is. Download **NEURON-Setup-0.16.4.exe** below.

## What NEURON is

Your computer runs an AI chat locally — fast, private, nothing you type leaves your machine.
While you are not using it, your machine lends spare capacity to a shared network that runs
models too big for any single computer. You earn **NRN** for that contribution.

## What you should know before installing

- **NRN has no cash value.** It is a record of compute contributed, nothing more.
- **The network is small today**, so earnings are small. They grow as more machines join.
- **CPU only.** No GPU, no crypto wallet, no payment details, no personal data collected.
- Your node **pauses automatically** when you use your machine, and on battery.
- First start downloads about **1.4 GB** (your slice of the model) and takes a few minutes.
- Your node is **verified automatically within about a minute** of joining, then it earns.
- The Windows installer is **not code-signed**, so Windows will call it "unrecognized". The
  full source is in this repository — read it before you trust it.

## Install

See **[INSTALL.md](INSTALL.md)**. Source install works on Windows, Linux and macOS.

## What's in this release

- One-command install; your node picks its own layers, downloads only its slice, and joins.
- Works from behind a normal home router — no port forwarding, no VPN. All traffic is outbound
  through a public relay.
- Nodes restart themselves after a reboot or a crash.
- Automatic verification of joining nodes, so earning starts without anyone intervening.
- A local chat UI and an OpenAI-compatible API on your own machine.

## Known limits

- Small model (Qwen2.5-1.5B). Bigger models need more machines to join.
- Answers run locally when your machine can hold the model, so the shared network is used
  mainly for models that don't fit — which is the point of it.
- Single coordinator; no redundancy yet.
