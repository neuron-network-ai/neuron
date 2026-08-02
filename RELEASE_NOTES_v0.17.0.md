# NEURON v0.17.0

**Early alpha.** A small network (a handful of machines). It works, and it is honest about
what it is. Download **NEURON-Setup-0.17.0.exe** below.

**Upgrading from 0.16.5?** Install straight over the top — do not uninstall first. Your model
slice, settings and earnings live in `%LOCALAPPDATA%\NEURON` and are kept, so the upgrade takes
seconds instead of re-downloading gigabytes. **This is the last release you will have to install
by hand:** from 0.17.0 onward the app updates itself.

## What NEURON is

Your computer runs an AI chat locally — fast, private, nothing you type leaves your machine.
While you are not using it, your machine lends spare capacity to a shared network that runs
models too big for any single computer. You earn **NRN** for that contribution.

## What's new in 0.17.0

- **The local chat UI works again.** In 0.16.5 it silently never started — a database component
  was missing from the packaged build, so the Chat UI and API Docs entries in the tray stayed
  greyed out with no explanation. Fixed, and the packaging gap that allowed it is closed.
- **The app updates itself.** It asks the network once a day whether a newer version exists,
  checks the download against a published SHA-256 before running anything, and never updates
  while your machine is in the middle of serving a request. If the network is unreachable it
  does nothing and carries on. You will not have to come back here for the next fix.
- **Your node now has a payout address.** It creates an ordinary Ethereum key on first run and
  registers it as the destination for your earnings, so they belong to you rather than to a
  name in someone's database. You do not need a wallet and there is nothing to set up.
  **Back up `payout_key.json`** (in `%LOCALAPPDATA%\NEURON`) — whoever holds that file holds
  those earnings. Prefer your own address? Put it in `config.json` and run
  `python -m agent.bind_payout`.
- **Uninstall tells you the truth.** It used to report success even when it had failed to
  remove your node from the network, which left a dead entry behind that reinstalling could not
  clear. It now says so, and tells you exactly what to do.
- **"See agent.log" now means something.** The reason a component failed actually reaches that
  file; previously the most useful errors were discarded before they got there.

## What you should know before installing

- **NRN has no cash value.** It is a record of compute contributed, nothing more.
- **The network is small today**, so earnings are small. They grow as more machines join.
- **CPU only.** No GPU, no crypto wallet, no payment details, no personal data collected.
- Your node **pauses automatically** when you use your machine, and on battery.
- First start downloads **1.4–1.8 GB** (your slice of the model) and takes a few minutes. The
  exact size depends on which part of the model the network gives you.
- Your node is **verified automatically within about a minute** of joining, then it earns.
- The Windows installer is **not code-signed**, so Windows will call it "unrecognized". The
  full source is in this repository — read it before you trust it. If you want to check the
  download is the file we built:

      SHA-256  939925ee8ae7ca343e9e90295e8a40b21c2b61f5cdffe0965834067dec29bfe3

  In PowerShell: `Get-FileHash .\NEURON-Setup-0.17.0.exe -Algorithm SHA256`

## Install

See **[INSTALL.md](INSTALL.md)**. Source install works on Windows, Linux and macOS.

## What's in this release

- One-command install; your node picks its own layers, downloads only its slice, and joins.
- Works from behind a normal home router — no port forwarding, no VPN. All traffic is outbound
  through a public relay.
- Nodes restart themselves after a reboot or a crash, and now update themselves too.
- Automatic verification of joining nodes, so earning starts without anyone intervening.
- A local chat UI and an OpenAI-compatible API on your own machine.

## Known limits

- Small model (Qwen2.5-1.5B). Bigger models need more machines to join.
- Answers run locally when your machine can hold the model, so the shared network is used
  mainly for models that don't fit — which is the point of it.
- Single coordinator; no redundancy yet.
- Auto-update is Windows-only for now. A source checkout is never modified automatically —
  it tells you to `git pull` instead.
