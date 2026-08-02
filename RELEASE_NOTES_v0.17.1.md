# NEURON v0.17.1

**Early alpha.** A small network (a handful of machines). It works, and it is honest about
what it is. Download **NEURON-Setup-0.17.1.exe** below.

**If you have 0.17.0, install this one.** It carries a single change, and it is the kind that
cannot be added later: your node can now follow the network if its address ever moves. Without
it a node is pinned to one hostname for good — and if that hostname changed, the node would be
stranded with no way back. Not even reinstalling would fix it. 0.17.0 will also update itself
to this build automatically the next time it checks.

**Upgrading from 0.16.5 or 0.17.0?** Install straight over the top — do not uninstall first.
Your model slice, settings and earnings live in `%LOCALAPPDATA%\NEURON` and are kept, so the
upgrade takes seconds instead of re-downloading gigabytes.

## What NEURON is

Your computer runs an AI chat locally — fast, private, nothing you type leaves your machine.
While you are not using it, your machine lends spare capacity to a shared network that runs
models too big for any single computer. You earn **NRN** for that contribution.

## What's new in 0.17.1

- **Your node can follow the network if it moves.** The coordinator's address was fixed when
  you installed and nothing could change it afterwards. Your node now learns the current
  address from the network itself — and checks that the new address actually answers before
  switching, so a mistake at our end cannot take your node offline. If it does not answer,
  your node keeps talking to the address it already knows.

Everything else here arrived in 0.17.0: the local chat UI works again, the app updates itself,
your node holds the key its earnings are paid to, uninstall reports honestly, and `agent.log`
finally contains the reason when something fails. See
[RELEASE_NOTES_v0.17.0.md](RELEASE_NOTES_v0.17.0.md) for the detail — the payout note in
particular is worth reading: **back up `payout_key.json`**.

## What you should know before installing

- **NRN has no cash value.** It is a record of compute contributed, nothing more.
- **The network is small today**, so earnings are small. They grow as more machines join.
- **CPU only.** No GPU, no crypto wallet, no payment details, no personal data collected.
- Your node **pauses automatically** when you use your machine, and on battery.
- First start downloads **1.4–1.8 GB** (your slice of the model) and takes a few minutes. The
  exact size depends on which part of the model the network gives you.
- Your node is **verified automatically within about a minute** of joining, then it earns.
- The Windows installer is **not code-signed**, so Windows will call it "unrecognized". The
  full source is in this repository — read it before you trust it. To check the download is
  the file we built:

      SHA-256  befeddd3c83dbe00cd987310c47e15fe8a7035ed32fe831139ca93c92c87ab71

  In PowerShell: `Get-FileHash .\NEURON-Setup-0.17.1.exe -Algorithm SHA256`

## Install

See **[INSTALL.md](INSTALL.md)**. Source install works on Windows, Linux and macOS.

## Known limits

- Small model (Qwen2.5-1.5B). Bigger models need more machines to join.
- Answers run locally when your machine can hold the model, so the shared network is used
  mainly for models that don't fit — which is the point of it.
- Single coordinator; no redundancy yet. This release is the groundwork for changing that
  without asking anyone to reinstall.
- Auto-update is Windows-only for now. A source checkout is never modified automatically —
  it tells you to `git pull` instead.
