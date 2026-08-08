# NEURON v0.19.0

**Early alpha.** A small network of volunteer machines. Download **NEURON-Setup-0.19.0.exe** below.

**Upgrading from 0.17 or 0.18?** Install straight over the top — do not uninstall first. Your model slice, settings and earnings live in `%LOCALAPPDATA%\NEURON` and are kept. If you are on a working 0.18 the app checks daily and will update itself.

## Pause now actually stops your machine serving

This is the one to know about. Clicking **Pause** stopped the heartbeat but did not stop the node taking work, so the network kept sending it live requests for up to 90 seconds while the tray said "Paused". It now refuses new requests immediately and finishes only the answer already in flight — so nobody's reply is cut off, and your machine really does stop.

The refusal is a named reply rather than a dropped connection, on purpose: a closed socket is indistinguishable from a machine crashing, and the network would have rebuilt the whole chain and told the user a machine had dropped out. Paused and died are different events, and they now read differently.

## A correction: this build does not use your GPU

**v0.18.0 said it did. That was wrong, and this release says so plainly.**

The packages NEURON ships are CPU-only builds — `torch 2.4.1+cpu`, and a `llama-cpp-python` with no GPU backend compiled in — so there is no code path that can reach a graphics card. A machine with an RTX 4090 ran exactly like one without it, while the log claimed it was "offloading every layer". This is a packaging limitation, not a missing test machine.

Worse, the coordinator was sizing GPU machines by their **VRAM** while their weights sat in system RAM, which could hand a volunteer far more of the model than their machine could hold. **That is already fixed on the coordinator and needed no update from you** — it applied to every installed 0.18 the moment the server restarted.

What is true today: your node still steps aside while your GPU is busy, so a game or a render never competes with it. And every slice load now prints where the weights actually are (`weights on cpu`), so you can check rather than take our word for it.

Full detail: `PROBLEMS.md` [P31], and the dated correction on the v0.18.0 release notes.

## Also in this release

- **Compute device setting** — tray → *Compute device* (Automatic / CPU only / GPU). Wired end to end and ready for a build that can use a card; today choosing GPU tells you plainly that this build computes on the CPU rather than silently doing nothing. Changes apply on restart.
- **Migrating to a new model no longer holds two copies at once.** The old slice is released before the new one loads, instead of peaking at roughly 150% of a slice on machines chosen for having room for one.
- **The chat header no longer takes credit for an answer it did not serve.** It read "Powered by N nodes worldwide" directly above a reply whose own line said "this machine · Cost: 0.0000 NRN". When your machine answers locally it now says so.
- **Fresh installs contribute at the "balanced" level** — using spare capacity while you work, backing off above 50% CPU, and never on battery. Existing settings are untouched. Change it any time from the tray.

## Honest notes

- NRN records what you contributed. It has no cash value.
- The network is small, so earnings are small. They grow as more machines join.
- Windows will warn about an "unrecognized publisher" — the installer is not code-signed yet. The full source is public; verify the SHA-256 below before running.
- If your 0.18 install crashes on startup it cannot auto-update, because auto-update only runs inside a working agent. Install this build manually over it.

**SHA-256:** `80eb83a2b669b05e3a84016726e84c55e1b86cc227ac2205636cd33bdfa69b81`

Full changelog: [`CHANGELOG.md`](https://github.com/neuron-network-ai/neuron/blob/main/CHANGELOG.md)
