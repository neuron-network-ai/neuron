# Changelog

Versions are the **installer/agent** version — the number in `NEURON-Setup-<x>.exe`, in
`config.AGENT_VERSION` and in `updater.LOCAL_VERSION`, which must all agree or a node either
never updates or tries to update to itself forever.

This project is early alpha. NRN has no cash value, and the network is a handful of machines.

## Unreleased

- **NVIDIA GPU capability is detected and reported.** A node reports `has_gpu`, `gpu_vram_gb`
  and `gpu_name` at registration; the coordinator stores them, and the balancer prefers a GPU
  node when two candidates tie on measured speed. `gpu_name` is operator-only in `/node/list`,
  like `platform` and node addresses — a card model is distinctive enough to correlate a roster
  on.
- **A node yields while its GPU is busy.** Each donation mode gained a `gpu_ceiling`, so a
  machine whose CPU looks idle while a game or a render saturates the GPU now pauses. Machines
  with no `nvidia-smi` are unaffected: unreadable utilisation never counts as busy.
- **Not in this change, and worth stating:** inference still runs on the CPU. `common.py`
  selects no device, so a GPU node computes exactly like a CPU one. VRAM-as-capacity is
  implemented in the balancer but gated off behind `balancer.GPU_EXECUTION`, because counting
  VRAM the pipeline cannot use would over-assign layers and OOM a volunteer's machine.
- `CONTRIBUTING.md` and this file added; README status refreshed.

## v0.17.1

- **A node can follow the coordinator if it moves.** The coordinator's public address is
  returned on every heartbeat and on registration. An agent adopts a new one only after
  probing it successfully (`GET <new>/node/<id>/ping` with its own token) and keeps the
  previous value — so a typo in one config value cannot strand the whole network at once.
- Insurance shipped before it is needed: `neuronnet.duckdns.org` is intended to stay the
  stable public name, with Cloudflare going behind it rather than replacing it.

## v0.17.0

- **The app updates itself.** It asks the coordinator once a day whether a newer version
  exists, verifies a published SHA-256 before running anything, refuses and deletes the file
  on a mismatch, and never updates mid-request. An empty published hash means no node installs
  anything — the correct failure direction for a mechanism that runs binaries unattended.
- **The local chat UI works again.** 0.16.5 shipped without `_sqlite3`, so the Chat UI and API
  Docs tray entries stayed greyed out with no explanation.
- **Payout keys ship.** `eth_account`/`eth_keys`/`eth_utils` are bundled, so a node can
  generate a secp256k1 key on first run and bind an address it proves it controls.
- Log lines carry the logger name, so `agent.log` says which component spoke.
- `config.AGENT_VERSION` corrected from `0.3.0`, where it had drifted for fourteen versions.

## v0.16.5

- **First public installer.** Published as a GitHub release, with the project moved to the
  `neuron-network-ai` organisation and personal identity stripped from the repository.
- Autostart is ticked by default in the installer, and the support/disclosure URLs are correct.
- Windows still calls the installer "unrecognized" — it is not code-signed, and that needs a
  certificate rather than a code change.

## Earlier (0.1.0 – 0.16.3)

Internal development only; none of these were distributed to anyone outside the project. The
work is logged session by session in [sessions.md](sessions.md) — layer-split inference proven
bit-exact, the KV cache, the coordinator and NRN ledger, the byte-range slice downloader, the
NAT relay, open join with proof-of-compute, peer quorum verification, the non-executable wire
codec, and the fixed-supply ledger.
