# NEURON v0.18.0

**Early alpha.** A small network (a handful of machines). It works, and it is honest about
what it is. Download **NEURON-Setup-0.18.0.exe** below.

**Upgrading from 0.16.5, 0.17.0 or 0.17.1?** Install straight over the top — do not uninstall
first. Your model slice, settings and earnings live in `%LOCALAPPDATA%\NEURON` and are kept, so
the upgrade takes seconds instead of re-downloading gigabytes. If you are on 0.17.0 or 0.17.1
you do not need to do anything at all: the app checks daily and will update itself.

## What NEURON is

Your computer runs an AI chat locally — fast, private, nothing you type leaves your machine.
While you are not using it, your machine lends spare capacity to a shared network that runs
models too big for any single computer. You earn **NRN** for that contribution.

## What's new in 0.18.0

- **GPU support.** If your machine has an NVIDIA GPU with CUDA installed, NEURON now uses it
  for inference automatically. Nothing to configure, no flag to set. The coordinator also
  assigns GPU machines a larger share of the model, sized against the VRAM you actually have,
  so a card with more memory carries more of the network.
- **Your node steps aside while the GPU is busy.** Start a game or a render and the node
  pauses, the same way it already yields when you touch the keyboard. A machine with no
  `nvidia-smi` is unaffected.
- **CPU machines work exactly as before.** This is the important half: the GPU path is
  additive, and if there is no CUDA device the code takes the same route it always did. The
  split-inference correctness check still reports a bit-exact match against running the whole
  model on one machine.

## Honest notes on the GPU support

**No machine in this project has an NVIDIA GPU**, and the PyTorch we build against is a
CPU-only build. So the CUDA path has been written and reviewed, and it is proven to change
nothing on a CPU machine — but **it has never actually executed on a GPU.** No speed figure is
claimed for it anywhere, because none has been measured. If you are the first person to run
NEURON on a GPU, `agent.log` will say `device: cuda:0` on startup, and we would genuinely like
to hear whether it worked.

If it misbehaves, you can force the old behaviour by setting the environment variable
`NEURON_DEVICE=cpu` before starting the agent. Your node keeps working either way.

**On the wire codec.** An earlier draft of these notes claimed it was deployed; a later one
claimed it was not. Checked properly on 3 August: it **is** on every node — `wire_codec.py` is
byte-identical on the remote machine, and `common.py` imports it at module level so every
packaged build ships it. What is still true is that the benefit is unproven in production: the
network is missing layers 21–27, so no request has crossed the chain to exercise it, and the
4.25× has not been re-measured live. (This file is the source copy; the notes already published
with the v0.18.0 release still carry the older wording.)

## Still true, and worth repeating

- **NRN has no cash value.** It is a record of compute contributed, nothing more.
- **The network is small today**, so earnings are small. They grow as more machines join.
- No crypto wallet, no payment details, and nothing personal is collected.
- Your node pauses when you use your machine, and on battery.
- The Windows installer is **not code-signed**, so Windows will call it "unrecognized". The
  full source is public and the SHA-256 below is the one the auto-updater verifies against.

## Verifying this download

    SHA-256  dd33d317f92adb2ae5da27f46ac02912ba2106a66c4f21104b27ebd7345fb1a8
    Size     216,630,112 bytes (206.6 MB)

Every NEURON install checks this hash before running an update, and refuses — and deletes the
file — if it does not match.
