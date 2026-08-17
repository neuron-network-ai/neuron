# NEURON v0.20.3

**Early alpha.** A small network of volunteer machines. Download **NEURON-Setup-0.20.3.exe** below.

**Upgrading from 0.20.x?** Install straight over the top — do not uninstall first. Your model
slice, settings and earnings live in `%LOCALAPPDATA%\NEURON` and are kept. A working 0.20.1 or
later checks daily and will update itself.

## You can now say how much of your machine to lend

A 64 GB workstation used to face a bad choice: donate the whole thing, or nothing. **Set
`donate_ram_gb` in your config and the network never asks for more than that.** It is not a
promise the app tries to keep at runtime — the coordinator sizes your slice from the number you
gave it, so work that would not fit inside your cap is simply never sent to you.

```json
"donate_ram_gb": 8
```

Leave it unset and nothing changes.

## Your machine can now run a model that does not fit on it

The point of a distributed network, and until this release the software could not describe it.
Two changes made it possible:

- **The first machine in a chain also holds the vocabulary tables**, and nobody was counting
  them. On a 4B model that is 1.5 GB — 41% of a small machine's budget, spent invisibly.
- **A node can now say what precision it stores weights at.** Half-precision storage with
  full-precision arithmetic halves what a model weighs in memory without changing the answers,
  and the coordinator had no way to be told a machine was doing it.

Together those turn "this model is too big for any one of you" into a split that actually fits.

## An old processor is told why, instead of just dying

NEURON asks for spare and older hardware, which is exactly the population with processors that
predate the instruction set our maths library was built for. On Windows that failure is
`0xC000001D` — the app closes with no message at all.

It now checks before loading anything and says so in plain words, names the setting to try it
anyway, and admits this has not been reproduced on hardware like yours. **Your machine is not
broken and neither is the download.** If you are on a pre-2013 CPU and it works, we would like
to hear about it.

## Smaller, and mostly invisible

- A node that was handed layers it cannot hold is now refused rather than quietly OOM-killed
  mid-answer, and the reason is written down where an operator can read it.
- A slice whose download was truncated is caught at load instead of serving uninitialized
  weights and returning fluent nonsense.
- Where a chain splits is now decided per model rather than by one number that had to match on
  every machine at once — so it can change without anybody restarting anything.
- If you contribute a machine, the chat now offers to record that its earnings are **yours**,
  signed with a browser wallet. Skip it and your node runs exactly as before.

---

## Publishing this release (operator)

The coordinator advertises a version and a hash; **an empty hash means no node installs
anything**, which is the correct failure direction and also means the steps below are not
optional once `AGENT_VERSION` is 0.20.3.

1. Upload `dist/installer/NEURON-Setup-0.20.3.exe` to the GitHub release `v0.20.3`. Do this
   **first** — the hash is what nodes act on, and a hash pointing at a 404 makes every node
   retry hourly.
2. Set `NEURON_AGENT_SHA256` in the coordinator's systemd unit. Built 2026-08-17,
   216.9 MB:

   ```
   01867f43c7866bc491f1cef5fb20fb34fc5ea95dd777c41d8b3170f6c03581f9
   ```

   Re-hash rather than trusting this line if the exe is ever rebuilt — a PyInstaller build is
   not byte-reproducible, so a second build of the same source has a different hash and every
   node would refuse it.

3. Deploy the coordinator (`bash coordinator/deploy.sh`) and restart the unit.

**Until step 2 is done, deploying the coordinator will have every node see 0.20.3, find no
hash, and decline to install — daily.** That is safe and by design, but it will show up as
`update_check` failures across the fleet, so it is worth knowing which one you are looking at.
