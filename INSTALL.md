# Join NEURON

Your computer runs AI for you, and lends its spare time to a shared network.

**What you get:** a private AI chat on your own machine. It runs locally, so it's fast and
nothing you type leaves your computer.

**What the network gets:** while you're not using your machine, it helps run AI models that are
too big for any one computer. You earn **NRN** for that work.

**Honest, before you install:**
- **NRN has no cash value.** It records what you contributed. That's all it is today.
- **The network is small right now** (a handful of machines), so earnings are small. They grow
  as more people join.
- Your node pauses the moment you touch your keyboard, and on battery.
- **Your GPU is not used. This build computes on the CPU, whatever card you have.** The
  packages NEURON ships are CPU-only builds — `torch 2.4.1+cpu`, and a `llama-cpp-python` with
  no GPU backend compiled in — so there is no code path that can reach a card, and a machine
  with an RTX 4090 runs exactly like one without it. This is a **packaging** limitation, not a
  missing feature and not a question of nobody having tested it: your card is detected and
  reported, and it is ignored.

  Earlier releases said the opposite. See the correction in `RELEASE_NOTES_v0.18.0.md`.

  Two things that ARE true today: your node **steps aside while your GPU is busy**, so a game
  or a render is never competing with it; and the node prints where its weights actually live
  on every load — `weights on cpu` in the log, which is the line to check if you want to
  confirm any of this rather than take our word for it.

  There is a **Compute device** setting (tray menu, or `"device"` in your config: `auto`, `cpu`
  or `gpu`). It is wired end to end and ready for a build that can use a card; today picking
  `gpu` tells you plainly that this build computes on the CPU. Changes apply on restart.
- **Your machine contributes while you work, not only when it is idle.** A fresh install
  donates at the **balanced** level: it uses spare capacity up to 50% CPU, backs off above
  that, and never runs on battery. Change it any time from the tray's **Donation level** menu —
  `Idle` only contributes when you are away from the machine.
- First start downloads about **1.4 GB** and takes a few minutes.
- Windows may warn the installer is "unrecognized" — it isn't signed yet, and for the same
  reason your **browser may block the download itself** ("blocked", "not commonly downloaded").
  Neither means the file is bad: an unsigned installer few people have fetched has no
  reputation to check. Both are click-through, and the README explains how, along with the one
  command that proves what you downloaded:
  [If your browser or Windows blocks it](README.md#if-your-browser-or-windows-blocks-it).
- **If Windows 11's Smart App Control is on, the installer will not run at all** — not a
  warning you can dismiss, a block with no "run anyway" button. It requires a code-signing
  certificate we do not have yet. **Use the source install below instead**; it is not affected,
  and it is the recommended route on those machines. Don't switch Smart App Control off for us:
  on most builds that cannot be undone without reinstalling Windows.

**Install:**
```
git clone https://github.com/neuron-network-ai/neuron.git
cd neuron
python -m venv .venv && .venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate
pip install -r agent/requirements.txt
python agent/agent.py
```

**Then:** open http://localhost:8080, sign in, and start chatting. Your node is checked
automatically within a minute and starts earning after that — nothing to ask anyone for.

**Check earnings:** the balance is shown at the top of your chat page.

**Your payout address:** the agent creates a key for you on first run and tells the network
that your NRN belongs to it — you don't have to do anything, and you don't need a crypto
wallet. The key is saved as `payout_key.json` next to your config; **back it up**, because if
NRN ever moves on-chain, whoever holds that file holds those earnings. It's an ordinary
Ethereum key, so you can import it into any wallet.

Already have a wallet and want to use it instead? Put its address in `payout_address` in
`config.json`, then run `python -m agent.bind_payout --address 0xYourAddress`. It prints a
short message; sign that in your wallet and pass it back with `--signature`. Your private key
never leaves your wallet. (Changing an address later needs a signature from the old one — that
way, someone who copies your config file still can't redirect your earnings.)

**Automatic updates.** The agent checks once a day for a newer build, verifies its published
SHA-256 before running anything, and never updates while your node is mid-request. It is on by
default, because otherwise a fix never reaches you. To disable automatic updates, set
`auto_update: false` in `%LOCALAPPDATA%\NEURON\config.json` (on Linux/macOS,
`~/.local/share/NEURON/config.json`) and restart the agent — it is read at startup. With it off,
your node never contacts the network about versions at all, and updating is up to you.

**Remove everything:** `python agent/uninstall.py`
