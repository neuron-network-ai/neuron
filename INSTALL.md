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
- **An NVIDIA GPU with CUDA is used automatically if available — no configuration needed.**
  NEURON detects the card, runs your layers on it, and asks the network for a larger share of
  the model to match the VRAM you have. Your node also steps aside while the GPU is busy, so a
  game or a render is never competing with it. No GPU is fine: everything works on the CPU
  exactly as before. One caveat worth stating — no machine in this project has an NVIDIA card,
  so the GPU path is written and tested but has not yet run on real hardware. If you are the
  first, `agent.log` will say `device: cuda:0`, and we would like to hear how it went.
- First start downloads about **1.4 GB** and takes a few minutes.
- Windows may warn the installer is "unrecognized" — it isn't signed yet. The source is public.

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
