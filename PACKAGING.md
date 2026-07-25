# Packaging the NEURON agent (desktop builds)

Turns the Python agent into a runnable bundle a less-technical person can start without setting up
Python. **Best-effort, honest caveats first:**

- **It's big.** The agent bundles PyTorch, so the output is **~1–2 GB**. This is not the "tiny
  1 MB agent" — that requires the llama.cpp engine (see `SCALING.md`, parked). Today it's a
  Python+PyTorch node, packaged.
- **It's unsigned.** Windows SmartScreen and some antivirus will warn on an unsigned executable.
  Code-signing needs a paid certificate (~$100–400/yr). Until then, users click through the warning
  (or use the "install Python + run" path in `INSTALL.md`, which is friction-free on that front).
- **Per-platform.** A Windows `.exe` must be built on Windows; a Linux `AppImage`/folder on Linux;
  macOS on macOS. There is no cross-compile.

For a first stranger, `INSTALL.md` (install Python + `python agent/agent.py`) is the reliable path.
This packaging is a convenience for non-technical users.

---

## Build (Windows `.exe`)

From the repo root, with the venv active:

```bash
pip install pyinstaller
pyinstaller packaging/neuron-agent.spec
```

Output: `dist/neuron-agent/` containing `neuron-agent.exe` and an `_internal/` folder of libraries.
This is **onedir** (a folder), not a single file — correct for a PyTorch app (a one-file build would
re-extract ~2 GB to a temp dir on every launch).

### Stage the config and ship it

The agent reads `config.json` **next to the executable**. Copy the template in and zip the folder:

```bash
cp agent/config.json dist/neuron-agent/config.json
# then zip dist/neuron-agent/ -> neuron-agent-windows.zip
```

The default `config.json` already points at the live coordinator, auto-places the node's layers, and
enables the relay (`behind_nat: true`), so the user just unzips and runs `neuron-agent.exe`.

## Build (Linux folder / AppImage)

Run the **same** `pyinstaller packaging/neuron-agent.spec` on a Linux machine to get
`dist/neuron-agent/neuron-agent` (an ELF binary + `_internal/`). Wrapping that into a single
`.AppImage` additionally needs `appimagetool` and an `AppDir` layout — a later step; the onedir
folder is already runnable and distributable as a tarball.

## Notes

- `build/` and `dist/` are git-ignored (multi-GB).
- The frozen agent finds `config.json`, `agent.log` and its model slice next to the executable
  (see the `getattr(sys, "frozen", False)` branch in `agent/agent.py`).
- Rebuild whenever agent code or its dependencies change.
