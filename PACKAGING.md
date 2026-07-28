# Packaging the NEURON desktop app (Windows installer)

Builds a real Windows app: a **system-tray application** (live NRN balance / status / Pause /
Dashboard / Quit) delivered by a proper **Inno Setup installer** (Start-menu + desktop shortcuts,
Add/Remove Programs entry, optional auto-start, clean uninstall that deregisters the node).

**Honest caveats first:**
- **It's big.** The app bundles PyTorch → ~1–2 GB installed. This is not the "tiny 1 MB agent";
  that needs the llama.cpp engine (see `SCALING.md`, parked). Today it's a Python+PyTorch node, packaged.
- **It's unsigned.** Windows SmartScreen / some antivirus will warn on an unsigned app and setup.
  Removing that needs a code-signing certificate (~$100–400/yr). Until then users click through.
- **Per-platform.** A Windows build must be built on Windows; Linux/macOS separately (no cross-compile).

## Architecture

- **One exe, three modes** (`packaging/neuron_app_entry.py`): no args → **tray app** (agent + icon);
  `--headless` → agent only (servers); `--deregister` → deregister node + delete slice/config
  (the uninstaller calls this).
- **Program vs data.** Installed program → `%LOCALAPPDATA%\Programs\NEURON` (per-user, no admin/UAC).
  Writable state (`config.json`, `agent.log`, `model_slice\`) → `%LOCALAPPDATA%\NEURON` — because an
  app folder can be read-only. The frozen app creates a default `config.json` on first run (open join,
  auto-placement, `donation_mode: idle`, relay on), so a fresh install just works.
- **Personal Chat UI bundled in** (`agent/local_chat.py`): every agent also runs its own local driver
  + Chat UI on `http://localhost:8080` by default (`local_chat: true` in config) — a separate, fixed
  driver slice downloaded alongside whatever compute range the coordinator assigns this machine. Pulls
  in the same stack `ui.app` needs standalone (fastapi/uvicorn/starlette/authlib) — see the spec's
  `datas`/`hiddenimports` for `ui.static` and `safety/blocklist.json`, which are plain data files
  PyInstaller won't discover on its own. Set `local_chat: false` to opt out (compute-only node).

## Build (two steps)

From the repo root, with the venv active:

```bash
# 1. build the app exe (onedir: neuron-agent.exe + _internal\ with torch, transformers, tray)
pip install pyinstaller pystray Pillow
pyinstaller packaging/neuron-agent.spec

# 2. compile the installer  ->  dist\installer\NEURON-Setup-0.12.0.exe
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\neuron.iss
```

That's it — ship `dist\installer\NEURON-Setup-<ver>.exe`. Double-clicking it installs the app,
creates shortcuts, and (optionally) sets it to start at sign-in. Uninstalling via Add/Remove Programs
deregisters the node and removes its slice/config.

## Linux

Run the same `pyinstaller packaging/neuron-agent.spec` on Linux for `dist/neuron-agent/neuron-agent`
(the tray needs a desktop session; servers use `--headless`). Wrapping into a single `.AppImage`
additionally needs `appimagetool` — a later step; the onedir folder is already runnable as a tarball.

## Notes

- `build/` and `dist/` are git-ignored (multi-GB). `packaging/*.spec`, `*.iss`, `neuron.ico`, and the
  entry scripts ARE committed (the reproducible recipe).
- The frozen app finds/creates its state in `%LOCALAPPDATA%\NEURON` (see the `getattr(sys,"frozen",…)`
  branches in `agent/agent.py` and `agent/uninstall.py`).
- Rebuild whenever agent code or dependencies change, then recompile the installer.
