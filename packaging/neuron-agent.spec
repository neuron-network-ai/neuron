# neuron-agent.spec — package the NEURON node agent into a distributable folder.
#
# Build from the repo root (with the venv active):
#     pyinstaller packaging/neuron-agent.spec
# Output: dist/neuron-agent/  (neuron-agent.exe + _internal/ with all libraries).
#
# This is ONEDIR (a folder you zip), NOT a single-file exe: the agent bundles PyTorch, which
# makes a one-file build a ~2 GB blob that re-extracts on every launch. A zipped folder is the
# sane distributable. See PACKAGING.md.
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules, copy_metadata

# The spec lives in packaging/; PyInstaller resolves script paths relative to the spec dir,
# so anchor everything to the repo root (the spec's parent) explicitly.
ROOT = os.path.dirname(SPECPATH)  # noqa: F821 (SPECPATH is injected by PyInstaller)

datas, binaries, hiddenimports = [], [], []

# Heavy / dynamically-imported packages: pull in all their submodules, data and binaries.
for pkg in ("torch", "transformers", "accelerate", "safetensors", "tokenizers",
            "huggingface_hub", "regex", "numpy", "psutil", "requests", "certifi",
            "filelock", "tqdm", "yaml", "packaging", "sympy", "networkx",
            "pystray", "PIL"):    # system-tray icon
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass  # an absent optional package must not break the build

# transformers checks installed versions via importlib.metadata — ship that metadata or it
# raises at import. Distribution names (not import names) go here.
for dist in ("transformers", "torch", "tokenizers", "huggingface-hub", "safetensors",
             "accelerate", "numpy", "tqdm", "regex", "requests", "filelock", "packaging",
             "pyyaml", "psutil"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# transformers loads model classes lazily — force the Qwen2 model module (our model) in.
hiddenimports += collect_submodules("transformers.models.qwen2")
# our own top-level modules, bundled so the frozen app can import them.
hiddenimports += ["common", "slice_downloader", "tunnel_client",
                  "agent", "agent.agent", "agent.resource_guard", "agent.node_server",
                  "agent.tray", "agent.uninstall"]

a = Analysis(
    # a distinct launcher (NOT agent/agent.py) so the top-level frozen script isn't named
    # 'agent' and doesn't shadow the agent package (would cause a circular import).
    # Dispatches: tray app (default) / --headless agent / --deregister uninstaller.
    [os.path.join(ROOT, "packaging", "neuron_app_entry.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter"],   # not used; trims bloat
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="neuron-agent",
    console=True,           # the node logs to this terminal window
)
coll = COLLECT(exe, a.binaries, a.datas, name="neuron-agent")
