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
            "pystray", "PIL",    # system-tray icon
            # Bundled personal Chat UI (agent/local_chat.py) — the same stack ui.app /
            # api.openai_compat need standalone: web server, sessions, OAuth client.
            "fastapi", "starlette", "uvicorn", "itsdangerous", "authlib", "anyio", "sniffio",
            # Local quantized engine (engine/local_gguf.py). llama_cpp is a thin Python wrapper
            # around a NATIVE library (llama.dll + ggml*.dll); collect_all pulls those in, and
            # without them the frozen app imports llama_cpp and dies at load time instead of
            # falling back cleanly.
            "llama_cpp", "diskcache",
            # Payout address binding (agent/payout_key.py). eth_account imports its backend
            # lazily, so without collect_all the frozen app finds no secp256k1 at runtime and
            # every node silently ships without an on-chain payout destination.
            "eth_account", "eth_keys", "eth_utils", "eth_abi", "rlp", "hexbytes",
            "eth_hash", "eth_typing", "eth_rlp", "ckzg", "cytoolz", "toolz",
            "pyunormalize", "bitarray", "parsimonious"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass  # an absent optional package must not break the build

# Belt and braces for the native side: collect_all misses DLLs that aren't declared as package
# data on some llama-cpp-python builds, and a missing ggml DLL is a silent runtime failure.
try:
    from PyInstaller.utils.hooks import collect_dynamic_libs
    binaries += collect_dynamic_libs("llama_cpp")
except Exception:
    pass

# transformers checks installed versions via importlib.metadata — ship that metadata or it
# raises at import. Distribution names (not import names) go here.
for dist in ("transformers", "torch", "tokenizers", "huggingface-hub", "safetensors",
             "accelerate", "numpy", "tqdm", "regex", "requests", "filelock", "packaging",
             "pyyaml", "psutil", "fastapi", "starlette", "uvicorn", "itsdangerous", "authlib"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# transformers loads model classes lazily — force the Qwen2 model module (our model) in.
hiddenimports += collect_submodules("transformers.models.qwen2")
# our own top-level modules, bundled so the frozen app can import them.
# sqlite3 is reached only through ui/conversations.py (chat history) and coordinator/models.py,
# both imported lazily off the local-chat path -- so it was never declared and its C extension
# was bundled or not depending on what else happened to drag it in. A build that dropped it
# shipped an app whose Chat UI dies with `No module named '_sqlite3'` while the node itself
# keeps serving, which is exactly how it reached a user. Declared now so it cannot silently go
# missing again.
hiddenimports += ["sqlite3", "_sqlite3"]
hiddenimports += ["common", "slice_downloader", "tunnel_client", "neuron_driver", "node_a",
                  "agent", "agent.agent", "agent.resource_guard", "agent.node_server",
                  "agent.tray", "agent.uninstall", "agent.local_chat",
                  "agent.payout_key", "agent.bind_payout",
                  "ui", "ui.app", "ui.oauth", "api", "api.openai_compat",
                  "safety", "safety.moderation", "rag", "rag.retriever",
                  "engine", "engine.local_gguf",
                  "coordinator", "coordinator.ledger", "coordinator.config"]

# ui.app's Chat page (static HTML/JS/CSS) and the moderation blocklist are plain data files,
# not Python — PyInstaller only follows imports, so these need to be listed explicitly or
# the frozen personal Chat UI serves a 404 / the moderation gate can't find its blocklist.
datas += [
    (os.path.join(ROOT, "ui", "static"), os.path.join("ui", "static")),
    (os.path.join(ROOT, "safety", "blocklist.json"), "safety"),
]

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
