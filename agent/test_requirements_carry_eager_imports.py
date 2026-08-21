"""agent/test_requirements_carry_eager_imports.py — run:
       python -m agent.test_requirements_carry_eager_imports

**[P54]'s fourth registration site, checked instead of remembered.** When `security/` was
created, four places had to learn about it and each failed silently in its own way — the
driver's imports, the coordinator's deploy file list, `coordinator/requirements.txt`, and the
PyInstaller spec. Three were fixed. `agent/requirements.txt` — the file that tells a volunteer
what a NODE needs — was not, because nothing checks that a new package reaches every manifest
obliged to name it.

It surfaced on `optiplex-server`, 2026-08-21, joining a third machine to the network: a venv
built from `agent/requirements.txt` produced

    ModuleNotFoundError: No module named 'cryptography'

for `neuron_driver`, `agent.node_server`, `agent.agent` and `api.openai_compat` at once. The
agent never reached its first log line. It had never shown up before because both existing
machines were set up by hand, months apart, by someone who installed what was missing as it
broke.

The rule this pins: **a package imported at MODULE SCOPE on the node's startup path is not
optional, and must be named in the file that claims to list the node's dependencies.** A
lazily-imported package (eth-account, inside `payout_key._eth()`) is genuinely optional — its
absence degrades one feature and the node still serves — and is exempt on purpose.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ok = fail = 0

# import name -> the distribution that provides it. Only names this repo actually imports
# eagerly on the node path; anything not here is treated as first-party or stdlib below.
DISTRIBUTION = {
    "torch": "torch", "transformers": "transformers", "requests": "requests",
    "psutil": "psutil", "numpy": "numpy", "cryptography": "cryptography",
    "safetensors": "safetensors", "accelerate": "accelerate",
    "eth_account": "eth-account", "huggingface_hub": "huggingface_hub",
    "pystray": "pystray", "PIL": "Pillow", "fastapi": "fastapi", "uvicorn": "uvicorn",
    "pydantic": "pydantic", "starlette": "starlette", "itsdangerous": "itsdangerous",
    "authlib": "authlib", "llama_cpp": "llama-cpp-python",
}

# Modules the agent pulls in at module scope on its way to its first log line. Kept explicit
# rather than discovered, because the point is to state what the startup path IS.
NODE_STARTUP_PATH = ["agent/agent.py", "agent/node_server.py", "neuron_driver.py",
                     "common.py", "security/wire_crypto.py", "agent/resource_guard.py",
                     "agent/updater.py"]


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def eager_imports(path):
    """Top-level `import x` / `from x import ...` only. An import inside a function or a
    try/except is a deliberate soft dependency and is not what this test is about."""
    tree = ast.parse(open(os.path.join(ROOT, path), encoding="utf-8").read())
    names = set()
    for node in tree.body:                       # body only == module scope only
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def listed_in(reqs_path):
    """Distribution names named in a requirements file, comments and pins stripped."""
    out = set()
    for line in open(os.path.join(ROOT, reqs_path), encoding="utf-8"):
        line = line.split("#")[0].strip()
        if line:
            out.add(re.split(r"[<>=\[!;]", line)[0].strip().lower())
    return out


def main():
    listed = listed_in("agent/requirements.txt")

    print("-- what the node imports before it can log anything")
    needed = set()
    for path in NODE_STARTUP_PATH:
        needed |= {DISTRIBUTION[n] for n in eager_imports(path) if n in DISTRIBUTION}
    print(f"     {', '.join(sorted(needed))}")

    print("\n-- every one of them is named in agent/requirements.txt")
    for dist in sorted(needed):
        check(f"{dist} is listed", dist.lower() in listed,
              "imported at module scope on the startup path, so its absence is not a degraded "
              "feature — it is an agent that never starts")

    print("\n-- the one that actually bit us, called out by name")
    check("cryptography is listed", "cryptography" in listed,
          "security/wire_crypto.py is imported at module scope by node_server and "
          "neuron_driver ([P52]); a venv without it crash-loops before the first log line")

    print("\n-- and it is still listed for the coordinator, which learned this first")
    check("coordinator/requirements.txt has it too",
          "cryptography" in listed_in("coordinator/requirements.txt"),
          "a deploy without it put the live coordinator into a restart loop on 2026-08-19 "
          "([P54]) — that line must never be tidied away")

    print("\n-- a lazily-imported dependency is exempt, and stays exempt")
    src = open(os.path.join(ROOT, "agent/payout_key.py"), encoding="utf-8").read()
    check("eth-account is imported lazily, so it is genuinely optional",
          "eth_account" not in eager_imports("agent/payout_key.py")
          and "from eth_account import Account" in src,
          "if this ever moves to module scope it stops being optional and this test's "
          "exemption becomes a hole")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
