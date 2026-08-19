"""test_encrypted_hop_is_reachable.py — run: python test_encrypted_hop_is_reachable.py

[P52] shipped an encrypted, authenticated hop and proved it against a byte-recording relay.
The proof was real. What nobody checked was whether the PRODUCT could reach that code.

It could not. `neuron_driver._connect()` called `wire_crypto.client_handshake(...)` and caught
`wire_crypto.HandshakeError` with **nothing importing the name**, so the first hop a coordinator
minted a grant for died with `NameError: name 'wire_crypto' is not defined`. Every test passed,
because the tests import `security.wire_crypto` themselves. It survived from 2026-08-18 to
2026-08-19 for one reason: local-first execution answers almost everything on the user's own
machine, so the driver path — and therefore the encryption — essentially never ran.

The same package went missing in three other places on the same day, each silent in its own way:

    coordinator/deploy.sh    FILES list        -> live coordinator in a restart loop, 502
    coordinator/requirements cryptography      -> same, one layer down
    packaging spec           hiddenimports     -> frozen app whose NETWORK path fails while
                                                  local answers keep working, so smoke tests pass

The lesson is not "remember four places". It is that a missing registration fails at RUNTIME on
a path nobody exercises, which is indistinguishable from working. So this file asserts the
registrations directly.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def names_used(path, attr):
    """Modules referenced as `<attr>.something` anywhere in the file."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == attr for n in ast.walk(tree))


def imports(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            found.update(a.asname or a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            found.update(a.asname or a.name for a in n.names)
    return found


def main():
    print("\n-- every module that USES wire_crypto also imports it")
    # The whole bug in one rule. Checked by parsing rather than by importing, so it holds even
    # for files that cannot be imported in a test process (torch, sockets, a real config).
    for rel in ("neuron_driver.py", "agent/node_server.py", "coordinator/router.py"):
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            continue
        if names_used(p, "wire_crypto"):
            check(f"{rel} imports the wire_crypto it uses",
                  "wire_crypto" in imports(p),
                  "this is exactly the NameError that broke the network path")

    print("\n-- the frozen app carries the package")
    spec = open(os.path.join(HERE, "packaging/neuron-agent.spec"), encoding="utf-8").read()
    check("the PyInstaller spec lists security in hiddenimports",
          '"security"' in spec and '"security.wire_crypto"' in spec,
          "PyInstaller follows imports; a package reached only at runtime must be declared, "
          "and omitting it breaks the NETWORK path while local answers keep working")

    print("\n-- the coordinator deploy carries it too")
    dep = open(os.path.join(HERE, "coordinator/deploy.sh"), encoding="utf-8").read()
    check("deploy.sh ships the security package",
          "FILES=(coordinator security" in dep,
          "router.py imports it at module scope, so a deploy without it is a restart loop")
    req = open(os.path.join(HERE, "coordinator/requirements.txt"), encoding="utf-8").read()
    check("the coordinator requires cryptography", "cryptography" in req,
          "wire_crypto needs it, and unlike eth-account it is imported eagerly")

    print("\n-- a node too old to handshake is not sent a grant")
    # The other half of the rolling upgrade. [P52] covered old-coordinator/new-node; this is
    # new-coordinator/old-node, which failed live on 2026-08-19 as "socket closed during
    # handshake" — a version mismatch wearing the costume of a network fault.
    from coordinator.router import _speaks_secure_hop as speaks
    check("0.20.5, the first build with the hop, is sent one",
          speaks({"agent_version": "0.20.5"}) is True)
    check("0.20.6 is sent one", speaks({"agent_version": "0.20.6"}) is True)
    check("0.21.0 is sent one", speaks({"agent_version": "0.21.0"}) is True)
    check("0.20.3 — the Pavilion, which actually broke — is NOT",
          speaks({"agent_version": "0.20.3"}) is False)
    check("0.19.9 is not", speaks({"agent_version": "0.19.9"}) is False)
    check("a node that cannot say what it runs is treated as too old",
          speaks({"agent_version": None}) is False and speaks({}) is False,
          "assuming about the one node that will not tell you is how a fleet partitions")
    check("garbage does not crash the router",
          speaks({"agent_version": "not-a-version"}) is False)

    print("\n-- and it actually resolves at runtime")
    try:
        import neuron_driver
        check("neuron_driver.wire_crypto resolves", hasattr(neuron_driver, "wire_crypto"))
    except ImportError as e:                      # torch missing in a bare checkout
        print(f"  SKIP  importing neuron_driver ({e.__class__.__name__})")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
