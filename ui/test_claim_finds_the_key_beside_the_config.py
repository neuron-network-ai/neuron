"""ui/test_claim_finds_the_key_beside_the_config.py — run:
       python -m ui.test_claim_finds_the_key_beside_the_config

**The account claim has to look for the payout key where the AGENT put it, and until now it
guessed.** `agent/agent.py` binds with `state_dir=os.path.dirname(self.config_path) or HERE`
— the key lives BESIDE the config that names the node it pays for. `ui/app.py::_node_identity`
knew the config could be in either of two places; `_local_payout_key` hard-coded
`LOCALAPPDATA/NEURON`. On a Windows installer install those coincide, which is why this ran
green for five sessions on the only machine anyone tested it on.

On every other node they differ. Live on the Pavilion (2026-08-21, the first genuinely
unclaimed node the flow was ever run against): config and key both in `~/neuron/agent/`, node
bound to `0xA39F…E3Ee`, the key for that exact address on disk — and `POST /node/claim`
answered

    409  this node pays out to 0xA39F…E3Ee, and this machine holds no key at all —
         nothing here can sign for it. Use the browser-wallet claim…

which is [P53]'s failure returning one layer down: a claim refused over a key the machine is
holding, by an error that blames the operator for losing it.

What is pinned here:

  * the key is found beside the config, wherever that config turned out to be;
  * a keyless machine mints its first key in THAT directory too, not in a second one the
    agent will never read (a key nobody knows exists is [P53]'s other half);
  * `_node_identity` and `_local_payout_key` answer about the SAME machine — the invariant
    whose absence was the whole bug.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("NEURON_SESSION_SECRET", "test-secret-not-a-real-one")

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {label}")
    else:
        fail += 1
        print(f"  FAIL {label}{('  — ' + detail) if detail else ''}")


def _machine(root, with_key=True):
    """A node's state directory exactly as a source-run agent leaves it: config.json naming
    the node, payout_key.json beside it. Deliberately NOT under LOCALAPPDATA — that is the
    layout the bug could not see."""
    from agent import payout_key
    d = Path(root) / "home" / "someone" / "neuron" / "agent"
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"node_id": "agent-under-test", "node_token": "tok-under-test"}),
        encoding="utf-8")
    address = None
    if with_key:
        address, _priv = payout_key.load_or_create(str(d))
    return d, address


def main():
    # The env fallback would otherwise answer for a machine these tests are describing as
    # having no node at all.
    os.environ.pop("NEURON_NODE_ID", None)
    os.environ.pop("NEURON_NODE_TOKEN", None)
    from ui import app as uiapp

    with tempfile.TemporaryDirectory() as root:
        # LOCALAPPDATA points somewhere real and EMPTY, which is the honest reproduction:
        # the machine has a state dir by the old rule, it just is not where the node lives.
        elsewhere = Path(root) / "elsewhere"
        (elsewhere / "NEURON").mkdir(parents=True)
        os.environ["LOCALAPPDATA"] = str(elsewhere)

        print("-- a node whose config and key sit outside LOCALAPPDATA")
        d, address = _machine(root)
        uiapp._config_candidates = lambda: (d / "config.json",)

        nid, tok = uiapp._node_identity()
        check("the node is identified from that config", nid == "agent-under-test", repr(nid))

        found, priv = uiapp._local_payout_key()
        check("the key beside that config is found", found == address,
              f"got {found!r}, key on disk is {address!r}")
        check("the private key comes back with it", bool(priv))

        path, _cfg = uiapp._node_config()
        check("the key directory IS the config directory — the rule agent.py uses",
              path is not None and Path(path).parent == d)

    print("\n-- the same machine with no key yet: create=True must mint it HERE")
    with tempfile.TemporaryDirectory() as root2:
        elsewhere = Path(root2) / "elsewhere"
        (elsewhere / "NEURON").mkdir(parents=True)
        os.environ["LOCALAPPDATA"] = str(elsewhere)
        d2, _none = _machine(root2, with_key=False)
        uiapp._config_candidates = lambda: (d2 / "config.json",)

        check("nothing is reported before one exists", uiapp._local_payout_key()[0] is None)
        made, _priv = uiapp._local_payout_key(create=True)
        check("create=True mints a key", bool(made))
        check("and mints it beside the config, not under LOCALAPPDATA",
              (d2 / "payout_key.json").exists()
              and not (elsewhere / "NEURON" / "payout_key.json").exists())
        check("reading it back gives the same address", uiapp._local_payout_key()[0] == made)

    print("\n-- a machine that serves no node at all")
    with tempfile.TemporaryDirectory() as root3:
        os.environ["LOCALAPPDATA"] = str(Path(root3) / "empty")
        uiapp._config_candidates = lambda: (Path(root3) / "nope" / "config.json",)
        check("no config means no identity", uiapp._node_identity() == (None, None))
        check("and no key is invented for it", uiapp._local_payout_key()[0] is None)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
