"""ui/test_node_identity_is_current.py — the Chat UI must not cache a rotated node token.

    python -m ui.test_node_identity_is_current

**Live 2026-08-18.** `/node/owner` answered `401 Client Error: Unauthorized` for this machine's
own node, while `node_server` on the same box answered proof-of-compute challenges correctly on
the same identity. Two snapshots of a value the agent ROTATES:

    local_chat.start_local_chat()   os.environ.setdefault("NEURON_NODE_TOKEN", …)  once, at
                                    agent startup
    ui/app.py                       NODE_TOKEN = os.environ.get(...)               once, at
                                    import

`register()` issues a fresh token on a relay-ticket refresh, on stale-token recovery, and on any
re-registration (`agent.py:1103`). From that moment the Chat UI carried a dead credential until
the whole process was restarted — and `/node/owner` is what the claim panel reads, so a token
rotation silently switched off the one feature that UI exists to offer. The same blank panel
[P39] was about, reached by a different road.

The fix is that identity is resolved WHEN USED, from the agent's config.json — which is what
`_save()` writes on rotation — with the environment kept as a fallback for a dev override.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()

    print("\n-- the identity is not a module-level snapshot")
    check("there is no import-time NODE_TOKEN constant",
          "\nNODE_TOKEN = os.environ.get" not in src,
          "read once at import, it goes stale the moment the agent re-registers")
    check("there is no import-time NODE_ID constant either",
          "\nNODE_ID = os.environ.get" not in src)
    check("a resolver exists", "def _node_identity():" in src)

    print("\n-- and every route that uses it resolves it per call")
    # A resolver nothing calls is the same bug with more code. Each of these three endpoints
    # authenticates to the coordinator AS this node; a stale token 401s all of them.
    for route in ("def node_owner(", "def node_payout_challenge(", "def node_payout_bind("):
        i = src.find(route)
        check(f"{route.split('(')[0][4:]} resolves the identity", i != -1
              and "_node_identity()" in src[i:i + 900],
              "this route still closes over a value captured at import")

    print("\n-- config.json is the authority, because it is what rotation writes")
    check("the resolver reads config.json", "config.json" in
          src[src.index("def _node_identity():"):src.index("def _node_identity():") + 1400])
    check("...and keeps the environment as a fallback",
          "NEURON_NODE_TOKEN" in src[src.index("def _node_identity():"):
                                     src.index("def _node_identity():") + 1400],
          "a dev shell override must still work, and a driver-only machine has neither")
    check("...and never raises",
          "except (OSError, ValueError):" in src,
          "a half-written config is 'this machine serves no node', a state already rendered")

    print("\n-- the behaviour, against real files")
    # Import the resolver without dragging in torch: exec just that function.
    start = src.index("def _node_identity():")
    end = src.index("\n\n", src.index("return env_id, env_tok"))
    # __file__ is referenced by the repo-checkout fallback branch; supply the real one.
    ns = {"os": os, "json": json, "__file__": os.path.join(HERE, "app.py")}
    from pathlib import Path
    ns["Path"] = Path
    exec(compile(src[start:end], "<resolver>", "exec"), ns)
    resolve = ns["_node_identity"]

    tmp = tempfile.mkdtemp(prefix="neuron-ident-")
    os.makedirs(os.path.join(tmp, "NEURON"), exist_ok=True)
    cfgp = os.path.join(tmp, "NEURON", "config.json")
    old_local, old_id, old_tok = (os.environ.get("LOCALAPPDATA"),
                                  os.environ.get("NEURON_NODE_ID"),
                                  os.environ.get("NEURON_NODE_TOKEN"))
    try:
        os.environ["LOCALAPPDATA"] = tmp
        os.environ["NEURON_NODE_ID"] = "stale-node"
        os.environ["NEURON_NODE_TOKEN"] = "STALE-TOKEN"

        json.dump({"node_id": "n1", "node_token": "TOKEN-A"}, open(cfgp, "w"))
        check("it reads the config, not the stale environment", resolve() == ("n1", "TOKEN-A"),
              str(resolve()))

        # THE ROTATION. Same process, no restart, no re-import.
        json.dump({"node_id": "n1", "node_token": "TOKEN-B"}, open(cfgp, "w"))
        check("a rotated token is picked up with no restart", resolve() == ("n1", "TOKEN-B"),
              f"{resolve()} — this is the 401 that disabled the claim panel")

        json.dump({"node_id": "n1"}, open(cfgp, "w"))
        check("a config with no token falls back rather than half-authenticating",
              resolve() == ("stale-node", "STALE-TOKEN"), str(resolve()))

        open(cfgp, "w").write("{ not json")
        check("a half-written config does not raise",
              resolve() == ("stale-node", "STALE-TOKEN"), str(resolve()))

        os.remove(cfgp)
        for k in ("NEURON_NODE_ID", "NEURON_NODE_TOKEN"):
            os.environ.pop(k, None)
        check("no config and no environment is 'this machine serves no node'",
              resolve() == (None, None), str(resolve()))
    finally:
        for k, v in (("LOCALAPPDATA", old_local), ("NEURON_NODE_ID", old_id),
                     ("NEURON_NODE_TOKEN", old_tok)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
