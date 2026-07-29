"""agent/test_stale_token_recovery.py — run: python -m agent.test_stale_token_recovery

Found live: the installed tray app sat in an endless loop logging

    [WARNING] coordinator unreachable, retrying in 60s:
              409 Client Error: Conflict for url: .../node/register

It was not unreachable. The coordinator mints a FRESH node_token on every registration, so a
second copy of the agent registering the same node_id silently invalidates the token the first
copy is holding in memory. (Here that second copy was a test run of the freshly built exe --
`dist/` and the installed app share %LOCALAPPDATA%\\NEURON\\config.json.) The running app then
retried forever with its dead in-memory token.

Two things were wrong, both fixed and pinned here:

  * calling a 409 "coordinator unreachable" sends everyone hunting a network fault that does not
    exist -- the log has to name the real cause and the actual fix;
  * the agent had no way out. The other copy wrote the CURRENT token to config.json, so the fix
    is simply to re-read it before retrying, and the loop becomes self-healing instead of
    permanent.
"""
import json
import os
import tempfile

import requests

import agent.agent as agentmod

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class _Stop(Exception):
    pass


def _agent(tmp, **over):
    path = os.path.join(tmp, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    # behind_nat with a ticket-less relay is what forces setup() to call register() on an
    # ALREADY-credentialed node -- i.e. exactly the situation the live agent was stuck in.
    cfg.update(node_id="agent-x", node_token="STALE-TOKEN", model_id="m",
               layer_start=0, layer_end=9, slice_dir="./slice/", behind_nat=True,
               relay={"host": "h", "control_port": 1, "data_port": 2, "public_port": 3})
    cfg.update(over)
    json.dump(cfg, open(path, "w"))
    return agentmod.Agent(config_path=path), path


def _http_error(status):
    resp = requests.Response()
    resp.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=resp)


def main():
    tmp = tempfile.mkdtemp(prefix="neuron_stale_token_")

    # ---- a 409 is reported as what it is, not as unreachability ---- #
    a, path = _agent(tmp)
    a.register = lambda: (_ for _ in ()).throw(_http_error(409))
    waits = []
    a._stop.wait = lambda s: (waits.append(s), a._stop.set())[0]
    a.setup()
    check("a 409 is NOT reported as 'coordinator unreachable'",
          "unreachable" not in a.state["detail"])
    check("the message names the real cause (token mismatch)",
          "different token" in a.state["detail"])
    check("the message says what actually fixes it (restart)",
          "Restart" in a.state["detail"] or "restart" in a.state["detail"])

    # ---- a genuine network failure still reads as unreachable ---- #
    a2, _ = _agent(tmp)
    a2.register = lambda: (_ for _ in ()).throw(requests.ConnectionError("no route"))
    a2._stop.wait = lambda s: a2._stop.set()
    a2.setup()
    check("a real connection failure still says 'unreachable'",
          "unreachable" in a2.state["detail"])

    # ---- other HTTP errors are named by status, not mislabelled ---- #
    a3, _ = _agent(tmp)
    a3.register = lambda: (_ for _ in ()).throw(_http_error(500))
    a3._stop.wait = lambda s: a3._stop.set()
    a3.setup()
    check("a 500 is reported by status, not as unreachable",
          "HTTP 500" in a3.state["detail"] and "unreachable" not in a3.state["detail"])

    # ---- THE RECOVERY: another copy wrote a fresh token; we pick it up ---- #
    a4, path4 = _agent(tmp)
    cfg = json.load(open(path4))
    cfg["node_token"] = "FRESH-TOKEN-FROM-OTHER-COPY"     # what the other agent wrote
    json.dump(cfg, open(path4, "w"))
    check("setup starts with the stale token in memory", a4.cfg["node_token"] == "STALE-TOKEN")
    a4.register = lambda: (_ for _ in ()).throw(_http_error(409))
    a4._stop.wait = lambda s: a4._stop.set()
    a4.setup()
    check("a 409 makes the agent re-read config.json and adopt the current token",
          a4.cfg["node_token"] == "FRESH-TOKEN-FROM-OTHER-COPY")

    # ---- and it does not clobber a good token when nothing changed ---- #
    a5, _ = _agent(tmp)
    a5.register = lambda: (_ for _ in ()).throw(_http_error(409))
    a5._stop.wait = lambda s: a5._stop.set()
    a5.setup()
    check("no newer token on disk -> keeps what it had, no crash",
          a5.cfg["node_token"] == "STALE-TOKEN")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
