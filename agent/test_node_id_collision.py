"""agent/test_node_id_collision.py — run: python -m agent.test_node_id_collision

Node ids used to be exactly `agent-{hostname}`, which collides deterministically. Windows ships
defaults like DESKTOP-8F3K2P1, plenty of machines are called "laptop", and the SAME machine
reinstalling produces the same id it had before. The coordinator refuses a secret-less
registration of an id that is already trusted/verified (the hijack guard — correct), so a
collision meant a 409 on every attempt, forever, and that machine could never join.

Observed live as an endless retry:
    this node_id is registered with a different token — another copy of the agent
    probably re-registered it. Restart this agent to pick up the current token.

Restarting could not help: no copy of the agent held that token.
"""
import json
import os
import tempfile

import requests

from agent import agent as agentmod

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


class _Resp:
    def __init__(self, status):
        self.status_code = status

    def raise_for_status(self):
        # only 4xx/5xx raise — an unconditional raise here made a SUCCESSFUL registration look
        # like a failure, and setup() then retried every 60s forever (the test hung, not the code)
        if self.status_code >= 400:
            e = requests.HTTPError(f"{self.status_code}")
            e.response = self
            raise e

    def json(self):
        return {}


def _agent(tmpdir, **over):
    path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(layer_start=0, layer_end=9, **over)
    json.dump(cfg, open(path, "w"))
    return agentmod.Agent(config_path=path), path


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron-collide-")

    # 1) a generated id is unique per install, not just per hostname
    a, _ = _agent(tmpdir)
    id1, id2 = a.new_node_id(), a.new_node_id()
    check("generated node ids embed the hostname", id1.startswith("agent-"))
    check("...but two installs on the same host differ", id1 != id2)

    # 2) a 409 with NO token must take a new identity, not retry forever
    b, path = _agent(tmpdir, node_id="agent-taken", node_token=None)
    calls = []
    real_post, real_get = agentmod.requests.post, agentmod.requests.get

    def fake_post(url, **kw):
        calls.append(kw.get("json", {}).get("node_id"))
        # the first identity is already claimed; any fresh one is accepted
        if calls[-1] == "agent-taken":
            return _Resp(409)
        r = _Resp(200)
        r.json = lambda: {"node_token": "fresh-token", "assigned_layers": [0, 9],
                          "standing": "probationary"}
        return r

    agentmod.requests.post = fake_post
    agentmod.requests.get = lambda *a, **k: _Resp(200)
    # stop setup() after registration succeeds, so we test only the identity logic
    b.slice_info = lambda: (_ for _ in ()).throw(KeyboardInterrupt)
    try:
        b.setup()
    except KeyboardInterrupt:
        pass
    finally:
        agentmod.requests.post, agentmod.requests.get = real_post, real_get

    check("the taken id was tried first", calls and calls[0] == "agent-taken")
    check("a 409 with no token leads to a NEW id being tried, not an endless retry",
          len(calls) >= 2 and calls[1] != "agent-taken")
    saved = json.load(open(path))
    check("the new id is persisted so the next start reuses it",
          saved.get("node_id") == calls[1])

    # 3) a 409 while we DO hold a token is a different situation (another copy re-registered);
    #    that must NOT rotate identity, or a node would abandon its id on a transient clash.
    c, cpath = _agent(tmpdir, node_id="agent-mine", node_token="i-own-this")
    posts = []

    def fake_post2(url, **kw):
        posts.append(kw.get("json", {}).get("node_id"))
        return _Resp(409)

    agentmod.requests.post = fake_post2
    agentmod.requests.get = lambda *a, **k: _Resp(200)
    c._stop.set()          # one pass through the retry loop, then exit
    try:
        c.setup()
    finally:
        agentmod.requests.post, agentmod.requests.get = real_post, real_get
    check("a 409 while holding a token does NOT abandon the id",
          json.load(open(cpath)).get("node_id") == "agent-mine")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
