"""agent/test_coordinator_migration.py — run: python -m agent.test_coordinator_migration

The coordinator's address is written into config.json once, at install time, and nothing could
ever revise it. Moving the public hostname would therefore strand every node that already
exists — permanently, and a new installer could not rescue them, because ensure_config only
writes defaults when there is no config.json at all. The only thing that can tell nodes where
the new address is, is the old host, while it is still up. So the mechanism has to be in place
BEFORE a move, not during one. That is what this covers.

It is also a redirect primitive pointed at every node at once, so the interesting cases are the
refusals: an address that does not answer, and an address that is not an address. Adopting
blindly would replace one way to strand the network with a faster one — a single typo in the
coordinator's PUBLIC_URL, obeyed everywhere simultaneously, with nothing left able to correct
it.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import agent as agent_mod           # noqa: E402

ok = fail = 0
OLD = "https://neuronnet.duckdns.org"
NEW = "https://neuron-lb.example.org"


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class Resp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise agent_mod.requests.HTTPError(str(self.status_code))


def make_agent(tmp, coordinator=OLD):
    path = os.path.join(tmp, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"coordinator": coordinator, "node_id": "agent-x", "node_token": "tok"}, f)
    a = agent_mod.Agent.__new__(agent_mod.Agent)     # no __init__: no network, no threads
    a.config_path = path
    a.cfg = json.load(open(path, encoding="utf-8"))
    a.base = a.cfg["coordinator"].rstrip("/")
    return a, path


def saved(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    real_get = agent_mod.requests.get
    tmp = tempfile.mkdtemp(prefix="neuron-coordmove-")

    print("\n-- a URL has to be a URL")
    for bad in (None, "", "   ", "not-a-url", "/relative/path", "ftp://x/y",
                "javascript:alert(1)", 42, "https://"):
        check(f"rejected: {bad!r}", agent_mod.Agent._normalize_url(bad) is None)
    check("a good one survives, trailing slash trimmed",
          agent_mod.Agent._normalize_url("https://x.example/ ") == "https://x.example")

    print("\n-- the same address is not a move")
    a, path = make_agent(tmp)
    calls = []
    agent_mod.requests.get = lambda *ar, **kw: calls.append(ar) or Resp({"status": "alive"})
    try:
        check("identical URL -> no change",
              a.adopt_coordinator_url({"coordinator_url": OLD}) is False)
        check("a trailing slash is not a change either",
              a.adopt_coordinator_url({"coordinator_url": OLD + "/"}) is False)
        check("nothing was probed", calls == [], str(calls))
        check("nothing was written", saved(path)["coordinator"] == OLD)
        check("a response with no coordinator_url is fine",
              a.adopt_coordinator_url({"status": "alive"}) is False)
        check("a non-dict payload is fine", a.adopt_coordinator_url("nope") is False)
    finally:
        agent_mod.requests.get = real_get

    print("\n-- a real move, probed first")
    a, path = make_agent(tmp)
    probed = []

    def probe_ok(url, headers=None, timeout=None, **kw):
        probed.append(url)
        return Resp({"status": "alive"})

    agent_mod.requests.get = probe_ok
    try:
        moved = a.adopt_coordinator_url({"coordinator_url": NEW})
    finally:
        agent_mod.requests.get = real_get
    check("the move is taken", moved is True)
    check("the new address was probed before being kept",
          probed and probed[0].startswith(NEW), str(probed))
    check("the probe used this node's own ping endpoint",
          probed and probed[0] == f"{NEW}/node/agent-x/ping", str(probed))
    check("config.json now holds the new address", saved(path)["coordinator"] == NEW)
    check("and remembers where it came from", saved(path)["coordinator_previous"] == OLD)
    check("in-memory base is updated too — the NEXT call goes to the new host",
          a.base == NEW)

    print("\n-- it survives a restart (this is the whole point)")
    a2 = agent_mod.Agent.__new__(agent_mod.Agent)
    a2.config_path = path
    a2.cfg = json.load(open(path, encoding="utf-8"))
    a2.base = a2.cfg["coordinator"].rstrip("/")
    check("a fresh agent reads the new address from disk", a2.base == NEW)

    print("\n-- an address that does not answer is NOT adopted")
    a, path = make_agent(tmp)

    def probe_dead(url, headers=None, timeout=None, **kw):
        raise agent_mod.requests.ConnectionError("no route")

    agent_mod.requests.get = probe_dead
    try:
        moved = a.adopt_coordinator_url({"coordinator_url": NEW})
    finally:
        agent_mod.requests.get = real_get
    check("the move is refused", moved is False)
    check("config.json is untouched", saved(path)["coordinator"] == OLD)
    check("and we keep talking to the old host", a.base == OLD)

    print("\n-- an address that answers with an error is NOT adopted")
    a, path = make_agent(tmp)
    agent_mod.requests.get = lambda *ar, **kw: Resp({"detail": "nope"}, status=500)
    try:
        moved = a.adopt_coordinator_url({"coordinator_url": NEW})
    finally:
        agent_mod.requests.get = real_get
    check("the move is refused", moved is False)
    check("config.json is untouched", saved(path)["coordinator"] == OLD)

    print("\n-- a malformed address is never probed, let alone adopted")
    a, path = make_agent(tmp)
    calls = []
    agent_mod.requests.get = lambda *ar, **kw: calls.append(ar) or Resp({"status": "alive"})
    try:
        check("garbage is refused",
              a.adopt_coordinator_url({"coordinator_url": "not-a-url"}) is False)
    finally:
        agent_mod.requests.get = real_get
    check("and no probe was even attempted", calls == [], str(calls))
    check("config.json is untouched", saved(path)["coordinator"] == OLD)

    print("\n-- the heartbeat is what carries it")
    a, path = make_agent(tmp)
    seen = []

    def ping_then_probe(url, headers=None, timeout=None, **kw):
        seen.append(url)
        return Resp({"status": "alive", "coordinator_url": NEW})

    agent_mod.requests.get = ping_then_probe
    try:
        a.ping()
    finally:
        agent_mod.requests.get = real_get
    check("one heartbeat moves the node", a.base == NEW and saved(path)["coordinator"] == NEW)
    check("it pinged the old host, then probed the new one",
          len(seen) == 2 and seen[0].startswith(OLD) and seen[1].startswith(NEW), str(seen))

    print("\n-- a ping that isn't JSON is still a good heartbeat")
    a, path = make_agent(tmp)
    agent_mod.requests.get = lambda *ar, **kw: Resp(None)
    try:
        a.ping()                                  # must not raise
        check("no exception, no move", a.base == OLD)
    except Exception as e:                        # noqa: BLE001
        check("no exception, no move", False, f"{type(e).__name__}: {e}")
    finally:
        agent_mod.requests.get = real_get

    print("\n-- the coordinator actually publishes it")
    from coordinator import config as ccfg
    check("config.PUBLIC_URL exists and is a URL",
          agent_mod.Agent._normalize_url(ccfg.PUBLIC_URL) is not None, str(ccfg.PUBLIC_URL))
    check("and it is the stable public name (the plan is a load balancer BEHIND it)",
          "neuronnet.duckdns.org" in ccfg.PUBLIC_URL, ccfg.PUBLIC_URL)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
