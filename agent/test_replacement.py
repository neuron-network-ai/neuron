"""agent/test_replacement.py — run: python -m agent.test_replacement

A probationary node must be able to MOVE off a redundant layer range.

Live on 2026-08-07 the network had three machines all serving layers 0-13 of a 28-layer model
and nothing at all on 21-27 — permanently DEGRADED, no request could complete. Each machine had
asked the coordinator where to go, and each had been answered from a view that could not see the
other unverified machines, so all three were told "the first gap is 0-13". Once written to
config.json that answer was final: `ensure_placement` returns early whenever a range is already
configured, so nobody ever re-asked and nobody ever took 21-27.

The coordinator half of the fix is in coordinator/test_placement.py (placement now reasons over
every online node, and `exclude` lets a node ask where it would go if it weren't already there).
This is the agent half: re-ask while probationary, and re-register on the new range.

Moving is free at exactly this moment and no other — a probationary node receives no live
traffic, so nothing is being served off the old slice.
"""
import json
import os
import sys
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
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            e = requests.HTTPError(str(self.status_code))
            e.response = self
            raise e

    def json(self):
        return self._payload


def _agent(tmp, name, **over):
    path = os.path.join(tmp, f"{name}.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="n1", node_token="t", layer_start=0, layer_end=13, port=50999)
    cfg.update(over)
    json.dump(cfg, open(path, "w"))
    return agentmod.Agent(config_path=path), path


def _harness(agent, placement, standing="probationary"):
    """Stub the two calls register() makes. Returns (registrations, placement_queries)."""
    regs, queries = [], []

    def fake_post(url, **kw):
        body = kw.get("json", {})
        regs.append((body["layer_start"], body["layer_end"]))
        return _Resp({"node_token": "t", "standing": standing,
                      "assigned_layers": [body["layer_start"], body["layer_end"]]})

    def fake_get(url, **kw):
        queries.append(kw.get("params") or {})
        return _Resp(placement)

    agent.adopt_coordinator_url = lambda data: None
    agentmod.requests.post, agentmod.requests.get = fake_post, fake_get
    return regs, queries


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-replace-")
    real_post, real_get = agentmod.requests.post, agentmod.requests.get
    real_ip = agentmod.detect_tailscale_ip
    agentmod.detect_tailscale_ip = lambda: "127.0.0.1"

    try:
        # --- the live bug: a redundant probationary node moves --------------------- #
        a, path = _agent(tmp, "moves")
        regs, queries = _harness(a, {"layer_start": 21, "layer_end": 27, "role": "fill-gap",
                                     "reason": "chain is missing layers 21-27"})
        a.register()
        check("registers once on the stale range, then again on the advised one",
              regs == [(0, 13), (21, 27)])
        check("the move is persisted, so a restart does not undo it",
              json.load(open(path))["layer_start"] == 21)
        check("it asks with itself excluded — 'where would I go if I weren't here?'",
              queries and queries[-1].get("exclude") == "n1")

        # --- self-stabilising: a node that belongs where it is must NOT move -------- #
        b, _ = _agent(tmp, "stays")
        regs, _q = _harness(b, {"layer_start": 0, "layer_end": 13, "role": "fill-gap",
                                "reason": "chain is missing layers 0-13"})
        b.register()
        check("advice identical to the current range causes no second registration",
              regs == [(0, 13)])

        # --- a verified node is never moved: it is serving live traffic ------------- #
        c, _ = _agent(tmp, "verified")
        regs, queries = _harness(c, {"layer_start": 21, "layer_end": 27, "role": "fill-gap",
                                     "reason": "gap"}, standing="verified")
        c.register()
        check("a verified node keeps its range even when advice differs", regs == [(0, 13)])
        check("...and is not asked about placement at all", queries == [])

        # --- --layers is an operator instruction, not a suggestion ------------------ #
        d, _ = _agent(tmp, "pinned", layers_pinned=True)
        regs, queries = _harness(d, {"layer_start": 21, "layer_end": 27, "role": "fill-gap",
                                     "reason": "gap"})
        d.register()
        check("a pinned range is never second-guessed", regs == [(0, 13)] and queries == [])

        # --- bounded: one extra round trip, never a loop ---------------------------- #
        e, _ = _agent(tmp, "bounded")
        # advice that keeps changing would recurse forever without the _replaced guard
        seq = iter([{"layer_start": 21, "layer_end": 27, "role": "fill-gap", "reason": "a"},
                    {"layer_start": 14, "layer_end": 20, "role": "fill-gap", "reason": "b"},
                    {"layer_start": 0, "layer_end": 13, "role": "fill-gap", "reason": "c"}])
        regs, _q = _harness(e, None)
        agentmod.requests.get = lambda url, **kw: _Resp(next(seq))
        e.register()
        check("re-placement happens at most once per registration", len(regs) == 2)

        # --- advice must never break a registration that already succeeded ---------- #
        f, _ = _agent(tmp, "resilient")
        regs, _q = _harness(f, {})                 # an older coordinator: no range in the body
        f.register()
        check("an unusable placement answer leaves the node where it is", regs == [(0, 13)])

        g, _ = _agent(tmp, "offline")
        regs, _q = _harness(g, None)
        agentmod.requests.get = lambda url, **kw: (_ for _ in ()).throw(
            requests.ConnectionError("coordinator unreachable"))
        g.register()
        check("an unreachable coordinator does not fail the registration", regs == [(0, 13)])
    finally:
        agentmod.requests.post, agentmod.requests.get = real_post, real_get
        agentmod.detect_tailscale_ip = real_ip

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
