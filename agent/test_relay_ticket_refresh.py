"""agent/test_relay_ticket_refresh.py — run: python -m agent.test_relay_ticket_refresh

Real bug found while smoke-testing the rebuilt installer: a NAT'd node's cached config from
before relay_auth.py existed has a relay block with no "ticket" key. setup() only calls
register() when node_id/node_token are missing, so that node could NEVER self-heal a missing
ticket -- its tunnel would churn against the relay's "bad/missing ticket" check forever. Fixes
setup()'s registration-skip condition to also re-register when behind_nat and the cached relay
has no ticket. Mocks register()/slice_info()/ensure_slice/NodeServer -- no real network.
"""
import json
import os
import tempfile

import agent.agent as agentmod

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _agent(tmpdir, **cfg_overrides):
    cfg_path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="test-node", node_token="tok-1", model_id="m",
              layer_start=0, layer_end=9, slice_dir="./slice/")
    cfg.update(cfg_overrides)
    json.dump(cfg, open(cfg_path, "w"))
    return agentmod.Agent(config_path=cfg_path)


def _stub(a, register_calls):
    a.register = lambda: register_calls.append(1)
    a.slice_info = lambda: (_ for _ in ()).throw(a.StopTest if hasattr(a, "StopTest") else SystemExit)


class _StopSetup(Exception):
    pass


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron_relay_ticket_test_")

    # ---- credentials present + a ticket already cached -> register() NOT called ---- #
    a1 = _agent(tmpdir, behind_nat=True,
               relay={"host": "h", "control_port": 1, "data_port": 2,
                      "public_port": 3, "ticket": "real-ticket"})
    calls1 = []
    a1.register = lambda: (calls1.append(1) or (_ for _ in ()).throw(_StopSetup))
    a1.slice_info = lambda: (_ for _ in ()).throw(_StopSetup)
    try:
        a1.setup()
    except _StopSetup:
        pass
    check("credentials + a real cached ticket -> register() is NOT called", calls1 == [])

    # ---- credentials present but relay has NO ticket (the actual bug) -> re-register ---- #
    a2 = _agent(tmpdir, behind_nat=True,
               relay={"host": "h", "control_port": 1, "data_port": 2, "public_port": 3})
    calls2 = []
    a2.register = lambda: (calls2.append(1) or (_ for _ in ()).throw(_StopSetup))
    try:
        a2.setup()
    except _StopSetup:
        pass
    check("credentials present but relay ticket missing -> register() IS called to refresh it",
          calls2 == [1])

    # ---- not behind_nat at all -> a missing "ticket" key is irrelevant, no forced re-reg ---- #
    a3 = _agent(tmpdir, behind_nat=False, relay=None)
    calls3 = []
    a3.register = lambda: (calls3.append(1) or (_ for _ in ()).throw(_StopSetup))
    a3.slice_info = lambda: (_ for _ in ()).throw(_StopSetup)
    try:
        a3.setup()
    except _StopSetup:
        pass
    check("not behind_nat -> register() not forced (nothing to refresh)", calls3 == [])

    # ---- no credentials at all -> register() still called as before (unaffected) ---- #
    cfg_path = os.path.join(tmpdir, "fresh.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    json.dump(cfg, open(cfg_path, "w"))
    a4 = agentmod.Agent(config_path=cfg_path)
    calls4 = []
    a4.register = lambda: (calls4.append(1) or (_ for _ in ()).throw(_StopSetup))
    try:
        a4.setup()
    except _StopSetup:
        pass
    check("no node_id/token at all -> register() still called (pre-existing behavior unaffected)",
          calls4 == [1])

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
