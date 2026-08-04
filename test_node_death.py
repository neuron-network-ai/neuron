"""test_node_death.py — RESILIENCE.md [R1]: prove a request survives a node dying.

`test_junction_cache.py` says of itself that it is model-free and tests the recovery
BOOKKEEPING, and that replaying into a genuinely fresh chain "has to be proved on the real
three nodes". This is that proof, run locally so it can live in CI instead of needing three
machines.

WHAT IS REAL HERE
  * Real Qwen2.5-1.5B layer slices, loaded in three separate OS processes.
  * The real wire protocol (`common.send_msg` / `wire_codec`), over real TCP sockets.
  * The real driver: `neuron_driver._Driver.stream`, its junction cache and its `_reroute`.
  * A real kill: SIGKILL/TerminateProcess on the middle node mid-generation. Not a mocked
    exception, not a closed socket -- the process stops existing.

WHAT IS STUBBED, AND WHY THAT IS HONEST
  Only the coordinator's chain handout (`coord_get_chain`) and billing settlement
  (`coord_complete`). Those are HTTP endpoints with their own tests
  (`coordinator/test_replica.py`, `coordinator/test_complete_auth.py`); dragging the FastAPI
  app, SQLite, wallets and holds in here would test those instead of the thing under test.
  The stub's only job is to answer "give me a chain" -- first with the node that is about to
  die, then with its replacement. Everything the recovery actually depends on -- the cached
  activations, the replay block, the fresh node's KV cache -- is real.

THE ASSERTION THAT MATTERS
  Not "the request completed". A request that completes with different text has not recovered,
  it has degraded silently, which is the failure mode RESILIENCE.md exists to prevent. So the
  test requires the killed run to be TOKEN-IDENTICAL to an uninterrupted baseline.

Run: python test_node_death.py            (~3-5 min: four model loads)
     python test_node_death.py --keep-going   (don't stop at the first failure)
"""

import argparse
import os
import socket
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import common  # noqa: E402
import node_a  # noqa: E402
import neuron_driver  # noqa: E402

S1, S2 = 10, 19
PROMPT = "Why is the sky blue"
MAX_NEW = 24
KILL_AFTER = 6          # kill the middle node once this many tokens have streamed
PORT_C, PORT_C2, PORT_B = 51301, 51302, 51303

ok = fail = 0
_keep_going = False


def check(name, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))
        if not _keep_going:
            raise SystemExit(finish())
    return cond


# --------------------------------------------------------------------------- #
# process helpers
# --------------------------------------------------------------------------- #
def spawn(script, port):
    """Start node_b.py / node_c.py as a real subprocess on `port`."""
    p = subprocess.Popen(
        [sys.executable, os.path.join(REPO, script), "--host", "127.0.0.1",
         "--port", str(port)],
        cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p


def wait_listening(port, timeout=180):
    """The node binds its socket before loading any weights, so this returns quickly; the
    model load happens lazily on the first `config`."""
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def hard_kill(proc):
    """Stop the process without letting it close its sockets politely. This is the power-cut
    case, not the graceful-shutdown case -- graceful shutdown is RESILIENCE.md Layer 1 and is
    a different test."""
    proc.kill()
    proc.wait(timeout=30)


# --------------------------------------------------------------------------- #
# coordinator stub
# --------------------------------------------------------------------------- #
class ChainStub:
    """Hands out chains in a fixed order and counts how many were asked for."""

    def __init__(self, chains):
        self.chains = list(chains)
        self.calls = 0

    def get_chain(self, *_a, **_kw):
        i = min(self.calls, len(self.chains) - 1)
        self.calls += 1
        host_c, port_c = self.chains[i]
        return (host_c, port_c, "127.0.0.1", PORT_B, S2,
                [f"node_c@{port_c}", f"node_b@{PORT_B}"],
                f"req-{self.calls}", "tok", 1.0)


def run_stream(driver, stub, kill_at=None, proc=None):
    """Consume one generation. If `kill_at` is set, SIGKILL `proc` after that many tokens."""
    node_a.coord_get_chain = stub.get_chain
    node_a.coord_complete = lambda *a, **k: None

    ids = driver.encode_chat([{"role": "user", "content": PROMPT}])
    text, events, killed = "", [], False
    for ev in driver.stream(ids, MAX_NEW, "http://stub", PROMPT, wallet_id="w"):
        events.append(ev)
        if ev["type"] == "token":
            text += ev["text"]
            if kill_at is not None and not killed and \
                    sum(1 for e in events if e["type"] == "token") >= kill_at:
                print(f"        ... killing middle node after {kill_at} tokens", flush=True)
                hard_kill(proc)
                killed = True
        elif ev["type"] == "error":
            print(f"        error event: {ev.get('detail')}")
    return text, events


def finish():
    print(f"\nRESULT: {ok} passed, {fail} failed")
    return 0 if fail == 0 else 1


# --------------------------------------------------------------------------- #
def main():
    global _keep_going
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()
    _keep_going = args.keep_going

    procs = []
    try:
        print("starting nodes (node_b, node_c, node_c-replacement) ...")
        pb = spawn("node_b.py", PORT_B)
        pc = spawn("node_c.py", PORT_C)
        pc2 = spawn("node_c.py", PORT_C2)
        procs = [pb, pc, pc2]
        for port in (PORT_B, PORT_C, PORT_C2):
            if not wait_listening(port):
                print(f"node on :{port} never listened")
                return 1
        print(f"  node_b :{PORT_B}   node_c :{PORT_C}   replacement :{PORT_C2}")

        print("loading driver shard (embed + layers 0..%d + head) ..." % (S1 - 1))
        driver = neuron_driver._Driver()
        driver.ensure_loaded()

        # ---- baseline: nobody dies -------------------------------------- #
        print("\n[1] baseline, uninterrupted")
        t0 = time.time()
        base_text, base_events = run_stream(
            driver, ChainStub([("127.0.0.1", PORT_C)]))
        base_done = [e for e in base_events if e["type"] == "done"]
        base_tokens = sum(1 for e in base_events if e["type"] == "token")
        print(f"        {base_tokens} tokens in {time.time() - t0:.1f}s: {base_text!r}")
        check("baseline completes", bool(base_done),
              "no done event -- the harness is wrong, not the recovery")
        check("baseline produced tokens", base_tokens > KILL_AFTER,
              f"only {base_tokens} tokens, need more than KILL_AFTER={KILL_AFTER}")

        # ---- kill run: node_c dies mid-generation ----------------------- #
        print(f"\n[2] middle node SIGKILLed after {KILL_AFTER} tokens")
        t0 = time.time()
        stub = ChainStub([("127.0.0.1", PORT_C), ("127.0.0.1", PORT_C2)])
        kill_text, kill_events = run_stream(driver, stub, kill_at=KILL_AFTER, proc=pc)
        kill_done = [e for e in kill_events if e["type"] == "done"]
        kill_err = [e for e in kill_events if e["type"] == "error"]
        kill_tokens = sum(1 for e in kill_events if e["type"] == "token")
        print(f"        {kill_tokens} tokens in {time.time() - t0:.1f}s: {kill_text!r}")

        check("node_c process is actually dead", pc.poll() is not None,
              "the kill did not take -- the test proves nothing")
        check("request survived the death", bool(kill_done) and not kill_err,
              f"errors={[e.get('detail') for e in kill_err]}")
        check("a reroute actually happened", stub.calls >= 2,
              f"coordinator asked for {stub.calls} chain(s); recovery never engaged, so the "
              f"run may have finished before the kill landed")
        check("recovery is EXACT, not approximate", kill_text == base_text,
              f"baseline : {base_text!r}\n        after kill: {kill_text!r}")
        if kill_done:
            check("reroute is reported to the caller",
                  kill_done[0].get("reroutes"),
                  f"done event carries no reroute record: {kill_done[0]}")

        return finish()
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.kill()
                    p.wait(timeout=10)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
