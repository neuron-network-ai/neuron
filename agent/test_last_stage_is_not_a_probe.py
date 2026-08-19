"""agent/test_last_stage_is_not_a_probe.py — run: python -m agent.test_last_stage_is_not_a_probe

**[P55]: the network returned garbage and billed for it, for a day, while every component was
correct.** The last node in a real chain must run `layers[s2:]` AND the final norm; a verifier
probing a range must run `layers[s1:s2]` and NO norm. `agent/node_server.py` has to tell those
two apart from the `config` message alone, because `is_true_last` asks what the node HOLDS and
a node holding the model's final layer is "true last" for every question it is ever asked.

It has now been wrong in both directions, and neither failure looked like a failure:

  * [P49] — a full-model node answered a verifier's probe about layers 0-9 by running
    `layers[10:]` + norm. Confident, deterministic, wrong by max_err 28.6.
  * [P55] — the fix for that read the mere PRESENCE of `s1` as "a verifier is probing me".
    `neuron_driver._connect` sends `s1` on every config, and a TWO-stage chain (driver 0-9,
    one node 10-27 — what the live network actually runs) sends no `host_b`. So every real
    request landed in the probe branch, came back with no final norm, and the driver applied
    `lm_head` to an un-normed hidden state. The user got `'  1  2   3'` and whitespace to the
    128-token cap and paid ~0.13 NRN for it.

**Proof-of-compute could not see it and never will**: the probe path is the only path it
exercises, so the node it certifies healthy is answering a different question from the one the
network asks. That is why this file drives `serve()` with the message the DRIVER really sends,
rather than the one the verifier sends.

No model is loaded; the batcher is stubbed and records which role was chosen.
"""
import re
import socket
import sys
import threading
import types

import torch

import batching
import common
from agent import node_server

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class _FakeBatcher:
    def __init__(self, hidden_size=64):
        self.h = hidden_size

    def submit(self, hidden, cache, past):
        return torch.zeros(hidden.shape[0], hidden.shape[1], self.h)


class _Server(node_server.NodeServer):
    """A NodeServer with no weights. `_batcher` records the role/range serve() picked, which
    is the whole observable difference between answering the question that was asked and
    answering a different one."""

    def __init__(self, lo, hi, n):
        self.lo, self.hi, self.n = lo, hi, n
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=64))
        self._batchers = {}
        self._batcher_lock = threading.Lock()
        self.paused = threading.Event()
        self.listening = threading.Event()
        self.bind_error = None
        self.picked = []

    def _batcher(self, role, lo, hi):
        self.picked.append((role, lo, hi))
        return _FakeBatcher()


def serve_on(server):
    a, b = socket.socketpair()

    def run():
        try:
            server.serve(b, ("test", 0))
        except (ConnectionError, TimeoutError, EOFError, OSError):
            pass
        finally:
            b.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return a, t


def role_for(server, cfg, hidden_size=64):
    """Send `cfg`, then one `act`, and report (ack, the (role, lo, hi) serve() chose)."""
    cli, t = serve_on(server)
    try:
        common.send_msg(cli, cfg)
        ack = common.recv_msg(cli)
        if not ack.get("ok"):
            return ack, None
        common.send_msg(cli, {"type": "act", "hidden": torch.zeros(1, 3, hidden_size)})
        common.recv_msg(cli)
        return ack, (server.picked[-1] if server.picked else None)
    finally:
        cli.close()
        t.join(timeout=5)


# The messages that actually reach a `config` on a node with no `host_b`, verbatim from their
# senders. If one of these drifts, this file is the thing that should break.
def driver_two_stage(s2=10):
    """neuron_driver._connect, chain [[0,9],[10,27]] — the live network's own shape.

    No `s1`: a last stage has never read it (its own start is implied by `s2`), and not sending
    a field the recipient cannot use is what repairs an ALREADY-INSTALLED node with no update.
    """
    return {"type": "config", "s2": s2, "stage": "last", "wire": ["f32"]}


def driver_two_stage_pre_0_20_9(s1=10, s2=10):
    """The same hop from a driver still in the field, which sends `s1` and no `stage`. This is
    the message that produced [P55], and the node has to serve it correctly on its own."""
    return {"type": "config", "s1": s1, "s2": s2, "wire": ["f32"]}


def middle_relay(s2=10, n=28):
    """node_server's own MIDDLE arm, forwarding to the stage after it."""
    return {"type": "config", "s2": s2, "n": n, "wire": ["f32"]}


def verifier_last(s2=10, n=28):
    """proof_of_compute.challenge_node."""
    return {"type": "config", "s2": s2, "n": n}


def verifier_probe(s1, s2, explicit=True):
    """proof_of_compute.challenge_middle_node."""
    m = {"type": "config", "s1": s1, "s2": s2}
    if explicit:
        m["probe"] = True
    return m


def main():
    # ---- 1. the discriminator, read off each caller's real message ----------- #
    P = node_server._is_range_probe
    check("the driver's two-stage config is NOT a probe (this is [P55])",
          P(driver_two_stage()) is False, driver_two_stage())
    check("...and is still not one from a driver too old to send `stage`",
          P(driver_two_stage_pre_0_20_9()) is False)
    check("a middle relay's config is not a probe", P(middle_relay()) is False)
    check("challenge_node's config is not a probe", P(verifier_last()) is False)
    check("challenge_middle_node's config IS a probe (this is [P49])",
          P(verifier_probe(0, 10)) is True)
    check("...and is still one from a verifier too old to send `probe`",
          P(verifier_probe(0, 10, explicit=False)) is True)
    check("a junction is a junction at any depth (s1 == s2 == 19)",
          P({"s1": 19, "s2": 19}) is False)

    # ---- 2. the same messages through serve(), on the live topology ---------- #
    # The Pavilion: holds 10-27 of 28, so it IS the true last stage.
    ack, picked = role_for(_Server(10, 27, 28), driver_two_stage())
    check("driver -> a 10-27 node runs the LAST stage, norm included",
          picked == ("last", 10, 28), f"ack={ack} picked={picked}")
    ack, picked = role_for(_Server(10, 27, 28), driver_two_stage_pre_0_20_9())
    check("...same for an installed driver that predates `stage`",
          picked == ("last", 10, 28), f"picked={picked}")
    ack, picked = role_for(_Server(10, 27, 28), middle_relay())
    check("a middle relay's onward hop runs the LAST stage",
          picked == ("last", 10, 28), f"picked={picked}")
    ack, picked = role_for(_Server(10, 27, 28), verifier_last())
    check("challenge_node reaches the LAST stage, which is what it scores",
          picked == ("last", 10, 28), f"picked={picked}")

    # A full-model node: `is_true_last` is true for every question, which is [P49]'s trap.
    ack, picked = role_for(_Server(0, 27, 28), verifier_probe(0, 10))
    check("a probe of a full-model node still runs the PROBE role, no norm ([P49])",
          picked == ("probe", 0, 28), f"picked={picked}")
    check("...and the ack reports what it HOLDS, so the verifier can see the mismatch",
          ack.get("holds") == [0, 27], ack)
    ack, picked = role_for(_Server(0, 27, 28), driver_two_stage(s2=10))
    check("but real traffic to that same full-model node runs the LAST stage",
          picked == ("last", 10, 28), f"picked={picked}")

    # ---- 2b. an un-upgraded node serves the new driver correctly ------------ #
    # A node still running 0.20.8 chooses the last stage with `is_true_last and "s1" not in
    # msg` -- there is no `_is_range_probe` on it to help. So the ONLY thing that repairs the
    # live network without waiting on an installer release is the driver not sending `s1` to a
    # hop that never reads it. This asserts against that literal rule, because the machine it
    # describes cannot be imported.
    def pre_0_20_9_picks_last(msg, is_true_last=True):
        return is_true_last and "s1" not in msg

    check("a 0.20.8 node serves the NEW driver's two-stage config as the last stage",
          pre_0_20_9_picks_last(driver_two_stage()) is True, driver_two_stage())
    check("...which is exactly what it did NOT do before ([P55] reproduced)",
          pre_0_20_9_picks_last(driver_two_stage_pre_0_20_9()) is False)
    check("a 0.20.8 node still reads a verifier probe as a probe",
          pre_0_20_9_picks_last(verifier_probe(0, 10)) is False)

    # A node whose range does not reach the final layer can only ever be probed.
    ack, picked = role_for(_Server(10, 18, 28), {"type": "config", "s1": 10, "s2": 19})
    check("a mid-range node is probed, never asked for a last stage",
          picked == ("probe", 10, 19), f"picked={picked}")

    # ---- 3. the two roles are not cosmetically different -------------------- #
    # If they were the same computation, picking wrong would not matter. They are not: the
    # last stage normalises and the probe does not, which is exactly the 247.09 in [P55].
    marker = torch.full((1, 1, 8), 3.0)
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(layers=[], norm=lambda h: h * 0.0 - 1.0))
    last = batching.last_stage_batched(model, 0, marker.clone(),
                                       batching.BatchedCache(lengths=[0]), [0])
    mid = batching.mid_stage_batched(model, 0, 0, marker.clone(),
                                     batching.BatchedCache(lengths=[0]), [0])
    check("last_stage_batched applies the final norm",
          torch.equal(last, torch.full((1, 1, 8), -1.0)), f"{last}")
    check("mid_stage_batched (the probe role) does NOT", torch.equal(mid, marker), f"{mid}")
    check("so answering with the wrong one returns a different tensor, not a rounding error",
          not torch.equal(last, mid))

    # ---- 4. the driver still sends the shape this file pins ----------------- #
    src = open("neuron_driver.py", encoding="utf-8").read()
    cfg_src = src[src.index('cfg = {"type": "config"'):]
    cfg_src = cfg_src[:cfg_src.index("common.send_msg")]
    # the dict LITERAL alone -- what every hop gets, before the host_b branch adds to it
    literal = cfg_src[:cfg_src.index('if chain["host_b"]:')]
    check("neuron_driver does NOT put `s1` in the config it builds for every hop",
          '"s1"' not in literal, literal)
    check("...it adds `s1` only inside the host_b branch, where a middle relay needs it",
          'cfg["s1"] = self.s1' in src and
          src.index('if chain["host_b"]:') < src.index('cfg["s1"] = self.s1'))
    check("neuron_driver names the stage outright",
          '"stage"' in cfg_src and '"last"' in cfg_src, cfg_src)
    check("...and calls the hop `middle` when it relays through one",
          '"middle" if chain["host_b"]' in cfg_src, cfg_src)
    # node_a.py is the driver neuron_driver superseded, and it still builds configs. It had
    # the same bug for the same reason, so it has to move with the rule, or the older path
    # resurrects [P55] the next time somebody runs it.
    na = open("node_a.py", encoding="utf-8").read()
    check("the legacy node_a driver sends `s1` only alongside host_b",
          na.count('cfg["s1"] = s1') == 2 and '"s1": s1, "s2": s2' not in na, )
    check("...and names the stage the same way neuron_driver does",
          na.count('"stage": "middle" if host_b else "last"') == 2)

    poc = open("security/proof_of_compute.py", encoding="utf-8").read()
    check("challenge_middle_node marks itself a probe",
          re.search(r'"type": "config", "s1": s1, "s2": s2, "probe": True', poc) is not None)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
