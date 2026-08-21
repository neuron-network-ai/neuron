"""
security/proof_of_compute.py — verify a node actually did the work  [Session 16]

A lazy or malicious node could return garbage to farm NRN without computing. Proof-of-
compute catches that: a verifier sends a node a challenge (a known input for its layer
range), the node runs its slice, and the verifier checks the output against the locally
computed expected result. Wrong output ⇒ the node failed ⇒ the coordinator drops its
reputation and withholds NRN (see coordinator reputation, Session 16).

The verifier needs torch (it computes the expected output); the coordinator only
aggregates pass/fail into reputation, so it stays torch-free. This challenges a
LAST-stage node (layers[s2:n] + final norm — node_b's wire protocol). Challenging a
middle node needs a no-relay mode on node_c (extension).

    python -m security.proof_of_compute --host <ip> --port 50999 --s2 19 --n 28
"""
import argparse
import json
import socket
import time

import requests
import torch

import common


def make_challenge(s2, n, seed=0):
    """Deterministic (input, expected) for a last-stage node holding layers[s2:n] + norm."""
    _tok, model, _N = common.load_model_shard(s2, n, norm=True)
    H = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, H, generator=g, dtype=common.DTYPE)
    expected = common.last_stage(model, s2, inp, common.new_cache(), 0)
    return inp, expected


def make_middle_challenge(s1, s2, seed=0):
    """Deterministic (input, expected) for a MIDDLE node holding layers[s1:s2] only (no norm,
    no head) -- the isolated no-relay probe (node_c.py / agent/node_server.py's "probe" role)."""
    _tok, model, _N = common.load_model_shard(s1, s2)
    H = model.config.hidden_size
    g = torch.Generator().manual_seed(seed)
    inp = torch.randn(1, 1, H, generator=g, dtype=common.DTYPE)
    expected = common.mid_stage(model, s1, s2, inp, common.new_cache(), 0)
    return inp, expected


def verify(output, expected, atol=0.05):
    """(passed, max_abs_err). The tolerance absorbs any cross-hardware fp jitter while
    still rejecting garbage, which is off by many orders of magnitude.

    DO NOT let the challenge path negotiate a lossy wire codec. The challenge functions
    below deliberately send NO "wire" field, so the node answers in the legacy (lossless)
    format -- see wire_codec.negotiate. A quantized reply would spend the atol budget on
    transport noise instead of on hardware jitter: on a randn(1,1,H) challenge, i8h's ~0.3%
    relative error lands right inside 0.05, so a cheating node would gain cover rather than
    the check failing loudly. Bandwidth is irrelevant here anyway -- one tensor per attest."""
    if not torch.is_tensor(output) or output.shape != expected.shape:
        return False, float("inf")
    err = (output.float() - expected.float()).abs().max().item()
    return err <= atol, err


class ChallengeRefused(RuntimeError):
    """The node declined the challenge and said why (paused, reloading, range_mismatch).

    Distinct from an exception raised while computing: a node that answers "no, and here is
    the reason" has told the truth about itself, which is the opposite of a proof-of-compute
    failure. The caller decides what each reason is worth."""


class RangeMismatch(ChallengeRefused):
    """The node does not hold the layers it was challenged on.

    This is a fact about PLACEMENT, not about the node, and conflating the two is what
    flagged three honest machines on 2026-08-10. A node challenged for layers it never
    downloaded either answers for a different range (a wrong-looking answer) or dies running
    uninitialized weights (a closed socket) -- and both were recorded against its reputation.
    Raised as its own type so the verifier can refuse to attest it either way."""


# A challenge blocks behind the peer's one-time model-shard load, which is legitimately slow:
# `common.COLD_CONNECT_TIMEOUT_S` is 120 for exactly this reason, while this path used a flat
# 30 -- less than the ~35s cold load common.py documents. A real PASS was measured at 14,725 ms
# on an office PC (verify_service.log, 2026-08-07), so the old budget had barely 2x of headroom
# over a node that was working correctly. Being slow is not being wrong ([P28]).
CHALLENGE_TIMEOUT_S = common.COLD_CONNECT_TIMEOUT_S


def _check_range(ack, lo, hi, requested):
    """Raise if the node cannot actually serve the range we challenged it on.

    COVERAGE, not equality. A node holding MORE than it was asked for answers correctly:
    `last_stage(model, lo)` is `layers[lo:]` on a full skeleton, so the extra layers cost
    resident RAM and change no arithmetic -- and `ensure_slice` deliberately reuses a superset
    rather than re-downloading it ([P36]). Demanding equality here would refuse to verify every
    such node, which is a promotion that never happens: the harm this function exists to
    prevent, reintroduced by the function preventing it.

    `holds` is authoritative and reports the node's real [lo, hi]. Agents too old to send it
    still betray a disagreement on the probe path, where `s1`/`s2` have always been the node's
    own range -- and an agent that answers a LAST-stage config from the probe branch is itself
    the symptom, because it means its range does not reach the model's final layer."""
    held = ack.get("holds")
    if held is None and ack.get("s1") is not None:
        held = [ack["s1"], ack["s2"] - 1]
    if held is None:
        return                                   # pre-0.20 agent on the last-stage path
    held_lo, held_hi = int(held[0]), int(held[1])
    if held_lo > lo or held_hi < hi:
        raise RangeMismatch(
            f"challenged on layers {lo}-{hi} ({requested}) but the node holds "
            f"{held_lo}-{held_hi} -- placement disagreement, not a bad answer")


def challenge_node(host, port, s2, n, inp, timeout=CHALLENGE_TIMEOUT_S):
    """Speak the last-stage wire protocol: config -> act(challenge) -> read output."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        # THE DRIVER'S OWN MESSAGE, from the driver's own constructor. This used to be a dict
        # written out here, which meant the challenge and the request were two shapes that
        # merely happened to agree -- and [P55] is what happened when they stopped. The node
        # reached the last-stage branch for this message only because `s1` was absent, an
        # inference rather than a statement; `stage: "last"` now says it, exactly as a real
        # request does. `hidden_size` is deliberately not passed, so no `wire` field is sent
        # and the reply comes back losslessly -- see `verify`. [P56].
        common.send_msg(s, common.stage_config(s2, stage="last", n=n))
        ack = common.recv_msg(s)
        if not ack.get("ok"):
            # A NAMED refusal, surfaced as such. `range_mismatch` in particular must never be
            # scored as a failed challenge: the node is telling us the coordinator's placement
            # is stale, which is a fault in the network's bookkeeping, not in the machine.
            err = ack.get("error")
            detail = ack.get("detail") or ack
            if err == "range_mismatch":
                raise RangeMismatch(f"node refused config: {detail}")
            raise ChallengeRefused(f"node refused config ({err}): {detail}")
        # The middle-node challenge has always verified this and the last-stage one never did,
        # so a node answering for the WRONG RANGE passed silently into `verify()` and came back
        # as a plain wrong answer -- max_err in the tens, deterministic, and indistinguishable
        # from a node running corrupted weights. Checked here so the two stop looking alike.
        _check_range(ack, s2, n - 1, f"s2={s2}, n={n}")
        common.send_msg(s, {"type": "act", "hidden": inp})
        resp = common.recv_msg(s)
        common.send_msg(s, {"type": "bye"})
        return resp["hidden"]
    finally:
        s.close()


def challenge_middle_node(host, port, s1, s2, inp, timeout=CHALLENGE_TIMEOUT_S):
    """Speak the PROBE wire protocol: a config with NO host_b tells a middle-shard node
    (node_c.py / agent/node_server.py) to run its own layers in isolation and return the raw
    result, instead of relaying to a next hop. The node answers with ITS OWN actual s1/s2
    (self.lo/self.hi) in the ack -- checked here so a node lying about its own range fails
    loudly instead of silently passing a challenge for a range it doesn't really hold."""
    s = socket.create_connection((host, port), timeout=timeout)
    try:
        # `probe: True` says outright what `s1`/`s2` only implied. A node below 0.20.9 ignores
        # the key and reads the range instead (s1 < s2), which means the same thing -- see
        # node_server._is_range_probe and [P55], where the inference went the other way and
        # sent real pipeline traffic down this path.
        common.send_msg(s, {"type": "config", "s1": s1, "s2": s2, "probe": True})
        ack = common.recv_msg(s)
        if not ack.get("ok"):
            err = ack.get("error")
            detail = ack.get("detail") or ack
            if err == "range_mismatch":
                raise RangeMismatch(f"node refused config: {detail}")
            raise ChallengeRefused(f"node refused config ({err}): {detail}")
        # COMPARE ONLY WHAT THE NODE ACTUALLY CLAIMS. Equality (not coverage) is right here --
        # the probe makes the node run ITS OWN lo/hi in isolation, so a node holding more would
        # compute more layers and the arithmetic genuinely would not match. But a field the node
        # never sent is not a disagreement, it is silence, and `_check_range` already says so for
        # the other path ("agents too old to send it").
        #
        # Live 2026-08-11: the DRIVER (`agent-optinovate-6ff49d`, 26/26, the healthiest node on
        # the network) acks `s2` correctly and omits `s1`, so the whole-tuple compare read
        # `(None, 10) != (0, 10)` and raised PLACEMENT MISMATCH on every single sweep. It failed
        # into the safe "nothing recorded" branch, so nothing looked wrong -- and proof-of-compute
        # silently stopped checking the most important machine in the chain until 0.20 ships.
        # That is exactly [P35]'s disease returning through its own cure: running perfectly,
        # checking nothing. A check that CANNOT pass is not a strict check, it is a dead one.
        # CHECK `holds` TOO, because s1/s2 alone cannot see the case that has been costing the
        # driver every hour it has ever worked. Live 2026-08-17, `agent-optinovate-6ff49d`:
        # assigned 0-9, actually serving 0-27 (it is a 64 GB machine that loaded the whole
        # model), so `node_server`'s `is_true_last` is TRUE and a probe config falls into the
        # LAST-stage branch. That branch runs `layers[s2:]` + the final norm -- layers 10-27 for
        # a challenge asking about 0-9 -- and its ack omits `s1` entirely. So `a1` is None, which
        # the rule below correctly reads as silence rather than disagreement, and `a2` is the
        # caller's own `s2` echoed back, which can never disagree. Both checks pass and the node
        # returns a confident, deterministic, completely unrelated answer: max_err 28.5958, the
        # same figure as 2026-08-11, which [P47] recorded as unexplained.
        #
        # `holds` was in that ack the whole time. This is [P37] exactly -- "the ack that
        # explained it was on the wire from the beginning and `challenge_node` threw it away" --
        # and the fix that closed it there was never applied to this function. Checked FIRST, so
        # a node that reports its real range gets the accurate diagnosis rather than the weaker
        # s1/s2 one, and works against today's agents with no release.
        held = ack.get("holds")
        if isinstance(held, (list, tuple)) and len(held) == 2 and None not in held:
            if [int(held[0]), int(held[1])] != [s1, s2 - 1]:
                raise RangeMismatch(
                    f"node holds layers {int(held[0])}-{int(held[1])} but was challenged on "
                    f"{s1}-{s2 - 1} -- registration/slice mismatch. It answered, and it told "
                    f"the truth about what it has; the placement is what is stale")
        a1, a2 = ack.get("s1"), ack.get("s2")
        if (a1 is not None and a1 != s1) or (a2 is not None and a2 != s2):
            # Was a bare RuntimeError, so the verifier's blanket `except Exception` counted it
            # towards UNREACHABLE_STRIKES and eventually attested a failure -- for a node that
            # answered correctly and honestly about a range the coordinator had moved.
            raise RangeMismatch(f"node's actual range {a1, a2} does not "
                                f"match the expected {s1, s2} -- registration/slice mismatch")
        common.send_msg(s, {"type": "act", "hidden": inp})
        resp = common.recv_msg(s)
        common.send_msg(s, {"type": "bye"})
        return resp["hidden"]
    finally:
        s.close()


class _RelaySink:
    """A one-connection stand-in for the NEXT hop, so a middle node can be challenged in the
    role it actually serves.

    **Why this exists.** `challenge_middle_node` sends `probe: True`, which puts the node in a
    role NO user request ever produces: it runs `layers[s1:s2]` in isolation and answers the
    caller directly. Its production role is different code -- it dials a next hop, forwards
    what it computed, and relays the reply back. So the middle role has never been verified at
    all, only a role adjacent to it. That is [P56] in its sharpest form.

    A relay needs somewhere to relay TO. This is that somewhere: it accepts the node's onward
    connection, acks its config, and **captures the hidden state the node forwards** -- which
    is precisely the middle node's own output, computed through the same
    `_batcher("middle", s1, s2)` that serves users.

    The ack carries NO `wire` field on purpose. `node_server` negotiates the onward codec from
    what comes back here, so acking without one keeps the forwarded tensor in the lossless
    legacy framing -- the same reason `verify` keeps the challenge itself lossless, and for the
    same failure it would otherwise open.
    """

    def __init__(self, bind_host="0.0.0.0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((bind_host, 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.forwarded = None
        self.error = None

    def serve_once(self, timeout):
        """Accept one relay, answer it, and keep what it forwarded. Runs on its own thread —
        the node does not dial until it has taken our config, and it does not answer us until
        this has answered it, so doing it inline would deadlock."""
        try:
            self.sock.settimeout(timeout)
            try:
                conn, _addr = self.sock.accept()
            except (socket.timeout, TimeoutError):
                # NOT an error to re-raise. Nobody dialled: either the node cannot reach this
                # verifier (NAT, no route — a fact about the network between us) or it took the
                # config and relayed nothing. Both leave `forwarded` None, which
                # `challenge_relay_node` turns into a NAMED ChallengeRefused, so the caller can
                # fall back to the probe AND record why. A bare TimeoutError here would reach
                # the verifier's blanket `except Exception` and be scored against the node —
                # [P28]'s rule, reporting one event as another.
                return
            try:
                conn.settimeout(timeout)
                cfg = common.recv_msg(conn)
                if cfg.get("type") != "config":
                    raise ChallengeRefused(f"relay opened with {cfg.get('type')!r}, not config")
                common.send_msg(conn, {"ok": True, "layers": cfg.get("n")})
                act = common.recv_msg(conn)
                if act.get("type") != "act":
                    raise ChallengeRefused(f"relay sent {act.get('type')!r}, not act")
                self.forwarded = act["hidden"]
                # Hand back something the right shape so the node completes its exchange and
                # answers the challenger normally. Its content is never read: what is under
                # test is what the node SENT here, not what it does with a reply.
                common.send_msg(conn, {"hidden": self.forwarded, "b_compute_ms": 0.0})
            finally:
                conn.close()
        except Exception as e:                       # kept, not raised — this is a worker thread
            self.error = e

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def challenge_relay_node(host, port, s1, s2, inp, sink_host, timeout=CHALLENGE_TIMEOUT_S):
    """Challenge a MIDDLE node through the role it really serves: as a relay.

    `sink_host` is the address THIS verifier is reachable at from the node. There is no way to
    infer it -- the node may be across a relay, a tailnet or a LAN -- so it is required rather
    than guessed, and `attest_via_coordinator` falls back to the probe when the node cannot
    dial back. A fallback that is not recorded would be worse than no fallback at all, so the
    caller is told which path ran ([P56]: an attestation must not claim more than it checked).
    """
    import threading
    sink = _RelaySink()
    worker = threading.Thread(target=sink.serve_once, args=(timeout,), daemon=True)
    worker.start()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        try:
            # The DRIVER's middle-hop message, from the driver's own constructor — the whole
            # point. No `wire`, so the reply to us stays lossless; no `grant_b`, so the node
            # does not attempt a [P52] handshake with a sink that holds no channel.
            common.send_msg(s, common.stage_config(s2, stage="middle", s1=s1,
                                                   host_b=sink_host, port_b=sink.port))
            ack = common.recv_msg(s)
            if not ack.get("ok"):
                err = ack.get("error")
                detail = ack.get("detail") or ack
                if err == "range_mismatch":
                    raise RangeMismatch(f"node refused config: {detail}")
                raise ChallengeRefused(f"node refused config ({err}): {detail}")
            _check_range(ack, s1, s2 - 1, f"s1={s1}, s2={s2}")
            common.send_msg(s, {"type": "act", "hidden": inp})
            common.recv_msg(s)              # the relayed reply; the sink holds what we grade
            common.send_msg(s, {"type": "bye"})
        finally:
            s.close()
    finally:
        worker.join(timeout=timeout)
        sink.close()
    if sink.error is not None:
        raise sink.error
    if sink.forwarded is None:
        raise ChallengeRefused(
            f"the node took a middle-hop config but relayed nothing to {sink_host}:{sink.port} "
            f"within {timeout}s — it either cannot reach this verifier or did not forward")
    return sink.forwarded


def attest(host, port, s2, n, seed=0, atol=0.05):
    """Full challenge: make -> send -> verify. Returns a result dict."""
    inp, expected = make_challenge(s2, n, seed)
    t0 = time.time()
    output = challenge_node(host, port, s2, n, inp)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6), "path": "last",
            "ms": int((time.time() - t0) * 1000), "layers": [s2, n - 1]}


def attest_relay(host, port, s1, s2, sink_host, seed=0, atol=0.05):
    """Full MIDDLE-node challenge through the RELAY path — the one users are served by.

    The expected value is the same `common.mid_stage` the probe compares against, because the
    layers a relay runs and the layers a probe runs are the same layers. What differs is the
    code that decides to run them, and that is the half [P55] proved can be wrong on its own.
    """
    inp, expected = make_middle_challenge(s1, s2, seed)
    t0 = time.time()
    output = challenge_relay_node(host, port, s1, s2, inp, sink_host)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6), "path": "relay",
            "ms": int((time.time() - t0) * 1000), "layers": [s1, s2 - 1]}


def attest_middle(host, port, s1, s2, seed=0, atol=0.05):
    """Full MIDDLE-node challenge: make -> send -> verify. Returns a result dict."""
    inp, expected = make_middle_challenge(s1, s2, seed)
    t0 = time.time()
    output = challenge_middle_node(host, port, s1, s2, inp)
    passed, err = verify(output, expected, atol)
    return {"passed": passed, "max_err": round(err, 6), "path": "probe",
            "ms": int((time.time() - t0) * 1000), "layers": [s1, s2 - 1]}


def attest_via_coordinator(coordinator, node_id, register_secret, n=None,
                           seed=0, atol=0.05, sink_host=None):
    """Verify a node the coordinator knows about and record the result (Session 12 — open
    join). Looks the node up in /node/list, challenges it (last-stage nodes get the full
    challenge incl. norm; any other node gets the middle no-relay probe), then POSTs the
    pass/fail to /node/{id}/attest so a passing probationary node becomes eligible to serve
    and earn. Returns {'challenge', 'attestation'}."""
    coordinator = coordinator.rstrip("/")
    # node addresses are operator-private -> authenticate to see them (S25 privacy)
    nodes = requests.get(f"{coordinator}/node/list", timeout=10,
                         headers={"X-Register-Secret": register_secret}).json()["nodes"]
    node = next((x for x in nodes if x["node_id"] == node_id), None)
    if node is None:
        raise SystemExit(f"node '{node_id}' not found at {coordinator}")
    if "tailscale_ip" not in node:
        raise SystemExit("coordinator did not return node addresses — wrong register secret?")
    total = n or node.get("total_layers") or 28
    is_last = node["layer_end"] == total - 1
    if is_last:
        res = attest(node["tailscale_ip"], node["port"], node["layer_start"], total, seed, atol)
    else:
        # THE RELAY PATH FIRST, because it is the one a user's request takes. The probe is
        # kept only as a fallback for a node that cannot dial this verifier back -- and the
        # result says WHICH ran, because a probe pass is a weaker claim than a relay pass and
        # reporting them as the same number is the whole of [P56]. `sink_host` is required and
        # never guessed: only the operator knows the address this verifier is reachable at
        # from the node, so with none configured the probe is all that is honestly available.
        res, fallback_reason = None, None
        if sink_host:
            try:
                res = attest_relay(node["tailscale_ip"], node["port"], node["layer_start"],
                                   node["layer_end"] + 1, sink_host, seed, atol)
            except RangeMismatch:
                raise                     # placement, not compute — must not fall back to a
                                          # second challenge that would report it as a failure
            except (ChallengeRefused, OSError) as e:
                # Unreachable-for-relay is a fact about the network between us, not about the
                # node's arithmetic. Fall back, and say so.
                res, fallback_reason = None, f"{type(e).__name__}: {e}"
        if res is None:
            res = attest_middle(node["tailscale_ip"], node["port"],
                                node["layer_start"], node["layer_end"] + 1, seed, atol)
            res["relay_fallback"] = fallback_reason or "no sink_host configured"
    r = requests.post(f"{coordinator}/node/{node_id}/attest",
                      json={"passed": res["passed"], "max_err": res["max_err"]},
                      headers={"X-Register-Secret": register_secret}, timeout=10)
    r.raise_for_status()
    return {"challenge": res, "attestation": r.json()}


def verify_loop(coordinator, register_secret, interval=60, seed=0, atol=0.05, n=None,
                sink_host=None):
    """Continuously find probationary nodes and verify them -- no more running the CLI by
    hand for every new arrival. Skips already-flagged nodes (the coordinator already excludes
    them from routing after repeated failures, per Session 16 reputation; re-challenging a
    hopeless node forever would just waste cycles). A single node's failure (network hiccup,
    genuinely bad node, wrong secret) is logged and the loop moves on -- one bad node must
    never stop the whole verifier."""
    coordinator = coordinator.rstrip("/")
    print(f"auto-verify: checking {coordinator} for probationary nodes every {interval}s "
         f"(Ctrl-C to stop)")
    while True:
        try:
            nodes = requests.get(f"{coordinator}/node/list", timeout=10,
                                 headers={"X-Register-Secret": register_secret}).json()["nodes"]
            pending = [nd for nd in nodes
                      if nd.get("standing") == "probationary" and not nd.get("flagged")]
            if not pending:
                print(f"[{time.strftime('%H:%M:%S')}] nothing pending")
            for nd in pending:
                try:
                    out = attest_via_coordinator(coordinator, nd["node_id"], register_secret,
                                                 n=n, seed=seed, atol=atol,
                                                 sink_host=sink_host)
                    ch = out["challenge"]
                    print(f"[{time.strftime('%H:%M:%S')}] {nd['node_id']} "
                         f"({'PASSED' if ch['passed'] else 'FAILED'}, "
                         f"max_err={ch['max_err']}, layers={ch['layers']}, "
                         f"path={ch.get('path')})"
                         + (f"  relay fallback: {ch['relay_fallback']}"
                            if ch.get("relay_fallback") else ""))
                except Exception as e:
                    print(f"[{time.strftime('%H:%M:%S')}] could not verify "
                         f"{nd['node_id']}: {e}")
        except requests.RequestException as e:
            print(f"auto-verify: coordinator unreachable: {e}")
        time.sleep(interval)


def main():
    import os

    ap = argparse.ArgumentParser(
        description="Proof-of-compute verifier. Direct mode (--host + --s2, or --host + "
                    "--s1/--s2 for a middle node) challenges one node and prints the result; "
                    "coordinator mode (--coordinator/--node-id) also records the attestation "
                    "so a probationary node gets promoted; --auto runs continuously and "
                    "verifies every probationary node it finds, with no per-node command.")
    # coordinator mode (Session 12)
    ap.add_argument("--coordinator", help="coordinator URL; with --node-id, verify + attest")
    ap.add_argument("--node-id", help="node to verify (coordinator mode)")
    ap.add_argument("--register-secret", default=os.environ.get("NEURON_REGISTER_SECRET"),
                    help="secret to post the attestation (defaults to $NEURON_REGISTER_SECRET)")
    ap.add_argument("--auto", action="store_true",
                    help="with --coordinator (no --node-id): loop forever, auto-verifying "
                         "every probationary node found, instead of a one-shot check")
    ap.add_argument("--interval", type=int, default=60, help="--auto poll interval, seconds")
    ap.add_argument("--sink-host", default=os.environ.get("NEURON_VERIFY_SINK_HOST"),
                    help="the address THIS verifier is reachable at FROM a node. Given it, a "
                         "middle node is challenged as a RELAY -- the role a user's request "
                         "actually puts it in -- with the probe kept only as a recorded "
                         "fallback. Without it only the probe is available, which verifies a "
                         "path nobody is served by ([P56]). Defaults to "
                         "$NEURON_VERIFY_SINK_HOST.")
    # direct mode
    ap.add_argument("--host")
    ap.add_argument("--port", type=int, default=50999)
    ap.add_argument("--s1", type=int, help="middle node: holds layers[s1:s2) (no norm/head)")
    ap.add_argument("--s2", type=int, help="last node: holds layers[s2:n]+norm; "
                                          "middle node: end of its range (with --s1)")
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--atol", type=float, default=0.05)
    args = ap.parse_args()

    if args.coordinator and args.auto:
        if not args.register_secret:
            ap.error("--register-secret (or $NEURON_REGISTER_SECRET) is required for --auto")
        verify_loop(args.coordinator, args.register_secret, interval=args.interval,
                   seed=args.seed, atol=args.atol, sink_host=args.sink_host)
    elif args.coordinator:
        if not args.node_id:
            ap.error("--coordinator requires --node-id (or --auto to verify all pending)")
        if not args.register_secret:
            ap.error("--register-secret (or $NEURON_REGISTER_SECRET) is required to attest")
        out = attest_via_coordinator(args.coordinator, args.node_id, args.register_secret,
                                     n=args.n, seed=args.seed, atol=args.atol,
                                     sink_host=args.sink_host)
        print(json.dumps(out, indent=2))
    elif args.host is not None and args.s1 is not None and args.s2 is not None:
        if args.sink_host:
            print(json.dumps(attest_relay(args.host, args.port, args.s1, args.s2,
                                          args.sink_host, args.seed, args.atol)))
        else:
            print(json.dumps(attest_middle(args.host, args.port, args.s1, args.s2,
                                           args.seed, args.atol)))
    else:
        if args.host is None or args.s2 is None:
            ap.error("direct mode requires --host and --s2 (+ --s1 for a middle node), "
                     "or --coordinator/--node-id, or --coordinator/--auto")
        print(json.dumps(attest(args.host, args.port, args.s2, args.n, args.seed, args.atol)))


if __name__ == "__main__":
    main()
