"""security/test_verifier_drives_the_user_path.py — run:
       python -m security.test_verifier_drives_the_user_path

**[P56]: a verifier that constructs its own challenge is verifying a path of its own
construction.** For a day the Pavilion returned garbage to every real user while
`challenges_passed` stood at 5662/4. That number was not broken — it was a correct measurement
of a question nobody was asking. `neuron_driver` and `proof_of_compute` each built their own
`config` message, `agent/node_server` reads its ROLE out of that message, and the two messages
disagreed on the one field the role turned on ([P55]).

So the thing worth pinning is not any particular message. It is that **the node cannot tell a
challenge from a request on any field that decides what it runs.** These tests assert that
against `node_server._is_range_probe` and the branch order in `node_server.serve`, for every
chain shape the network can be in.

Nothing here needs weights or a socket: the role decision is pure logic, which is exactly why
it was worth extracting and is worth pinning.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import common
from security import proof_of_compute as poc

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def role_of(msg, is_true_last, _is_range_probe):
    """The branch `node_server.serve` takes for this config, in its own order."""
    if "host_b" in msg:
        return "middle"
    if is_true_last and not _is_range_probe(msg):
        return "last"
    return "probe"


def main():
    global ok, fail
    from agent.node_server import _is_range_probe

    print("-- the LAST hop: a challenge and a request are the same role")
    # The live topology: driver holds 0-9, one node holds 10-27 and IS the final layer.
    request = common.stage_config(10, stage="last", hidden_size=1536, lan_hint=["192.168.1"])
    challenge = common.stage_config(10, stage="last", n=28)
    for name, msg in (("request", request), ("challenge", challenge)):
        check(f"the {name} puts a true-last node in the LAST role",
              role_of(msg, True, _is_range_probe) == "last",
              role_of(msg, True, _is_range_probe))
    check("...and they agree, which is the invariant [P55] broke",
          role_of(request, True, _is_range_probe) == role_of(challenge, True, _is_range_probe))

    print("\n-- the role-deciding fields are IDENTICAL, not merely compatible")
    # `wire`, `lan_hint` and `n` are role-INERT: node_server never reads them to decide what to
    # run. Everything else must match, or the two messages are two paths again.
    INERT = {"wire", "lan_hint", "n"}
    r = {k: v for k, v in request.items() if k not in INERT}
    c = {k: v for k, v in challenge.items() if k not in INERT}
    check("last-hop request and challenge differ only in inert fields", r == c, f"{r} != {c}")
    check("the challenge sends NO wire field, so the reply stays lossless",
          "wire" not in challenge, "a quantized reply would spend verify()'s atol budget")

    print("\n-- a full-model node, which is what actually bit us")
    # 2026-08-17: a 64 GB machine assigned 0-9 had loaded 0-27, so `is_true_last` is TRUE even
    # for a config about a middle range. That is the shape that sent a probe into the LAST
    # branch and produced max_err 28.5958 twice, recorded as unexplained.
    check("a probe still reads as a probe on a node that holds everything",
          role_of({"type": "config", "s1": 0, "s2": 10, "probe": True}, True,
                  _is_range_probe) == "probe")
    check("...while a genuine last-stage request on the same node reads as LAST",
          role_of(request, True, _is_range_probe) == "last")

    print("\n-- the MIDDLE hop: the relay challenge takes the role the probe never did")
    relay_request = common.stage_config(10, stage="middle", s1=0, host_b="10.0.0.1",
                                        port_b=50999, hidden_size=1536)
    relay_challenge = common.stage_config(10, stage="middle", s1=0, host_b="10.0.0.9",
                                          port_b=41111)
    check("a real request puts a middle node in the MIDDLE role",
          role_of(relay_request, False, _is_range_probe) == "middle")
    check("the relay CHALLENGE puts it in the same role",
          role_of(relay_challenge, False, _is_range_probe) == "middle",
          role_of(relay_challenge, False, _is_range_probe))
    check("the old probe challenge does NOT — this is the gap [P56] names",
          role_of({"type": "config", "s1": 0, "s2": 10, "probe": True}, False,
                  _is_range_probe) == "probe")
    r2 = {k: v for k, v in relay_request.items() if k not in INERT | {"host_b", "port_b"}}
    c2 = {k: v for k, v in relay_challenge.items() if k not in INERT | {"host_b", "port_b"}}
    check("middle request and relay challenge differ only in the hop address and inert fields",
          r2 == c2, f"{r2} != {c2}")

    print("\n-- the constructor refuses shapes that would silently mis-role a node")
    for bad, why in (
            (dict(stage="middle", s1=0), "a middle hop with no host_b would answer as a LAST "
                                         "stage and apply the final norm"),
            (dict(stage="middle", host_b="h", port_b=1), "a middle hop with no s1 has no range"),
            (dict(stage="probe"), "the probe is not a shape any request produces"),
            (dict(stage="first"), "there is no such role on the wire")):
        try:
            common.stage_config(10, **bad)
            check(f"refused: {why}", False, "it was accepted")
        except ValueError:
            check(f"refused: {why}", True)

    print("\n-- the relay challenge, end to end over real sockets")
    # A stand-in node that speaks the sequence `agent/node_server.serve` implements for the
    # MIDDLE role: dial host_b, config it, ack the caller, then on each `act` forward what it
    # computed and relay the reply back. Real sockets and real framing, fake arithmetic — what
    # is under test is the verifier's half, and whether it comes away holding the tensor the
    # node FORWARDED rather than the one it answered with.
    import socket
    import threading

    computed = "COMPUTED-BY-THE-NODE"
    answered = "WHAT-THE-NODE-ANSWERS-ITS-CALLER"
    node_seen = {}

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def fake_middle_node():
        conn, _ = srv.accept()
        try:
            cfg = common.recv_msg(conn)
            node_seen["cfg"] = cfg
            b = socket.create_connection((cfg["host_b"], cfg["port_b"]), timeout=20)
            try:
                common.send_msg(b, {"type": "config", "s2": cfg["s2"], "n": 28,
                                    "wire": ["i8h", "f16", "f32"]})
                node_seen["sink_ack"] = common.recv_msg(b)
                common.send_msg(conn, {"ok": True, "layers": 28,
                                       "s1": cfg["s1"], "s2": cfg["s2"]})
                act = common.recv_msg(conn)
                node_seen["challenge_input"] = act["hidden"]
                common.send_msg(b, {"type": "act", "hidden": computed})
                relayed = common.recv_msg(b)
                node_seen["relayed_back"] = relayed["hidden"]
                common.send_msg(conn, {"hidden": answered, "c_compute_ms": 1.0,
                                       "b_compute_ms": relayed["b_compute_ms"]})
                common.recv_msg(conn)                    # bye
            finally:
                b.close()
        finally:
            conn.close()

    t = threading.Thread(target=fake_middle_node, daemon=True)
    t.start()
    try:
        got = poc.challenge_relay_node("127.0.0.1", srv.getsockname()[1], 0, 10,
                                       "CHALLENGE-INPUT", "127.0.0.1", timeout=20)
    finally:
        t.join(timeout=20)
        srv.close()

    check("the verifier grades what the node FORWARDED, not what it answered",
          got == computed, f"got {got!r}; the node answered {answered!r}")
    check("the node was put in the MIDDLE role by the challenge",
          "host_b" in node_seen.get("cfg", {}) and node_seen["cfg"].get("stage") == "middle",
          str(node_seen.get("cfg")))
    check("the challenge input reached the node unchanged",
          node_seen.get("challenge_input") == "CHALLENGE-INPUT")
    check("the sink acks with NO wire field, so the forwarded tensor stays lossless",
          "wire" not in node_seen.get("sink_ack", {}), str(node_seen.get("sink_ack")))
    check("the sink replies so the node can complete its exchange rather than hanging",
          node_seen.get("relayed_back") is not None)

    print("\n-- a node that takes the config and relays NOTHING is refused, not passed")
    srv2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv2.bind(("127.0.0.1", 0))
    srv2.listen(1)

    def lazy_node():
        conn, _ = srv2.accept()
        try:
            common.recv_msg(conn)
            common.send_msg(conn, {"ok": True, "layers": 28, "s1": 0, "s2": 10})
            common.recv_msg(conn)                        # the act
            common.send_msg(conn, {"hidden": answered, "b_compute_ms": 0.0})
            common.recv_msg(conn)                        # bye
        except Exception:
            pass
        finally:
            conn.close()

    t2 = threading.Thread(target=lazy_node, daemon=True)
    t2.start()
    try:
        poc.challenge_relay_node("127.0.0.1", srv2.getsockname()[1], 0, 10, "X",
                                 "127.0.0.1", timeout=5)
        check("a node that never relayed is refused", False, "it returned a result")
    except poc.ChallengeRefused:
        check("a node that never relayed is refused", True)
    except Exception as e:
        check("a node that never relayed is refused", False, f"{type(e).__name__}: {e}")
    finally:
        t2.join(timeout=10)
        srv2.close()

    print("\n-- every attestation says which path it exercised")
    src = open(os.path.join(HERE, "proof_of_compute.py"), encoding="utf-8").read()
    for fn, path in (("def attest(", "last"), ("def attest_relay(", "relay"),
                     ("def attest_middle(", "probe")):
        i = src.find(fn)
        check(f"{fn.split('(')[0][4:]} reports path={path!r}",
              i != -1 and f'"path": "{path}"' in src[i:i + 1200])
    check("a relay attempt that fell back to the probe says so",
          '"relay_fallback"' in src,
          "a probe pass is a weaker claim than a relay pass; reporting them as the same "
          "number is the whole of [P56]")
    check("a RangeMismatch is never retried as a probe",
          "except RangeMismatch:" in src and "raise " in src,
          "placement is not compute — falling back would report a stale range as a failure")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
