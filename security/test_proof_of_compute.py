"""security/test_proof_of_compute.py — run: python -m security.test_proof_of_compute

Covers the wiring around proof-of-compute that doesn't need real model weights or a live
node: verify()'s tolerance, attest_via_coordinator's LAST-vs-MIDDLE branching (mocked HTTP +
mocked attest/attest_middle), and verify_loop's one-pass filtering logic (mocked HTTP, no
real sleep). The actual challenge/response protocol (make_challenge/make_middle_challenge,
challenge_node/challenge_middle_node against a REAL running node_server.py, including the
range-mismatch rejection) was verified manually end-to-end against real downloaded model
slices -- not repeated here since that needs real weights + a real socket server.
"""
import torch

import security.proof_of_compute as poc

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload or {}, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def main():
    # ---- verify(): tolerance behavior ---- #
    exp = torch.zeros(1, 1, 4)
    check("identical tensors pass", poc.verify(exp.clone(), exp)[0])
    close = exp.clone()
    close[0, 0, 0] = 0.01
    check("small diff within atol passes", poc.verify(close, exp, atol=0.05)[0])
    far = exp.clone()
    far[0, 0, 0] = 27.0
    check("large diff fails", not poc.verify(far, exp, atol=0.05)[0])
    check("wrong shape fails, doesn't crash", not poc.verify(torch.zeros(1, 1, 5), exp)[0])
    check("non-tensor output fails, doesn't crash", not poc.verify("garbage", exp)[0])

    # ---- attest_via_coordinator: picks the LAST-stage path for a true last node ---- #
    calls = {"attest": [], "attest_middle": []}
    real_attest, real_attest_middle = poc.attest, poc.attest_middle
    real_get, real_post = poc.requests.get, poc.requests.post
    poc.attest = lambda *a, **k: calls["attest"].append((a, k)) or {"passed": True, "max_err": 0.0}
    poc.attest_middle = lambda *a, **k: calls["attest_middle"].append((a, k)) or {"passed": True, "max_err": 0.0}
    try:
        last_node = {"node_id": "n-last", "tailscale_ip": "1.2.3.4", "port": 9,
                    "layer_start": 19, "layer_end": 27}
        mid_node = {"node_id": "n-mid", "tailscale_ip": "1.2.3.4", "port": 9,
                   "layer_start": 10, "layer_end": 18}

        def fake_get(url, timeout=None, headers=None):
            return FakeResp({"nodes": [last_node, mid_node]})

        def fake_post(url, json=None, headers=None, timeout=None):
            return FakeResp({"ok": True})

        poc.requests.get, poc.requests.post = fake_get, fake_post

        poc.attest_via_coordinator("http://c", "n-last", "secret", n=28)
        check("a node ending at the model's last layer uses attest() (full last-stage)",
              len(calls["attest"]) == 1 and calls["attest_middle"] == [])
        args, _ = calls["attest"][0]
        check("attest() called with this node's own layer_start as s2",
              args[2] == 19)

        poc.attest_via_coordinator("http://c", "n-mid", "secret", n=28)
        check("a node NOT ending at the model's last layer uses attest_middle()",
              len(calls["attest_middle"]) == 1)
        args, _ = calls["attest_middle"][0]
        check("attest_middle() called with [layer_start, layer_end+1)",
              args[2:4] == (10, 19))
    finally:
        poc.attest, poc.attest_middle = real_attest, real_attest_middle
        poc.requests.get, poc.requests.post = real_get, real_post

    # ---- verify_loop: one pass filters correctly, skips flagged, survives a bad node ---- #
    class StopLoop(Exception):
        pass

    seen = []
    real_attest_via_coordinator = poc.attest_via_coordinator
    poc.attest_via_coordinator = lambda coordinator, node_id, secret, n=None, seed=0, atol=0.05: (
        seen.append(node_id),
        (_ for _ in ()).throw(RuntimeError("boom")) if node_id == "n-broken"
        else {"challenge": {"passed": True, "max_err": 0.0, "layers": [0, 1]}}
    )[-1]
    real_get2 = poc.requests.get
    real_sleep = poc.time.sleep
    nodes_payload = {"nodes": [
        {"node_id": "n-probationary", "standing": "probationary", "flagged": False},
        {"node_id": "n-flagged", "standing": "probationary", "flagged": True},
        {"node_id": "n-trusted", "standing": "trusted", "flagged": False},
        {"node_id": "n-broken", "standing": "probationary", "flagged": False},
    ]}
    poc.requests.get = lambda url, timeout=None, headers=None: FakeResp(nodes_payload)

    def sleep_once_then_stop(_):
        raise StopLoop()

    poc.time.sleep = sleep_once_then_stop
    try:
        poc.verify_loop("http://c", "secret", interval=1)
        check("verify_loop should have stopped via StopLoop", False)
    except StopLoop:
        check("verify_loop attempted the probationary, non-flagged node",
              "n-probationary" in seen)
        check("verify_loop skipped the flagged probationary node", "n-flagged" not in seen)
        check("verify_loop skipped the already-trusted node", "n-trusted" not in seen)
        check("verify_loop attempted the node that then failed", "n-broken" in seen)
        check("verify_loop survived that failure and reached time.sleep (loop didn't crash)", True)
    finally:
        poc.attest_via_coordinator = real_attest_via_coordinator
        poc.requests.get = real_get2
        poc.time.sleep = real_sleep

    # ---- challenge_middle_node: an absent field is SILENCE, not disagreement ---- #
    # Live 2026-08-11: the driver acks `s2` and omits `s1`, so a whole-tuple compare read
    # `(None, 10) != (0, 10)` and raised PLACEMENT MISMATCH every sweep. It lands in the
    # verifier's "nothing recorded" branch, so the log looked benign while proof-of-compute
    # quietly stopped checking the most important node in the chain. No socket needed: the
    # comparison is pure logic, which is why it is worth pinning even though the wire protocol
    # itself still needs real weights.
    real_conn, real_send, real_recv = (poc.socket.create_connection,
                                       poc.common.send_msg, poc.common.recv_msg)

    class _Sock:
        def close(self):
            pass

    def _probe(ack, s1=0, s2=10):
        """Run the probe against a scripted ack. Returns the hidden state, or the exception."""
        seq = [ack, {"hidden": "H"}]
        poc.socket.create_connection = lambda *a, **k: _Sock()
        poc.common.send_msg = lambda *a, **k: None
        poc.common.recv_msg = lambda *a, **k: seq.pop(0)
        try:
            return poc.challenge_middle_node("h", 1, s1, s2, "X")
        except Exception as e:                       # returned, not raised, so check() can read it
            return e

    try:
        check("the live driver case: s1 omitted, s2 agrees — verified, not a mismatch",
              _probe({"ok": True, "s1": None, "s2": 10}) == "H")
        check("a pre-0.20 agent claiming nothing at all is challenged, not refused",
              _probe({"ok": True}) == "H")
        check("an exact match still passes",
              _probe({"ok": True, "s1": 0, "s2": 10}) == "H")
        check("a node claiming a DIFFERENT start is still a mismatch",
              isinstance(_probe({"ok": True, "s1": 5, "s2": 10}), poc.RangeMismatch))
        check("a node claiming a DIFFERENT end is still a mismatch",
              isinstance(_probe({"ok": True, "s1": 0, "s2": 14}), poc.RangeMismatch))
        check("the live 18f1da shape (assigned 19-27, holds 10-13) is a mismatch",
              isinstance(_probe({"ok": True, "s1": 10, "s2": 14}, s1=19, s2=28),
                         poc.RangeMismatch))
        check("a typed range_mismatch refusal is still surfaced as RangeMismatch",
              isinstance(_probe({"ok": False, "error": "range_mismatch", "detail": "x"}),
                         poc.RangeMismatch))
    finally:
        poc.socket.create_connection = real_conn
        poc.common.send_msg = real_send
        poc.common.recv_msg = real_recv

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
