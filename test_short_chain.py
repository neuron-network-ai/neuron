"""test_short_chain.py — run: python -m test_short_chain

A three-machine network with one machine away is a TWO-stage chain, and that is the ordinary
evening state of a network built from other people's spare computers, not an exotic one.

The coordinator has always handled it: self-heal re-splits the model across whoever is left and
reports the network healthy. Observed live 2026-08-07 with two nodes up:

    gap-heal: mode=resplit  plan=[pavilion 0-13, optinovate 14-27]  capacity_shortfall=0

...and then every request died on `expected a 3-node chain, got 2`. The recovery machinery
worked perfectly and nothing could use what it produced. These tests pin the shapes the driver
must accept, and — the part that actually broke — what it puts on the wire for each, because
`agent/node_server.py` picks its role from whether `host_b` is present.
"""
import sys

import node_a

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
        self._p, self.status_code = payload, status

    def json(self):
        return self._p


def hop(node_id, ip, lo, hi):
    return {"node_id": node_id, "ip": ip, "port": 50999, "layers": [lo, hi]}


def get_chain(chain, expected_s1, monkeypatch_post=None):
    payload = {"chain": chain, "request_id": "r1", "complete_token": "t", "hold_amount": 1.0}
    real = node_a.requests.post
    node_a.requests.post = lambda *a, **k: _Resp(payload)
    try:
        return node_a.coord_get_chain("http://c", "hi", 8, expected_s1, "w1")
    finally:
        node_a.requests.post = real


def main():
    # ---- three stages: unchanged behaviour ----------------------------------- #
    three = [hop("a", "10.0.0.1", 0, 9), hop("c", "10.0.0.2", 10, 18),
             hop("b", "10.0.0.3", 19, 27)]
    nh, np_, hb, pb, s2, ids, rid, tok, hold = get_chain(three, 10)
    check("3-stage: next hop is the middle", (nh, np_) == ("10.0.0.2", 50999))
    check("3-stage: host_b is the final stage", (hb, pb) == ("10.0.0.3", 50999))
    check("3-stage: s2 is where the last stage begins", s2 == 19)
    check("3-stage: every node is billed", ids == ["a", "c", "b"])

    # ---- two stages: the case that broke the live network -------------------- #
    two = [hop("a", "10.0.0.1", 0, 13), hop("b", "10.0.0.3", 14, 27)]
    nh, np_, hb, pb, s2, ids, rid, tok, hold = get_chain(two, 14)
    check("2-stage: accepted at all (was: expected a 3-node chain, got 2)", True)
    check("2-stage: next hop is the last stage", (nh, np_) == ("10.0.0.3", 50999))
    check("2-stage: NO host_b, so that hop plays the last-stage role", (hb, pb) == (None, None))
    check("2-stage: s2 is where the last stage begins", s2 == 14)
    check("2-stage: both nodes are billed", ids == ["a", "b"])

    # ---- the driver must still be the first stage ---------------------------- #
    try:
        get_chain(two, 10)                       # our shard is 0..9, chain says 0..13
        check("a chain that does not match our own shard is refused", False)
    except RuntimeError as e:
        check("a chain that does not match our own shard is refused", "shard is 0..9" in str(e))

    # ---- shapes this driver genuinely cannot route --------------------------- #
    one = [hop("a", "10.0.0.1", 0, 27)]
    try:
        get_chain(one, 28)
        check("a 1-stage chain is refused with a useful message", False)
    except RuntimeError as e:
        check("a 1-stage chain is refused with a useful message",
              "1-stage" in str(e) and "2 or 3" in str(e))

    four = three + [hop("d", "10.0.0.4", 28, 30)]
    try:
        get_chain(four, 10)
        check("a 4-stage chain is refused rather than silently mis-routed", False)
    except RuntimeError as e:
        check("a 4-stage chain is refused rather than silently mis-routed", "4-stage" in str(e))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
