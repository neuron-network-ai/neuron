"""test_relay_auth.py — HMAC ticket that lets the relay verify a node's tunnel registration
without a DB or a callback to the coordinator (post-launch-audit fix). Run: python test_relay_auth.py
"""
import relay_auth

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    secret = "test-secret"
    t = relay_auth.make_ticket(secret, "node-a", 9001)

    check("a correctly-minted ticket verifies", relay_auth.verify_ticket(secret, "node-a", 9001, t))
    check("wrong node_id is rejected", not relay_auth.verify_ticket(secret, "node-b", 9001, t))
    check("wrong port is rejected (can't replay onto another port)",
          not relay_auth.verify_ticket(secret, "node-a", 9002, t))
    check("wrong secret is rejected", not relay_auth.verify_ticket("other-secret", "node-a", 9001, t))
    check("missing ticket is rejected", not relay_auth.verify_ticket(secret, "node-a", 9001, None))
    check("empty-string ticket is rejected", not relay_auth.verify_ticket(secret, "node-a", 9001, ""))
    check("tampered ticket (one char flipped) is rejected",
          not relay_auth.verify_ticket(secret, "node-a", 9001, t[:-1] + ("0" if t[-1] != "0" else "1")))
    check("same node_id+port+secret is deterministic (relay recomputes independently)",
          relay_auth.make_ticket(secret, "node-a", 9001) == t)
    check("different port for the same node -> different ticket (no cross-port reuse)",
          relay_auth.make_ticket(secret, "node-a", 9002) != t)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
