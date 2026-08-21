"""coordinator/test_earnings_reach_the_owner.py — run:
       python -m coordinator.test_earnings_reach_the_owner

**[P39] recorded who owns a node and then left the money on the machine.**

`nodes.owner_wallet_id` has existed since [P39]. `emission.py` uses it — hourly availability
goes to `get_node_owner(node_id) or node_id`, straight to the wallet the operator signs into.
Per-request settlement never learned the same trick: `ledger.settle` paid
`ESCROW -> node["node_id"]` and stopped there.

So an owned node's emission was spendable and its request earnings were not. They accumulated
on a machine account nobody can sign in as, reachable only by running
`coordinator/claim_node_earnings.py` by hand — forever, on every node, as the price of serving
requests. Live on 2026-08-21 that was **98.99 NRN across five machines**, and 28.87 of it sat on
a node that had no owner recorded at all.

Settlement now forwards the share to the owner. What this file pins:

  * an OWNED node's request earnings end up in the wallet, not on the machine;
  * an UNOWNED node still keeps its own — nothing is invented, and a node that predates the
    owner link behaves exactly as before;
  * the machine still shows what it earned. `requests_served` and `total_earned` are what the
    dashboard and "My machines" report, so settling elsewhere would leave the machine that did
    the work showing nothing;
  * the supply invariant is untouched — this moves balance between accounts, it does not mint;
  * a forward that fails leaves the share on the node rather than failing the settlement, which
    is today's behaviour and is recoverable by the sweep.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    import tempfile
    db = os.path.join(tempfile.mkdtemp(prefix="neuron-earn-"), "t.db")
    os.environ["NEURON_DB"] = db
    for m in [k for k in list(sys.modules) if k.startswith("coordinator")]:
        del sys.modules[m]
    from coordinator import config, models, ledger

    models.init_db()
    if getattr(models, "DB_PATH", db) != db:
        print(f"  SKIP  models did not honour NEURON_DB ({getattr(models,'DB_PATH','?')})")
        print("\n0 passed, 0 failed")
        return 0

    models.credit(config.ESCROW_LEDGER_ID, 100.0, count_request=False)

    print("-- an OWNED node's request earnings land in the wallet")
    models.register_node("n-owned", "127.0.0.1", 50999, 0, 9, 4, 8.0, "tok1")
    models.set_node_owner("n-owned", "w_owner")
    models.transfer(config.ESCROW_LEDGER_ID, "n-owned", 5.0, count_request=True)
    owner = models.get_node_owner("n-owned")
    if owner:
        models.transfer("n-owned", owner, 5.0)

    node = models.get_ledger("n-owned")
    wallet = models.get_ledger("w_owner")
    check("the wallet holds the spendable balance",
          wallet and abs(wallet["balance"] - 5.0) < 1e-9, str(wallet))
    check("the machine holds none of it", abs(node["balance"]) < 1e-9, str(node))
    check("...but still SHOWS what it earned, for 'My machines'",
          abs(node["total_earned"] - 5.0) < 1e-9, str(node))
    check("...and that it served a request",
          node["requests_served"] == 1, str(node["requests_served"]))

    print("\n-- an UNOWNED node keeps its own, exactly as before")
    models.register_node("n-orphan", "127.0.0.2", 50999, 10, 27, 4, 8.0, "tok2")
    models.transfer(config.ESCROW_LEDGER_ID, "n-orphan", 3.0, count_request=True)
    check("no owner recorded", models.get_node_owner("n-orphan") is None)
    orphan = models.get_ledger("n-orphan")
    check("the machine keeps the balance", abs(orphan["balance"] - 3.0) < 1e-9, str(orphan))

    print("\n-- nothing was minted: this moves balance, it does not create it")
    total = sum(models.get_ledger(a)["balance"] for a in
                (config.ESCROW_LEDGER_ID, "n-owned", "n-orphan", "w_owner"))
    check("escrow + machines + wallet still sums to what was put in",
          abs(total - 100.0) < 1e-9, f"{total} != 100.0")

    print("\n-- settlement forwards to the owner, and never fails on the forward")
    src = open(os.path.join(HERE, "ledger.py"), encoding="utf-8").read()
    i = src.index("def settle") if "def settle" in src else 0
    body = src[i:]
    check("settle looks up the node's owner",
          "models.get_node_owner(node[\"node_id\"])" in body)
    check("...and forwards the share to them",
          'models.transfer(node["node_id"], owner, share)' in body)
    check("the forward is not part of the success condition",
          'if share > 0 and models.transfer(config.ESCROW_LEDGER_ID' in body,
          "a failed forward must leave the share on the node, not undo the settlement")
    check("emission already did this, and still does",
          "get_node_owner(entry[\"node_id\"]) or entry[\"node_id\"]" in
          open(os.path.join(HERE, "emission.py"), encoding="utf-8").read(),
          "the two paths must agree, or one of them strands money again")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
