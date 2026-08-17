"""coordinator/test_node_retirement.py — deleting a node leaves a receipt ([P39] item 5, [P50]).

`delete_node` used to be one DELETE. Two things followed from that, and both cost real NRN or
came within one file write of it:

  * **It stranded money.** A node account's only credential is the `node_token` in one
    config.json on one disk, and the coordinator's copy lives in the row being deleted. After
    that nothing can prove who owned the balance. `agent-bhpc012104-82cbee` was recovered only
    because the machine belonged to the founder.
  * **It left an unattributable hole.** `attendance` and `ledger` survive, so the reconciliation
    finds settled hours belonging to no node — and cannot tell a deliberate retirement from its
    own bookkeeping losing a machine. It has said so since 2026-08-02 and could never stop,
    because that machine is not coming back. [P40] item 2 wants that check running as a standing
    assertion, and a check that always warns is a check nobody reads.

Run:  python -m coordinator.test_node_retirement     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, models   # noqa: E402

models.init_db()

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def _mk(node_id, balance=0.0):
    models.register_node(node_id, "1.1.1.1", 50999, 0, 9, 8, 16, f"tok-{node_id}", trusted=True)
    with models._db() as c:
        c.execute("INSERT OR REPLACE INTO ledger (node_id, balance, total_earned) VALUES (?,?,?)",
                  (node_id, balance, balance))


def main():
    with models._db() as c:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM retired_nodes")

    print("\n-- a funded node is not deleted")
    _mk("rich", balance=5.25)
    try:
        models.delete_node("rich")
        check("deleting a node that still holds NRN is refused", False, "it was deleted")
    except models.NodeStillFunded as e:
        check("deleting a node that still holds NRN is refused", True)
        check("...and the message says how to proceed properly",
              "sweep" in str(e).lower() and "5.25" in str(e), str(e))
    check("...so the node is still there", models.get_node("rich") is not None)
    check("...and no tombstone was written for a node that still exists",
          "rich" not in models.retired_nodes())

    print("\n-- force is for the operator who has already swept it")
    check("force deletes a funded node", models.delete_node("rich", force=True) is True)
    check("...and the tombstone records the balance it had",
          abs(models.retired_nodes()["rich"]["final_balance"] - 5.25) < 1e-9,
          str(models.retired_nodes().get("rich")))

    print("\n-- an ordinary retirement leaves a receipt")
    _mk("gone", balance=0.0)
    check("a node with no balance deletes", models.delete_node("gone", reason="work PC, gone") is True)
    check("...the nodes row is gone", models.get_node("gone") is None)
    t = models.retired_nodes()["gone"]
    check("...a tombstone remains", t is not None)
    check("...carrying the reason", t["reason"] == "work PC, gone", str(t))
    check("...and the range it held, which is unrecoverable once the row is gone",
          (t["layer_start"], t["layer_end"]) == (0, 9), str(t))
    check("...and when", t["retired_at"] > 0)

    print("\n-- the ledger and attendance are deliberately NOT touched")
    # Deleting must stay exactly as destructive as it was for the `nodes` row and no more: the
    # balance is somebody's money and the attendance is the record of hours actually worked.
    with models._db() as c:
        led = c.execute("SELECT balance FROM ledger WHERE node_id='gone'").fetchone()
    check("the ledger row survives a retirement", led is not None)

    print("\n-- deleting something that was never there")
    check("deleting an unknown node returns False", models.delete_node("never-existed") is False)
    check("...and writes no tombstone for it",
          "never-existed" not in models.retired_nodes())

    print("\n-- a re-registered node id can be retired twice")
    # INSERT OR REPLACE, not INSERT: node ids are reused (one PC registered eight times, [P50]),
    # and a second retirement must not raise on the primary key.
    _mk("recycled")
    models.delete_node("recycled", reason="first")
    _mk("recycled")
    models.delete_node("recycled", reason="second")
    check("the second retirement replaces the first rather than raising",
          models.retired_nodes()["recycled"]["reason"] == "second")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
