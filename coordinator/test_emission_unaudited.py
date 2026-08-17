"""coordinator/test_emission_unaudited.py — what emission pays for when nobody checked ([P47]).

Emission's rule was "a proof-of-compute challenge passed inside this slot". That is the right
shape and it was measuring the wrong thing: whether a challenge lands inside a given hour is
decided by the verifier's rotation and by whether the operator's PC is awake, neither of which a
node can influence. `verify_service.log` says the verifier was not running for 207 of the 390
hours of its own history, and 57 of `node-c-pavilion`'s 64 unpaid hours fall in hours when it was
down or could not read the roster. That machine has passed 4,523 challenges.

So the rule is now "the node's proof is current, OR the reason it isn't is provably ours". The
danger in the second half is obvious and these tests exist to bound it — **every** way of being
paid must still have a real challenge behind it, and no node may be able to engineer itself into
the excused state. The three that would break it, pinned below:

  * an audited slot with no pass still pays NOTHING (presence never pays — the whole point);
  * slots predating the mechanism are not excused, or deploying it would read as a network-wide
    blackout and pay every node for the whole of history;
  * the excuse runs out, because "we could not check" stops being an excuse eventually.

Run:  python -m coordinator.test_emission_unaudited     (from repo root)
"""
import os
import tempfile
import time

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, emission, genesis, models   # noqa: E402

models.init_db()
genesis.seed_genesis()

N = config.TOTAL_LAYERS
SLOT = float(config.SLOT_SECONDS)
T0 = 1_800_000_000.0                      # a fixed slot boundary; no wall clock in these tests
SLOT0 = models.slot_start_for(T0)
MAXRUN = config.EMISSION_MAX_UNAUDITED_SLOTS


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM attendance")
        c.execute("DELETE FROM audit_slots")


def _supply():
    with models._db() as c:
        return round(c.execute("SELECT COALESCE(SUM(balance),0) AS s FROM ledger")
                     .fetchone()["s"], 6)


def _attend(node_id, lo, hi, frac=1.0, poc=False, slot=SLOT0):
    models.register_node(node_id, "1.1.1.1", 50999, lo, hi, 8, 16, f"tok-{node_id}",
                         trusted=True)
    with models._db() as c:
        c.execute("INSERT OR REPLACE INTO attendance "
                  "(node_id, slot_start, seconds_online, block_start, block_end, poc_ok) "
                  "VALUES (?,?,?,?,?,?)",
                  (node_id, slot, SLOT * frac, lo, hi, 1 if poc else 0))


def _audited(slot, ts=None):
    """Mark a slot as one the coordinator did audit."""
    models.record_verifier_heartbeat(ts=slot + 1.0 if ts is None else ts)


def _row(node_id, slot=SLOT0):
    with models._db() as c:
        r = c.execute("SELECT * FROM attendance WHERE node_id=? AND slot_start=?",
                      (node_id, slot)).fetchone()
    return dict(r) if r else None


# --------------------------------------------------------------------------- #
# The property that must NOT change: presence alone still earns nothing.
# --------------------------------------------------------------------------- #
def test_an_audited_slot_with_no_pass_still_pays_nothing():
    """The original rule, intact. This is the exploit emission exists to refuse: a script that
    heartbeats and computes nothing must never be the most profitable node on the network."""
    _clear()
    _audited(SLOT0)
    _attend("n1", 0, 9, poc=False)
    plan = emission.plan_slot([_row("n1")], N, unaudited_run=models.unaudited_run(SLOT0, MAXRUN + 1))
    assert plan[0]["reward"] == 0.0, "a node that failed to prove anything was paid"
    assert "no proof-of-compute" in plan[0]["reason"]
    assert plan[0]["poc_excused"] is False


def test_the_excuse_cannot_reach_a_node_that_was_not_present():
    """An unaudited slot upgrades only nodes that were demonstrably there. The attendance floor
    is evidence the NODE produced, and nothing about our downtime makes it true."""
    _clear()
    plan = emission.plan_slot([dict(node_id="n1", slot_start=SLOT0, seconds_online=SLOT * 0.1,
                                    block_start=0, block_end=9, poc_ok=0)], N, unaudited_run=1)
    assert plan[0]["reward"] == 0.0, "an absent node was paid for an hour nobody audited"
    assert "floor" in plan[0]["reason"]


def test_the_excuse_cannot_reach_a_node_holding_no_block_of_the_model():
    _clear()
    plan = emission.plan_slot([dict(node_id="n1", slot_start=SLOT0, seconds_online=SLOT,
                                    block_start=900, block_end=999, poc_ok=0)], N,
                              unaudited_run=1)
    assert plan[0]["reward"] == 0.0
    assert "held no block" in plan[0]["reason"]


# --------------------------------------------------------------------------- #
# The new behaviour
# --------------------------------------------------------------------------- #
def test_an_unaudited_slot_pays_a_present_correctly_placed_node():
    """The Pavilion's case: online the whole hour, correctly placed, and unpaid only because the
    verifier was asleep."""
    _clear()
    plan = emission.plan_slot([dict(node_id="pav", slot_start=SLOT0, seconds_online=SLOT,
                                    block_start=10, block_end=27, poc_ok=0)], N, unaudited_run=1)
    assert plan[0]["reward"] > 0.0, "the network billed a volunteer for its own downtime"
    assert plan[0]["reason"] is None
    assert plan[0]["poc_excused"] is True


def test_an_excused_hour_pays_the_same_as_a_proved_one():
    """Bounded by COUNT, not discounted by RATE. A discount would be a penalty for our failure,
    which is the thing being fixed."""
    _clear()
    row = dict(node_id="a", slot_start=SLOT0, seconds_online=SLOT, block_start=0, block_end=9,
               poc_ok=0)
    excused = emission.plan_slot([dict(row)], N, unaudited_run=1)[0]
    proved = emission.plan_slot([dict(row, poc_ok=1)], N, unaudited_run=0)[0]
    assert excused["reward"] == proved["reward"], "an excused hour was priced differently"


def test_a_node_challenged_inside_an_unaudited_slot_is_not_marked_excused():
    """`poc_excused` must count what it says it counts, or the payout log misreports how much of
    the network's spend is going out on trust rather than on evidence."""
    _clear()
    plan = emission.plan_slot([dict(node_id="a", slot_start=SLOT0, seconds_online=SLOT,
                                    block_start=0, block_end=9, poc_ok=1)], N, unaudited_run=1)
    assert plan[0]["reward"] > 0.0
    assert plan[0]["poc_excused"] is False, "a node that proved itself was logged as excused"


def test_excused_rows_count_toward_replica_depth():
    """If they did not, our own outage would read as a network-wide scarcity spike and pay MORE
    per node at exactly the moment the coordinator knows least — turning downtime into a payout
    event, which is a strictly worse failure than the one being fixed."""
    _clear()
    rows = [dict(node_id=f"n{i}", slot_start=SLOT0, seconds_online=SLOT, block_start=0,
                 block_end=9, poc_ok=0) for i in range(config.EMISSION_TARGET_REPLICAS)]
    plan = emission.plan_slot(rows, N, unaudited_run=1)
    assert all(p["replicas"] == config.EMISSION_TARGET_REPLICAS for p in plan)
    assert all(p["multiplier"] == 1.0 for p in plan), \
        "an outage priced itself as scarcity and paid a premium for it"


def test_the_excuse_runs_out():
    """Past the bound nobody has verified this network in hours and the honest answer is that we
    do not know. The reason says so rather than blaming the node."""
    _clear()
    row = dict(node_id="a", slot_start=SLOT0, seconds_online=SLOT, block_start=0, block_end=9,
               poc_ok=0)
    assert emission.plan_slot([dict(row)], N, unaudited_run=MAXRUN)[0]["reward"] > 0.0
    over = emission.plan_slot([dict(row)], N, unaudited_run=MAXRUN + 1)[0]
    assert over["reward"] == 0.0, "an unbounded outage kept paying"
    assert "not audited" in over["reason"] and "nobody checked" in over["reason"]


# --------------------------------------------------------------------------- #
# audit_slots: the coordinator's record of its OWN diligence
# --------------------------------------------------------------------------- #
def test_a_heartbeat_makes_the_slot_audited():
    _clear()
    _audited(SLOT0)
    assert models.unaudited_run(SLOT0, MAXRUN + 1) == 0


def test_slots_before_the_first_heartbeat_are_not_outages():
    """THE DEPLOY MUST NOT LOOK LIKE A BLACKOUT. Absence of a row is the signal, and every slot
    in history has no row — so without the epoch, shipping this would excuse the whole of the
    past and pay every present node for it. That is paying for presence, the one thing emission
    exists to refuse, arriving through the door built to protect volunteers."""
    _clear()
    _audited(SLOT0)
    for back in (1, 5, 50, 5000):
        assert models.unaudited_run(SLOT0 - back * SLOT, MAXRUN + 1) == 0, \
            f"a slot {back} before the mechanism existed was treated as an outage"


def test_unaudited_run_counts_consecutive_slots_and_is_bounded():
    _clear()
    _audited(SLOT0)                                    # the epoch
    for k in (1, 2, 3):                                # three silent slots after it
        assert models.unaudited_run(SLOT0 + k * SLOT, MAXRUN + 1) == k
    _audited(SLOT0 + 4 * SLOT)                         # verifier came back
    assert models.unaudited_run(SLOT0 + 4 * SLOT, MAXRUN + 1) == 0
    assert models.unaudited_run(SLOT0 + 5 * SLOT, MAXRUN + 1) == 1, \
        "the run must restart after an audited slot, not accumulate over all history"


def test_the_walk_back_stops_at_the_limit():
    """A long outage must not turn every sweep into a thousand-row scan."""
    _clear()
    _audited(SLOT0)
    assert models.unaudited_run(SLOT0 + 10_000 * SLOT, MAXRUN + 1) == MAXRUN + 1


def test_heartbeats_accumulate_within_a_slot():
    _clear()
    for i in range(5):
        models.record_verifier_heartbeat(ts=SLOT0 + 60.0 * i)
    with models._db() as c:
        r = c.execute("SELECT * FROM audit_slots WHERE slot_start=?", (SLOT0,)).fetchone()
    assert r["heartbeats"] == 5
    assert r["last_at"] >= r["first_at"]


# --------------------------------------------------------------------------- #
# A pass carries beyond the minute it landed in
# --------------------------------------------------------------------------- #
def test_a_recent_pass_proves_the_current_slot():
    """Real clock, because touch_node reads it. A challenge that passed a moment ago must still
    count for the hour now running: whether one lands inside any given hour is the verifier's
    rotation, not the node's behaviour."""
    _clear()
    models.register_node("carry", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-carry", trusted=True)
    now = time.time()
    with models._db() as c:
        c.execute("UPDATE nodes SET last_seen=?, last_poc_at=? WHERE node_id='carry'",
                  (now - 30.0, now - 60.0))
    models.touch_node("carry")
    row = _row("carry", models.slot_start_for(now))
    assert row is not None and row["poc_ok"] == 1, "a pass one minute old proved nothing"
    assert row["poc_at"] is not None, "the proving instant was not recorded"


def test_a_stale_pass_does_not_prove_anything():
    """The other half, and the one that keeps this honest: evidence ages out. Without this the
    carry-forward is just 'passed once, paid forever'."""
    _clear()
    models.register_node("stale", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-stale", trusted=True)
    now = time.time()
    old = now - (config.EMISSION_POC_VALID_SLOTS * SLOT) - 600.0
    with models._db() as c:
        c.execute("UPDATE nodes SET last_seen=?, last_poc_at=? WHERE node_id='stale'",
                  (now - 30.0, old))
    models.touch_node("stale")
    row = _row("stale", models.slot_start_for(now))
    assert row is not None and row["poc_ok"] == 0, "a stale proof was still being paid on"


def test_mark_slot_poc_only_moves_last_poc_at_forward():
    """Attestations can arrive out of order — a retry, two verifiers. A late one carrying an
    older instant must not make a node's proof look staler than it is."""
    _clear()
    models.register_node("ord", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-ord", trusted=True)
    models.mark_slot_poc("ord", ts=T0 + 500.0)
    models.mark_slot_poc("ord", ts=T0 + 100.0)          # the late, older one
    with models._db() as c:
        got = c.execute("SELECT last_poc_at FROM nodes WHERE node_id='ord'").fetchone()[0]
    assert got == T0 + 500.0, "an out-of-order attestation aged a node's proof backwards"


# --------------------------------------------------------------------------- #
# Settlement writes the reason down, and the supply survives
# --------------------------------------------------------------------------- #
def test_settlement_records_that_the_hour_was_excused():
    """The reason a payment happened must survive with the payment. `--replay` re-prices a row
    from its own frozen values ([P40]), so a row paid without a proof has to carry why, or the
    reconciliation would read it as a mispricing forever after."""
    _clear()
    _audited(SLOT0 - SLOT)                      # epoch: the slot BEFORE the outage was audited
    _attend("pav", 10, 27, frac=1.0, poc=False)
    out = emission.close_slots(N, now=T0 + SLOT + 1.0, log=lambda *_: None)
    row = _row("pav")
    assert row["paid_at"] is not None and row["reward"] > 0.0, "the excused hour did not pay"
    assert row["poc_excused"] == 1, "the row does not say why it was paid"
    assert out["excused"] == 1, "the sweep did not report paying for an unaudited hour"


def test_supply_is_conserved_across_an_excused_payout():
    """Nothing is minted, ever. An excused hour is a transfer out of the pool like any other."""
    _clear()
    before = _supply()
    _audited(SLOT0 - SLOT)
    _attend("s1", 0, 9, frac=1.0, poc=False)
    emission.close_slots(N, now=T0 + SLOT + 1.0, log=lambda *_: None)
    assert _supply() == before, "the excused path minted NRN"


def test_an_audited_slot_settles_at_zero_and_says_the_ordinary_thing():
    """End to end through close_slots, so the wiring is pinned at its call site and not only in
    plan_slot: a slot we DID audit, with a node that proved nothing, still pays nothing."""
    _clear()
    _audited(SLOT0)
    _attend("lazy", 0, 9, frac=1.0, poc=False)
    emission.close_slots(N, now=T0 + SLOT + 1.0, log=lambda *_: None)
    row = _row("lazy")
    assert row["paid_at"] is not None and row["reward"] == 0.0, \
        "an audited, unproven hour was paid"
    assert row["poc_excused"] == 0


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
