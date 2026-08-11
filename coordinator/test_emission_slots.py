"""coordinator/test_emission_slots.py — availability emission (TOKENOMICS.md §11.4).

Paying nodes for being AVAILABLE is a different economic object from paying them for work
served, and it fails in different ways. These tests pin the three that matter:

  * nothing is minted — the fixed 1,000,000,000 supply survives every payout;
  * presence alone earns nothing — a node that heartbeats and computes nothing is exactly the
    exploit this reward invents, so proof-of-compute must land inside the slot;
  * a slot settles at most once — a replayed sweep is a no-op, not a second payment.

Run:  python -m coordinator.test_emission_slots     (from repo root)
"""
import os
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, emission, genesis, models

models.init_db()
genesis.seed_genesis()

N = config.TOTAL_LAYERS
SLOT = float(config.SLOT_SECONDS)
T0 = 1_800_000_000.0                      # a fixed slot boundary; no wall clock in tests
SLOT0 = models.slot_start_for(T0)


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM attendance")


def _supply():
    with models._db() as c:
        return round(c.execute("SELECT COALESCE(SUM(balance),0) AS s FROM ledger")
                     .fetchone()["s"], 6)


def _pool():
    row = models.get_ledger(config.GENESIS_BUCKETS_EMISSION_ID)
    return round(row["balance"], 6)


def _attend(node_id, lo, hi, frac=1.0, poc=True, slot=SLOT0):
    """Write an attendance row directly — the heartbeat path is exercised separately."""
    models.register_node(node_id, "1.1.1.1", 50999, lo, hi, 8, 16, f"tok-{node_id}",
                         trusted=True)
    with models._db() as c:
        c.execute("INSERT OR REPLACE INTO attendance "
                  "(node_id, slot_start, seconds_online, block_start, block_end, poc_ok) "
                  "VALUES (?,?,?,?,?,?)",
                  (node_id, slot, SLOT * frac, lo, hi, 1 if poc else 0))


# --------------------------------------------------------------------------- #
# The pricing curve — pure, no DB
# --------------------------------------------------------------------------- #
def test_scarcity_multiplier_is_bounded_and_monotonic():
    cap = config.EMISSION_SCARCITY_MAX
    tgt = config.EMISSION_TARGET_REPLICAS
    assert emission.scarcity_multiplier(0) == cap, "an uncovered block advertises the cap"
    assert emission.scarcity_multiplier(1) > emission.scarcity_multiplier(tgt)
    assert emission.scarcity_multiplier(tgt) == 1.0
    assert emission.scarcity_multiplier(tgt * 10) == 1.0, "over-covered never pays below base"
    for r in range(0, 25):
        assert 1.0 <= emission.scarcity_multiplier(r) <= cap


def test_thin_slot_pays_more_than_a_crowded_one():
    _clear()
    _attend("lonely", 0, 9)
    thin = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    _clear()
    for i in range(config.EMISSION_TARGET_REPLICAS * 2):
        _attend(f"crowd{i}", 0, 9)
    crowded = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert thin[0]["reward"] > crowded[0]["reward"]
    assert crowded[0]["multiplier"] == 1.0


# --------------------------------------------------------------------------- #
# The three gates
# --------------------------------------------------------------------------- #
def test_presence_without_proof_of_compute_earns_nothing():
    """THE exploit this reward invents: heartbeat all night, compute nothing."""
    _clear()
    _attend("faker", 0, 9, frac=1.0, poc=False)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert plan[0]["reward"] == 0.0
    assert "proof-of-compute" in plan[0]["reason"]


def test_a_block_outside_the_serving_model_earns_nothing():
    _clear()
    _attend("stale", N + 5, N + 9)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert plan[0]["reward"] == 0.0
    assert "no block" in plan[0]["reason"]


def test_below_the_attendance_floor_earns_nothing():
    _clear()
    _attend("blinker", 0, 9, frac=config.SLOT_MIN_ATTENDANCE_FRAC / 2)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert plan[0]["reward"] == 0.0
    assert "floor" in plan[0]["reason"]


def test_a_full_qualifying_slot_earns_the_base_rate_times_the_multiplier():
    _clear()
    for i in range(config.EMISSION_TARGET_REPLICAS):
        _attend(f"n{i}", 0, 9)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert all(p["reason"] is None for p in plan)
    assert all(abs(p["reward"] - config.EMISSION_BASE_NRN_PER_HOUR) < 1e-9 for p in plan)


def test_partial_attendance_is_prorated():
    _clear()
    for i in range(config.EMISSION_TARGET_REPLICAS):
        _attend(f"n{i}", 0, 9, frac=0.75)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert abs(plan[0]["reward"] - config.EMISSION_BASE_NRN_PER_HOUR * 0.75) < 1e-9


def test_unverified_nodes_do_not_make_a_block_look_covered():
    """If a failed challenge still counted toward replica depth, a block held only by
    unverifiable nodes would price itself as healthy and never attract a real one."""
    _clear()
    _attend("real", 0, 9, poc=True)
    for i in range(6):
        _attend(f"ghost{i}", 0, 9, poc=False)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    real = [p for p in plan if p["node_id"] == "real"][0]
    assert real["replicas"] == 1, "only the verified node counts"
    assert real["multiplier"] == emission.scarcity_multiplier(1)


# --------------------------------------------------------------------------- #
# Settlement — money, and the invariants around it
# --------------------------------------------------------------------------- #
def test_supply_is_conserved_and_the_pool_pays_for_it():
    _clear()
    before_supply, before_pool = _supply(), _pool()
    for i in range(config.EMISSION_TARGET_REPLICAS):
        _attend(f"n{i}", 0, 9)
    res = emission.close_slots(N, now=SLOT0 + SLOT * 1.5, log=lambda *_: None)
    assert res["nodes"] == config.EMISSION_TARGET_REPLICAS
    assert res["paid"] > 0
    assert _supply() == before_supply, "emission must MOVE NRN, never mint it"
    assert abs((before_pool - _pool()) - res["paid"]) < 1e-6


def test_a_replayed_sweep_pays_nothing_extra():
    _clear()
    _attend("n0", 0, 9)
    first = emission.close_slots(N, now=SLOT0 + SLOT * 1.5, log=lambda *_: None)
    pool_after_first = _pool()
    second = emission.close_slots(N, now=SLOT0 + SLOT * 1.5, log=lambda *_: None)
    assert first["paid"] > 0
    assert second["paid"] == 0.0 and second["nodes"] == 0
    assert _pool() == pool_after_first


def test_an_open_slot_is_never_settled():
    """Paying a slot still accruing would settle a partial hour with nowhere to put the rest."""
    _clear()
    _attend("openslot", 0, 9)
    before = models.get_ledger("openslot")
    before = before["balance"] if before else 0.0
    res = emission.close_slots(N, now=SLOT0 + SLOT * 0.5, log=lambda *_: None)
    assert res["slots"] == 0 and res["paid"] == 0.0
    after = models.get_ledger("openslot")
    assert (after["balance"] if after else 0.0) == before


def test_the_daily_cap_bounds_a_scarcity_spike():
    _clear()
    original = config.EMISSION_DAILY_CAP_NRN
    config.EMISSION_DAILY_CAP_NRN = 1.5
    try:
        for i in range(10):
            _attend(f"n{i}", 0, 9)
        res = emission.close_slots(N, now=SLOT0 + SLOT * 1.5, log=lambda *_: None)
        assert res["capped"] is True
        assert res["paid"] <= 1.5 + 1e-9
    finally:
        config.EMISSION_DAILY_CAP_NRN = original


def test_zero_rows_are_reported_not_silently_dropped():
    """"I was up all night and earned nothing" is the question an operator will ask; a payout
    log that omits the misses cannot answer it."""
    _clear()
    _attend("good", 0, 9, poc=True)
    _attend("faker", 0, 9, poc=False)
    plan = emission.plan_slot(models.unpaid_attendance(SLOT0 + SLOT), N)
    assert len(plan) == 2
    assert {p["node_id"] for p in plan} == {"good", "faker"}


# --------------------------------------------------------------------------- #
# Attendance accrual from the real heartbeat path
# --------------------------------------------------------------------------- #
def test_heartbeat_accrues_attendance_and_caps_a_long_absence():
    """A node that vanished for hours must not bank the gap as presence on its first beat."""
    _clear()
    models.register_node("hb", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-hb", trusted=True)
    import time as _t
    with models._db() as c:                      # away for an hour, then one beat
        c.execute("UPDATE nodes SET last_seen=? WHERE node_id='hb'", (_t.time() - 3600.0,))
    models.touch_node("hb")
    with models._db() as c:
        row = c.execute("SELECT seconds_online FROM attendance WHERE node_id='hb'").fetchone()
    assert row is not None
    assert row["seconds_online"] <= config.PING_INTERVAL_S * 2 + 1e-6, \
        "an absence must not be credited as presence"


def test_coverage_report_names_the_uncovered_layers():
    _clear()
    models.register_node("a", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-a", trusted=True)
    rep = emission.coverage_report(models.list_nodes(), N)
    assert rep["uncovered_layers"], "10..27 are uncovered and must be reported"
    assert rep["uncovered_multiplier"] == config.EMISSION_SCARCITY_MAX
    assert rep["blocks"][0]["block"] == [0, 9]


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
