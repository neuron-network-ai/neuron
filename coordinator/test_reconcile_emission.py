"""coordinator/test_reconcile_emission.py — the emission reconciliation ([P40] item 1).

Two kinds of test here, and the split is deliberate.

**Against the real code.** A database is built by driving `models` and `emission.close_slots`
exactly as production does — register, attend, settle — and the reconciler is then pointed at
the resulting FILE. If the reconciler and the coordinator disagree about what a correct ledger
looks like, one of them is wrong, and this is the only test that can tell.

**Against hand-built databases.** Every discrepancy the reconciler claims to detect is written
into a database directly, because the coordinator cannot produce these states on purpose and a
detector nobody has seen fire is not a detector. A check that has only ever returned "clean" is
exactly what [P40] is about.

Plus the equivalence tests that keep the duplicated pricing honest: `reconcile_emission` carries
its own copy of `plan_slot` and of the emission constants so it can run against a snapshot
without importing the coordinator's live config, and copies drift. These pin them.

Run:  python -m coordinator.test_reconcile_emission     (from repo root)
"""
import os
import random
import sqlite3
import tempfile

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, emission, genesis, models, reconcile_emission as R

models.init_db()
genesis.seed_genesis()

DB = config.DB_PATH
SLOT = float(config.SLOT_SECONDS)
T0 = 1_800_000_000.0
SLOT0 = models.slot_start_for(T0)
N = config.TOTAL_LAYERS
PARAMS = dict(R.DEFAULTS)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _clear():
    """Back to a freshly-seeded network: no nodes, no attendance, every bucket at its genesis
    balance. Restoring ALL of them, not just the pool, so a test that parks NRN elsewhere to
    empty the pool cannot leave the supply invariant broken for the next one."""
    with models._db() as c:
        c.execute("DELETE FROM nodes")
        c.execute("DELETE FROM attendance")
        c.execute("DELETE FROM ledger WHERE account_type='node' OR account_type='wallet'")
        for bucket, balance in BUCKETS.items():
            c.execute("UPDATE ledger SET balance=? WHERE node_id=?", (balance, bucket))


def _attend(node_id, lo, hi, frac=1.0, poc=True, slot=SLOT0):
    models.register_node(node_id, "1.1.1.1", 50999, lo, hi, 8, 16, f"tok-{node_id}",
                         trusted=True)
    with models._db() as c:
        c.execute("INSERT OR REPLACE INTO attendance "
                  "(node_id, slot_start, seconds_online, block_start, block_end, poc_ok) "
                  "VALUES (?,?,?,?,?,?)",
                  (node_id, slot, SLOT * frac, lo, hi, 1 if poc else 0))


def _read(db=None):
    con = sqlite3.connect(f"file:{db or DB}?mode=ro", uri=True)
    try:
        return R.read_all(con)
    finally:
        con.close()


def _run_checks(db=None, seed=None, now=None, **kw):
    return R.reconcile(_read(db), PARAMS, seed=seed if seed is not None else SEED,
                       now=now if now is not None else T0 + SLOT * 2, **kw)


def _codes(res, level=None):
    return {f["code"] for f in res["findings"] if level is None or f["level"] == level}


SEED = models.get_ledger(config.GENESIS_BUCKETS_EMISSION_ID)["balance"]
with models._db() as _c:
    BUCKETS = {r["node_id"]: r["balance"] for r in
               _c.execute("SELECT node_id, balance FROM ledger WHERE account_type='bucket'")}


# --------------------------------------------------------------------------- #
# A hand-built ledger, for states the coordinator cannot produce on purpose
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE ledger (node_id TEXT PRIMARY KEY, balance REAL NOT NULL DEFAULT 0,
    total_earned REAL NOT NULL DEFAULT 0, account_type TEXT NOT NULL DEFAULT 'node');
CREATE TABLE attendance (node_id TEXT NOT NULL, slot_start REAL NOT NULL,
    seconds_online REAL NOT NULL DEFAULT 0, block_start INTEGER, block_end INTEGER,
    poc_ok INTEGER NOT NULL DEFAULT 0, reward REAL, paid_at REAL,
    PRIMARY KEY (node_id, slot_start));
CREATE TABLE nodes (node_id TEXT PRIMARY KEY, owner_wallet_id TEXT);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _fake(rows, ledger=None, nodes=None, pool=None, settings=None, slack=0.0):
    """A database with exactly the rows given. `rows` are dicts of attendance columns; a
    `__rest__` bucket absorbs whatever makes the supply come to exactly 1,000,000,000, so a
    test only has to state the part it cares about. `slack` breaks that on purpose."""
    path = tempfile.mktemp(suffix=".db")
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    paid = sum(float(r.get("reward") or 0.0) for r in rows if r.get("paid_at") is not None)
    led = list(ledger or [])
    known = {r[0] for r in led}
    for r in rows:
        if r["node_id"] not in known:
            led.append((r["node_id"], 0.0, 0.0, "node"))
            known.add(r["node_id"])
    pool_balance = R.KNOWN_LIVE_SEED - paid if pool is None else pool
    rest = R.TOTAL_SUPPLY - R.KNOWN_LIVE_SEED
    led += [(R.EMISSION_POOL, pool_balance, 0.0, "bucket"),
            ("__rest__", rest + paid - sum(float(x[1]) for x in led) + slack, 0.0, "bucket")]
    con.executemany("INSERT INTO ledger (node_id, balance, total_earned, account_type) "
                    "VALUES (?,?,?,?)", led)
    con.executemany(
        "INSERT INTO attendance (node_id, slot_start, seconds_online, block_start, "
        "block_end, poc_ok, reward, paid_at) VALUES (?,?,?,?,?,?,?,?)",
        [(r["node_id"], r["slot_start"], r.get("seconds_online", SLOT),
          r.get("block_start", 0), r.get("block_end", 9), r.get("poc_ok", 1),
          r.get("reward"), r.get("paid_at")) for r in rows])
    # `is not None`, not `or` -- nodes={} is a test asking for attendance with no node rows.
    con.executemany("INSERT INTO nodes (node_id, owner_wallet_id) VALUES (?,?)",
                    list((nodes if nodes is not None else
                          {n: None for n in known if not n.startswith("__")}).items()))
    con.executemany("INSERT INTO settings (key, value) VALUES (?,?)",
                    list((settings or {}).items()))
    con.commit()
    con.close()
    return path


def _row(node_id, reward, paid=True, **kw):
    r = {"node_id": node_id, "slot_start": SLOT0, "seconds_online": SLOT,
         "block_start": 0, "block_end": 9, "poc_ok": 1, "reward": reward,
         "paid_at": (SLOT0 + SLOT + 1) if paid else None}
    r.update(kw)
    return r


def _check_fake(rows, seed=R.KNOWN_LIVE_SEED, **kw):
    unknown = set(kw) - {"ledger", "nodes", "pool", "settings", "slack",
                         "replay", "total_layers", "now"}
    assert not unknown, f"_check_fake got {unknown} and would have silently ignored them"
    path = _fake(rows, **{k: v for k, v in kw.items()
                          if k in ("ledger", "nodes", "pool", "settings", "slack")})
    return R.reconcile(_read(path), PARAMS, seed=seed,
                       now=kw.get("now", SLOT0 + SLOT * 2),
                       replay=kw.get("replay", False),
                       total_layers=kw.get("total_layers"))


# --------------------------------------------------------------------------- #
# The copies cannot drift
# --------------------------------------------------------------------------- #
def test_constants_match_the_coordinators():
    assert R.DEFAULTS["slot_seconds"] == config.SLOT_SECONDS
    assert R.DEFAULTS["base_nrn_per_hour"] == config.EMISSION_BASE_NRN_PER_HOUR
    assert R.DEFAULTS["scarcity_max"] == config.EMISSION_SCARCITY_MAX
    assert R.DEFAULTS["target_replicas"] == config.EMISSION_TARGET_REPLICAS
    assert R.DEFAULTS["daily_cap"] == config.EMISSION_DAILY_CAP_NRN
    assert R.DEFAULTS["min_attendance_frac"] == config.SLOT_MIN_ATTENDANCE_FRAC
    assert R.DEFAULTS["total_layers"] == config.TOTAL_LAYERS


def test_replay_agrees_with_plan_slot_on_randomised_slots():
    """The reconciler prices rows with its own copy of plan_slot. If the two ever disagree the
    replay leg is testing the copy, not the coordinator."""
    rnd = random.Random(20260817)
    for trial in range(200):
        rows = []
        for i in range(rnd.randint(1, 6)):
            lo = rnd.choice([0, 10, 19, 27])
            rows.append({"node_id": f"n{i}", "slot_start": SLOT0,
                         "seconds_online": SLOT * rnd.choice([0.0, 0.2, 0.5, 0.75, 1.0, 1.4]),
                         "block_start": lo, "block_end": rnd.choice([lo, lo + 8, N + 4]),
                         "poc_ok": rnd.choice([0, 1])})
        mine = R.replay_slot([dict(r) for r in rows], N, PARAMS)
        theirs = emission.plan_slot([dict(r) for r in rows], N)
        assert [e["reward"] for e in mine] == [e["reward"] for e in theirs], \
            f"trial {trial}: rewards differ\n{mine}\n{theirs}"
        assert [e["replicas"] for e in mine] == [e["replicas"] for e in theirs]
        assert [e["reason"] for e in mine] == [e["reason"] for e in theirs]


# --------------------------------------------------------------------------- #
# Against the real coordinator
# --------------------------------------------------------------------------- #
def test_a_real_settled_sweep_reconciles():
    """The load-bearing test: build the ledger the way production does, then reconcile it."""
    _clear()
    _attend("n1", 0, 9)
    _attend("n2", 10, 27)
    _attend("n3", 10, 27, frac=0.75)
    res = emission.close_slots(N, now=T0 + SLOT + 1, log=lambda *_: None)
    assert res["nodes"] == 3, res

    out = _run_checks(replay=True)
    assert out["ok"], [f for f in out["findings"] if f["level"] == "error"]
    assert "pool-drop-matches" in _codes(out)
    assert "emission-arrived" in _codes(out)
    assert "replay-matches" in _codes(out)
    # A is the sum, B is the pool's drop, and they are the same number.
    assert abs(float(out["a"]["settled_sum"]) - float(out["b"]["drop"])) < 1e-9
    assert float(out["a"]["settled_sum"]) > 0


def test_zero_paying_rows_still_reconcile():
    """Presence without proof-of-compute settles at zero. Nothing leaves the pool, and the
    reconciler must call that reconciled rather than 'nothing happened'."""
    _clear()
    _attend("n1", 0, 9, poc=False)
    emission.close_slots(N, now=T0 + SLOT + 1, log=lambda *_: None)
    out = _run_checks(replay=True)
    assert out["ok"], out["findings"]
    assert out["a"]["rows_settled"] == 1 and out["a"]["rows_paying"] == 0
    assert float(out["a"]["settled_sum"]) == 0.0
    assert float(out["b"]["drop"]) == 0.0


def test_an_exhausted_pool_leaves_a_row_that_still_reconciles():
    """The bug this reconciliation was written to look for, now fixed and pinned here.

    `close_slots` claims the row before it moves the money, then walks the reward back if the
    transfer fails. The walk-back used to be `settle_attendance(..., 0.0)`, whose UPDATE carries
    `WHERE paid_at IS NULL` — falsified by the claim four lines earlier. So the row kept its
    full reward while the pool paid nothing, permanently, and `emitted_since` counted money that
    never moved. `models.void_settlement` is the fix; this drives it with the real code."""
    _clear()
    _attend("n1", 0, 9)
    # Park the pool's balance in another bucket rather than deleting it, so this scenario has
    # an empty pool AND an intact 1,000,000,000 supply — otherwise the broken invariant is the
    # fixture's doing and would mask what the test is actually asking about.
    with models._db() as c:
        c.execute("UPDATE ledger SET balance=balance+? WHERE node_id=?",
                  (SEED, config.GENESIS_BUCKETS_FOUNDER_ID))
        c.execute("UPDATE ledger SET balance=0 WHERE node_id=?",
                  (config.GENESIS_BUCKETS_EMISSION_ID,))
    logs = []
    res = emission.close_slots(N, now=T0 + SLOT + 1, log=logs.append)

    with models._db() as c:
        row = dict(c.execute("SELECT * FROM attendance WHERE node_id='n1'").fetchone())
    assert row["paid_at"] is not None, "the hour must stay claimed, or the next sweep retries it"
    assert row["reward"] == 0, f"nothing was paid, so the row must say 0 — got {row['reward']}"
    assert models.emitted_since(0.0) == 0.0, \
        "the daily cap must not count money that never left the pool"
    assert res["unpaid"] == 1 and "POOL EXHAUSTED" in logs[0]

    out = _run_checks(seed=0.0)          # this run's pool started drained, so its seed is 0
    assert out["ok"], [f for f in out["findings"] if f["level"] == "error"]


def test_it_would_still_catch_a_settled_row_the_pool_never_paid_for():
    """The detector, kept honest now that the coordinator no longer produces this shape: a row
    marked paid at a price that never left the pool, written by hand."""
    out = _check_fake([_row("n1", 3.0)], ledger=[("n1", 0.0, 0.0, "node")],
                      pool=R.KNOWN_LIVE_SEED)
    assert not out["ok"], "a recorded payment that never left the pool must not reconcile"
    assert "pool-drop-mismatch" in _codes(out, "error")
    assert "emission-did-not-arrive" in _codes(out, "error")


def test_the_settled_row_is_frozen_so_replay_is_sound():
    """The replay leg rests on settlement freezing its inputs. Assert it directly: a heartbeat
    and a proof-of-compute after settlement must not move the row."""
    _clear()
    _attend("n1", 0, 9, frac=0.6)
    emission.close_slots(N, now=T0 + SLOT + 1, log=lambda *_: None)
    with models._db() as c:
        before = dict(c.execute("SELECT * FROM attendance WHERE node_id='n1'").fetchone())
    models.touch_node("n1")
    models.mark_slot_poc("n1", ts=T0)
    with models._db() as c:
        after = dict(c.execute("SELECT * FROM attendance WHERE node_id='n1' "
                               "AND slot_start=?", (SLOT0,)).fetchone())
    assert before == after, f"a settled row moved under a later heartbeat:\n{before}\n{after}"


# --------------------------------------------------------------------------- #
# Every discrepancy the reconciler claims to detect
# --------------------------------------------------------------------------- #
def test_clean_hand_built_ledger_reconciles():
    out = _check_fake([_row("n1", 1.0), _row("n2", 1.0)],
                      ledger=[("n1", 1.0, 1.0, "node"), ("n2", 1.0, 1.0, "node")])
    assert out["ok"], out["findings"]


def test_pool_drop_mismatch_is_caught_in_both_directions():
    base = [_row("n1", 1.0)]
    led = [("n1", 1.0, 1.0, "node")]
    over = _check_fake(base, ledger=led, pool=R.KNOWN_LIVE_SEED - 5.0)   # pool paid more
    under = _check_fake(base, ledger=led, pool=R.KNOWN_LIVE_SEED)        # pool paid nothing
    assert "pool-drop-mismatch" in _codes(over, "error")
    assert "pool-drop-mismatch" in _codes(under, "error")


def test_settled_with_null_reward_is_caught():
    out = _check_fake([_row("n1", None)])
    assert "settled-with-null-reward" in _codes(out, "error")


def test_reward_without_paid_at_is_caught():
    out = _check_fake([_row("n1", 1.0, paid=False)])
    assert "reward-without-paid-at" in _codes(out, "error")


def test_negative_reward_is_caught():
    out = _check_fake([_row("n1", -1.0)], ledger=[("n1", 0.0, 0.0, "node")])
    assert "negative-reward" in _codes(out, "error")


def test_reward_above_the_ceiling_is_caught():
    """base x scarcity_max is 3.0 NRN. No attendance fraction and no replica count can beat it,
    so 4.0 is arithmetically impossible however the slot was priced."""
    out = _check_fake([_row("n1", 4.0)], ledger=[("n1", 4.0, 4.0, "node")])
    assert "reward-over-ceiling" in _codes(out, "error")


def test_paying_a_row_that_does_not_qualify_is_caught():
    """Presence paid without proof-of-compute — rule 2 of emission.py, checked against the
    data rather than the formula."""
    no_poc = _check_fake([_row("n1", 1.0, poc_ok=0)], ledger=[("n1", 1.0, 1.0, "node")])
    thin = _check_fake([_row("n1", 1.0, seconds_online=SLOT * 0.1)],
                       ledger=[("n1", 1.0, 1.0, "node")])
    no_block = _check_fake([_row("n1", 1.0, block_start=None, block_end=None)],
                           ledger=[("n1", 1.0, 1.0, "node")])
    for out in (no_poc, thin, no_block):
        assert "paid-unqualified" in _codes(out, "error"), out["findings"]


def test_emission_that_never_arrived_is_caught():
    """The pool dropped, the row says paid, and the payee's total_earned does not cover it."""
    out = _check_fake([_row("n1", 2.0)], ledger=[("n1", 2.0, 0.5, "node")])
    assert "emission-did-not-arrive" in _codes(out, "error")


def test_a_payee_with_no_ledger_row_at_all_is_caught():
    path = _fake([_row("ghost", 1.0)], ledger=[("__placeholder__", 0.0, 0.0, "node")])
    con = sqlite3.connect(path)
    con.execute("DELETE FROM ledger WHERE node_id='ghost'")
    con.commit()
    con.close()
    out = R.reconcile(_read(path), PARAMS, seed=R.KNOWN_LIVE_SEED, now=SLOT0 + SLOT * 2)
    assert "emission-did-not-arrive" in _codes(out, "error")


def test_a_larger_residual_than_emission_is_fine():
    """total_earned above the emission paid is per-request earnings, a faucet grant or a sweep.
    Expected, and must not be reported as a discrepancy."""
    out = _check_fake([_row("n1", 1.0)], ledger=[("n1", 40.0, 40.0, "node")])
    assert out["ok"], out["findings"]
    assert out["c"]["checked"][0]["residual"] > 0


def test_the_daily_cap_is_checked_against_paid_at():
    rows, at = [], SLOT0 + SLOT
    for i in range(9):
        rows.append(_row(f"n{i}", 3.0, slot_start=SLOT0 + SLOT * i, paid_at=at + i))
    out = R.reconcile(_read(_fake(rows, ledger=[(f"n{i}", 3.0, 3.0, "node")
                                                for i in range(9)])),
                      dict(PARAMS, daily_cap=20.0), seed=R.KNOWN_LIVE_SEED,
                      now=SLOT0 + SLOT * 20)
    assert "daily-cap-exceeded" in _codes(out, "error")
    assert float(out["cap"]["max_window"]) == 27.0


def test_settlements_a_day_apart_do_not_stack_into_one_window():
    rows = [_row("n1", 3.0, slot_start=SLOT0, paid_at=SLOT0 + SLOT),
            _row("n2", 3.0, slot_start=SLOT0, paid_at=SLOT0 + SLOT + 90000.0)]
    out = R.reconcile(_read(_fake(rows, ledger=[("n1", 3.0, 3.0, "node"),
                                                ("n2", 3.0, 3.0, "node")])),
                      dict(PARAMS, daily_cap=5.0), seed=R.KNOWN_LIVE_SEED,
                      now=SLOT0 + SLOT + 200000.0)
    assert "daily-cap-exceeded" not in _codes(out, "error")
    assert float(out["cap"]["max_window"]) == 3.0


def test_a_stopped_sweep_is_caught():
    """Unsettled rows in slots that closed long ago. Invisible in the logs — an idle sweep
    returns early and says nothing — which is exactly why this is worth asserting."""
    out = _check_fake([_row("n1", None, paid=False, slot_start=SLOT0)])
    assert "sweep-backlog" in _codes(out, "error")


def test_the_slot_that_just_closed_is_not_a_backlog():
    path = _fake([_row("n1", None, paid=False, slot_start=SLOT0)])
    out = R.reconcile(_read(path), PARAMS, seed=R.KNOWN_LIVE_SEED, now=SLOT0 + SLOT + 5)
    assert "sweep-backlog" not in _codes(out, "error"), out["findings"]


def test_a_broken_supply_invariant_is_caught():
    out = _check_fake([_row("n1", 1.0)], ledger=[("n1", 1.0, 1.0, "node")], slack=500.0)
    assert "supply-invariant-broken" in _codes(out, "error")


# --------------------------------------------------------------------------- #
# The things it must refuse to claim
# --------------------------------------------------------------------------- #
def test_an_inferred_seed_is_reported_as_inferred_not_as_agreement():
    """Without a seed, leg B is A by construction. Saying 'reconciled' there would be the
    check lying about its own strength."""
    out = _check_fake([_row("n1", 1.0)], ledger=[("n1", 1.0, 1.0, "node")],
                      pool=600_000_000.0 - 1.0, seed=None)
    assert "seed-inferred" in _codes(out, "warn"), out["findings"]
    assert "pool-drop-matches" not in _codes(out)
    assert out["b"]["seed_source"] == "inferred"


def test_an_inferred_seed_still_catches_a_pool_that_lost_more_than_emission_explains():
    """The bound that survives without a seed: 600,000,000 - implied_seed is what genesis
    backdated, and it cannot exceed what the nodes have ever earned."""
    # A pool 5,000,000 NRN below its allocation with only 1 NRN of settled emission implies
    # genesis backdated 5,000,000 — against nodes that have ever earned 1.0 in total.
    out = _check_fake([_row("n1", 1.0)],
                      ledger=[("n1", 1.0, 1.0, "node")],
                      pool=600_000_000.0 - 5_000_000.0 - 1.0, seed=None)
    assert "implied-seed-exceeds-earnings" in _codes(out, "error")


def test_a_pool_above_its_allocation_is_impossible_and_says_so():
    out = _check_fake([_row("n1", 1.0)], ledger=[("n1", 1.0, 1.0, "node")],
                      pool=600_000_100.0, seed=None)
    assert "implied-seed-impossible" in _codes(out, "error")


def test_an_owned_node_is_ambiguous_rather_than_assumed():
    """[P39] phase 3 pays the owner, decided at settle time, and nothing records when the link
    was made. Guessing would be worse than declining."""
    out = _check_fake([_row("n1", 1.0)],
                      ledger=[("n1", 0.0, 0.0, "node"), ("w_x", 1.0, 1.0, "wallet")],
                      nodes={"n1": "w_x"})
    assert "payee-ambiguous" in _codes(out, "warn")
    assert "emission-did-not-arrive" not in _codes(out, "error"), \
        "n1's own total_earned is 0, but the money went to its owner — not a shortfall"


def test_attendance_outliving_its_node_is_reported_not_dropped():
    path = _fake([_row("n1", 1.0)], ledger=[("n1", 1.0, 1.0, "node")], nodes={})
    out = R.reconcile(_read(path), PARAMS, seed=R.KNOWN_LIVE_SEED, now=SLOT0 + SLOT * 2)
    assert "attendance-without-node" in _codes(out, "warn")
    # Live 2026-08-17: agent-bhpc012104-82cbee, 8.686871 NRN. Naming the amount is not enough
    # to act on -- where it LANDED is the question, so the finding reads the ledger row.
    finding = next(f for f in out["findings"] if f["code"] == "attendance-without-node")
    assert "1.000000 NRN" in finding["extra"]["n1"]


def test_the_reasons_settled_hours_earned_nothing_are_counted():
    """198 of 316 live node-hours paid zero and the first run could not say why. Every one is
    emission refusing to pay for presence — but which gate, and whose machine, is the question
    an operator asks, so it is counted per code and per node."""
    out = _check_fake([
        _row("n1", 0.0, poc_ok=0),
        _row("n2", 0.0, poc_ok=0, slot_start=SLOT0),
        _row("n1", 0.0, seconds_online=SLOT * 0.1, slot_start=SLOT0 + SLOT),
        _row("n1", 0.0, block_start=90, block_end=95, slot_start=SLOT0 + SLOT * 2),
        _row("n3", 1.0),
    ], ledger=[("n1", 0.0, 0.0, "node"), ("n2", 0.0, 0.0, "node"), ("n3", 1.0, 1.0, "node")],
        now=SLOT0 + SLOT * 4)
    assert out["ok"], out["findings"]
    assert out["rows"]["zero_by_code"] == {"no-proof-of-compute": 2,
                                           "below-attendance-floor": 1,
                                           "no-block-of-serving-model": 1}
    assert {k: v["n"] for k, v in out["rows"]["zero_by_node"]["n1"].items()} == {
        "no-proof-of-compute": 1, "below-attendance-floor": 1,
        "no-block-of-serving-model": 1}
    assert "n3" not in out["rows"]["zero_by_node"], "a paid hour is not a zero"
    assert "why-hours-earned-nothing" in _codes(out, "info")


def test_refused_hours_carry_the_window_they_fell_in():
    """A count with no dates cannot be acted on: 64 hours with no proof-of-compute is an
    incident already closed if they all predate [P37]'s fix, and money being lost today if
    they do not. Live 2026-08-17 is what raised this."""
    out = _check_fake([
        _row("n1", 0.0, poc_ok=0, slot_start=SLOT0),
        _row("n1", 0.0, poc_ok=0, slot_start=SLOT0 + SLOT * 5),
        _row("n2", 0.0, poc_ok=0, slot_start=SLOT0 + SLOT * 2),
    ], ledger=[("n1", 0.0, 0.0, "node"), ("n2", 0.0, 0.0, "node")],
        now=SLOT0 + SLOT * 8)
    seen = out["rows"]["zero_by_node"]["n1"]["no-proof-of-compute"]
    assert seen == {"n": 2, "first": SLOT0, "last": SLOT0 + SLOT * 5}
    assert R._span(seen) == "2 (2027-01-15 08:00 .. 2027-01-15 13:00 UTC)"
    # A single refused hour reads as an instant, not a zero-width range.
    assert R._span(out["rows"]["zero_by_node"]["n2"]["no-proof-of-compute"]) == \
        "1 (2027-01-15 10:00 UTC)"


def test_a_qualifying_hour_paid_zero_is_counted_apart_from_a_refused_one():
    """The cap and an exhausted pool zero a row that DID earn. Folding that in with the rows
    that failed a gate would report a payment problem as a proof-of-compute problem."""
    out = _check_fake([_row("n1", 0.0)], ledger=[("n1", 0.0, 0.0, "node")])
    assert out["rows"]["zero_by_code"] == {"qualified-but-paid-zero": 1}


def test_a_database_predating_emission_is_not_an_error():
    """backups-offbox/neuron-20260802-115534.db is exactly this shape."""
    path = tempfile.mktemp(suffix=".db")
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE ledger (node_id TEXT PRIMARY KEY, balance REAL, "
                      "total_earned REAL, account_type TEXT);")
    con.execute("INSERT INTO ledger VALUES ('__emission_pool__', ?, 0, 'bucket')",
                (R.KNOWN_LIVE_SEED,))
    con.commit()
    con.close()
    out = R.reconcile(_read(path), PARAMS, seed=None, now=T0)
    assert out["ok"]
    assert "no-attendance-table" in _codes(out, "info")


def test_the_tolerance_scales_with_the_pool_not_with_a_fixed_epsilon():
    """One ULP of a 6e8 float is ~1.2e-7, so a few hundred payments cannot be held to 1e-6."""
    assert R.ab_tolerance(600_000_000.0, 300) > 1e-6
    assert R.ab_tolerance(600_000_000.0, 300) < 1e-4
    assert R.ab_tolerance(600_000_000.0, 0) == 1e-6


# --------------------------------------------------------------------------- #
# Replay
# --------------------------------------------------------------------------- #
def test_replay_catches_a_reward_priced_for_the_wrong_replica_count():
    """Two nodes on the same block is 2 replicas, so the multiplier is 1.5 and each earns 1.5.
    A row carrying the lone-node price of 3.0 is a mispricing the sums cannot see — leg A and
    leg B agree perfectly, because the pool really did pay it."""
    rows = [_row("n1", 3.0), _row("n2", 1.5)]
    out = _check_fake(rows, ledger=[("n1", 3.0, 3.0, "node"), ("n2", 1.5, 1.5, "node")],
                      replay=True)
    assert "replay-mismatch" in _codes(out, "error")
    assert out["replay"]["mismatches"][0]["node_id"] == "n1"
    assert out["replay"]["mismatches"][0]["expected"] == 1.5


def test_replay_separates_a_capped_zero_from_a_mismatch():
    out = _check_fake([_row("n1", 0.0)], replay=True)
    assert not out["replay"]["mismatches"]
    assert out["replay"]["cap_explained"], "settled at 0 where pricing says it earned"
    assert out["ok"], "a capped zero is not a discrepancy"


def test_replay_flags_rows_whose_verdict_depends_on_the_serving_model():
    """total_layers is the one input settlement does not freeze. A block outside the assumed
    range is where a wrong assumption would silently change the answer, so it is named."""
    out = _check_fake([_row("n1", 1.0, block_start=30, block_end=35)],
                      ledger=[("n1", 1.0, 1.0, "node")], replay=True, total_layers=28)
    assert "replay-layer-sensitive" in _codes(out, "warn")
    ok36 = _check_fake([_row("n1", 3.0, block_start=30, block_end=35)],
                       ledger=[("n1", 3.0, 3.0, "node")], replay=True, total_layers=36)
    assert not ok36["replay"]["mismatches"], "at 36 layers the same row prices correctly"


# --------------------------------------------------------------------------- #
# It cannot write
# --------------------------------------------------------------------------- #
def test_the_script_has_no_write_path():
    """Read the module's syntax tree rather than grepping it: the docstring explains that there
    is deliberately no `--execute`, and a grep cannot tell that sentence from the flag.

    Three properties, each the way a write could actually get in — a connection that is not
    read-only, a commit, or an SQL statement that is not a SELECT."""
    import ast
    path = os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "reconcile_emission.py")
    tree = ast.parse(open(path, encoding="utf-8").read())

    connects = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("commit", "rollback"), \
                f"reconcile_emission.py calls .{node.attr}()"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "connect":
            connects += 1
            assert "mode=ro" in ast.unparse(node), f"not read-only: {ast.unparse(node)}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("execute", "executemany", "executescript"):
            sql = ast.unparse(node.args[0]) if node.args else ""
            assert sql.lstrip("'\"f").upper().startswith(("SELECT", "PRAGMA TABLE_INFO")), \
                f"reconcile_emission.py runs a non-SELECT statement: {sql}"
    assert connects == 1, "expected exactly one sqlite3.connect"
    assert "--execute" not in {n.value for n in ast.walk(tree)
                               if isinstance(n, ast.Constant) and isinstance(n.value, str)
                               and len(n.value) < 40}


def test_running_it_does_not_change_a_byte_of_the_database():
    """The property the mode=ro connection is there for, asserted rather than assumed."""
    import hashlib
    import io
    import contextlib
    import shutil
    src = _fake([_row("n1", 1.0)], ledger=[("n1", 1.0, 1.0, "node")])
    copy = tempfile.mktemp(suffix=".db")
    shutil.copyfile(src, copy)
    before = hashlib.sha256(open(copy, "rb").read()).hexdigest()
    with contextlib.redirect_stdout(io.StringIO()):
        R.main(["--db", copy, "--seed", repr(R.KNOWN_LIVE_SEED), "--replay",
                "--now", repr(SLOT0 + SLOT * 2)])
    assert hashlib.sha256(open(copy, "rb").read()).hexdigest() == before


def test_the_known_live_seed_is_what_the_backup_actually_holds():
    """KNOWN_LIVE_SEED is only usable as a seed because that backup predates emission. Both
    halves of that claim are checked here, so editing either the constant or the file is
    caught rather than quietly believed."""
    backup = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(R.__file__))),
                          "backups-offbox", "neuron-20260802-115534.db")
    if not os.path.exists(backup):
        return                                   # not every checkout carries the backups
    con = sqlite3.connect(f"file:{backup}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        pool = con.execute("SELECT balance FROM ledger WHERE node_id=?",
                           (R.EMISSION_POOL,)).fetchone()[0]
    finally:
        con.close()
    assert "attendance" not in tables, \
        "the backup has an attendance table, so it no longer predates emission"
    assert pool == R.KNOWN_LIVE_SEED, f"backup pool is {pool!r}, constant says {R.KNOWN_LIVE_SEED!r}"


def test_the_cli_returns_the_verdict_as_an_exit_status():
    _clear()
    _attend("n1", 0, 9)
    emission.close_slots(N, now=T0 + SLOT + 1, log=lambda *_: None)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["--db", DB, "--seed", repr(SEED), "--now", repr(T0 + SLOT * 2),
                     "--replay"])
    assert rc == 0, buf.getvalue()
    assert "RESULT: reconciled" in buf.getvalue()

    bad = _fake([_row("n1", 4.0)], ledger=[("n1", 4.0, 4.0, "node")])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = R.main(["--db", bad, "--seed", repr(R.KNOWN_LIVE_SEED),
                     "--now", repr(SLOT0 + SLOT * 2)])
    assert rc == 1, buf.getvalue()
    assert "reward-over-ceiling" in buf.getvalue()

    assert R.main(["--db", os.path.join(tempfile.gettempdir(), "nope-does-not-exist.db")]) == 2


def test_json_output_is_machine_readable():
    import io
    import json as _json
    import contextlib
    bad = _fake([_row("n1", 4.0)], ledger=[("n1", 4.0, 4.0, "node")])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.main(["--db", bad, "--seed", repr(R.KNOWN_LIVE_SEED), "--json",
                "--now", repr(SLOT0 + SLOT * 2)])
    payload = _json.loads(buf.getvalue())
    assert payload["ok"] is False
    assert payload["seed_source"] == "given"
    assert any(f["code"] == "reward-over-ceiling" for f in payload["findings"])


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
