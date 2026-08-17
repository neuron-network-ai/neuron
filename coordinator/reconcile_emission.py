"""coordinator/reconcile_emission.py — check the emission ledger against its own attendance rows.

    python coordinator/reconcile_emission.py --db /path/to/neuron.db
    python coordinator/reconcile_emission.py --db /path/to/neuron.db --replay
    python coordinator/reconcile_emission.py --db /path/to/neuron.db --json

**Read-only. It opens the database `mode=ro` and has no write path at all** — there is no
`--execute`, deliberately. A reconciliation that can also repair is a reconciliation nobody
can run casually against production, and running it casually against production is the entire
point ([P40] item 2: a periodic assertion, not a one-off).

Exit status is the answer: **0** reconciled, **1** discrepancy found, **2** could not run.
So it is usable directly from a health sweep or cron.

**Running it against a live coordinator.** The database is in WAL mode, so the recent
settlements live in `neuron.db-wal` until a checkpoint. Point this at the live path directly
(safe — it takes a read lock and the running coordinator is unaffected), or take a consistent
copy with `sqlite3 neuron.db ".backup snap.db"`. Do **not** reconcile a hand-copied `.db`
without its `-wal`: it will read a state the coordinator has already moved past, and report a
discrepancy that is the copy's, not the ledger's.

---

**Why this exists.** 262.89 NRN has been distributed by `emission.close_slots` and nothing has
ever compared what the ledger *holds* against what the attendance rows *say it should hold*.
`test_emission_slots.py` covers the economics properly, but on a fresh temp DB with fabricated
attendance: it proves the formula, never the data. The failure mode is silent by construction —
`total_earned` only ever grows, so an error does not appear as a spike, it compounds, and the
first symptom is an operator disputing a balance with no independent record to check them
against.

**The three numbers that must agree** ([P40] item 1). None of these re-tests the formula:

    A.  SUM(attendance.reward) over settled rows   — what emission RECORDED it paid
    B.  seed − ledger['__emission_pool__'].balance — what actually LEFT the pool
    C.  the rise in the payees' total_earned       — what actually ARRIVED

A and B are an exact identity: `emission.py:162` is the only writer in the entire codebase that
debits `__emission_pool__` (verified by grep, not assumed), and `close_slots` stamps the row and
moves the money in the same iteration. Any gap is money recorded-but-not-paid or paid-but-not-
recorded, and either one is a real defect.

**The seed is the awkward part, and it is worth being explicit about.** `genesis.seed_genesis`
computes `600,000,000 − already_minted` and stores the result nowhere, so "the drop in the
pool" is not directly computable from the database alone. Three ways out, in order of strength:

  1. `--seed` given by the operator;
  2. `settings['emission_pool_seed']`, which `genesis.py` now records — but only for databases
     seeded after that change, which does NOT include the live one;
  3. otherwise INFERRED as `pool_balance + settled_sum`, and then bounded rather than trusted:
     the implied `already_minted` must be ≥ 0 (`genesis.py` refuses to seed otherwise) and
     ≤ today's `SUM(total_earned) WHERE account_type='node'` (nothing ever decrements
     `total_earned`). A three-way check that quietly invented one of its three numbers would be
     worse than no check, so an inferred seed is reported as INFERRED and leg A↔B is downgraded
     to those bounds.

For the live database there is a fourth and better answer, which is why `--seed` exists:
`backups-offbox/neuron-20260802-115534.db` predates emission entirely (it has no `attendance`
table), so the pool balance it carries — **599999971.999972** — *is* the seed, and passing it
turns leg B back into an exact equality.

**Leg C cannot be exact, and the script says so instead of pretending.** The ledger has no
transactions table (`claim_node_earnings.py` says the same thing for the same reason), so a
single `total_earned` scalar cannot be decomposed into emission vs. per-request earnings vs.
faucet vs. an operator sweep. What IS assertible is a floor: an account must have earned at
least the emission it was paid. `total_earned < emission_paid` is an error — money that was
debited from the pool and never arrived. A residual above it is expected and is reported, not
flagged.

Two attribution hazards leg C handles rather than ignores:
  * **[P39] phase 3 pays the OWNER, not the machine** (`get_node_owner(node) or node`), decided
    at settle time. A node that has an owner recorded *today* may have been paid as itself
    yesterday, so its historical rows cannot be attributed either way. Those rows are reported
    as AMBIGUOUS and excluded from the floor check rather than silently assigned.
  * **A sweep credits `total_earned` twice.** `claim_node_earnings.py` mirrors `models.transfer`
    — the destination wallet's `total_earned` rises and the source node's does not fall — so
    summing `total_earned` across all accounts double-counts every swept NRN. Leg C therefore
    works per-payee and never over the whole ledger.

**And the checks that would actually catch a bug**, which the sums alone would not: rows settled
with no reward, negative rewards, rewards above the per-row ceiling, a payee with no ledger row,
a backlog of closed-but-unsettled slots (the sweep has stopped), rewards paid to rows that do
not qualify, and the rolling daily cap. `--replay` goes further and re-prices every settled slot
from its own frozen rows — sound because `touch_node` and `mark_slot_poc` both update only
`WHERE paid_at IS NULL`, so a settled row's inputs are frozen at the values the payment was
computed from.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from decimal import Decimal

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = os.path.join(HERE, "neuron.db")

EMISSION_POOL = "__emission_pool__"
COORDINATOR = "__coordinator__"
TOTAL_SUPPLY = 1_000_000_000

# The pool balance carried by backups-offbox/neuron-20260802-115534.db, which has no
# `attendance` table and therefore predates any emission payment. Offered by --seed-from-backup
# rather than baked in as a default: a hardcoded seed that is silently wrong turns every leg-B
# result into a confident lie.
KNOWN_LIVE_SEED = 599999971.999972

# Emission parameters. Duplicated from coordinator/config.py on purpose — this script must run
# against a SNAPSHOT on a machine where importing the coordinator would read the live NEURON_DB
# and the live env. `test_reconcile_emission.py` asserts these stay equal to config's, and that
# `replay_slot` stays equal to `emission.plan_slot`, so the copy cannot drift unnoticed.
DEFAULTS = {
    "slot_seconds": int(os.environ.get("NEURON_SLOT_SECONDS", "3600")),
    "base_nrn_per_hour": float(os.environ.get("NEURON_EMISSION_BASE", "1.0")),
    "scarcity_max": float(os.environ.get("NEURON_EMISSION_SCARCITY_MAX", "3.0")),
    "target_replicas": int(os.environ.get("NEURON_EMISSION_TARGET_REPLICAS", "3")),
    "daily_cap": float(os.environ.get("NEURON_EMISSION_DAILY_CAP", "5000.0")),
    "min_attendance_frac": float(os.environ.get("NEURON_SLOT_MIN_ATTENDANCE", "0.5")),
    "total_layers": int(os.environ.get("NEURON_TOTAL_LAYERS", "28")),
    "max_unaudited_slots": int(os.environ.get("NEURON_EMISSION_MAX_UNAUDITED_SLOTS", "6")),
}

# Below this a difference is float residue, not NRN. Leg A↔B needs more than this because the
# pool is a ~6e8 float that has been decremented once per payment; see `ab_tolerance`.
DUST = 1e-9


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def read_all(con):
    """Everything the checks need, in one pass. Missing tables are reported rather than raised:
    a database old enough to have no `attendance` table is a legitimate state (every backup
    before 2026-08-09 is one) and the right answer is "nothing to reconcile", not a traceback."""
    con.row_factory = sqlite3.Row
    have = {r["name"] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}

    ledger = [dict(r) for r in con.execute(
        "SELECT node_id, balance, total_earned, account_type FROM ledger")] \
        if "ledger" in have else []
    # `poc_excused` arrived by ALTER TABLE ([P47] cause 2), so a snapshot can legitimately have
    # the table without the column. Defaulted to 0 rather than skipped: every row settled before
    # it existed was priced on a proof-of-compute, which is exactly what 0 means.
    attendance = []
    if "attendance" in have:
        acols = {r["name"] for r in con.execute("PRAGMA table_info(attendance)")}
        excused = "poc_excused" if "poc_excused" in acols else "0 AS poc_excused"
        attendance = [dict(r) for r in con.execute(
            "SELECT node_id, slot_start, seconds_online, block_start, block_end, poc_ok, "
            f"{excused}, reward, paid_at FROM attendance ORDER BY slot_start, node_id")]
    settings = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM settings")} \
        if "settings" in have else {}
    # owner_wallet_id arrived by ALTER TABLE ([P39]); a snapshot older than that has the table
    # without the column, which is a different failure from having no table at all.
    owners = {}
    if "nodes" in have:
        cols = {r["name"] for r in con.execute("PRAGMA table_info(nodes)")}
        if "owner_wallet_id" in cols:
            owners = {r["node_id"]: r["owner_wallet_id"] for r in con.execute(
                "SELECT node_id, owner_wallet_id FROM nodes")}
        else:
            owners = {r["node_id"]: None for r in con.execute("SELECT node_id FROM nodes")}
    # Computed here because this is where the connection lives, and it is the only new fact the
    # checks below need that is not already on a row. Empty for any snapshot with no audit_slots
    # table, which reads as "every slot was audited" -- i.e. exactly the pre-[P47] behaviour, so
    # an older database reconciles the way it always did.
    runs = unaudited_runs(con, DEFAULTS) if "audit_slots" in have else {}
    # Machines known to be gone for good ([P39] item 5). Absent on a snapshot older than the
    # table, which reads as "no retirement was ever recorded" -- true of every such database,
    # and it simply leaves the orphan warning behaving exactly as it did before.
    retired = ({r["node_id"]: dict(r) for r in con.execute("SELECT * FROM retired_nodes")}
               if "retired_nodes" in have else {})
    return {"have": have, "ledger": ledger, "attendance": attendance,
            "settings": settings, "owners": owners, "unaudited_runs": runs,
            "retired": retired}


def dsum(values):
    """Sum as Decimal over the exact repr of each float. Adding a few hundred REAL rewards in
    float64 accumulates error of its own, and this check exists to measure somebody else's."""
    return sum((Decimal(repr(float(v))) for v in values), Decimal(0))


# --------------------------------------------------------------------------- #
# The pricing replay — a faithful copy of emission.plan_slot, pinned by test
# --------------------------------------------------------------------------- #
def scarcity_multiplier(replicas, params):
    if replicas <= 0:
        return params["scarcity_max"]
    return max(1.0, min(params["scarcity_max"],
                        params["target_replicas"] / float(replicas)))


def unaudited_runs(conn, params):
    """{slot_start: consecutive unaudited slots ending there} for every slot with attendance.

    Read from `audit_slots`, which is append-only history: a heartbeat can only ever be recorded
    for the slot it arrives in, so a CLOSED slot's audit record is as frozen as the attendance
    row itself. That is what makes this safe to consult from a leg whose soundness rests on
    replaying frozen inputs ([P40]).

    It refines the REASON a row earned nothing; it never decides whether one was owed. The
    payment justification stays `poc_excused`, written into the row by settlement. Reporting
    "no proof-of-compute" for an hour when the truth is "nobody was watching for two days" is
    [P47]'s own conflation reappearing in the reporting layer, which is the one place it would
    be least visible.
    """
    slot_seconds = float(params["slot_seconds"])
    limit = int(params.get("max_unaudited_slots", 6)) + 1
    try:
        audited = {round(float(r[0]), 6) for r in
                   conn.execute("SELECT slot_start FROM audit_slots")}
    except sqlite3.OperationalError:
        return {}                       # a snapshot predating the table -- nothing to say
    if not audited:
        return {}
    epoch = min(audited)
    slots = [float(r[0]) for r in
             conn.execute("SELECT DISTINCT slot_start FROM attendance ORDER BY slot_start")]
    out = {}
    for s in slots:
        if s < epoch:
            out[s] = 0                  # before the mechanism existed -- not an outage
            continue
        run, cur = 0, s
        while run < limit and cur >= epoch and round(cur, 6) not in audited:
            run += 1
            cur -= slot_seconds
        out[s] = run
    return out


def qualifies(row, total_layers, params, unaudited_run=0):
    """(ok, reason, code) — the three gates emission.plan_slot applies, against the row's own
    frozen values. Sound to replay because `touch_node` and `mark_slot_poc` both carry
    `WHERE paid_at IS NULL`, so settlement freezes the inputs alongside the output.

    `reason` is emission.plan_slot's wording, character for character, because the equivalence
    test compares them. `code` is this script's own addition: the reason strings interpolate a
    percentage, so they cannot be counted, and "how many node-hours failed WHICH gate" turned
    out to be the first question the live run raised."""
    frac = (row.get("seconds_online") or 0.0) / float(params["slot_seconds"])
    if frac < params["min_attendance_frac"]:
        return False, "attended %.0f%% of the slot, floor is %.0f%%" % (
            frac * 100, params["min_attendance_frac"] * 100), "below-attendance-floor"
    # `poc_excused` ([P47] cause 2): the slot was one the coordinator never audited, and the row
    # was paid on presence-plus-placement because the reason there is no proof is OURS, not the
    # node's. Read from the ROW rather than re-derived from `audit_slots`, which keeps this leg
    # working the way every other one does — from values settlement froze — and means the
    # reconciliation of a historical payment can never be changed by a later heartbeat.
    #
    # Without this the reconciler would report every excused hour as a reward paid to a row that
    # does not qualify, i.e. [P40]'s standing assertion would start alarming on [P47]'s fix. The
    # zero-reason counts keep it separate from a genuine miss, because "we did not look" and
    # "the node did not answer" are the whole distinction [P47] exists to draw.
    # Falls THROUGH to the block check rather than returning, mirroring plan_slot's order. An
    # excused hour skips only the proof-of-compute gate; being present during an outage says
    # nothing about whether the node held a block of the model actually being served, and an
    # early return here paid a node for holding layers 27-32 of a 28-layer model.
    if not row.get("poc_ok") and not row.get("poc_excused"):
        if unaudited_run:
            return (False,
                    "the network was not audited for %d consecutive slots, beyond the %d this "
                    "pays through — nobody checked, so nobody can say" % (
                        unaudited_run, int(params.get("max_unaudited_slots", 6))),
                    "network-not-audited")
        return (False, "no proof-of-compute challenge passed inside the slot",
                "no-proof-of-compute")
    lo, hi = row.get("block_start"), row.get("block_end")
    if lo is None or hi is None or not (0 <= lo <= hi < total_layers):
        return False, "held no block of the serving model", "no-block-of-serving-model"
    return True, None, None


def replay_slot(rows, total_layers, params, unaudited_run=0):
    """Re-price one slot from its own rows. Same shape as emission.plan_slot's output, and
    test_reconcile_emission asserts the two agree on randomised input — including across the
    unaudited threshold, which is where two independently-written copies would drift first."""
    slot_seconds = float(params["slot_seconds"])
    marks = []
    for r in rows:
        ok, reason, _code = qualifies(r, total_layers, params, unaudited_run)
        marks.append((r, ok, reason, (r.get("seconds_online") or 0.0) / slot_seconds))

    depth = {}
    for r, ok, _reason, _frac in marks:
        if ok:
            key = (r["block_start"], r["block_end"])
            depth[key] = depth.get(key, 0) + 1

    out = []
    for r, ok, reason, frac in marks:
        replicas = depth.get((r.get("block_start"), r.get("block_end")), 0)
        if not ok:
            out.append({"node_id": r["node_id"], "slot_start": r["slot_start"], "reward": 0.0,
                        "replicas": replicas, "multiplier": 0.0,
                        "attended_frac": round(frac, 4), "reason": reason})
            continue
        mult = scarcity_multiplier(replicas, params)
        reward = params["base_nrn_per_hour"] * min(frac, 1.0) * mult
        out.append({"node_id": r["node_id"], "slot_start": r["slot_start"],
                    "reward": round(reward, 6), "replicas": replicas,
                    "multiplier": round(mult, 4), "attended_frac": round(frac, 4),
                    "reason": None})
    return out


# --------------------------------------------------------------------------- #
# Legs
# --------------------------------------------------------------------------- #
def ab_tolerance(seed, n_payments):
    """How far apart legs A and B may be before it means something.

    Not a fixed epsilon. Leg B reads a ~6e8 float that has been decremented once per payment,
    and one ULP at that magnitude is ~1.2e-7 — so 300 payments can legitimately disagree with
    an exactly-summed leg A by ~3.6e-5, which a 1e-6 epsilon would report as a discrepancy on
    the very first honest run. Scale with the arithmetic actually performed instead."""
    return max(1e-6, float(n_payments) * math.ulp(float(seed or TOTAL_SUPPLY)))


def leg_a(attendance):
    """What emission RECORDED it paid, plus the row-shape violations that make the sum a lie."""
    settled = [r for r in attendance if r["paid_at"] is not None]
    unsettled = [r for r in attendance if r["paid_at"] is None]
    settled_null = [r for r in settled if r["reward"] is None]
    priced = [r for r in settled if r["reward"] is not None]
    return {
        "rows_total": len(attendance),
        "rows_settled": len(settled),
        "rows_unsettled": len(unsettled),
        "rows_paying": len([r for r in priced if float(r["reward"]) > 0]),
        "settled_sum": dsum(r["reward"] for r in priced),
        "settled_null_reward": settled_null,
        "reward_without_paid_at": [r for r in unsettled if r["reward"] is not None],
        "negative": [r for r in priced if float(r["reward"]) < 0],
        "unsettled_rows": unsettled,
    }


def leg_b(ledger, settled_sum, seed=None, node_total_earned=None):
    """What actually LEFT the pool. Returns the seed's provenance alongside the number, because
    an inferred seed makes this an inequality and a given one makes it an equality."""
    pool = next((r for r in ledger if r["node_id"] == EMISSION_POOL), None)
    balance = Decimal(repr(float(pool["balance"]))) if pool else None
    if balance is None:
        return {"pool_present": False, "balance": None, "seed": None, "seed_source": None,
                "drop": None, "implied_seed": None, "implied_already_minted": None}
    implied_seed = balance + settled_sum
    out = {
        "pool_present": True,
        "balance": balance,
        "implied_seed": implied_seed,
        # genesis: seed = 600,000,000 - already_minted. Invert it, and the result is a number
        # with two hard bounds even when the seed itself is unknown.
        "implied_already_minted": Decimal(TOTAL_SUPPLY) * Decimal("0.6") - implied_seed,
        "node_total_earned": node_total_earned,
    }
    if seed is None:
        out.update({"seed": implied_seed, "seed_source": "inferred",
                    "drop": settled_sum})       # inferred seed makes A==B true by construction
    else:
        s = Decimal(repr(float(seed)))
        out.update({"seed": s, "seed_source": "given", "drop": s - balance})
    return out


def leg_c(attendance, ledger, owners):
    """What actually ARRIVED. Per-payee, never in aggregate — see the module docstring on why
    summing total_earned across the ledger double-counts an operator sweep."""
    by_ledger = {r["node_id"]: r for r in ledger}
    paid_by_node, ambiguous, orphan_node = {}, {}, {}
    for r in attendance:
        if r["paid_at"] is None or r["reward"] is None or float(r["reward"]) <= 0:
            continue
        nid = r["node_id"]
        amt = Decimal(repr(float(r["reward"])))
        if nid not in owners:
            # The node row is gone (models.delete_node removes `nodes`, never `attendance` or
            # `ledger`), so whether this was paid to the machine or to an owner is unknowable.
            orphan_node[nid] = orphan_node.get(nid, Decimal(0)) + amt
        elif owners.get(nid):
            # An owner is recorded TODAY. set_node_owner only ever writes, so this row was paid
            # to the node if it predates the link and to the wallet if it follows it — and
            # nothing records when the link was made. Not attributable either way.
            ambiguous[nid] = ambiguous.get(nid, Decimal(0)) + amt
        else:
            paid_by_node[nid] = paid_by_node.get(nid, Decimal(0)) + amt

    shortfalls, checked = [], []
    for nid, amt in sorted(paid_by_node.items()):
        row = by_ledger.get(nid)
        if row is None:
            shortfalls.append({"account": nid, "emission_paid": amt, "total_earned": None,
                               "detail": "no ledger row at all — the transfer credited nobody"})
            continue
        te = Decimal(repr(float(row["total_earned"])))
        if te + Decimal(repr(DUST)) < amt:
            shortfalls.append({"account": nid, "emission_paid": amt, "total_earned": te,
                               "detail": "total_earned is below the emission this account was "
                                         "paid — money left the pool and did not arrive"})
        else:
            checked.append({"account": nid, "emission_paid": amt, "total_earned": te,
                            "residual": te - amt})
    return {"checked": checked, "shortfalls": shortfalls,
            "ambiguous": ambiguous, "orphan_node": orphan_node,
            "attributable_sum": sum(paid_by_node.values(), Decimal(0))}


# --------------------------------------------------------------------------- #
# Integrity checks
# --------------------------------------------------------------------------- #
def _tally(by_code, by_node, node_id, code, slot_start):
    """Count a refused hour, and remember WHEN. The count alone is not actionable: 64 hours
    with no proof-of-compute is an incident already fixed if they all predate 2026-08-10
    ([P37] excluded flagged nodes from the verifier's own re-check list, so they could never
    earn again), and money a volunteer is losing today if they do not."""
    by_code[code] = by_code.get(code, 0) + 1
    per_node = by_node.setdefault(node_id, {})
    seen = per_node.setdefault(code, {"n": 0, "first": slot_start, "last": slot_start})
    seen["n"] += 1
    seen["first"] = min(seen["first"], slot_start)
    seen["last"] = max(seen["last"], slot_start)


def _span(seen):
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.gmtime(float(t)))
    if seen["first"] == seen["last"]:
        return f"{seen['n']} ({fmt(seen['first'])} UTC)"
    return f"{seen['n']} ({fmt(seen['first'])} .. {fmt(seen['last'])} UTC)"


def check_rows(attendance, params, total_layers, runs=None):
    """Per-row checks that a sum cannot see. `over_ceiling` is the useful one: frac ≤ 1 and the
    multiplier ≤ scarcity_max, so no honest row can exceed base × cap however scarce the slot."""
    ceiling = params["base_nrn_per_hour"] * params["scarcity_max"]
    over_ceiling, paid_unqualified, zeroed_qualified = [], [], []
    # Why settled node-hours earned nothing, per gate and per node. Not a discrepancy check --
    # every one of these is emission working as designed -- but it is the number an operator
    # actually asks about ("I was up all night and earned nothing"), and the first live run
    # could not answer it: 198 of 316 settled hours paid zero with no way to see which gate.
    zero_by_code, zero_by_node = {}, {}
    for r in attendance:
        if r["paid_at"] is None or r["reward"] is None:
            continue
        reward = float(r["reward"])
        if reward > ceiling + DUST:
            over_ceiling.append({"row": r, "ceiling": ceiling})
        ok, reason, code = qualifies(r, total_layers, params,
                                     (runs or {}).get(float(r["slot_start"]), 0))
        if reward > 0 and not ok:
            paid_unqualified.append({"row": r, "reason": reason})
        elif reward <= 0 and ok:
            # Not an error on its own: the daily cap zeroes a qualifying row deliberately, and
            # so does a pool that cannot pay. Reported so the count can be explained, not
            # flagged as a discrepancy.
            zeroed_qualified.append({"row": r})
            _tally(zero_by_code, zero_by_node, r["node_id"], "qualified-but-paid-zero",
                   r["slot_start"])
        elif reward <= 0:
            _tally(zero_by_code, zero_by_node, r["node_id"], code, r["slot_start"])
    return {"over_ceiling": over_ceiling, "paid_unqualified": paid_unqualified,
            "zeroed_qualified": zeroed_qualified,
            "zero_by_code": zero_by_code, "zero_by_node": zero_by_node}


def check_daily_cap(attendance, params):
    """The heaviest rolling 24h window of settlements, against EMISSION_DAILY_CAP_NRN.

    Windowed on `paid_at` rather than `slot_start` because that is what `emitted_since` — the
    figure close_slots actually checks the cap against — reads. A backlog settled in one sweep
    is therefore correctly counted as one day's spend."""
    priced = sorted(((float(r["paid_at"]), float(r["reward"])) for r in attendance
                     if r["paid_at"] is not None and r["reward"] is not None
                     and float(r["reward"]) > 0), key=lambda t: t[0])
    worst, worst_at, i = Decimal(0), None, 0
    running = Decimal(0)
    for t, amt in priced:
        running += Decimal(repr(amt))
        while priced[i][0] <= t - 86400.0:
            running -= Decimal(repr(priced[i][1]))
            i += 1
        if running > worst:
            worst, worst_at = running, t
    return {"max_window": worst, "at": worst_at, "cap": params["daily_cap"],
            "exceeded": worst > Decimal(repr(params["daily_cap"])) + Decimal(repr(DUST))}


def check_sweep_backlog(attendance, params, now):
    """Closed slots nobody settled. One is normal — the slot that just closed is settled on the
    next sweep. A backlog means close_slots is not running, which is invisible in the logs
    because a sweep with nothing to do is deliberately silent."""
    slot = float(params["slot_seconds"])
    current = float(int(now // slot) * slot)
    stale = [r for r in attendance
             if r["paid_at"] is None and float(r["slot_start"]) < current - slot]
    return {"stale": stale, "current_slot": current}


def check_supply(ledger):
    total = dsum(r["balance"] for r in ledger)
    return {"total": total, "ok": abs(total - Decimal(TOTAL_SUPPLY)) < Decimal("0.000001")}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def reconcile(data, params, seed=None, now=None, replay=False, total_layers=None):
    """Every check, one dict. Pure — takes rows, returns findings — so the tests drive it
    directly and `main` stays a printer."""
    now = time.time() if now is None else now
    attendance, ledger = data["attendance"], data["ledger"]
    if total_layers is None:
        total_layers = int(data["settings"].get("serving_layers") or params["total_layers"])

    node_te = dsum(r["total_earned"] for r in ledger
                   if r.get("account_type") == "node" and r["node_id"] != COORDINATOR)

    a = leg_a(attendance)
    b = leg_b(ledger, a["settled_sum"], seed=seed, node_total_earned=node_te)
    c = leg_c(attendance, ledger, data["owners"])
    rows = check_rows(attendance, params, total_layers, data.get("unaudited_runs"))
    cap = check_daily_cap(attendance, params)
    backlog = check_sweep_backlog(attendance, params, now)
    supply = check_supply(ledger)

    findings = []

    def add(level, code, detail, extra=None):
        findings.append({"level": level, "code": code, "detail": detail, "extra": extra or {}})

    if "attendance" not in data["have"]:
        add("info", "no-attendance-table",
            "this database has no `attendance` table — it predates availability emission, "
            "so there is nothing to reconcile")
        return {"findings": findings, "a": a, "b": b, "c": c, "rows": rows, "cap": cap,
                "backlog": backlog, "supply": supply, "total_layers": total_layers,
                "replay": None, "ok": True}

    # --- leg A vs leg B ----------------------------------------------------- #
    if not b["pool_present"]:
        add("error", "no-emission-pool",
            f"no ledger row for {EMISSION_POOL}; genesis has not run on this database")
    elif b["seed_source"] == "given":
        tol = ab_tolerance(b["seed"], a["rows_paying"])
        gap = a["settled_sum"] - b["drop"]
        if abs(gap) > Decimal(repr(tol)):
            add("error", "pool-drop-mismatch",
                f"attendance says {a['settled_sum']} NRN was paid, the pool dropped "
                f"{b['drop']} NRN — a gap of {gap} against a tolerance of {tol:g}",
                {"gap": str(gap), "tolerance": tol})
        else:
            add("info", "pool-drop-matches",
                f"settled attendance ({a['settled_sum']} NRN) matches the pool's drop "
                f"({b['drop']} NRN) to within {tol:g}")
    else:
        # No seed to compare against, so bound the seed the data implies instead.
        implied = b["implied_already_minted"]
        if implied < Decimal(0):
            add("error", "implied-seed-impossible",
                f"the pool balance and the settled rows imply a genesis seed of "
                f"{b['implied_seed']}, i.e. already_minted = {implied}, which is negative — "
                f"genesis.seed_genesis refuses to seed on that, so something else has debited "
                f"{EMISSION_POOL}")
        elif node_te is not None and implied > node_te + Decimal(repr(DUST)):
            add("error", "implied-seed-exceeds-earnings",
                f"the implied already-minted figure at genesis ({implied}) exceeds today's "
                f"total node earnings ({node_te}). total_earned is never decremented, so the "
                f"genesis figure cannot be larger — the pool has lost NRN that no attendance "
                f"row explains")
        else:
            add("warn", "seed-inferred",
                f"no seed given and none recorded, so leg B was INFERRED as "
                f"{b['implied_seed']} and A==B is true by construction. The bounds it must "
                f"satisfy do hold (0 <= {implied} <= {node_te}). Pass --seed (or "
                f"--seed-from-backup) to make this an equality rather than an inference")

    # --- row shape ---------------------------------------------------------- #
    if a["settled_null_reward"]:
        add("error", "settled-with-null-reward",
            f"{len(a['settled_null_reward'])} row(s) have paid_at set and reward NULL. "
            f"settle_attendance always writes both, so something else stamped these",
            {"rows": [_r(r) for r in a["settled_null_reward"][:20]]})
    if a["reward_without_paid_at"]:
        add("error", "reward-without-paid-at",
            f"{len(a['reward_without_paid_at'])} row(s) carry a reward but no paid_at — "
            f"unsettled rows nothing should have priced",
            {"rows": [_r(r) for r in a["reward_without_paid_at"][:20]]})
    if a["negative"]:
        add("error", "negative-reward",
            f"{len(a['negative'])} row(s) settled at a negative reward",
            {"rows": [_r(r) for r in a["negative"][:20]]})
    if rows["over_ceiling"]:
        add("error", "reward-over-ceiling",
            f"{len(rows['over_ceiling'])} row(s) paid more than base x scarcity_max "
            f"({params['base_nrn_per_hour']} x {params['scarcity_max']} = "
            f"{params['base_nrn_per_hour'] * params['scarcity_max']} NRN), which no attendance "
            f"fraction or replica count can produce",
            {"rows": [_r(x["row"]) for x in rows["over_ceiling"][:20]]})
    if rows["paid_unqualified"]:
        add("error", "paid-unqualified",
            f"{len(rows['paid_unqualified'])} row(s) were paid although their own frozen "
            f"values fail a qualification gate — presence paid without proof of compute is "
            f"the exploit emission exists to refuse",
            {"rows": [dict(_r(x["row"]), reason=x["reason"])
                      for x in rows["paid_unqualified"][:20]]})
    if rows["zeroed_qualified"]:
        add("info", "zeroed-though-qualified",
            f"{len(rows['zeroed_qualified'])} qualifying row(s) settled at 0 — expected when "
            f"the daily cap bound the sweep or the pool could not pay; not a discrepancy on "
            f"its own")
    if rows["zero_by_code"]:
        total_zero = sum(rows["zero_by_code"].values())
        share = 100.0 * total_zero / max(1, a["rows_settled"])
        add("info", "why-hours-earned-nothing",
            f"{total_zero} of {a['rows_settled']} settled node-hour(s) ({share:.0f}%) paid "
            f"nothing. Each is emission refusing to pay for presence, which is what it is for "
            f"— but it is also the question an operator asks first, so: "
            + ", ".join(f"{v} {k}" for k, v in
                        sorted(rows["zero_by_code"].items(), key=lambda kv: -kv[1])),
            {f"  {n}": ", ".join(f"{code} {_span(seen)}"
                                 for code, seen in sorted(codes.items()))
             for n, codes in sorted(rows["zero_by_node"].items())})

    # --- leg C -------------------------------------------------------------- #
    for s in c["shortfalls"]:
        add("error", "emission-did-not-arrive",
            f"{s['account']}: paid {s['emission_paid']} NRN of emission but total_earned is "
            f"{s['total_earned']} — {s['detail']}")
    if c["ambiguous"]:
        add("warn", "payee-ambiguous",
            f"{len(c['ambiguous'])} node(s) have an owner recorded today, so their "
            f"{sum(c['ambiguous'].values(), Decimal(0))} NRN of settled emission cannot be "
            f"attributed to the node or the owner — nothing records when the link was made. "
            f"Excluded from the arrival check",
            {"nodes": {k: str(v) for k, v in c["ambiguous"].items()}})
    if c["orphan_node"]:
        by_ledger = {r["node_id"]: r for r in ledger}
        # A RETIRED node is not an anomaly, it is a machine we were told about ([P39] item 5).
        # Splitting these apart is what lets this run as a standing assertion: the orphan warning
        # has been true and unactionable since 2026-08-02 and would never clear, because the
        # machine is not coming back -- and a check that always warns is a check nobody reads,
        # which is the failure [P40] exists to prevent arriving through [P40]'s own output.
        retired = data.get("retired", {})
        known = {k: v for k, v in c["orphan_node"].items() if k in retired}
        unknown = {k: v for k, v in c["orphan_node"].items() if k not in retired}

        def _detail(items):
            out = {}
            for nid, amt in sorted(items.items()):
                row = by_ledger.get(nid)
                out[nid] = (
                    f"{amt} NRN paid; ledger row holds {float(row['balance']):.6f} NRN "
                    f"(total_earned {float(row['total_earned']):.6f})" if row else
                    f"{amt} NRN paid, and there is no ledger row either — it went nowhere")
                r = retired.get(nid)
                if r:
                    when = time.strftime("%Y-%m-%d", time.gmtime(float(r["retired_at"])))
                    out[nid] += f" — retired {when}" + (f": {r['reason']}" if r.get("reason") else "")
            return out

        if known:
            add("ok", "attendance-from-retired-nodes",
                f"{len(known)} retired node(s) hold settled attendance totalling "
                f"{sum(known.values(), Decimal(0))} NRN. Accounted for: the machine was "
                f"deliberately unregistered and left a tombstone, so its hours are explained "
                f"rather than missing. Still excluded from the arrival check, because the payee "
                f"at settle time remains unknowable",
                _detail(known))
        if unknown:
            add("warn", "attendance-without-node",
                f"{len(unknown)} node(s) have settled attendance but no row in `nodes` and no "
                f"retirement record. `delete_node` removes the node and leaves its attendance "
                f"and its ledger, so the machine kept being paid after it ceased to exist — and "
                f"since the payee is resolved as `get_node_owner(id) or id`, and a deleted node "
                f"has no owner, the NRN landed in an account whose only credential was that "
                f"machine's node_token ([P39]). Payee unknowable at settle time; excluded from "
                f"the arrival check",
                _detail(unknown))
    if c["checked"]:
        add("info", "emission-arrived",
            f"{len(c['checked'])} payee(s) hold total_earned >= the emission they were paid")

    # --- bounds and liveness ------------------------------------------------ #
    if cap["exceeded"]:
        add("error", "daily-cap-exceeded",
            f"the heaviest rolling 24h of settlements is {cap['max_window']} NRN against a cap "
            f"of {cap['cap']} NRN", {"at": cap["at"]})
    if backlog["stale"]:
        add("error", "sweep-backlog",
            f"{len(backlog['stale'])} attendance row(s) sit unsettled in slots that closed "
            f"more than one slot ago — close_slots is not keeping up, and an idle sweep is "
            f"silent by design so the logs will not say so",
            {"oldest_slot": min(float(r["slot_start"]) for r in backlog["stale"])})
    if not supply["ok"]:
        add("error", "supply-invariant-broken",
            f"SUM(ledger.balance) = {supply['total']}, expected exactly {TOTAL_SUPPLY}")

    # --- optional full replay ----------------------------------------------- #
    rep = None
    if replay:
        rep = run_replay(attendance, total_layers, params, data.get("unaudited_runs"))
        if rep["mismatches"]:
            add("error", "replay-mismatch",
                f"{len(rep['mismatches'])} settled row(s) do not match the reward re-priced "
                f"from their own frozen inputs at total_layers={total_layers}",
                {"rows": rep["mismatches"][:20]})
        else:
            add("info", "replay-matches",
                f"every one of {rep['compared']} settled row(s) matches the reward re-priced "
                f"from its own frozen inputs at total_layers={total_layers}")
        if rep["cap_explained"]:
            add("info", "replay-cap-explained",
                f"{len(rep['cap_explained'])} row(s) differ only by being settled at 0 where "
                f"pricing says they had earned — consistent with the daily cap or an "
                f"exhausted pool, and counted separately from a mismatch")
        if rep["layer_sensitive"]:
            add("warn", "replay-layer-sensitive",
                f"{len(rep['layer_sensitive'])} row(s) hold a block outside "
                f"[0, {total_layers}) and would change verdict under a different serving "
                f"model. total_layers is the one input settlement does not freeze — pass "
                f"--total-layers if the network was serving a different model then",
                {"rows": rep["layer_sensitive"][:20]})

    ok = not any(f["level"] == "error" for f in findings)
    return {"findings": findings, "a": a, "b": b, "c": c, "rows": rows, "cap": cap,
            "backlog": backlog, "supply": supply, "total_layers": total_layers,
            "replay": rep, "ok": ok}


def run_replay(attendance, total_layers, params, runs=None):
    """Re-price every settled slot from its own rows and diff against what was recorded."""
    runs = runs or {}
    by_slot = {}
    for r in attendance:
        if r["paid_at"] is not None:
            by_slot.setdefault(float(r["slot_start"]), []).append(r)

    mismatches, cap_explained, layer_sensitive, compared = [], [], [], 0
    for slot in sorted(by_slot):
        rows = by_slot[slot]
        priced = {e["node_id"]: e for e in
                  replay_slot(rows, total_layers, params, runs.get(slot, 0))}
        for r in rows:
            if r["reward"] is None:
                continue
            compared += 1
            want = priced[r["node_id"]]["reward"]
            got = float(r["reward"])
            hi = r.get("block_end")
            if hi is not None and hi >= total_layers:
                layer_sensitive.append(dict(_r(r), expected=want))
            if abs(got - want) <= 1e-6:
                continue
            # Settled at zero where pricing says it earned: exactly what the daily cap does,
            # and what an exhausted pool leaves behind. A real mismatch is anything else.
            if got == 0.0 and want > 0:
                cap_explained.append(dict(_r(r), expected=want))
            else:
                mismatches.append(dict(_r(r), expected=want))
    return {"compared": compared, "mismatches": mismatches,
            "cap_explained": cap_explained, "layer_sensitive": layer_sensitive}


def _r(row):
    """A row, trimmed for display."""
    return {"node_id": row["node_id"], "slot_start": row["slot_start"],
            "seconds_online": row["seconds_online"], "block": [row["block_start"],
                                                               row["block_end"]],
            "poc_ok": row["poc_ok"], "reward": row["reward"], "paid_at": row["paid_at"]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _fmt(d):
    return "n/a" if d is None else f"{float(d):,.6f}"


def emit(line=""):
    """Print, folded to ASCII.

    This may be run from the founder's Windows console, whose stdout encoding is cp1252, where
    the em-dashes the findings are written in arrive as replacement characters. Folding at the
    point of printing rather than writing the messages in ASCII keeps the prose readable in the
    source, which is where it is read most."""
    print(str(line).replace("—", "--").replace("–", "-").replace("−", "-")
          .encode("ascii", "replace").decode("ascii"))


def report(res, db_path, params, seed_note):
    a, b, c = res["a"], res["b"], res["c"]
    emit(f"ledger            : {db_path}")
    emit("mode              : READ-ONLY (this script has no write path)")
    emit(f"serving layers    : {res['total_layers']}")
    emit(f"slot              : {params['slot_seconds']}s, base "
         f"{params['base_nrn_per_hour']} NRN/h, cap x{params['scarcity_max']}, "
         f"target {params['target_replicas']} replicas")
    emit()
    emit("attendance rows   : "
         f"{a['rows_total']} total, {a['rows_settled']} settled, "
         f"{a['rows_paying']} paying, {a['rows_unsettled']} unsettled")
    emit()
    emit("  A  recorded     : " + _fmt(a["settled_sum"]) + " NRN  (sum of settled "
         "attendance.reward)")
    emit("  B  left pool    : " + _fmt(b.get("drop")) + f" NRN  (seed {_fmt(b.get('seed'))}"
         f" [{b.get('seed_source') or 'n/a'}] - balance {_fmt(b.get('balance'))})")
    emit("  C  arrived      : " + _fmt(c["attributable_sum"]) + " NRN attributable across "
         f"{len(c['checked'])} payee(s); "
         f"{_fmt(sum(c['ambiguous'].values(), Decimal(0)))} ambiguous, "
         f"{_fmt(sum(c['orphan_node'].values(), Decimal(0)))} orphaned")
    if seed_note:
        emit(f"\n  {seed_note}")
    emit()
    emit(f"supply            : {_fmt(res['supply']['total'])} NRN "
         f"({'invariant holds' if res['supply']['ok'] else 'INVARIANT BROKEN'})")
    emit(f"heaviest 24h      : {_fmt(res['cap']['max_window'])} NRN against a "
         f"{res['cap']['cap']:,.0f} NRN cap")
    emit()
    order = {"error": 0, "warn": 1, "info": 2}
    mark = {"error": "FAIL", "warn": "WARN", "info": "ok  "}
    for f in sorted(res["findings"], key=lambda f: order[f["level"]]):
        emit(f"  {mark[f['level']]}  [{f['code']}] {f['detail']}")
        for k, v in (f["extra"] or {}).items():
            if k == "rows":
                for row in v:
                    emit(f"          {row}")
            else:
                emit(f"          {k}: {v}")
    emit()
    errors = [f for f in res["findings"] if f["level"] == "error"]
    if errors:
        emit(f"RESULT: {len(errors)} discrepancy(ies) — the ledger and the attendance rows "
             f"do NOT agree.")
    else:
        emit("RESULT: reconciled — settled attendance, the emission pool and the payees' "
             "earnings agree.")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Reconcile the emission ledger against its own attendance rows (read-only).")
    p.add_argument("--db", default=DEFAULT_DB, help=f"database to read (default {DEFAULT_DB})")
    p.add_argument("--seed", type=float, default=None,
                   help="the genesis balance of __emission_pool__. Without it leg B is "
                        "inferred and can only be bounded, not checked.")
    p.add_argument("--seed-from-backup", action="store_true",
                   help=f"use {KNOWN_LIVE_SEED}, the pool balance in "
                        f"backups-offbox/neuron-20260802-115534.db, which predates emission")
    p.add_argument("--total-layers", type=int, default=None,
                   help="layer count of the model that was serving (default: "
                        "settings['serving_layers'], else 28)")
    p.add_argument("--replay", action="store_true",
                   help="also re-price every settled slot from its own frozen rows")
    p.add_argument("--now", type=float, default=None, help="override the clock (testing)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"no ledger at {args.db}", file=sys.stderr)
        return 2

    seed = args.seed
    seed_note = ""
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        data = read_all(con)
    finally:
        con.close()

    if seed is None and args.seed_from_backup:
        seed, seed_note = KNOWN_LIVE_SEED, (
            f"seed {KNOWN_LIVE_SEED} taken from the 2026-08-02 backup, which has no "
            f"`attendance` table and therefore predates every emission payment.")
    if seed is None and data["settings"].get("emission_pool_seed"):
        try:
            seed = float(data["settings"]["emission_pool_seed"])
            seed_note = "seed read from settings['emission_pool_seed'], recorded at genesis."
        except (TypeError, ValueError):
            pass

    params = dict(DEFAULTS)
    res = reconcile(data, params, seed=seed, now=args.now, replay=args.replay,
                    total_layers=args.total_layers)

    if args.json:
        print(json.dumps({
            "db": args.db, "ok": res["ok"], "total_layers": res["total_layers"],
            "recorded": str(res["a"]["settled_sum"]),
            "left_pool": str(res["b"].get("drop")),
            "seed": str(res["b"].get("seed")), "seed_source": res["b"].get("seed_source"),
            "rows": {k: res["a"][k] for k in
                     ("rows_total", "rows_settled", "rows_paying", "rows_unsettled")},
            "supply_ok": res["supply"]["ok"],
            "findings": res["findings"],
        }, indent=2, default=str))
    else:
        report(res, args.db, params, seed_note)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
