"""NEURON coordinator — availability emission (TOKENOMICS.md §11.4).

Nodes have only ever earned by SERVING. Nothing paid a machine to simply be there, so coverage
was whatever happened to be awake, and the coordinator reacted to that by re-splitting layers —
which costs every affected node a delete and a re-download. §11.4 already specified the fix and
it was never built: emission paid **per device-hour**, deliberately decoupled from traffic,
because anything paying above the spend for metered work can be farmed by generating your own
traffic.

This adds the part §11.4 lacks: paying for hours **where and when coverage is short**, so
availability is rostered rather than accidental. Phones are why it matters —
`agent/android/SAFETY_LIMITS.md` has them contribute only at battery >= 80% and plugged in, so
they arrive and leave in correlated nightly waves, which is exactly when coverage is thinnest.

Three rules this module exists to enforce:

1. **Nothing is minted.** Payment is a `models.transfer` out of `__emission_pool__`. The fixed
   1,000,000,000 supply invariant is not negotiable (`test_escrow_conservation`).
2. **Presence alone earns nothing.** Paying for uptime invents a reason to fake uptime — a
   script that heartbeats and computes nothing would otherwise be the most profitable node on
   the network. A slot pays only if a proof-of-compute challenge landed *inside* it.
3. **At most once per slot.** Settlement claims the row atomically; a replayed sweep pays
   nothing extra.
"""
import time

from coordinator import config, models


def scarcity_multiplier(replicas, target=None, cap=None):
    """How much more a thin slot pays than a well-covered one.

    Deliberately a bounded, monotonic function of replica count and nothing else — not an
    auction. Uncapped, a slot with a single eligible node would price itself arbitrarily high
    at exactly the moment the network is least able to afford it, and drain a pool that has to
    last years. `target` replicas earns the base rate; scarcer earns more, up to `cap`.

    Zero replicas returns the cap rather than dividing by zero: an uncovered block is the most
    valuable hour the network can advertise, and this figure is what `/network/slots` publishes
    to recruit for it.
    """
    target = config.EMISSION_TARGET_REPLICAS if target is None else target
    cap = config.EMISSION_SCARCITY_MAX if cap is None else cap
    if replicas <= 0:
        return cap
    return max(1.0, min(cap, target / float(replicas)))


def _block_is_needed(row, total_layers):
    """Does the block this node held during the slot belong to the model being served?

    A node re-asserting a stale range from an older model, or one that never had a range, is
    online but useless — paying it would be paying for presence, which is the thing rule 2
    exists to prevent.
    """
    lo, hi = row.get("block_start"), row.get("block_end")
    if lo is None or hi is None:
        return False
    return 0 <= lo <= hi < total_layers


def plan_slot(rows, total_layers, now=None, unaudited_run=0):
    """Price one closed slot. Pure — no DB, no clock, no payment — so the arithmetic can be
    tested directly and the sweep below stays a thin shell around it.

    Returns [{node_id, slot_start, reward, replicas, multiplier, attended_frac, reason,
    poc_excused}], one entry per attendance row, including the ones that earn nothing. Carrying
    the zeroes (with a `reason`) is deliberate: "you were up all night and earned nothing" is
    exactly the question an operator will ask, and a payout log that silently omits the misses
    cannot answer it.

    `unaudited_run` is how many consecutive slots ending at this one the coordinator heard
    nothing from any verifier — 0 whenever the slot was audited, and 0 for every slot predating
    the mechanism. It is what separates "this node failed" from "we did not look" ([P47] cause
    2): the first is a fact about the node, the second is a fact about us, and only the first
    should cost a volunteer their hour.
    """
    slot_seconds = float(config.SLOT_SECONDS)
    # An hour nobody audited is excused only while the outage is short enough that presence plus
    # a recent history of passing still means something. Past the bound we genuinely do not know
    # what these machines were doing, and saying so is worth more than paying on a guess.
    excusable = 0 < unaudited_run <= config.EMISSION_MAX_UNAUDITED_SLOTS
    qualified = []
    for r in rows:
        frac = (r.get("seconds_online") or 0.0) / slot_seconds
        r["_excused"] = False
        if frac < config.SLOT_MIN_ATTENDANCE_FRAC:
            r["_reason"] = "attended %.0f%% of the slot, floor is %.0f%%" % (
                frac * 100, config.SLOT_MIN_ATTENDANCE_FRAC * 100)
        elif not r.get("poc_ok") and not excusable:
            # Two different events, and until [P47] they were the same zero. The wording says
            # which, because "no challenge passed" reads as an accusation when the reason is
            # that the verifier was asleep.
            r["_reason"] = (
                "no proof-of-compute challenge passed inside the slot"
                if unaudited_run == 0 else
                "the network was not audited for %d consecutive slots, beyond the %d this "
                "pays through — nobody checked, so nobody can say" % (
                    unaudited_run, config.EMISSION_MAX_UNAUDITED_SLOTS))
        elif not _block_is_needed(r, total_layers):
            r["_reason"] = "held no block of the serving model"
        else:
            r["_reason"] = None
            # Excused only if it was NOT proved the ordinary way. A node challenged inside an
            # otherwise-unaudited slot earns on its own proof and is not marked excused, so the
            # flag counts what it claims to count.
            r["_excused"] = excusable and not r.get("poc_ok")
            # Excused rows count toward replica depth. If they did not, the coordinator's own
            # outage would read as a network-wide scarcity spike and pay MORE per node at exactly
            # the moment it knows least -- turning our downtime into a payout event.
            qualified.append(r)
        r["_frac"] = frac

    # Replica depth is counted over the QUALIFYING rows only. A node that was present but
    # failed its challenge does not make a block look covered -- if it did, a block held by
    # nothing but unverifiable nodes would price itself as healthy and never attract a real one.
    depth = {}
    for r in qualified:
        key = (r["block_start"], r["block_end"])
        depth[key] = depth.get(key, 0) + 1

    out = []
    for r in rows:
        key = (r.get("block_start"), r.get("block_end"))
        replicas = depth.get(key, 0)
        if r["_reason"] is not None:
            out.append({"node_id": r["node_id"], "slot_start": r["slot_start"], "reward": 0.0,
                        "replicas": replicas, "multiplier": 0.0,
                        "attended_frac": round(r["_frac"], 4), "reason": r["_reason"],
                        "poc_excused": False})
            continue
        mult = scarcity_multiplier(replicas)
        reward = config.EMISSION_BASE_NRN_PER_HOUR * min(r["_frac"], 1.0) * mult
        # An excused hour pays the FULL rate, not a discounted one. A discount would be a
        # penalty for our own downtime, which is the thing being corrected; the honest lever is
        # how MANY such hours are covered (EMISSION_MAX_UNAUDITED_SLOTS), not what they are
        # worth. A node that did exactly what was asked should not be paid less because the
        # auditor's PC went to sleep.
        out.append({"node_id": r["node_id"], "slot_start": r["slot_start"],
                    "reward": round(reward, 6), "replicas": replicas,
                    "multiplier": round(mult, 4),
                    "attended_frac": round(r["_frac"], 4), "reason": None,
                    "poc_excused": bool(r["_excused"])})
    return out


def close_slots(total_layers, now=None, log=print):
    """Settle every slot that has finished. Called from the health sweep.

    Ordering is load-bearing: the row is CLAIMED (`settle_attendance`) before any money moves.
    Claim-then-pay can at worst pay nothing for a claimed slot, which is visible and recoverable;
    pay-then-claim can pay twice for the same hour, which is neither.
    """
    now = time.time() if now is None else now
    current = models.slot_start_for(now)
    rows = models.unpaid_attendance(current)
    if not rows:
        return {"slots": 0, "paid": 0.0, "nodes": 0, "capped": False}

    by_slot = {}
    for r in rows:
        by_slot.setdefault(r["slot_start"], []).append(r)

    spent_today = models.emitted_since(now - 86400.0)
    total_paid, nodes_paid, capped = 0.0, 0, False
    settled_zero = 0
    unpaid = 0                # claimed, priced above zero, and the pool could not pay it
    excused = 0               # paid for an hour the coordinator never audited

    for slot in sorted(by_slot):
        # Did WE do our job during this slot? Read per slot rather than once, because a sweep
        # that has been down for a while settles a backlog spanning both audited and unaudited
        # hours, and pricing them alike is the conflation this exists to end.
        run = models.unaudited_run(slot, config.EMISSION_MAX_UNAUDITED_SLOTS + 1)
        for entry in plan_slot(by_slot[slot], total_layers, now=now, unaudited_run=run):
            reward = entry["reward"]
            # The daily cap is checked per payment, not per sweep: the multiplier rises with
            # scarcity, and a mass outage is simultaneously the scarcest and the most expensive
            # the network ever gets. This is the backstop for exactly that moment.
            if reward > 0 and spent_today + reward > config.EMISSION_DAILY_CAP_NRN:
                reward, capped = 0.0, True
                entry["reason"] = "daily emission cap reached"
                entry["reward"] = 0.0
            if not models.settle_attendance(entry["node_id"], slot, reward, now=now,
                                            poc_excused=entry.get("poc_excused", False)):
                continue                      # already settled by another pass -- never pay twice
            if reward > 0 and entry.get("poc_excused"):
                excused += 1
            if reward <= 0:
                settled_zero += 1             # claimed the hour, paid nothing -- counted so the
                continue                      # log can tell this apart from "nothing to do"
            # [P39] phase 3: pay the person, not the machine. A node account's only
            # credential is the node_token in one config.json, so crediting it means the
            # money is lost with the file and skipped as `unmapped` by migrate_ledger. When
            # an owner has been recorded (which requires a verified signature, see phase 1),
            # the reward goes straight to the wallet they sign into and no sweep is ever
            # needed. Falls back to the node account when no owner is known, so nodes that
            # predate this keep earning exactly as before.
            #
            # The attendance row is still claimed against the NODE -- that is which machine
            # was present, and settling it elsewhere would let one owner's two nodes settle
            # each other's hours.
            payee = models.get_node_owner(entry["node_id"]) or entry["node_id"]
            entry["payee"] = payee
            if not models.transfer(config.GENESIS_BUCKETS_EMISSION_ID, payee, reward):
                # The pool is empty. The row stays CLAIMED so it is not retried forever, but
                # its reward is corrected to 0, because nothing was paid.
                #
                # This used to call `settle_attendance(..., 0.0)`, which cannot work: that
                # UPDATE carries `WHERE paid_at IS NULL` and the successful claim four lines
                # above has just falsified it. So the row kept its full reward while the pool
                # paid nothing -- `emitted_since` (which is what the daily cap is read against)
                # counted money that never moved, and the ledger disagreed with the attendance
                # rows by exactly this amount, permanently. Found by writing the reconciliation
                # in `reconcile_emission.py` and asking what would make its two legs differ.
                models.void_settlement(entry["node_id"], slot)
                entry["reward"] = 0.0
                unpaid += 1
                log(f"[emission] POOL EXHAUSTED — cannot pay {reward:.6f} NRN to "
                    f"'{payee}' for slot {int(slot)}. Availability rewards have "
                    f"stopped; per-request earnings are unaffected.")
                continue
            spent_today += reward
            total_paid += reward
            nodes_paid += 1

    # Log every sweep that had rows, not only the ones that paid. A sweep that settled ten
    # node-slots at zero and a sweep that did nothing were previously indistinguishable in
    # the log, which is half of why [P40] could not be answered from the logs alone. The
    # early return above still keeps a genuinely idle sweep silent.
    tail = " (daily emission cap reached)" if capped else ""
    # `unpaid` is counted apart from `settled_zero`: a row that earned nothing and a row the
    # pool could not pay are the same zero in the ledger and completely different events.
    tail += f", {unpaid} unpayable (pool exhausted)" if unpaid else ""
    # Say it out loud. An excused hour is the network paying for its own blind spot, and that is
    # exactly the kind of spending that must never become invisible -- if this line is showing up
    # every sweep, the fix is to keep the verifier up, not to keep paying around it.
    tail += (f", {excused} paid unaudited (the verifier was down, not the node)"
             if excused else "")
    if nodes_paid:
        log(f"[emission] paid {total_paid:.4f} NRN to {nodes_paid} node-slot(s) across "
            f"{len(by_slot)} slot(s)"
            + (f", {settled_zero} settled at 0" if settled_zero else "") + tail)
    else:
        log(f"[emission] settled {settled_zero} node-slot(s) at 0 across "
            f"{len(by_slot)} slot(s), paid nothing" + tail)
    return {"slots": len(by_slot), "paid": round(total_paid, 6), "nodes": nodes_paid,
            "zero": settled_zero, "unpaid": unpaid, "excused": excused, "capped": capped}


def coverage_report(nodes, total_layers, now=None):
    """What the network is short of, per block, right now — the recruiting signal.

    New capability, not a restatement of /status: this says where cover is needed *before* the
    chain breaks, which is the whole point of rostering availability instead of reacting to it.
    """
    now = time.time() if now is None else now
    live = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    depth = {}
    for n in live:
        key = (n["layer_start"], n["layer_end"])
        depth[key] = depth.get(key, 0) + 1
    covered = set()
    for n in live:
        covered.update(range(n["layer_start"], min(n["layer_end"] + 1, total_layers)))
    blocks = [{"block": [lo, hi], "replicas": r,
               "multiplier": round(scarcity_multiplier(r), 4)}
              for (lo, hi), r in sorted(depth.items())]
    uncovered = sorted(set(range(total_layers)) - covered)
    return {
        "slot_start": models.slot_start_for(now),
        "slot_seconds": config.SLOT_SECONDS,
        "base_nrn_per_hour": config.EMISSION_BASE_NRN_PER_HOUR,
        "target_replicas": config.EMISSION_TARGET_REPLICAS,
        "blocks": blocks,
        "uncovered_layers": uncovered,
        # An uncovered layer pays the cap, which is the strongest signal this endpoint can send.
        "uncovered_multiplier": config.EMISSION_SCARCITY_MAX if uncovered else None,
    }
