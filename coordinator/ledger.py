"""NEURON coordinator — NRN settlement (fixed-supply, TOKENOMICS.md §11 Phase 0).

NRN is transfer-only from here on — this file never mints. A request's cost is HELD from the
payer's wallet into __escrow__ at /infer time (models.hold, worst-case quote from max_tokens),
then settle() here moves the REAL metered cost out of escrow to the serving nodes + the
coordinator's fee, refunding whatever wasn't used. See coordinator/genesis.py for the fixed
1,000,000,000 supply this all conserves.

    actual cost   = (completion_tokens + prompt_tokens * INPUT_WEIGHT) / 1000 * PRICE_PER_1K
    coordinator fee = actual * COORDINATOR_FEE           (0.10, from the payment, never minted)
    node pool        = actual * (1 - COORDINATOR_FEE)     (0.90)
    a node's share    = pool * its_layer_equivalents / sum(layer_equivalents in this plan)

layer_equivalents is layer count + a bonus for whoever holds the lm_head (meaningfully more
compute than a plain layer). Summing over the ACTUAL plan (not a hardcoded layer count) makes
this self-adjusting -- correct whether the network is serving a 28-layer or an 80-layer model.

The property this file has to keep, and did not: **every NRN a hold puts into __escrow__ comes
back out when the request settles.** Escrow is a staging area, never a destination. The live
ledger disproved that -- it held 0.056001 NRN with zero holds in state 'held' -- because the
node payout sat inside `if total_le > 0` and was neither paid nor refunded when no planned node
was eligible, and because independently-rounded shares did not add up to the pool. Both are
fixed below, and settle() now checks the invariant itself rather than trusting the arithmetic.
"""
import logging

from coordinator import config, models

log = logging.getLogger("neuron.coordinator.ledger")


def layer_equivalents(node):
    """A node's reward-split weight. layer_end/layer_start define its slice; a bonus is added
    if it holds the lm_head (head_ms > 0), since that costs far more than one plain layer
    (~38ms vs ~9ms/layer, S14 benchmark data) -- without the bonus the head-holder is
    underpaid relative to its real compute burden."""
    layers = node["layer_end"] - node["layer_start"] + 1
    bonus = config.HEAD_BONUS_LE if (node.get("head_ms") or 0) > 0 else 0.0
    return layers + bonus


def quote(max_tokens, input_tokens_estimate):
    """Worst-case hold amount for a not-yet-completed request -- prices the FULL max_tokens
    as if every one gets generated. settle() refunds the difference once the real usage is
    known, so this only needs to be an upper bound, not exact."""
    weighted = max(0, int(input_tokens_estimate)) * config.INPUT_WEIGHT + max(0, int(max_tokens))
    return round(weighted / 1000.0 * config.PRICE_PER_1K_WEIGHTED, 6)


def settle(request_id, wallet_id, hold_amount, prompt_tokens, completion_tokens,
          plan_nodes, model_multiplier=1.0):
    """Settle a completed request against its existing hold. Every movement below is a
    transfer OUT of __escrow__ (where the hold already put the money) -- settlement never
    touches the wallet's balance column directly, which is what keeps SUM(ledger.balance)
    trivially conserved through the whole hold -> settle lifecycle. Returns a breakdown dict
    {node_id: amount, ..., '__coordinator__': fee, '__refund__': unused_hold (if any)}.
    Ineligible/unknown nodes in `plan_nodes` are silently excluded, same as before (S12
    probation / S16 flagged nodes earn nothing) -- but their share of the pool is now REFUNDED
    to the payer rather than left in escrow: the network did not deliver it, so it goes back to
    whoever paid, not to a bucket nobody is watching."""
    weighted = max(0, int(completion_tokens)) + max(0, int(prompt_tokens)) * config.INPUT_WEIGHT
    actual = round(weighted / 1000.0 * config.PRICE_PER_1K_WEIGHTED * model_multiplier, 6)
    actual = min(actual, hold_amount)   # never charge more than what was actually held

    pool = round(actual * (1.0 - config.COORDINATOR_FEE), 6)
    fee = round(actual - pool, 6)

    eligible = [n for n in plan_nodes if n and n.get("eligible")]
    total_le = sum(layer_equivalents(n) for n in eligible)

    breakdown = {}
    paid = 0.0            # what actually LEFT escrow for this request, not what we intended
    if total_le > 0:
        allocated = 0.0
        last = len(eligible) - 1
        for i, node in enumerate(eligible):
            # The last node takes the remainder instead of its own rounded share, so the
            # shares sum to `pool` exactly. Rounding each independently left a sub-cent
            # residue behind in escrow on every settlement -- small, permanent, cumulative.
            share = (round(pool - allocated, 6) if i == last
                     else round(pool * layer_equivalents(node) / total_le, 6))
            allocated = round(allocated + share, 6)
            if share > 0 and models.transfer(config.ESCROW_LEDGER_ID, node["node_id"], share,
                                             count_request=True):
                breakdown[node["node_id"]] = share
                paid = round(paid + share, 6)

    if fee > 0 and models.transfer(config.ESCROW_LEDGER_ID, config.COORDINATOR_LEDGER_ID, fee):
        breakdown[config.COORDINATOR_LEDGER_ID] = fee
        paid = round(paid + fee, 6)

    # Whatever this hold put into escrow and did not pay out belongs to the payer. Deriving
    # the refund from what actually LEFT (rather than from `actual`) is what makes it total:
    # it covers the ordinary unused-hold refund, the entire node pool when nobody was
    # eligible, and anything a transfer failed to move. Scoped to this hold's own amount --
    # escrow is a shared pot, and sweeping its BALANCE would raid requests still in flight.
    refund = round(hold_amount - paid, 6)
    # Clamp to what escrow can actually give. `refund` is computed in rounded decimal while
    # escrow's balance is the result of its own chain of float additions and subtractions, so
    # the two disagree in the 15th digit -- and models.transfer's `balance >= amount` guard
    # then refuses the whole refund over a 1e-15 shortfall, stranding it. This can only ever
    # shave dust: `refund` is this hold's own unpaid remainder, which is by construction no
    # larger than the escrow balance, so the clamp never reaches another request's money.
    escrow_balance, _ = models.escrow_state()
    refund = min(refund, escrow_balance)
    if refund > 0 and models.transfer(config.ESCROW_LEDGER_ID, wallet_id, refund):
        breakdown["__refund__"] = refund
        paid = round(paid + refund, 6)

    models.mark_hold_settled(request_id)

    # Assert it rather than assume it. Note this is NOT "escrow == 0": escrow legitimately
    # holds the sum of every request still in flight, and this network runs concurrent
    # requests by design, so zero is only the idle case of the real invariant.
    balance, still_held = models.escrow_state()
    drift = round(balance - still_held, 6)
    if abs(drift) > 1e-9:
        log.error("ESCROW DRIFT after settling %s: escrow holds %.6f, live holds total %.6f "
                  "(drift %+.6f); paid out %.6f of a %.6f hold. NRN is stranded in __escrow__ "
                  "-- it is not lost (the supply invariant still holds) but it belongs to "
                  "somebody and is sitting in a bucket nobody watches.",
                  request_id, balance, still_held, drift, paid, hold_amount)
    return breakdown
