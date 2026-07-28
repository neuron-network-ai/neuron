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
"""
from coordinator import config, models


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
    probation / S16 flagged nodes earn nothing)."""
    weighted = max(0, int(completion_tokens)) + max(0, int(prompt_tokens)) * config.INPUT_WEIGHT
    actual = round(weighted / 1000.0 * config.PRICE_PER_1K_WEIGHTED * model_multiplier, 6)
    actual = min(actual, hold_amount)   # never charge more than what was actually held

    pool = round(actual * (1.0 - config.COORDINATOR_FEE), 6)
    fee = round(actual - pool, 6)

    eligible = [n for n in plan_nodes if n and n.get("eligible")]
    total_le = sum(layer_equivalents(n) for n in eligible)

    breakdown = {}
    if total_le > 0:
        for node in eligible:
            share = round(pool * layer_equivalents(node) / total_le, 6)
            if share > 0 and models.transfer(config.ESCROW_LEDGER_ID, node["node_id"], share,
                                             count_request=True):
                breakdown[node["node_id"]] = share

    if fee > 0 and models.transfer(config.ESCROW_LEDGER_ID, config.COORDINATOR_LEDGER_ID, fee):
        breakdown[config.COORDINATOR_LEDGER_ID] = fee

    refund = round(hold_amount - actual, 6)
    if refund > 0 and models.transfer(config.ESCROW_LEDGER_ID, wallet_id, refund):
        breakdown["__refund__"] = refund

    models.mark_hold_settled(request_id)
    return breakdown
