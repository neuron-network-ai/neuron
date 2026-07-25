"""NEURON coordinator — NRN earnings.

Reward per completed request (NRN = the network coin):

    total minted           = NRN_PER_REQUEST            (1.0)
    coordinator fee (kept)  = NRN_PER_REQUEST * FEE       (0.10, always)
    distributed to nodes    = NRN_PER_REQUEST * (1-FEE)   (0.90)
    a node holding L layers = distributed * L / TOTAL_LAYERS

So a 10-layer node earns 0.9 * 10/28 = 0.321 NRN; a 9-layer node 0.9 * 9/28 = 0.289.
Across a full 28-layer chain the nodes share 0.90 and the coordinator keeps 0.10.
(This reconciles the two rules in the spec — "layers/28 of the pool" AND "coordinator
keeps 10% always" — and matches the /status example: 47 requests -> 42.3 NRN to nodes.)
"""
from coordinator import config, models


def reward_for_layers(layers_held, total_layers=None):
    total = total_layers if total_layers is not None else config.TOTAL_LAYERS
    distributable = config.NRN_PER_REQUEST * (1.0 - config.COORDINATOR_FEE)
    return distributable * (layers_held / total)


def distribute(node_ids):
    """Credit each participating node its layer-proportional share and the
    coordinator its fee. Layer counts come from the registry. Returns a breakdown
    dict {node_id: nrn, ..., '__coordinator__': fee}."""
    breakdown = {}
    for node_id in node_ids:
        node = models.get_node(node_id)
        if node is None:
            continue  # unknown node in the report -> earns nothing
        layers = node["layer_end"] - node["layer_start"] + 1
        amount = round(reward_for_layers(layers), 6)
        models.credit(node_id, amount, count_request=True)
        breakdown[node_id] = amount

    fee = round(config.NRN_PER_REQUEST * config.COORDINATOR_FEE, 6)
    models.credit(config.COORDINATOR_LEDGER_ID, fee, count_request=False)
    breakdown[config.COORDINATOR_LEDGER_ID] = fee
    return breakdown
