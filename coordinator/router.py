"""NEURON coordinator — request routing.

Given the currently-online nodes, assemble an ordered pipeline that covers every
layer 0..TOTAL_LAYERS-1 contiguously, or report which layers are uncovered.
"""
from coordinator import config, models


def build_chain(now=None):
    """Return (chain, missing).

    chain   : online nodes ordered by layer_start that contiguously cover layers
              0..TOTAL_LAYERS-1 (usable only if `missing` is empty).
    missing : list of (start, end) layer ranges with no online node.
    """
    nodes = models.online_nodes(now)
    total = config.TOTAL_LAYERS

    by_start = {}
    for n in nodes:
        by_start.setdefault(n["layer_start"], []).append(n)

    chain, missing, cursor = [], [], 0
    while cursor < total:
        candidates = by_start.get(cursor)
        if not candidates:
            later = [s for s in by_start if s > cursor]
            gap_end = (min(later) - 1) if later else (total - 1)
            missing.append((cursor, gap_end))
            cursor = gap_end + 1
            continue
        best = max(candidates, key=lambda n: n["layer_end"])   # reach farthest
        chain.append(best)
        cursor = best["layer_end"] + 1

    return chain, missing


def chain_public(chain):
    """Client-facing view of the chain (node_id + address + layer range)."""
    return [
        {
            "node_id": n["node_id"],
            "ip": n["tailscale_ip"],
            "port": n["port"],
            "layers": [n["layer_start"], n["layer_end"]],
        }
        for n in chain
    ]


def missing_str(missing):
    return ", ".join(f"{a}-{b}" for a, b in missing)
