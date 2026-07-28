"""NEURON coordinator — request routing.

Given the currently-online nodes, assemble an ordered pipeline that covers every
layer 0..TOTAL_LAYERS-1 contiguously, or report which layers are uncovered.

Replication (Session 18): when more than one eligible node covers the SAME farthest
segment from a given cursor, those nodes are REPLICAS. `build_chain` picks one per call
(default: at random), so concurrent requests spread across the replicas — the way to add
a machine that lifts throughput rather than deepening the pipeline (PROBLEMS.md [P8]). Each
assembled chain is still the usual driver -> middle -> last shape, so the drivers are
unchanged; only which node fills a slot varies per request.
"""
import random

from coordinator import config, models


def _walk(nodes, total, pick):
    """Cursor-walk covering 0..total-1 over an ALREADY eligible+online-filtered `nodes` list.
    Returns (chain, missing, covering_ids) -- covering_ids includes EVERY node tied at the
    farthest layer_end for each visited cursor, not just whichever `pick` chose, so a replica
    that wasn't picked THIS call is still correctly counted as covering its segment (used by
    self-heal to never mistake an un-picked replica for idle surplus)."""
    by_start = {}
    for n in nodes:
        by_start.setdefault(n["layer_start"], []).append(n)

    chain, missing, covering_ids, cursor = [], [], set(), 0
    while cursor < total:
        candidates = by_start.get(cursor)
        if not candidates:
            later = [s for s in by_start if s > cursor]
            gap_end = (min(later) - 1) if later else (total - 1)
            missing.append((cursor, gap_end))
            cursor = gap_end + 1
            continue
        # advance as far as possible; nodes tied at the farthest layer_end are replicas of
        # that segment -> choose one (default random) so load spreads across them.
        farthest = max(n["layer_end"] for n in candidates)
        replicas = [n for n in candidates if n["layer_end"] == farthest]
        chosen = pick(replicas)
        chain.append(chosen)
        covering_ids.update(n["node_id"] for n in replicas)
        cursor = chosen["layer_end"] + 1

    return chain, missing, covering_ids


def build_chain(now=None, pick=None, total=None):
    """Return (chain, missing).

    chain   : eligible online nodes ordered by layer_start that contiguously cover layers
              0..total-1 (usable only if `missing` is empty).
    missing : list of (start, end) layer ranges with no eligible online node.
    pick    : chooser used to break replica ties, `pick(list) -> node` (default random.choice;
              injectable for tests / deterministic routing).
    total   : layer count of the serving model (defaults to config.TOTAL_LAYERS; the
              coordinator passes the active serving model's layer count so routing tracks
              whichever model the network is serving).
    """
    pick = pick or random.choice
    # Only nodes cleared for live traffic are routed: excludes flagged nodes (failed
    # proof-of-compute, Session 16) AND probationary nodes (open join, Session 12 — not
    # yet verified). `eligible` = trusted or PoC-passed, and not flagged.
    nodes = [n for n in models.online_nodes(now) if n.get("eligible")]
    total = total if total is not None else config.TOTAL_LAYERS
    chain, missing, _covering_ids = _walk(nodes, total, pick)
    return chain, missing


def covering_and_missing(nodes, total, pick=None):
    """Pure (no DB access) equivalent of build_chain's gap detection, for callers that
    already have an in-memory nodes list (self-heal, called once per health sweep with the
    same list update() already fetched). `nodes` is the FULL roster (unfiltered) -- filtered
    to online+eligible internally, same convention as plan_migration. Returns (missing,
    covering_ids): covering_ids are node_ids currently covering (or tied to cover) some
    segment -- anything else eligible+online is true idle surplus."""
    pick = pick or random.choice
    elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    _chain, missing, covering_ids = _walk(elig, total, pick)
    return missing, covering_ids


def suggest_placement(now=None, total=None):
    """Advise a JOINING node which layer slice to serve (Session 20 — zero-config open join).

    A stranger shouldn't pick layer numbers. Policy: if the eligible chain has a coverage GAP,
    fill the first one; otherwise the chain is complete, so replicate the LAST segment — it adds
    throughput (S18 replica routing) and is the segment proof-of-compute can verify (S16). Returns
    {layer_start, layer_end, role, reason}. Advisory only; the node still registers normally.
    `total` = serving model's layer count (defaults to config.TOTAL_LAYERS).
    """
    total = total if total is not None else config.TOTAL_LAYERS
    chain, missing = build_chain(now, total=total)
    if missing:
        start, end = missing[0]
        return {"layer_start": start, "layer_end": end, "role": "fill-gap",
                "reason": f"chain is missing layers {start}-{end}"}
    last = chain[-1]      # complete chain => non-empty; last node defines the final segment
    return {"layer_start": last["layer_start"], "layer_end": last["layer_end"],
            "role": "replica-last",
            "reason": "chain is complete; replicate the last segment to add throughput "
                      "(verifiable via proof-of-compute)"}


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
