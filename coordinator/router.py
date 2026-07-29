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
import collections
import random

from coordinator import config, models

# What we assume about a node that has never self-measured (`ms_per_layer` is NULL until the
# node runs benchmark.py). Deliberately pessimistic-but-not-crippling: an unmeasured node
# should still get traffic, just not be preferred over one known to be fast.
DEFAULT_MS_PER_LAYER = 40.0


def stage_ms(n):
    """Expected time for this node to run its OWN segment once, in ms.

    Petals scores a server by min(network, compute) throughput. We only have the compute half
    -- `ms_per_layer`, self-measured by benchmark.py (Session 14) -- because nothing measures
    per-node RTT yet. That makes this an underestimate of cost for a distant node, which is
    the honest limitation to fix when node-to-node latency is actually measured. It is still
    strictly better than the uniform assumption it replaces.
    """
    layers = max(int(n["layer_end"]) - int(n["layer_start"]) + 1, 1)
    ms = n.get("ms_per_layer")
    try:
        ms = float(ms) if ms else DEFAULT_MS_PER_LAYER
    except (TypeError, ValueError):
        ms = DEFAULT_MS_PER_LAYER
    return layers * max(ms, 1e-6)


def throughput(n):
    """Requests/sec this node can push through its own segment. Petals' `server throughput`."""
    return 1000.0 / stage_ms(n)


def segment_throughput(nodes):
    """Petals' `block throughput`: the summed throughput of everyone serving this segment."""
    return sum(throughput(n) for n in nodes)


def fastest_pick(rng=random):
    """Replica chooser weighted by measured throughput -- Petals mechanism 2, adapted.

    The paper has each CLIENT ping servers and beam-search for the lowest-latency path. That
    works there because routing is decentralised, so different clients naturally pick
    different servers. NEURON routes centrally, so a straight argmin would send *every*
    request to whichever node is fastest, serialise behind that node's `compute_lock`, and
    undo [P16]'s whole point about spreading load. Weighted-random keeps both properties: a
    node twice as fast gets twice the traffic, and a slow node still contributes instead of
    being starved.

    Note the segment cursor walk is greedy per segment, and for an additive path cost with
    independent per-segment choices that IS the optimal path -- no beam needed at this shape.
    """
    def pick(replicas):
        if len(replicas) == 1:
            return replicas[0]
        weights = [throughput(n) for n in replicas]
        total = sum(weights)
        if total <= 0:
            return rng.choice(replicas)
        return rng.choices(replicas, weights=weights, k=1)[0]
    return pick


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
    pick = pick or fastest_pick()
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
    fill the first one; otherwise the chain is complete, so replicate the segment that has the
    FEWEST replicas today. Returns {layer_start, layer_end, role, reason}. Advisory only; the
    node still registers normally. `total` = serving model's layer count.

    This used to always replicate the LAST segment, which silently capped the whole network's
    throughput at one request at a time. A pipeline is only as parallel as its least-replicated
    stage: with layers split 0-9 / 10-18 / 19-27, seven machines joining a 3-node network all
    piled onto 19-27, so every request still funnelled through the single node holding 0-9 and
    the single node holding 10-18 -- and node_server.py's module-level `compute_lock` serialises
    each machine's forward pass, so those two ran strictly one request at a time. Ten machines
    delivered one machine's throughput. Balancing replicas across stages is what actually turns
    added machines into added concurrency (each complete extra set of replicas = one more
    request served in parallel). Ties break toward the EARLIEST segment: every request traverses
    the front of the pipeline first, so a shortfall there throttles everything behind it.
    """
    total = total if total is not None else config.TOTAL_LAYERS
    chain, missing = build_chain(now, total=total)
    if missing:
        start, end = missing[0]
        return {"layer_start": start, "layer_end": end, "role": "fill-gap",
                "reason": f"chain is missing layers {start}-{end}"}
    # Petals mechanism 1: a joining server takes the interval whose current total THROUGHPUT
    # is lowest -- i.e. it removes the actual bottleneck. Counting replicas (what this did
    # before) treats one slow laptop as equal to one fast desktop, so a stage could look
    # well-replicated while still being the slowest thing in the pipeline. With no
    # ms_per_layer data anywhere this reduces to the old count-based behaviour, since every
    # node then scores identically -- so an unmeasured network behaves exactly as before.
    nodes = [n for n in models.online_nodes(now) if n.get("eligible")]
    by_seg = collections.defaultdict(list)
    for n in nodes:
        by_seg[(n["layer_start"], n["layer_end"])].append(n)
    idx, seg = min(enumerate(chain),
                   key=lambda t: (segment_throughput(
                       by_seg[(t[1]["layer_start"], t[1]["layer_end"])]), t[0]))
    key = (seg["layer_start"], seg["layer_end"])
    members = by_seg[key]
    tput = segment_throughput(members)
    return {"layer_start": seg["layer_start"], "layer_end": seg["layer_end"],
            "role": "replica-balance",
            "reason": f"chain is complete; layers {seg['layer_start']}-{seg['layer_end']} are "
                      f"the lowest-throughput stage ({tput:.2f} req/s across {len(members)} "
                      f"node(s), stage {idx + 1} of {len(chain)}) -- copying it removes the "
                      f"current bottleneck"}


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
