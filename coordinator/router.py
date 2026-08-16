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
import time

from coordinator import balancer, config, model_tiers, models

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
    # ROUTING USES THE MEASUREMENT WHATEVER ITS AGE. An old number measured on the machine beats
    # a fresh guess about it: `DEFAULT_MS_PER_LAYER` is 40.0, so ageing a figure out REPLACES
    # evidence with a pessimistic assumption.
    #
    # Live 2026-08-11, and it was a self-inflicted product regression. Expiry reached the live
    # coordinator for the first time and scored `node-c-pavilion` -- 4 cores, 99% reliable,
    # measured at 11.0 -- at 40.0 because its reading was 12.9 h old, while `82cbee` kept its
    # fresher 20.4. That INVERTS the preference between them, and chat fell to 0.05 tok/s.
    #
    # The TTL exists for [P34]'s outlier trap: a figure taken while a machine thrashed excludes
    # it via REPLICA_SLOWDOWN_LIMIT, so it never serves, so nothing revises it. That reasoning
    # assumed the hourly re-measurement in `agent.remeasure_loop` -- which is a **0.20** feature.
    # On a 0.19 fleet nothing re-measures on a timer, so expiry has no second measurement to fall
    # back to and is pure loss. The outlier case is still covered: REPLICA_SLOWDOWN_LIMIT drops
    # an 8x replica before weighting, and an agent re-measures on restart.
    #
    # REVISIT WHEN 0.20 IS ON EVERY NODE: with hourly re-measurement the TTL becomes correct
    # again, and this should go back to `ms_per_layer_fresh`.
    ms = n.get("ms_per_layer")
    try:
        ms = float(ms) if ms else None
    except (TypeError, ValueError):
        ms = None
    return layers * max(DEFAULT_MS_PER_LAYER if ms is None else ms, 1e-6)


def ms_per_layer_fresh(n):
    """This node's measured ms/layer if it is still believed, else None.

    A measurement EXPIRES. It is taken once at agent startup, and one taken while the machine
    happened to be thrashing was believed forever -- 4150 ms/layer against 8-22 for its peers,
    live 2026-08-10 ([P34]). Worse together with REPLICA_SLOWDOWN_LIMIT: an outlier is excluded
    from routing, so it never serves, so nothing ever revises it. Ageing back to the default
    prior is the way back in, and it is the same treatment a never-measured node already gets.

    ONE definition of "believed", because two of them drifted. Routing expired the figure here
    while BOTH dashboards went on printing the raw column as current -- so on 2026-08-11 an
    operator read `4146.6` off the node table as live evidence about a node the router had
    already stopped scoring on it, and that node's own dashboard told its volunteer the same
    thing about their hardware. That is [P37]'s lesson -- a stale diagnostic field read as a
    live one -- reappearing in the surface a person actually looks at.

    A NULL `ms_per_layer_at` is NOT stale: it is a row written before the column existed
    (models.py:225 backfills it, but a legacy or hand-inserted row can still carry NULL).
    Reading it as expired would silently reset every such node to the default prior.
    """
    ms = n.get("ms_per_layer")
    if not ms:
        return None
    at = n.get("ms_per_layer_at")
    if at is not None:
        try:
            if (time.time() - float(at)) > config.MS_PER_LAYER_TTL_S:
                return None
        except (TypeError, ValueError):
            pass
    try:
        return float(ms)
    except (TypeError, ValueError):
        return None


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
        # Drop catastrophic outliers BEFORE weighting. Weighted-random keeps a slow node
        # contributing, which is right for "somewhat slower" -- but it also means a node orders
        # of magnitude slower still wins occasionally, and when it does the whole request crawls
        # because a chain is only as fast as its slowest stage. Live 2026-08-10:
        # `agent-bhpc012101` reported 4150 ms/layer against 8-22 ms for its peers (a stale
        # measurement taken while it was thrashing under a migration), and answers came back at
        # 0.24 tok/s. A node that far out is not "slow", it is broken or lying, and its share of
        # traffic should be zero rather than small.
        best = max(throughput(n) for n in replicas)
        usable = [n for n in replicas
                  if best <= 0 or throughput(n) * config.REPLICA_SLOWDOWN_LIMIT >= best]
        if not usable:                       # every replica is an outlier of the best; keep all
            usable = replicas                # rather than emptying a segment of the chain
        if len(usable) == 1:
            return usable[0]
        weights = [throughput(n) for n in usable]
        total = sum(weights)
        if total <= 0:
            return rng.choice(usable)
        return rng.choices(usable, weights=weights, k=1)[0]
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
            # Clamp: a node holding a layer_start at or beyond `total` must not stretch the gap
            # past the end of the model. Reachable whenever a range outlives the model it was
            # cut for -- a migration onto a SMALLER model leaves stale ranges behind until each
            # node reloads. Unclamped, self-heal read the gap as (start .. that node's start - 1)
            # and planned a slice tens of layers longer than the model has.
            gap_end = min(gap_end, total - 1)
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


def chain_shape(nodes, total, pick=None):
    """The SHAPE of the chain a driver would be handed: how many stages, and whether that count
    is one any driver can actually route.

    Deliberately a DIFFERENT question from coverage, because conflating the two is a live bug.
    `covering_and_missing` answers "is every layer served somewhere" -- what /status, the
    dashboard and `self_heal` have always asked -- and a roster can answer that perfectly while
    being unroutable. Two eligible nodes on 0-27 and 0-9 cover all 28 layers, but `_walk` takes
    `max(layer_end)` from cursor 0, so the chain is ONE stage and `node_a.coord_get_chain`
    refuses it. That exact state was live on 2026-08-09 for hours, reporting
    `total_layers_covered: 28/28`, `uncovered_layers: []` and `network_healthy: true` the whole
    time, while every chat died and each attempt still took a wallet hold. PROBLEMS.md [P32].

    `nodes` is the FULL roster, filtered internally -- same convention as
    `covering_and_missing`, so callers that already hold the roster need no extra DB read.

    The chooser is deterministic (`_first_replica`) for the same reason placement's is: this
    DESCRIBES a roster rather than routing a request, and a description that varies between two
    calls a second apart is not a description. Replica choice cannot change the answer anyway --
    every replica tied at a cursor shares its `layer_end`, so the walk advances identically.
    """
    pick = pick or _first_replica
    elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    chain, missing, _covering = _walk(elig, total, pick)
    stages = len(chain)
    ranges = [[n["layer_start"], n["layer_end"]] for n in chain]
    # Stage 1 must be EXACTLY the driver's shard. node_a.coord_get_chain refuses anything else,
    # so a chain that is the right length over full coverage is still dead if stage 1 is the
    # wrong width. Live on 2026-08-10: a halted migration left [[0,16],[17,23],[24,27]] — three
    # stages, 28/28 covered, and this function called it routable while every chat would have
    # been refused. Counting stages answered two thirds of the question.
    want_stage1 = [0, config.DRIVER_STAGE1_LAYERS - 1]
    stage1_ok = bool(ranges) and ranges[0] == want_stage1
    return {
        "stages": stages,
        "ranges": ranges,
        "stage1_ok": stage1_ok,
        "expected_stage1": want_stage1,
        # `missing` is checked as well as the stage count, not instead of it: a chain that stops
        # at a gap still has a plausible-looking number of stages, and reporting that as routable
        # would be the same class of half-answer this function exists to end.
        "routable": (not missing and stage1_ok
                     and config.MIN_PIPELINE_STAGES <= stages <= config.PIPELINE_STAGES),
    }


def canonical_assignment(nodes, total, s1=None, max_stages=None, serving_model_id=None):
    """The ONE routable shape for this roster: stage 1 is exactly the driver's shard, the rest
    of the model is split across at most `max_stages - 1` more nodes, and every remaining
    machine REPLICATES a stage instead of becoming another one.

    This is `pin_layers.sh` expressed server-side, and it exists because that script being a
    manual step is the actual bug. Every join and every leave could push the chain into a shape
    no driver accepts -- four stages, or a stage 1 of the wrong width -- and the only repair was
    a human noticing and running a command. On 2026-08-10 that happened repeatedly overnight
    while nobody was awake.

    Safe to apply automatically only because placement ownership landed first (PROBLEMS.md
    [P32]): before that, a node re-registering would overwrite whatever this decided, and an
    auto-repair would have fought the roster every 60 seconds instead of fixing it.

    Stability is deliberate, not incidental:
      * whoever already holds stage 1 KEEPS it -- that node is the driver, and moving the driver
        is what breaks chat even when the chain looks legal;
      * everyone else is ordered by their current layer_start, so a node stays near the range it
        already has on disk and re-splits move as few slices as possible.

    Returns [] when the roster genuinely cannot form a chain (fewer than MIN_PIPELINE_STAGES
    eligible nodes) -- there is no shape to apply, and saying so is better than inventing one.
    """
    s1 = config.DRIVER_STAGE1_LAYERS if s1 is None else s1
    max_stages = config.PIPELINE_STAGES if max_stages is None else max_stages
    elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    if len(elig) < config.MIN_PIPELINE_STAGES or total <= s1:
        return []

    # MEMORY. An even split is a fine default and a bad promise -- the same lesson [P26] cost a
    # live outage for, and this function reproduced it: on the 1.5B floor an even split is
    # harmless, but on the 7B tier it would hand an 8 GB office PC 9 layers (~8.4 GB at fp32).
    # Each stage is capped at what its node can actually hold, and whatever that cap sheds
    # spills onto the next stage. gb_per_layer of None (an unknown model) means no constraint,
    # which is exactly the behaviour from before tiers carried the figure.
    gpl = model_tiers.gb_per_layer_for(serving_model_id) if serving_model_id else None
    head = model_tiers.head_gb_for(serving_model_id) if serving_model_id else None

    def _can_drive(n):
        """Can this node hold stage 1 AND the head it comes with? ([P44])

        The driver's capacity was the one this function never asked about: `out[0]` was emitted
        unconditionally while `caps` covered only `rest`. That put the largest fixed cost in the
        model -- the embedding and lm_head, which stage 1 runs -- on the single machine nothing
        checked. Unknown footprint means unknown, i.e. no constraint, as everywhere else here.
        """
        if not gpl:
            return True
        cap = balancer.max_layers_for(n, gpl, head_gb=head)
        return cap is None or cap >= s1

    want1 = (0, s1 - 1)
    holding = [n for n in elig if (n["layer_start"], n["layer_end"]) == want1]
    # The incumbent keeps the seat -- UNLESS it cannot hold it. Stability is the reason the
    # incumbent rule exists, and it is worth a great deal; it is not worth keeping a driver that
    # will be OOM-killed on the first token, because that is not stability, it is a chain that
    # breaks every time it is repaired.
    holding = [n for n in holding if _can_drive(n)]
    if holding:
        driver = sorted(holding, key=lambda n: n["node_id"])[0]
    else:
        # No incumbent. Prefer the machine most able to carry the head, deterministically --
        # stage 1 also runs the embedding and the lm_head, so it is the worst place for the
        # weakest node. Deterministic ties keep two sweeps a second apart from disagreeing.
        # `balancer.head_node_index` mirrors this ordering and is pinned to it by a test.
        by_ram = sorted(elig, key=lambda n: (-(n.get("ram_gb") or 0.0), n["node_id"]))
        able = [n for n in by_ram if _can_drive(n)]
        if not able:
            # Nobody can hold stage 1 of this model. There IS no routable shape, which is the
            # same answer this function already gives for too few nodes -- and the remedy is a
            # smaller model (the TierController's demotion), not a plan that OOM-kills whoever
            # drew the short straw. Returning a plan here would be the 2026-08-07 promotion
            # arriving through the repair path instead of the promote path.
            return []
        driver = able[0]

    rest = sorted([n for n in elig if n["node_id"] != driver["node_id"]],
                  key=lambda n: (n["layer_start"], n["node_id"]))
    n_stages = min(max_stages - 1, len(rest))
    remaining = total - s1
    base, extra = divmod(remaining, n_stages)

    caps = [balancer.max_layers_for(rest[i], gpl) if gpl else None for i in range(n_stages)]

    out = [{"node_id": driver["node_id"], "layer_start": 0, "layer_end": s1 - 1}]
    segments, cur = [], s1
    for i in range(n_stages):
        cnt = base + (1 if i < extra else 0)
        left = total - cur                       # never leave the tail uncovered
        if caps[i] is not None and i < n_stages - 1:
            cnt = max(1, min(cnt, int(caps[i])))
        cnt = min(cnt, left) if i < n_stages - 1 else left
        seg = (cur, cur + cnt - 1)
        segments.append(seg)
        out.append({"node_id": rest[i]["node_id"], "layer_start": seg[0], "layer_end": seg[1]})
        cur += cnt
        if cur >= total:
            n_stages = i + 1                     # the model ran out before the stages did
            break

    # Everyone left over replicates the least-replicated segment. Added machines become
    # throughput rather than a deeper pipeline ([P16]) -- and a fourth STAGE is precisely the
    # unroutable state this function exists to prevent.
    depth = {seg: 1 for seg in segments}
    for n in rest[n_stages:]:
        seg = min(segments, key=lambda s: (depth[s], s[0]))
        depth[seg] += 1
        out.append({"node_id": n["node_id"], "layer_start": seg[0], "layer_end": seg[1]})
    return out


def assignment_overflow(nodes, assignment, serving_model_id):
    """Layers this assignment gives nodes that cannot hold them. [] when every node fits.

    **`canonical_assignment` deliberately overfills the last stage and this does not change
    that.** `cnt = min(cnt, left) if i < n_stages - 1 else left` gives the tail to the final
    node whatever its cap says, and that is the right call: a gap means not one request
    completes, while an over-full node might still swap rather than die. Covering the tail
    beats refusing to.

    What was wrong is that it did so SILENTLY. `main.py` applied the plan and printed
    `routable=True`, which is exactly the half-answer `chain_shape` was written to end -- a
    roster can satisfy every structural check and still be dead, and "the chain is repaired" is
    not the same claim as "every node can hold what it was given". So the overflow is returned
    rather than swallowed, and the repair log says which node is over and by how much.

    Separate from `canonical_assignment` rather than folded into it because that function
    returns a LIST and every caller and test unpacks it as one. A pure follow-up query costs
    nothing and changes no signature.

    Returns [{node_id, layers, max_layers, over}], worst first.
    """
    gpl = model_tiers.gb_per_layer_for(serving_model_id) if serving_model_id else None
    if not gpl:
        return []
    head = model_tiers.head_gb_for(serving_model_id)
    by_id = {n["node_id"]: n for n in nodes}
    out = []
    for a in assignment:
        n = by_id.get(a["node_id"])
        if not n:
            continue
        got = a["layer_end"] - a["layer_start"] + 1
        # The head follows layer 0, wherever this plan put it -- not whichever machine is
        # biggest. This describes an assignment that already exists.
        cap = balancer.max_layers_for(n, gpl, head_gb=(head if a["layer_start"] == 0 else None))
        if cap is not None and got > cap:
            out.append({"node_id": a["node_id"], "layers": got, "max_layers": cap,
                        "over": got - cap})
    return sorted(out, key=lambda x: -x["over"])


def _first_replica(replicas):
    """Deterministic replica chooser, for placement only.

    `pick` cannot change which segments exist or where the gaps are -- every replica tied at a
    cursor shares the same `layer_end`, so the walk advances identically whoever is chosen --
    and placement only ever reads segment boundaries. Using a fixed chooser instead of the
    weighted-random routing one keeps the ADVICE reproducible: two nodes asking the same
    question a second apart get the same answer, and the reason string doesn't shuffle.
    """
    return replicas[0]


def placement_roster(now=None, exclude=None):
    """Every ONLINE node -- eligible or not -- optionally minus one node id.

    Routing must only ever use eligible nodes (`build_chain`); PLACEMENT must not. A
    probationary node has already been handed a layer range and has already downloaded that
    slice, so it is a real claim on that segment even though it cannot serve traffic yet.
    Advising newcomers from the eligible-only view makes every unverified node invisible, and
    the consequence is not subtle: on 2026-08-07 the live network had three machines all sitting
    on layers 0-13 and layers 21-27 covered by nobody. Each had joined while only the trusted
    node (14-20) was eligible, so each was told "the first gap is 0-13" -- the second and third
    joiners could not see that the first had already taken it. Three downloads of the same slice,
    a permanently incomplete chain, and DEGRADED on the dashboard.

    `exclude` drops one node from the view, which answers a different and useful question: where
    would this node be placed if it weren't already here? If its current range is genuinely
    needed the walk without it shows a gap exactly there and the advice is unchanged, so a node
    can safely re-ask without churning (see the agent's re-placement path).
    """
    return [n for n in models.online_nodes(now) if n["node_id"] != exclude]


def suggest_placement(now=None, total=None, exclude=None):
    """Advise a JOINING node which layer slice to serve (Session 20 — zero-config open join).

    A stranger shouldn't pick layer numbers. Policy: if the chain has a coverage GAP, fill the
    first one; otherwise the chain is complete, so replicate the segment that has the FEWEST
    replicas today. Returns {layer_start, layer_end, role, reason}. Advisory only; the node
    still registers normally. `total` = serving model's layer count.

    Computed over `placement_roster` -- ALL online nodes, not just the eligible ones routing
    would use. See that function for why; the short version is that an unverified node's range
    is still taken.

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
    nodes = placement_roster(now, exclude)
    chain, missing, _covering = _walk(nodes, total, _first_replica)
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
