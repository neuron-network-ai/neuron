"""
coordinator/balancer.py — heterogeneity-aware layer assignment  [Session 14]

Nodes are not equal: on this trio the Pavilion is ~14.7 ms/layer, the OptiPlex ~13.5,
the Windows PC ~11.8 — and node_a additionally carries the lm_head (a fixed per-token
cost). A naive equal split makes the slowest/most-loaded stage the bottleneck.

This solver assigns each node a contiguous slice of the L model layers so that every
stage takes ~equal wall time (minimizing the bottleneck). The driver (node_a) gets
fewer layers to pay for its head cost.

Model: node i does k_i layers at s_i ms/layer plus fixed cost H_i (head on the driver,
0 elsewhere). Stage time t_i = s_i*k_i + H_i. Equalize all t_i = T with sum k_i = L:

    k_i = (T - H_i) / s_i ,   sum_i (T - H_i)/s_i = L
    =>  T = (L + sum_i H_i/s_i) / sum_i (1/s_i)

Then round the fractional k_i to integers summing to L (>=1 each) by largest remainder.
Pure Python, no torch — safe to import in the lightweight coordinator.
"""


def _apportion(raw, total, min_each=1):
    """Round fractional targets `raw` to non-negative ints summing to `total`, each
    >= min_each, using the largest-remainder (Hamilton) method."""
    n = len(raw)
    if total < min_each * n:
        raise ValueError(f"{total} layers cannot cover {n} nodes with >={min_each} each")
    extra = total - min_each * n
    targets = [max(r - min_each, 0.0) for r in raw]
    tsum = sum(targets) or 1.0
    quotas = [t / tsum * extra for t in targets]
    floors = [int(q) for q in quotas]
    order = sorted(range(n), key=lambda i: quotas[i] - floors[i], reverse=True)
    for i in range(extra - sum(floors)):
        floors[order[i]] += 1
    return [min_each + floors[i] for i in range(n)]


def max_layers_for(node, gb_per_layer, headroom=0.75):
    """How many layers this node can actually HOLD, from its reported free RAM.

    The balancer optimises for TIME and knew nothing about memory, which is fine until the
    model stops fitting. Measured 2026-07-30: Llama-3.1-8B at fp32 is 0.87 GB/layer, so an
    equal 3-way split assigns 9.3 GB to a machine with 5-6 GB free -- the balancer would
    have proposed it and the node would have been OOM-killed. A pipeline stage that dies is
    infinitely slower than a slow one, so memory is a HARD constraint and speed is the thing
    to optimise inside it.

    `headroom` leaves a quarter of free RAM alone: these are machines somebody is using, and
    the resident figure excludes the KV cache, the transient fp32 cast in CastLinear, and
    the process itself.
    """
    free = node.get("ram_free_gb")
    if not free or not gb_per_layer:
        return None                     # unknown -> no constraint, same as before
    return max(int((float(free) * headroom) / gb_per_layer), 1)


def solve(nodes, total_layers, gb_per_layer=None):
    """nodes: list of {"node_id", "ms_per_layer", "head_ms"(optional), "ram_free_gb"(optional)}
    in PIPELINE ORDER (driver first). Returns a list of assignments with contiguous layer
    ranges and the predicted per-stage time.

    `gb_per_layer` (when known) turns each node's free RAM into a hard cap on its layer
    count -- see max_layers_for. Without it the behaviour is exactly as before.
    """
    if not nodes:
        return []
    s = [max(float(n["ms_per_layer"]), 1e-6) for n in nodes]
    H = [float(n.get("head_ms", 0.0)) for n in nodes]
    inv = [1.0 / si for si in s]
    T = (total_layers + sum(H[i] * inv[i] for i in range(len(nodes)))) / sum(inv)
    raw = [max((T - H[i]) * inv[i], 0.0) for i in range(len(nodes))]
    ks = _apportion(raw, total_layers, min_each=1)

    # Memory is a hard constraint; speed is optimised inside it. Shift layers off any node
    # that cannot hold its time-optimal share onto nodes with room, cheapest-first. If the
    # network genuinely cannot hold the model, `capacity_shortfall` says so rather than
    # returning a plan that OOM-kills a volunteer's machine.
    caps = [max_layers_for(n, gb_per_layer) for n in nodes]
    if any(c is not None for c in caps):
        caps = [c if c is not None else total_layers for c in caps]
        for _ in range(total_layers):
            over = [i for i in range(len(ks)) if ks[i] > caps[i]]
            if not over:
                break
            room = [i for i in range(len(ks)) if ks[i] < caps[i]]
            if not room:
                break                      # nowhere left to put it -- reported below
            src = max(over, key=lambda i: ks[i] - caps[i])
            dst = min(room, key=lambda i: s[i])     # the fastest node with space
            ks[src] -= 1
            ks[dst] += 1

    out, start = [], 0
    for i, n in enumerate(nodes):
        end = start + ks[i] - 1
        out.append({
            "node_id": n["node_id"],
            "layer_start": start,
            "layer_end": end,
            "layers": ks[i],
            "stage_ms": round(s[i] * ks[i] + H[i], 2),
        })
        start = end + 1
    return out


def bottleneck_ms(assignment):
    """The slowest stage = the pipeline's per-token wall-clock floor."""
    return max((a["stage_ms"] for a in assignment), default=0.0)


def equal_split(nodes, total_layers):
    """Naive baseline: split layers as evenly as possible, ignoring speed/head."""
    return solve([{"node_id": n["node_id"], "ms_per_layer": 1.0} for n in nodes],
                 total_layers)


def plan(nodes, total_layers):
    """Full comparison: the balanced assignment vs. the naive equal split, scored by
    predicted bottleneck (lower = faster)."""
    # No node has self-measured yet -- the normal state of a freshly deployed network, since
    # ms_per_layer stays NULL until benchmark.py runs. This used to fall through to the
    # max() below, which raises on an empty sequence, so GET /network/plan answered 500
    # instead of "no data yet" -- and its own guard for that case sat one line further down,
    # unreachable. Found on the live coordinator with three brand-new agents online.
    if not nodes:
        return {"assignment": [], "balanced_bottleneck_ms": 0.0,
                "equal_split_bottleneck_ms": 0.0, "speedup_vs_equal": 1.0,
                "total_layers": total_layers,
                "note": "no online eligible node has reported ms_per_layer yet"}
    balanced = solve(nodes, total_layers)
    # score the equal split using the REAL speeds so the comparison is apples-to-apples
    eq_layers = [a["layers"] for a in equal_split(nodes, total_layers)]
    s = [max(float(n["ms_per_layer"]), 1e-6) for n in nodes]
    H = [float(n.get("head_ms", 0.0)) for n in nodes]
    eq_bottleneck = max(s[i] * eq_layers[i] + H[i] for i in range(len(nodes)))
    bal_bottleneck = bottleneck_ms(balanced)
    return {
        "assignment": balanced,
        "balanced_bottleneck_ms": round(bal_bottleneck, 2),
        "equal_split_bottleneck_ms": round(eq_bottleneck, 2),
        "speedup_vs_equal": round(eq_bottleneck / bal_bottleneck, 3) if bal_bottleneck else 1.0,
        "total_layers": total_layers,
    }
