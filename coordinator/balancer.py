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

**On GPUs.** Nodes report `has_gpu` / `gpu_vram_gb`, and this solver uses them in exactly one
honest way: as a tie-break when layers have to move, so a GPU machine collects them ahead of an
equally-fast CPU machine. It deliberately does NOT treat a GPU as a speed multiplier — a node's
speed is its MEASURED `ms_per_layer`, which already reflects whatever hardware it actually ran
on, and multiplying a measured number by a guessed factor would double-count.
VRAM as *capacity* is implemented and switched OFF; see GPU_EXECUTION below.
"""
import math
import os

# --------------------------------------------------------------------------- #
# Weight dtype: the tier table's figures and the runtime disagreed, by exactly 2x
# --------------------------------------------------------------------------- #
# `model_tiers.gb_per_layer` is computed at the fp16 STORAGE dtype — 2 bytes/param — and its
# comment says so explicitly, citing `common.WEIGHT_DTYPE=fp16` as the justification.
#
# **That justification is false.** `common.py:62` reads `NEURON_WEIGHT_DTYPE` and defaults to
# `"fp32"`, so every node stores weights at **4 bytes/param** unless somebody deliberately set
# that variable — and nothing in the agent, the installer or the packaged build ever does.
# `cast_linears` is a no-op at fp32 (`common.py:305`), which is the other half of the same
# story: the half-precision path exists and is not switched on.
#
# The result was that EVERY node on EVERY tier was sized against half its real footprint. Not
# only a GPU node, and not only in theory: at the 7B tier an 8 GB machine was cleared for 8
# layers = 7.46 GB of weights against a 3.75 GB budget. It has been latent only because the
# network serves the 1.5B, where 28 layers is small enough that the cap never binds.
#
# So the coordinator scales the table's figure by the dtype it believes nodes run at. The
# default is the PESSIMISTIC one (fp32), because that is what they actually run, and because
# over-estimating a slice costs throughput while under-estimating it costs a volunteer's
# machine. A node may override per-registration once it reports `weight_dtype`; until then the
# assumption is uniform and stated, rather than implicit and wrong.
TIER_BASIS_BYTES = 2.0
ASSUMED_WEIGHT_BYTES = float(os.environ.get("NEURON_ASSUMED_WEIGHT_BYTES", "4"))

_DTYPE_BYTES = {"fp32": 4.0, "float32": 4.0, "fp16": 2.0, "float16": 2.0,
                "bf16": 2.0, "bfloat16": 2.0, "int8": 1.0, "fp8": 1.0}


def weight_bytes_for(node=None):
    """Bytes per parameter this node stores weights at.

    Reads `weight_dtype` when a node reports one (agents do not yet; the field is accepted so
    the agent-side change is additive and needs no coordinator release). Anything unknown or
    absent falls back to ASSUMED_WEIGHT_BYTES — never to the optimistic figure, because a
    wrong guess in that direction is an OOM on somebody's personal machine.
    """
    if node:
        b = _DTYPE_BYTES.get(str(node.get("weight_dtype") or "").strip().lower())
        if b:
            return b
    return ASSUMED_WEIGHT_BYTES


def effective_gb_per_layer(gb_per_layer, node=None):
    """The tier's per-layer figure corrected to the dtype the node really stores at.

    None in, None out: an unknown footprint is not a constraint, which is what every consumer
    already assumed and is the right answer for a model injected via NEURON_MODEL_TIERS with
    no measurement behind it.
    """
    if not gb_per_layer:
        return gb_per_layer
    return float(gb_per_layer) * (weight_bytes_for(node) / TIER_BASIS_BYTES)

# May a node's VRAM be counted toward how many layers it can HOLD?
#
# **No.** Turned back off on 2026-08-08, having been on since Session 42.
#
# Session 42 turned it on because `common.py` had gained a device resolver that moves a shard
# onto CUDA. That reasoning was checked against the code that actually runs on a volunteer's
# machine and does not survive it:
#
#   `agent/node_server.py` loads its shard through `slice_downloader.load_slice_model`, and
#   `slice_downloader.py:299` returns `common.cast_linears(model)` — with NO device move.
#   `common.move_model_to_device` has one caller in the whole repo (`common.py:256`), on the
#   bench/verifier/dev path that no agent ever reaches. On top of that the shipped wheel is
#   `torch 2.4.1+cpu` (`dist/neuron-agent/_internal/torch/version.py:4`), which cannot use a
#   card at all.
#
# So a GPU node computes in SYSTEM RAM while this flag sized it by VRAM, and with no OS reserve
# on that branch at all. A volunteer with a 12 GB card and 8 GB of RAM was handed 19 layers of
# Qwen2.5-7B (12 * 0.75 / 0.466 GB per layer) — 8.9 GB of weights at the tier's fp16 basis, and
# ~17.7 GB at the fp32 the runtime actually defaults to, to be held in 8 GB. That is the exact
# OOM `max_layers_for` exists to prevent, caused by the function that prevents it.
#
# **Before flipping this back to True**, confirm the loader really moves weights to the device
# (see the Phase 4b tripwire in `test_device_path.py`) AND that the shipped torch is a CUDA
# build. A device resolver existing is not the same as the pipeline using it.
GPU_EXECUTION = False

# How much of a node's TOTAL RAM is assumed to be already spoken for — the OS plus whatever the
# volunteer is actually doing with their own machine.
#
# Only used when a node reports total RAM and no free figure, which today is EVERY node:
# `agent.py` sends `ram_gb` (psutil *total*) at registration and the coordinator has no column
# for a free one. Until this fallback existed `max_layers_for` returned None for every real node,
# so the hard memory cap below was live in the unit tests and DORMANT IN PRODUCTION — which is
# how the 2026-08-07 auto-promotion to Qwen2.5-7B handed 9–10 layers (~4.5 GB of weights) to
# machines with 8 GB of total RAM.
#
# 3 GB is Windows 11 idling with a browser open, on the machines in this project. It is
# subtracted BEFORE `headroom`, which then covers the KV cache, the transient fp32 cast in
# CastLinear, and the python process itself.
RAM_OS_RESERVE_GB = 3.0

# The same idea for VRAM, and it is not optional: this is somebody's own graphics card and they
# are looking at a desktop it is drawing. `engine/local_gguf.py:154` reserves exactly this for
# exactly this reason on the local path. The VRAM branch had NO reserve at all while the RAM
# branch had 3 GB, so a GPU node was sized more aggressively than a CPU one — backwards, since
# a card that runs out of memory takes the display with it.
VRAM_OS_RESERVE_GB = 1.5

# Above this, a reported VRAM figure is not believed. The largest accelerator anyone could
# plausibly volunteer today is ~141 GB (H200); 192 leaves room to be wrong without leaving room
# to be absurd.
MAX_PLAUSIBLE_VRAM_GB = 192.0


def sane_vram_gb(value):
    """A VRAM figure worth sizing a slice from, or None if it is not believable.

    `gpu_vram_gb` arrives in the body of `POST /node/register`, which under open join needs no
    credential — and it is the only reported field that turns directly into a memory budget.
    Nothing else a node claims can make the coordinator hand it more work: `cores` and `ram_gb`
    are advisory, and `ms_per_layer` is bounded by having to actually serve at that speed.

    Unbelievable means: not a number, NaN or infinite, non-positive, or larger than
    MAX_PLAUSIBLE_VRAM_GB. None means "size this node from its system RAM instead", which is
    the honest answer to a figure we do not trust, and never a refusal — see the clamp in
    `coordinator/main.py`, and [P24] for what refusing a registration over a cosmetic field
    costs a volunteer.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0 or v > MAX_PLAUSIBLE_VRAM_GB:
        return None
    return v


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

    **A GPU node is sized by its system RAM, like every other node**, because that is where its
    weights actually sit: `slice_downloader.py:299` loads a shard and never moves it to a
    device. VRAM would be counted instead — not in addition; the weights live in one place or
    the other, and summing them roughly doubles the true ceiling — but only once the loader
    really moves them and the shipped torch can use a card. `GPU_EXECUTION` is that switch, and
    it is off. Sizing by memory the pipeline cannot reach is how this function came to *cause*
    the OOM it exists to prevent.

    Memory figures in order of preference: VRAM (only with GPU_EXECUTION on, and only if
    `sane_vram_gb` believes it), then reported FREE system RAM, then TOTAL system RAM minus
    RAM_OS_RESERVE_GB. The last one is the only figure real nodes actually report — see
    RAM_OS_RESERVE_GB for why that mattered.

    Never returns 0: `solve` gives every node at least one layer, and a node that cannot hold
    even one layer is a node that should not be in the pipeline at all — an eligibility
    decision, not one for this function. `capacity_shortfall` is what says "this network cannot
    hold this model".
    """
    free = node.get("ram_free_gb")
    total = node.get("ram_gb")
    vram = sane_vram_gb(node.get("gpu_vram_gb")) if node.get("has_gpu") else None
    if GPU_EXECUTION and vram:
        usable = max(vram - VRAM_OS_RESERVE_GB, 0.0)
    elif free:
        usable = float(free)
    elif total:
        usable = max(float(total) - RAM_OS_RESERVE_GB, 0.0)
    else:
        return None                     # unknown -> no constraint, same as before
    # Corrected to the dtype this node actually stores weights at. The tier table's column is
    # on an fp16 basis that nothing in the shipped build uses -- see effective_gb_per_layer.
    gb_per_layer = effective_gb_per_layer(gb_per_layer, node)
    if not gb_per_layer:
        return None
    return max(int((usable * headroom) / gb_per_layer), 1)


def layer_caps(nodes, gb_per_layer, unconstrained):
    """Per-node hard caps in layers, in `nodes` order. A node that reports no memory at all is
    left `unconstrained` (pass the model's layer count) — exactly the behaviour from before
    memory was considered at all."""
    return [c if c is not None else unconstrained
            for c in (max_layers_for(n, gb_per_layer) for n in nodes)]


def fit_to_capacity(counts, caps, prefer=None):
    """Shift layers off any node that is over its cap onto the nodes with room.

    `prefer(i)` chooses WHICH node with room receives a displaced layer (lowest key wins);
    the default sends it to the node with the most unused headroom. `solve` passes a
    speed-based key instead, so a displaced layer lands on the fastest machine that can hold it.

    Returns `(counts, shortfall)`. A non-zero shortfall means layers are left on a node that
    cannot hold them because there is nowhere else to put them — the caller's signal to refuse
    the plan rather than propose one that OOM-kills a volunteer's machine.
    """
    counts = list(counts)
    if prefer is None:
        def prefer(i):
            return (counts[i] - caps[i], i)          # most unused headroom first
    for _ in range(sum(counts) + 1):
        over = [i for i in range(len(counts)) if counts[i] > caps[i]]
        if not over:
            break
        room = [i for i in range(len(counts)) if counts[i] < caps[i]]
        if not room:
            break                                    # nowhere left to put it -- reported below
        src = max(over, key=lambda i: counts[i] - caps[i])
        dst = min(room, key=prefer)
        counts[src] -= 1
        counts[dst] += 1
    return counts, sum(max(0, counts[i] - caps[i]) for i in range(len(counts)))


def capacity_shortfall(nodes, total_layers, gb_per_layer):
    """How many layers this set of nodes cannot hold BETWEEN THEM. 0 means a valid per-node
    partition exists.

    Sufficient as well as necessary: layers are interchangeable and every node's slice is
    contiguous only *within the pipeline order*, so any per-node distribution that respects the
    caps can be laid out as contiguous ranges. If the caps sum to at least `total_layers`, some
    plan fits; `fit_to_capacity` finds one.

    Unknown `gb_per_layer` (a model not in the tier table) means unknown footprint, which means
    no constraint — the same answer this module gave before it knew about memory at all.
    """
    if not nodes or total_layers <= 0 or not gb_per_layer:
        return 0
    return max(0, total_layers - sum(layer_caps(nodes, gb_per_layer, total_layers)))


def _prefers_gpu(nodes):
    """Sort key factory: among candidates that are otherwise equal, put GPU nodes first.

    This is the whole of "GPU nodes are preferred for larger layer counts" — a tie-break, not
    a weighting. It only ever changes WHICH of two equally-suitable nodes receives a layer that
    had to move anyway, so on an all-CPU network (every network today) it is a no-op.
    """
    return lambda i: (0 if nodes[i].get("has_gpu") else 1)


def solve(nodes, total_layers, gb_per_layer=None):
    """nodes: list of {"node_id", "ms_per_layer", "head_ms"(optional), "ram_free_gb"(optional),
    "has_gpu"(optional), "gpu_vram_gb"(optional)} in PIPELINE ORDER (driver first). Returns a
    list of assignments with contiguous layer ranges and the predicted per-stage time.

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
    gpu_first = _prefers_gpu(nodes)
    if any(c is not None for c in caps):
        caps = [c if c is not None else total_layers for c in caps]
        # fastest node with space receives a displaced layer; a GPU node wins a tie on speed
        ks, _ = fit_to_capacity(ks, caps, prefer=lambda i: (s[i], gpu_first(i)))
    else:
        caps = [total_layers] * len(nodes)

    out, start = [], 0
    for i, n in enumerate(nodes):
        end = start + ks[i] - 1
        out.append({
            "node_id": n["node_id"],
            "layer_start": start,
            "layer_end": end,
            "layers": ks[i],
            "stage_ms": round(s[i] * ks[i] + H[i], 2),
            "has_gpu": bool(n.get("has_gpu")),
            # What this node could hold; > layers means the plan is over its head and the
            # network has nowhere else to put those layers (see plan()'s capacity_shortfall).
            "max_layers": caps[i],
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


def plan(nodes, total_layers, gb_per_layer=None):
    """Full comparison: the balanced assignment vs. the naive equal split, scored by
    predicted bottleneck (lower = faster).

    `gb_per_layer` makes the assignment memory-aware and adds `capacity_shortfall`: the layers
    this network cannot hold at all, which is a REFUSAL signal, not a slow plan.
    """
    # No node has self-measured yet -- the normal state of a freshly deployed network, since
    # ms_per_layer stays NULL until benchmark.py runs. This used to fall through to the
    # max() below, which raises on an empty sequence, so GET /network/plan answered 500
    # instead of "no data yet" -- and its own guard for that case sat one line further down,
    # unreachable. Found on the live coordinator with three brand-new agents online.
    if not nodes:
        return {"assignment": [], "balanced_bottleneck_ms": 0.0,
                "equal_split_bottleneck_ms": 0.0, "speedup_vs_equal": 1.0,
                "total_layers": total_layers, "capacity_shortfall": 0,
                "note": "no online eligible node has reported ms_per_layer yet"}
    balanced = solve(nodes, total_layers, gb_per_layer)
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
        "capacity_shortfall": capacity_shortfall(nodes, total_layers, gb_per_layer),
    }
