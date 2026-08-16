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


# Storage dtypes the RUNTIME can actually be in. `common.WEIGHT_DTYPE` maps exactly these three
# and nothing else, so a node claiming int8 or fp8 storage is claiming something no build of
# this software can do — and the claim would QUARTER the divisor and quadruple the layers it is
# handed. Believing it is the `gpu_vram_gb` mistake in a new field: under open join, registration
# needs no credential, and this is now the second reported value that turns straight into a
# memory budget. `sane_weight_dtype` is the `sane_vram_gb` of this field, and for the same reason.
BELIEVABLE_WEIGHT_DTYPES = {"fp32", "float32", "fp16", "float16", "bf16", "bfloat16"}


def sane_weight_dtype(value):
    """A storage dtype worth sizing from, normalised — or None, meaning "size this node the
    pessimistic way instead", which is the honest answer to a claim we cannot believe.

    Never a refusal: a cosmetic field must not cost a volunteer their registration ([P24]).
    """
    v = str(value or "").strip().lower()
    return v if v in BELIEVABLE_WEIGHT_DTYPES and v in _DTYPE_BYTES else None


def weight_bytes_for(node=None):
    """Bytes per parameter this node stores weights at.

    Reads `weight_dtype`, which agents report from 0.20.3. Anything unknown, absent or not
    believable falls back to ASSUMED_WEIGHT_BYTES — never to the optimistic figure, because a
    wrong guess in that direction is an OOM on somebody's personal machine.
    """
    if node:
        b = _DTYPE_BYTES.get(sane_weight_dtype(node.get("weight_dtype")))
        if b:
            return b
    return ASSUMED_WEIGHT_BYTES


def effective_gb(gb, node=None):
    """Any fp16-basis GB figure from the tier table, corrected to the dtype a node really
    stores at.

    `gb_per_layer` and `head_gb` are both on that basis and both scale by the same factor,
    because the correction is a property of the DTYPE, not of which weights are being counted.
    Kept as one function so the two figures cannot drift onto different bases -- which is the
    exact fault this correction exists to fix, one level up.

    None in, None out: an unknown footprint is not a constraint, which is what every consumer
    already assumed and is the right answer for a model injected via NEURON_MODEL_TIERS with
    no measurement behind it.
    """
    if not gb:
        return gb
    return float(gb) * (weight_bytes_for(node) / TIER_BASIS_BYTES)


def effective_gb_per_layer(gb_per_layer, node=None):
    """The tier's per-layer figure corrected to the dtype the node really stores at."""
    return effective_gb(gb_per_layer, node)

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
# Only used when a node reports total RAM and no free figure, which is still EVERY node:
# `agent.py` sends `ram_gb` (psutil *total*) and the coordinator has no column for a free one.
#
# **That is a decision, not an omission — do not "finish the job" by adding one.** The
# `ram_free_gb` branch below skips this reserve entirely (`usable = float(free)`), so a node
# reporting a free figure is sized MORE generously than one reporting a total: a 12 GB Linux box
# freshly booted reports ~11 GB free and gets an 8.25 GB budget against this path's 6.75 GB.
# That figure is a SNAPSHOT taken during registration, it carries no age, and nothing re-reads
# it — so a machine that registered at 3am idle keeps a 3am-idle budget all day, and the layers
# it was handed on that basis are still resident when its owner opens a browser. A number that
# was true when captured and false when applied is [P34], and this would be the third instance.
# Reporting it needs an age and a re-read (heartbeat, not registration) before it is safe, and
# the pessimistic path costs throughput rather than somebody's machine.
#
# Until this fallback existed `max_layers_for` returned None for every real node,
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


def max_layers_for(node, gb_per_layer, headroom=0.75, head_gb=None):
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

    `head_gb` is weight this node must hold ON TOP of its layers — the embedding, and the
    `lm_head` when the model does not tie them. It belongs to the DRIVER, which runs the output
    head (`common.apply_lm_head`), and it is charged against the same post-headroom budget the
    layers come out of. `model_tiers` called this out as unmodelled and said so in a comment;
    the figure it wanted now exists, measured, via `tools/measure_model.py`.
    Why it matters at all: it is a fixed cost, so it hurts most on the SMALLEST machine and on
    the FEWEST nodes — i.e. exactly the two-machine capacity case, where it is the difference
    between a plan and an OOM. On Qwen3-4B the embedding is 389M params: 1.56 GB at fp32,
    against an 8 GB machine's 3.75 GB budget.

    Never returns 0: `solve` gives every node at least one layer, and a node that cannot hold
    even one layer is a node that should not be in the pipeline at all — an eligibility
    decision, not one for this function. `capacity_shortfall` is what says "this network cannot
    hold this model".

    The floor is why a driver charged a head BIGGER than its whole budget still reports 1 rather
    than 0, understating the shortfall by a layer. Left deliberately: dropping the floor would
    let `solve` emit a zero-width range for stage 1, and a degenerate assignment on the live
    placement path is a worse failure than a one-layer error in a refusal that is already going
    to refuse. `head_node_index` puts the head on the biggest machine anyway, mirroring the
    router, which is what keeps that case remote.
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
    # on an fp16 basis that nothing in the shipped build uses -- see effective_gb.
    gb_per_layer = effective_gb(gb_per_layer, node)
    if not gb_per_layer:
        return None
    # The head is weight like any other, so it comes out of the same post-headroom budget the
    # layers do. The quarter held back by `headroom` stays held back on top of it.
    budget = usable * headroom - (effective_gb(head_gb, node) or 0.0)
    return max(int(budget / gb_per_layer), 1)


def head_node_index(nodes):
    """Which node in `nodes` would carry the embedding + lm_head, i.e. be the driver.

    **`coordinator/router.canonical_assignment` is the authority and this function only mirrors it** — the
    machine with the most RAM, tie-broken by node_id, for the reason stated there: stage 1 runs
    the embedding and the lm_head, so it is the worst place for the weakest node. Mirrored
    rather than imported because `router` needs config and a DB and this module is deliberately
    pure; `test_head_cost.py` pins the two together so they cannot drift.

    Drift would be the expensive kind: a feasibility check that charges the head to a different
    machine from the one that ends up holding it either clears a plan that OOMs the driver, or
    refuses one that would have fitted.

    **The router's other rule — an incumbent stage-1 holder KEEPS the driver seat — is
    deliberately not mirrored.** That rule exists for STABILITY (moving the driver breaks chat)
    and is tested against the ranges nodes hold for the model being served now. Every caller
    here is asking about a DIFFERENT model, whose layer count makes the current stage-1 range
    meaningless, and whose adoption re-solves every range anyway. The residual gap is real and
    worth naming: if a small machine is the incumbent driver and a big one is not, the router
    will keep the head on the small machine while this charges it to the big one. That is a
    feasibility answer one machine too generous, not a placement that runs — `solve` charges
    the head where the pipeline order actually puts it.

    Returns None for an empty roster.
    """
    if not nodes:
        return None
    return min(range(len(nodes)),
               key=lambda i: (-(nodes[i].get("ram_gb") or 0.0),
                              str(nodes[i].get("node_id") or "")))


def layer_caps(nodes, gb_per_layer, unconstrained, head_gb=None, head_index=None):
    """Per-node hard caps in layers, in `nodes` order. A node that reports no memory at all is
    left `unconstrained` (pass the model's layer count) — exactly the behaviour from before
    memory was considered at all.

    `head_gb` is charged to exactly ONE node: `head_index` when the caller knows the pipeline
    order (`solve` does — its contract is driver-first), otherwise whichever node the router
    would make the driver. Charging it to every node would refuse networks that fit; charging
    it to none is the behaviour this replaces.
    """
    hi = head_index if head_index is not None else (head_node_index(nodes) if head_gb else None)
    return [c if c is not None else unconstrained
            for c in (max_layers_for(n, gb_per_layer, head_gb=(head_gb if i == hi else None))
                      for i, n in enumerate(nodes))]


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


def capacity_shortfall(nodes, total_layers, gb_per_layer, head_gb=None):
    """How many layers this set of nodes cannot hold BETWEEN THEM. 0 means a valid per-node
    partition exists.

    Sufficient as well as necessary: layers are interchangeable and every node's slice is
    contiguous only *within the pipeline order*, so any per-node distribution that respects the
    caps can be laid out as contiguous ranges. If the caps sum to at least `total_layers`, some
    plan fits; `fit_to_capacity` finds one.

    `head_gb` is charged to ONE node — the one `head_node_index` says would be the driver — so
    a network is not refused for a cost only one of its machines pays.

    Unknown `gb_per_layer` (a model not in the tier table) means unknown footprint, which means
    no constraint — the same answer this module gave before it knew about memory at all.
    """
    if not nodes or total_layers <= 0 or not gb_per_layer:
        return 0
    return max(0, total_layers - sum(layer_caps(nodes, gb_per_layer, total_layers, head_gb)))


def _prefers_gpu(nodes):
    """Sort key factory: among candidates that are otherwise equal, put GPU nodes first.

    This is the whole of "GPU nodes are preferred for larger layer counts" — a tie-break, not
    a weighting. It only ever changes WHICH of two equally-suitable nodes receives a layer that
    had to move anyway, so on an all-CPU network (every network today) it is a no-op.
    """
    return lambda i: (0 if nodes[i].get("has_gpu") else 1)


def solve(nodes, total_layers, gb_per_layer=None, head_gb=None):
    """nodes: list of {"node_id", "ms_per_layer", "head_ms"(optional), "ram_free_gb"(optional),
    "has_gpu"(optional), "gpu_vram_gb"(optional)} in PIPELINE ORDER (driver first). Returns a
    list of assignments with contiguous layer ranges and the predicted per-stage time.

    `gb_per_layer` (when known) turns each node's free RAM into a hard cap on its layer
    count -- see max_layers_for. Without it the behaviour is exactly as before.

    `head_gb` is charged to nodes[0], because that is what "PIPELINE ORDER (driver first)"
    means and it is the same node `head_ms` is already keyed to: the head's cost in TIME and
    its cost in MEMORY are the same weights, and they must land on the same machine.
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
    caps = [max_layers_for(n, gb_per_layer, head_gb=(head_gb if i == 0 else None))
            for i, n in enumerate(nodes)]
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


def plan(nodes, total_layers, gb_per_layer=None, head_gb=None):
    """Full comparison: the balanced assignment vs. the naive equal split, scored by
    predicted bottleneck (lower = faster).

    `gb_per_layer` makes the assignment memory-aware and adds `capacity_shortfall`: the layers
    this network cannot hold at all, which is a REFUSAL signal, not a slow plan. `head_gb` adds
    the driver's embedding + lm_head to that arithmetic.
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
    balanced = solve(nodes, total_layers, gb_per_layer, head_gb)
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
        "capacity_shortfall": capacity_shortfall(nodes, total_layers, gb_per_layer, head_gb),
    }
