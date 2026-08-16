"""coordinator/model_tiers.py — capacity-driven model tiering.

The network serves the BIGGEST model its current capacity can back. As contributing
nodes join, capacity grows and the network PROMOTES to a larger model; as nodes leave,
it DEMOTES. The whole thesis in one line: "the more of us, the smarter the shared model."

Two layers, both deterministically unit-testable (no wall-clock inside the logic — the
caller passes `now`, so tests are reproducible and resume-safe):

  * pure     : capacity(nodes) -> which tiers are feasible -> the biggest one
  * stateful : TierController adds HYSTERESIS — a promote margin + dwell, and a demote
               grace period — so a laptop sleeping/waking doesn't flap the whole
               network's model (the Pavilion-nap problem at network scale).

Capacity is read from the SAME node data the router/balancer already use: only nodes
that are BOTH online AND eligible count (probationary/flagged nodes can't serve, so
they can't unlock a bigger model). Tiers are data (env-overridable via
NEURON_MODEL_TIERS), so adding a tier needs no code change.

INVARIANT: TIERS is ordered small -> large, with non-decreasing requirements. A bigger
model strictly needs more nodes/RAM, so "highest feasible index" == "biggest servable".
"""
import json
import os

from coordinator import balancer


# A tier = a model + the capacity needed to serve it WITH redundancy.
#   min_nodes    : online+eligible nodes required (enough to split the pipeline AND
#                  hold `min_replicas` copies of each segment).
#   min_ram_gb   : total RAM across those nodes (must hold every replica's slices).
#   min_replicas : redundant copies of each layer segment. >=2 means one node can drop
#                  without an outage — the resilience floor for a production tier.
#   gb_per_layer : one transformer layer's weights. This is the PER-NODE question, and it is
#                  a different question from min_ram_gb: "3 nodes · 20 GB" is satisfied by
#                  8+8+12, and none of those machines can hold a third of a 7B model. Without
#                  it the network promoted itself onto Qwen2.5-7B on 2026-08-07 and handed
#                  9-10 layers to an 8 GB laptop. Consumed by balancer.max_layers_for.
#   head_gb      : the DRIVER's extra weights — the embedding, plus lm_head when untied. A
#                  FIXED cost, so its share of a machine grows as the network shrinks: noise
#                  across ten nodes, decisive across two. Measured by tools/measure_model.py;
#                  absent means unmeasured and nothing is charged, exactly as before it existed.
#   manual_only  : this tier is NEVER chosen by the capacity ladder — `feasible_tier_index`
#                  skips it, so only an operator pin can reach it. For a model being TESTED
#                  rather than served. Without it, adding an experimental tier is how a live
#                  network migrates itself onto an experiment overnight, which is the
#                  2026-08-07 auto-promotion wearing a different hat.
#
# The gb_per_layer figures are computed from each model's config at the fp16 STORAGE dtype
# (2 bytes/param), which is the same basis the min_ram_gb column was re-derived on in b5b4f22.
#
# **THAT BASIS IS NOT WHAT NODES RUN.** This comment used to justify it with
# "common.WEIGHT_DTYPE=fp16". It is not: `common.py:62` reads NEURON_WEIGHT_DTYPE and defaults
# to **fp32**, nothing in the agent or installer sets it, and `cast_linears` is a no-op at fp32.
# So every figure below describes half of a real node's footprint, and the wrong half was the
# generous one. Corrected in ONE place -- `balancer.effective_gb_per_layer` scales by the dtype
# the node actually stores at -- rather than by doubling the numbers here, because the column
# is a property of the MODEL and the dtype is a property of the NODE. Keeping them separate is
# what lets a node that really does run fp16 be sized correctly once it says so.
#   1.5B  hidden 1536, ffn 8960,  28 layers ->  46.8M params/layer -> 0.094 GB
#   7B    hidden 3584, ffn 18944, 28 layers -> 233.0M params/layer -> 0.466 GB
#   72B   hidden 8192, ffn 29568, 80 layers -> 877.7M params/layer -> 1.755 GB
# Cross-check: 28 x 0.466 + embed/head = ~15.2 GB for the 7B, against its 20 GB tier minimum.
# A node running the fp32 default pays double; the headroom factor in max_layers_for does not
# cover that, so a network of fp32 nodes on a tier sized this way is still over its head. The
# 7B tier has been fp16-shaped since b5b4f22 either way.
#
# NOT modelled: the driver additionally holds the embedding and lm_head (~2.2 GB for the 7B),
# so the first node in a plan is under-charged by that much. Worth fixing with a `head_gb`
# column once a real measurement exists; recorded here rather than guessed.
# The numbers are illustrative starting points (env-overridable); tune them once real
# per-model slice sizes are measured. What matters here is the SELECTION LOGIC.
# GATED MODELS DO NOT WORK HERE. `slice_downloader` fetches byte ranges straight off
# huggingface.co with no auth, so a repo behind a license click returns 401 and the node
# fails at download with nothing useful to say. Verified 2026-07-30:
#     meta-llama/Llama-3.1-8B-Instruct    -> 401  (gated)
#     Qwen/Qwen2.5-7B-Instruct            -> 200
# The 8b tier pointed at the Meta repo, so promoting to it would have broken every node on
# the network. Any model added here must be publicly fetchable without a token.
_DEFAULT_TIERS = [
    {"name": "1.5b", "model_id": "Qwen/Qwen2.5-1.5B-Instruct", "layers": 28,
     "min_nodes": 2,  "min_ram_gb": 6.0,   "min_replicas": 1, "gb_per_layer": 0.094,
     "description": "Qwen2.5-1.5B — the always-available floor."},
    # THE CAPACITY CASE. 16.09 GB at fp32 — more than the 12 GB Pavilion has in total, so no
    # single machine here can hold it, which is the entire claim NEURON is built to make. At
    # fp16 STORAGE with fp32 compute (common.WEIGHT_DTYPE, measured by test_weight_dtype.py at
    # 2 B/param resident with the GEMMs untouched) it is 8.04 GB and fits across the 12 GB and
    # 8 GB machines with room: caps of 29 + 18 against 36 layers needed.
    #
    # `manual_only` because this is an EXPERIMENT, and because the ladder would take it: at
    # min_nodes 2 the live 3-node network clears the promote margin, and the 68 GB machine
    # makes it placeable at fp32 — so the network would migrate itself onto a 4B model on a
    # health sweep, which is not what "run a model your machine can't run" is supposed to mean.
    # Reach it with `pinned_model_id`, deliberately, on the roster you meant.
    #
    # Figures measured 2026-08-16 by tools/measure_model.py from the published safetensors
    # header: 36 layers x 100,930,816 params, embedding 388,956,160 and tied, Apache-2.0,
    # ungated. min_ram_gb is documentation here — a pinned target is gated by the per-node
    # memory check in migration.update(), not by the aggregate.
    {"name": "4b", "model_id": "Qwen/Qwen3-4B-Instruct-2507", "layers": 36,
     "min_nodes": 2, "min_ram_gb": 18.0, "min_replicas": 1,
     "gb_per_layer": 0.2019, "head_gb": 0.7779, "manual_only": True,
     "description": "Qwen3-4B — the capacity case: too big for any one machine here, "
                    "servable across two at fp16 storage."},
    {"name": "7b",   "model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28,
     "min_nodes": 3,  "min_ram_gb": 20.0,  "min_replicas": 1, "gb_per_layer": 0.466,
     "description": "Qwen2.5-7B — the first model no single volunteer machine can hold."},
    {"name": "70b",  "model_id": "Qwen/Qwen2.5-72B-Instruct", "layers": 80,
     "min_nodes": 20, "min_ram_gb": 180.0, "min_replicas": 2, "gb_per_layer": 1.755,
     "description": "Qwen2.5-72B — unlocked by a large network."},
]


def _load_tiers():
    raw = os.environ.get("NEURON_MODEL_TIERS")
    if raw:
        try:
            tiers = json.loads(raw)
            if tiers:
                return tiers
        except Exception:
            pass
    return [dict(t) for t in _DEFAULT_TIERS]


TIERS = _load_tiers()

# --- hysteresis knobs (env-overridable) ------------------------------------- #
# Promote only when a bigger tier is feasible with this much headroom, sustained for the
# dwell. Demote only after the current tier has been infeasible for the grace period.
PROMOTE_MARGIN = float(os.environ.get("NEURON_TIER_PROMOTE_MARGIN", "0.15"))
PROMOTE_DWELL_S = float(os.environ.get("NEURON_TIER_PROMOTE_DWELL_S", "300"))
DEMOTE_GRACE_S = float(os.environ.get("NEURON_TIER_DEMOTE_GRACE_S", "300"))


# --------------------------------------------------------------------------- #
# Pure capacity + feasibility
# --------------------------------------------------------------------------- #
def network_capacity(nodes):
    """Aggregate serving capacity from online+eligible nodes only."""
    live = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    return {
        "nodes": len(live),
        "total_ram_gb": round(sum((n.get("ram_gb") or 0.0) for n in live), 1),
    }


def _meets(cap, tier, margin=0.0):
    """Does `cap` satisfy `tier`'s requirements, optionally scaled up by `margin`?"""
    factor = 1.0 + margin
    return (cap["nodes"] >= tier["min_nodes"] * factor
            and cap["total_ram_gb"] >= tier["min_ram_gb"] * factor)


def tier_for(model_id):
    """The tier entry for a model id, or None if it is not one we know how to serve.

    None is the load-bearing answer: an operator pin naming a model that is not in the table
    must be IGNORED rather than acted on, because the coordinator would otherwise migrate the
    whole network toward a model it has no layer count or footprint for.
    """
    for t in TIERS:
        if t.get("model_id") == model_id:
            return dict(t)
    return None


def gb_per_layer_for(model_id):
    """One layer's weights for a model in the tier table, or None if it isn't one.

    None means "unknown footprint", which every consumer treats as "no memory constraint" —
    the behaviour from before tiers carried this figure, and the right answer for a model
    injected via NEURON_MODEL_TIERS with no measurement behind it.
    """
    for t in TIERS:
        if t.get("model_id") == model_id and t.get("gb_per_layer"):
            return float(t["gb_per_layer"])
    return None


def head_gb_for(model_id):
    """The DRIVER's extra weight for a model: the embedding, plus `lm_head` when untied.

    Same fp16 basis as `gb_per_layer`, and `balancer.effective_gb` corrects both together.
    None means unmeasured, which is the pre-existing behaviour — the head simply is not
    charged — so a tier added without this figure is sized exactly as it was before.
    """
    for t in TIERS:
        if t.get("model_id") == model_id and t.get("head_gb"):
            return float(t["head_gb"])
    return None


def partition_shortfall(nodes, tier):
    """Layers of `tier`'s model the online+eligible nodes cannot hold BETWEEN THEM.

    This is the per-node question `_meets` cannot ask. `_meets` sums RAM across the network;
    a pipeline stage lives on ONE machine, and a machine that cannot hold its slice does not
    run slowly, it gets OOM-killed.
    """
    live = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    return balancer.capacity_shortfall(live, int(tier.get("layers") or 0),
                                       tier.get("gb_per_layer"), tier.get("head_gb"))


def placeable(nodes, tier):
    """Can this tier's model actually be laid out across these nodes, one slice per machine?"""
    return partition_shortfall(nodes, tier) == 0


def feasible_tier_index(cap, margin=0.0, nodes=None):
    """Highest tier index whose requirements `cap` meets. -1 if even the floor fails.

    Pass `nodes` to require a real per-node partition as well as the aggregate numbers. Callers
    that only have a capacity summary keep the old aggregate-only answer.
    """
    best = -1
    for i, t in enumerate(TIERS):
        # A manual_only tier is invisible to the ladder in BOTH directions: it can never be
        # promoted to, and it can never be the answer a demotion falls back to. An experiment
        # the operator aimed the network at must not become the tier the network settles on.
        if t.get("manual_only"):
            continue
        if _meets(cap, t, margin) and (nodes is None or placeable(nodes, t)):
            best = i
    return best


def best_feasible(cap, nodes=None):
    """The biggest model this capacity can serve right now (no hysteresis). Or None."""
    i = feasible_tier_index(cap, 0.0, nodes)
    return TIERS[i] if i >= 0 else None


def next_tier_gap(cap, current_index):
    """What the network still needs to unlock the NEXT tier up. None if at the top.

    Powers the "you're N nodes away from a bigger model" growth prompt in the UI.
    """
    # Skip past any manual_only tier: the ladder will never climb to it, so telling an operator
    # they are "2 nodes away" from a model no amount of growth will select is a false promise
    # printed on the dashboard and in every growth prompt.
    nxt = current_index + 1
    while nxt < len(TIERS) and TIERS[nxt].get("manual_only"):
        nxt += 1
    if nxt >= len(TIERS):
        return None
    t = TIERS[nxt]
    return {
        "name": t["name"],
        "model_id": t["model_id"],
        "need_nodes": max(0, t["min_nodes"] - cap["nodes"]),
        "need_ram_gb": round(max(0.0, t["min_ram_gb"] - cap["total_ram_gb"]), 1),
    }


# --------------------------------------------------------------------------- #
# Stateful selection with hysteresis
# --------------------------------------------------------------------------- #
class TierController:
    """Model-tier selector with hysteresis.

    PROMOTE only after the bigger tier has been feasible-with-margin continuously for
    PROMOTE_DWELL_S (sustained growth, not a blip). DEMOTE only after the current tier
    has been infeasible continuously for DEMOTE_GRACE_S (a sleeping laptop gets a grace
    period before the whole network shrinks its model).

    Call `update(nodes, now)` periodically (e.g. from the coordinator's health sweep);
    it returns the active tier dict. All timing comes from the caller-supplied `now`.
    """

    def __init__(self, start_index=0):
        self.index = start_index
        self._promote_candidate = None   # (index, since_ts) — the tier we're waiting on
        self._infeasible_since = None    # ts the current tier first became infeasible

    def update(self, nodes, now):
        cap = network_capacity(nodes)

        # ---- demotion: is the CURRENT tier still feasible (no margin)? ---------
        # Feasible now means BOTH: the aggregate numbers, and a per-node partition that
        # actually fits. A network whose shape has drifted (the big machine left, four small
        # ones joined) can still clear "3 nodes · 20 GB" while no node can hold a slice —
        # and it is the demotion that fixes that, by pointing everyone at a smaller model.
        if _meets(cap, TIERS[self.index], 0.0) and placeable(nodes, TIERS[self.index]):
            self._infeasible_since = None
        else:
            if self._infeasible_since is None:
                self._infeasible_since = now
            if now - self._infeasible_since >= DEMOTE_GRACE_S:
                # biggest we can back right now -- and actually place
                target = feasible_tier_index(cap, 0.0, nodes)
                self.index = target if target >= 0 else 0
                self._infeasible_since = None
                self._promote_candidate = None
                return self.active()

        # ---- promotion: a bigger tier feasible-with-margin, sustained ----------
        # `nodes` is what stops the network qualifying for a model it cannot lay out. The
        # migration controller refuses such a target anyway, but a tier that is qualified and
        # permanently unreachable is a lie told on the dashboard and in every growth prompt.
        cand = feasible_tier_index(cap, PROMOTE_MARGIN, nodes)
        if cand > self.index:
            if self._promote_candidate is None or self._promote_candidate[0] != cand:
                self._promote_candidate = (cand, now)          # start the dwell clock
            elif now - self._promote_candidate[1] >= PROMOTE_DWELL_S:
                self.index = cand
                self._promote_candidate = None
        else:
            self._promote_candidate = None

        return self.active()

    def active(self):
        return TIERS[self.index]


def snapshot(nodes, controller, now=None):
    """A UI/endpoint-ready view: the active model, capacity, and the next-tier gap.

    Does NOT advance hysteresis unless `now` is given (then it refreshes the controller).
    """
    if now is not None:
        controller.update(nodes, now)
    cap = network_capacity(nodes)
    active = controller.active()
    return {
        "active_model": active["model_id"],
        "active_tier": active["name"],
        "layers": active["layers"],
        "capacity": cap,
        "next_tier": next_tier_gap(cap, controller.index),
        "tiers": [
            # `feasible` is the aggregate gate; `placeable` is whether any one machine can hold
            # a slice. feasible-but-not-placeable is a real state ("you have the RAM, but not
            # on any single node") and the reason a qualified promotion can sit still.
            # `manual_only` rides along so a reader is not left to infer why a tier that is
            # both feasible and placeable is sitting there unadopted.
            {"name": t["name"], "model_id": t["model_id"], "layers": t["layers"],
             "min_nodes": t["min_nodes"], "min_ram_gb": t["min_ram_gb"],
             "feasible": _meets(cap, t, 0.0), "placeable": placeable(nodes, t),
             "manual_only": bool(t.get("manual_only"))}
            for t in TIERS
        ],
    }
