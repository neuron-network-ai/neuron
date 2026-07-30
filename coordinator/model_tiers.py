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


# A tier = a model + the capacity needed to serve it WITH redundancy.
#   min_nodes    : online+eligible nodes required (enough to split the pipeline AND
#                  hold `min_replicas` copies of each segment).
#   min_ram_gb   : total RAM across those nodes (must hold every replica's slices).
#   min_replicas : redundant copies of each layer segment. >=2 means one node can drop
#                  without an outage — the resilience floor for a production tier.
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
     "min_nodes": 2,  "min_ram_gb": 6.0,   "min_replicas": 1,
     "description": "Qwen2.5-1.5B — the always-available floor."},
    {"name": "7b",   "model_id": "Qwen/Qwen2.5-7B-Instruct", "layers": 28,
     "min_nodes": 3,  "min_ram_gb": 20.0,  "min_replicas": 1,
     "description": "Qwen2.5-7B — the first model no single volunteer machine can hold."},
    {"name": "70b",  "model_id": "Qwen/Qwen2.5-72B-Instruct", "layers": 80,
     "min_nodes": 20, "min_ram_gb": 180.0, "min_replicas": 2,
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


def feasible_tier_index(cap, margin=0.0):
    """Highest tier index whose requirements `cap` meets. -1 if even the floor fails."""
    best = -1
    for i, t in enumerate(TIERS):
        if _meets(cap, t, margin):
            best = i
    return best


def best_feasible(cap):
    """The biggest model this capacity can serve right now (no hysteresis). Or None."""
    i = feasible_tier_index(cap, 0.0)
    return TIERS[i] if i >= 0 else None


def next_tier_gap(cap, current_index):
    """What the network still needs to unlock the NEXT tier up. None if at the top.

    Powers the "you're N nodes away from a bigger model" growth prompt in the UI.
    """
    nxt = current_index + 1
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
        if _meets(cap, TIERS[self.index], 0.0):
            self._infeasible_since = None
        else:
            if self._infeasible_since is None:
                self._infeasible_since = now
            if now - self._infeasible_since >= DEMOTE_GRACE_S:
                target = feasible_tier_index(cap, 0.0)   # biggest we can back right now
                self.index = target if target >= 0 else 0
                self._infeasible_since = None
                self._promote_candidate = None
                return self.active()

        # ---- promotion: a bigger tier feasible-with-margin, sustained ----------
        cand = feasible_tier_index(cap, PROMOTE_MARGIN)
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
            {"name": t["name"], "model_id": t["model_id"], "layers": t["layers"],
             "min_nodes": t["min_nodes"], "min_ram_gb": t["min_ram_gb"],
             "feasible": _meets(cap, t, 0.0)}
            for t in TIERS
        ],
    }
