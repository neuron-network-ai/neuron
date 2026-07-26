"""coordinator/migration.py — rolling model migration (Build 3).

Moves the network from its SERVING model to the TARGET model its capacity now qualifies for
(the TierController's pick), WITHOUT dropping service. Coordinator-side orchestration only;
the actual per-node download+reload is node-side (needs live nodes to validate).

State machine, advanced each health sweep via update():

  STEADY     serving == target — nothing to do.
  PREPARING  target != serving. The coordinator has partitioned the target model's layers
             across the eligible nodes and assigned each a target range. Nodes download the
             target slice in the BACKGROUND while STILL serving the old model, then report
             READY. Serving does NOT change yet — so the network keeps serving throughout.
  (cutover)  Once every planned node is READY, serving_model is flipped to the target in one
             step and the machine returns to STEADY on the new model. Nodes then reload to
             the target (the brief per-node reload window is covered by the small-model floor).

Safety properties:
  * serving_model is never flipped until the WHOLE target partition is ready → the old model
    keeps serving the entire time preparation is underway (no coverage gap from migration).
  * if the target stops qualifying before cutover (capacity dropped and the TierController
    demoted the target back to the serving model), the migration ABORTS and serving stays put.
  * if the target changes to a different model mid-flight, the plan is recomputed.

The layer partition here is a simple even contiguous split (deterministic + testable); the
speed-weighted balancer and replica-aware placement are refinements layered on top.
"""


def plan_migration(nodes, layers):
    """Even contiguous partition of `layers` across the eligible online nodes.

    Returns [{node_id, layer_start, layer_end}] covering 0..layers-1. The driver (holds the
    lm_head, head_ms>0) is placed first; remaining nodes follow by current layer_start. If
    there are more nodes than layers, the surplus nodes get no segment (candidate replicas).
    """
    elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    elig.sort(key=lambda n: (0 if (n.get("head_ms") or 0) > 0 else 1, n.get("layer_start", 0)))
    n = len(elig)
    if n == 0 or layers <= 0:
        return []
    base, rem = divmod(layers, n)
    plan, cur = [], 0
    for i, node in enumerate(elig):
        cnt = base + (1 if i < rem else 0)
        if cnt == 0:
            continue                       # more nodes than layers → surplus unassigned
        plan.append({"node_id": node["node_id"],
                     "layer_start": cur, "layer_end": cur + cnt - 1})
        cur += cnt
    return plan


class MigrationController:
    """Orchestrates one migration at a time. All timing is caller-supplied (`now`)."""

    def __init__(self):
        self.phase = "steady"
        self.target = None       # {model_id, layers} being migrated to, or None
        self.plan = []           # [{node_id, layer_start, layer_end}] for the target
        self.ready = set()       # node_ids that reported the target slice downloaded

    def update(self, nodes, target, serving, now, apply_serving):
        """Advance the machine.

        target / serving : {model_id, layers}. `apply_serving(model_id, layers)` performs the
        cutover (flips the coordinator's serving model). Returns status().
        """
        # Nothing to migrate (or a just-completed cutover): the target is what we serve.
        if target["model_id"] == serving["model_id"]:
            if self.phase != "steady":
                self._reset()
            return self.status()

        # A migration is warranted. (Re)start preparing if we aren't, or the target changed.
        if self.phase != "preparing" or (self.target or {}).get("model_id") != target["model_id"]:
            self.target = {"model_id": target["model_id"], "layers": int(target["layers"])}
            self.plan = plan_migration(nodes, self.target["layers"])
            self.ready = set()
            self.phase = "preparing"

        # Cutover when every planned node has the target slice ready.
        planned = {a["node_id"] for a in self.plan}
        if planned and planned <= self.ready:
            apply_serving(self.target["model_id"], self.target["layers"])
            self._reset()
        return self.status()

    def mark_ready(self, node_id):
        """A node reports its target slice is downloaded and it can serve the target range.
        Only meaningful while PREPARING and if the node is in the plan."""
        if self.phase != "preparing":
            return False
        if node_id not in {a["node_id"] for a in self.plan}:
            return False
        self.ready.add(node_id)
        return True

    def assignment_for(self, node_id):
        """The target slice this node should prepare, or None if it isn't migrating."""
        if self.phase != "preparing":
            return None
        for a in self.plan:
            if a["node_id"] == node_id:
                return {"migrating": True, "model_id": self.target["model_id"],
                        "layer_start": a["layer_start"], "layer_end": a["layer_end"],
                        "ready": node_id in self.ready}
        return None

    def _reset(self):
        self.phase, self.target, self.plan, self.ready = "steady", None, [], set()

    def status(self):
        planned = {a["node_id"] for a in self.plan}
        return {
            "phase": self.phase,
            "target": self.target,
            "plan": [{"node_id": a["node_id"], "layers": [a["layer_start"], a["layer_end"]],
                      "ready": a["node_id"] in self.ready} for a in self.plan],
            "ready_count": len(planned & self.ready),
            "plan_size": len(self.plan),
        }
