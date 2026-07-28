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

Self-heal (added later): a SEPARATE state machine, self_heal()/heal_status(), living on the
same MigrationController instance but never touching phase/target/plan/ready above. It closes
a coverage GAP (a segment with zero online+eligible nodes and no replica) by reassigning true
IDLE surplus nodes -- nodes that aren't covering, or tied to cover, any live segment -- never
by taking capacity away from a segment that's already working. Deliberately kept separate from
update() rather than folded in: update()'s target==serving branch unconditionally resets to
steady every tick, which is exactly the state self-heal operates in, so anything stored via
`phase` would be wiped before a heal could ever complete.
"""

from coordinator import router


def plan_migration(nodes, layers, start=0):
    """Even contiguous partition of `layers` across the eligible online nodes, beginning at
    layer `start` (default 0, every existing caller unaffected).

    Returns [{node_id, layer_start, layer_end}] covering start..start+layers-1. The driver
    (holds the lm_head, head_ms>0) is placed first; remaining nodes follow by current
    layer_start. If there are more nodes than layers, the surplus nodes get no segment
    (candidate replicas -- or, for self-heal, the already-idle nodes it was given to fill a gap
    with in the first place).
    """
    elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
    elig.sort(key=lambda n: (0 if (n.get("head_ms") or 0) > 0 else 1, n.get("layer_start", 0)))
    n = len(elig)
    if n == 0 or layers <= 0:
        return []
    base, rem = divmod(layers, n)
    plan, cur = [], start
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
        # Self-heal state -- entirely separate from the tier-migration fields above.
        self.heal_plan = []      # [{node_id, layer_start, layer_end}] closing the current gap
        self.heal_ready = set()  # node_ids that reported the heal slice downloaded
        self.heal_target = None  # {model_id, layers} -- always the CURRENT serving model

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

        # A migration is warranted. (Re)start/replan preparing if: we aren't preparing yet, the
        # target changed, OR a node already in the plan is no longer online+eligible (a real
        # stranger's laptop can drop mid-preparing under the idle donation mode — without this,
        # cutover requires `planned <= ready` forever with a planned node that will never report
        # ready again, wedging the migration silently for good; post-audit fix). Replanning
        # against the currently-eligible set lets a churny node's segment fall to whoever else
        # qualifies (or drop the migration back to idle-preparing if nobody currently does).
        elig_ids = {n["node_id"] for n in nodes if n.get("status") == "online" and n.get("eligible")}
        planned_ids = {a["node_id"] for a in self.plan}
        node_dropped = self.phase == "preparing" and not planned_ids.issubset(elig_ids)
        if self.phase != "preparing" or (self.target or {}).get("model_id") != target["model_id"] \
                or node_dropped:
            self.target = {"model_id": target["model_id"], "layers": int(target["layers"])}
            self.plan = plan_migration(nodes, self.target["layers"])
            self.ready = set()
            self.phase = "preparing"
            # A real tier migration always wins -- abandon any in-flight self-heal rather than
            # let a stale heal assignment linger through a cutover it was never part of.
            self.heal_plan, self.heal_ready, self.heal_target = [], set(), None

        # Cutover when every planned node has the target slice ready.
        planned = {a["node_id"] for a in self.plan}
        if planned and planned <= self.ready:
            apply_serving(self.target["model_id"], self.target["layers"])
            self._reset()
        return self.status()

    def mark_ready(self, node_id):
        """A node reports its target slice is downloaded and it can serve the target range.
        Checks a real tier migration first (PREPARING); falls back to a self-heal plan when
        STEADY -- the two never overlap (update() clears heal_* the moment it starts preparing),
        so this is unambiguous."""
        if self.phase == "preparing" and node_id in {a["node_id"] for a in self.plan}:
            self.ready.add(node_id)
            return True
        if self.phase == "steady" and node_id in {a["node_id"] for a in self.heal_plan}:
            self.heal_ready.add(node_id)
            return True
        return False

    def assignment_for(self, node_id):
        """The target slice this node should prepare, or None if it isn't migrating/healing."""
        if self.phase == "preparing":
            for a in self.plan:
                if a["node_id"] == node_id:
                    return {"migrating": True, "model_id": self.target["model_id"],
                            "total_layers": self.target["layers"],
                            "layer_start": a["layer_start"], "layer_end": a["layer_end"],
                            "ready": node_id in self.ready}
            return None
        if self.phase == "steady":
            for a in self.heal_plan:
                if a["node_id"] == node_id:
                    return {"migrating": True, "model_id": self.heal_target["model_id"],
                            "total_layers": self.heal_target["layers"],
                            "layer_start": a["layer_start"], "layer_end": a["layer_end"],
                            "ready": node_id in self.heal_ready}
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

    # ----------------------------------------------------------------------- #
    # Self-heal: close a coverage gap using true-idle surplus nodes only.
    # ----------------------------------------------------------------------- #
    def self_heal(self, nodes, serving, now, apply_layers):
        """Advance the self-heal machine. `nodes` is the SAME full roster update() already
        received this tick (no extra DB read). `serving` is {model_id, layers} for whatever
        the network currently serves. `apply_layers(assignments)` persists a completed heal.
        Returns heal_status(). No-op (and drops any prior heal) unless phase=="steady" -- a
        real tier migration always takes priority and owns coverage during its own transition."""
        if self.phase != "steady":
            return self.heal_status()

        missing, covering_ids = router.covering_and_missing(nodes, serving["layers"])
        if not missing:
            if self.heal_plan:                      # the gap closed some other way (node came
                self.heal_plan, self.heal_ready, self.heal_target = [], set(), None
            return self.heal_status()

        elig_ids = {n["node_id"] for n in nodes
                   if n.get("status") == "online" and n.get("eligible")}
        surplus_ids = elig_ids - covering_ids
        surplus = [n for n in nodes if n["node_id"] in surplus_ids]

        # Try each gap in order; heal the first one a proposal can actually cover so an
        # unhealable earlier gap never starves a later, healable one.
        proposal, target_gap = [], None
        for gap_start, gap_end in missing:
            candidate = plan_migration(surplus, gap_end - gap_start + 1, start=gap_start)
            if candidate:
                proposal, target_gap = candidate, (gap_start, gap_end)
                break

        if not proposal:
            if self.heal_plan and not ({a["node_id"] for a in self.heal_plan} <= surplus_ids):
                # what we were healing with is no longer available and nothing else fits either
                self.heal_plan, self.heal_ready, self.heal_target = [], set(), None
            return self.heal_status()

        planned_ids = {a["node_id"] for a in self.heal_plan}
        proposal_ids = {a["node_id"] for a in proposal}
        node_dropped = self.heal_plan and not planned_ids.issubset(surplus_ids)
        changed = proposal_ids != planned_ids or node_dropped
        if changed:
            self.heal_plan = proposal
            self.heal_ready = set()
            self.heal_target = {"model_id": serving["model_id"], "layers": serving["layers"]}

        planned = {a["node_id"] for a in self.heal_plan}
        if planned and planned <= self.heal_ready:
            apply_layers(self.heal_plan)
            self.heal_plan, self.heal_ready, self.heal_target = [], set(), None
        return self.heal_status()

    def heal_status(self):
        planned = {a["node_id"] for a in self.heal_plan}
        return {
            "healing": bool(self.heal_plan),
            "target": self.heal_target,
            "plan": [{"node_id": a["node_id"], "layers": [a["layer_start"], a["layer_end"]],
                      "ready": a["node_id"] in self.heal_ready} for a in self.heal_plan],
            "ready_count": len(planned & self.heal_ready),
            "plan_size": len(self.heal_plan),
        }
