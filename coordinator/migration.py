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
  * a target NO NODE PARTITION CAN HOLD is refused outright: the network keeps serving what it
    has rather than preparing a plan that OOM-kills the machines preparing it.

The layer partition here is an even contiguous split fitted to each node's memory (deterministic
+ testable); the speed-weighted balancer and replica-aware placement are refinements layered on
top. Memory is the one hard constraint — see plan_migration.

Self-heal (added later): a SEPARATE state machine, self_heal()/heal_status(), living on the
same MigrationController instance but never touching phase/target/plan/ready above. It closes
a coverage GAP (a segment with zero online+eligible nodes and no replica), preferring true IDLE
surplus nodes -- ones that aren't covering, or tied to cover, any live segment -- so a working
segment is never stripped to patch a broken one. When there is no idle node it falls back to
re-splitting the model across everyone who is left ([R6]); that DOES move covering nodes, which
is sound only because a gap means nothing can complete anyway. Deliberately kept separate from
update() rather than folded in: update()'s target==serving branch unconditionally resets to
steady every tick, which is exactly the state self-heal operates in, so anything stored via
`phase` would be wiped before a heal could ever complete.
"""

import collections

from coordinator import balancer, config, router


def eligible_nodes(nodes):
    """Online AND eligible, the same filter every planner here applies."""
    return [n for n in nodes if n.get("status") == "online" and n.get("eligible")]


def partition_shortfall(nodes, layers, gb_per_layer):
    """Layers the online+eligible nodes cannot hold BETWEEN THEM at `gb_per_layer`.

    0 means a valid per-node partition exists — and also means "unknown", when the model's
    per-layer footprint isn't known (an env-supplied tier with no `gb_per_layer`), because an
    unknown footprint can't justify refusing to serve.
    """
    return balancer.capacity_shortfall(eligible_nodes(nodes), layers, gb_per_layer)


def plan_migration(nodes, layers, start=0, gb_per_layer=None, max_stages=None):
    """Contiguous partition of `layers` across the eligible online nodes, beginning at
    layer `start` (default 0, every existing caller unaffected).

    Returns [{node_id, layer_start, layer_end}] covering start..start+layers-1. The driver
    (holds the lm_head, head_ms>0) is placed first; remaining nodes follow by current
    layer_start. If there are more nodes than layers, the surplus nodes get no segment
    (candidate replicas -- or, for self-heal, the already-idle nodes it was given to fill a gap
    with in the first place).

    The split starts EVEN and is then fitted to memory when `gb_per_layer` is known: each node's
    reported RAM (or VRAM) becomes a hard cap via `balancer.max_layers_for`, and layers land on
    the machines with room instead. An even split is a fine default and a bad promise --
    observed live 2026-08-07, an even 3-way split of Qwen2.5-7B handed 9-10 layers (~4.5 GB of
    weights) to machines with 8 GB of TOTAL RAM, because the tier gate qualified the network on
    AGGREGATE capacity and nothing downstream ever asked whether one node could hold its slice.
    A stage that gets OOM-killed is infinitely slower than a slow one.

    Under memory pressure a node that an even split left with no segment (surplus) CAN pick one
    up -- that is the whole point of the fit. This plan always covers every layer; callers that
    must not proceed when the network genuinely cannot hold the model check
    `partition_shortfall` first (MigrationController.update does).

    At most `max_stages` nodes become STAGES; any beyond that are assigned as REPLICAS of an
    existing stage, sharing its exact range. The stage count is not a tuning choice -- the
    inference path is three programs with three roles (config.PIPELINE_STAGES), and
    `node_a.coord_get_chain` rejects a chain of any other length. Splitting across "however many
    nodes are eligible" produced exactly that failure live on 2026-08-07: a migration cut over
    onto a one-stage plan, coverage read 28/28 healthy because one node held every layer, and
    every chat died on "expected a 3-node chain, got 1". Replicas are also what added machines
    are FOR -- they raise throughput instead of deepening the pipeline ([P16]) -- so nothing
    idles as a result of the cap.
    """
    elig = eligible_nodes(nodes)
    # Stage 1 is not just "the first slice". It holds the embedding and the lm_head, and the
    # driver (ui/app.py, neuron_driver.py) IS that node -- node_a.coord_get_chain refuses any
    # chain whose first stage is not its own shard. So the ordering here decides whether the
    # network is routable at all, not merely how fast it is.
    #
    #   1. a node that has measured a head_ms already carries the head -- keep it there, so the
    #      lm_head does not have to move machines;
    #   2. otherwise the FASTEST node takes it. The head is a fixed per-token cost on top of a
    #      stage's layers, so it belongs on the quickest machine -- and on this network that is
    #      also the machine running the driver. Sorting by layer_start alone put stage 1 on a
    #      node measured at 45.8 ms/layer while an 11.3 ms/layer machine took the tail;
    #   3. layer_start keeps a node near its current range, so fewer slices move;
    #   4. node_id breaks the tie last. Without it two nodes on the identical segment sort
    #      equal, the plan follows whatever order the roster arrived in, and a plan that changes
    #      between ticks resets every node's readiness and discards partial downloads -- on a
    #      network of machines that come and go, it can then never converge.
    elig.sort(key=lambda n: (0 if (n.get("head_ms") or 0) > 0 else 1,
                             float(n.get("ms_per_layer") or 1e9),
                             n.get("layer_start", 0), n.get("node_id", "")))
    if not elig or layers <= 0:
        return []
    max_stages = config.PIPELINE_STAGES if max_stages is None else max_stages
    stage_nodes, extra_nodes = elig[:max_stages], elig[max_stages:]

    n = len(stage_nodes)
    base, rem = divmod(layers, n)
    counts = [base + (1 if i < rem else 0) for i in range(n)]
    if gb_per_layer:
        counts, _ = balancer.fit_to_capacity(
            counts, balancer.layer_caps(stage_nodes, gb_per_layer, layers))
    plan, stages, cur = [], [], start
    for node, cnt in zip(stage_nodes, counts):
        if cnt == 0:
            continue                       # more stages than layers → this one takes no segment
        seg = {"layer_start": cur, "layer_end": cur + cnt - 1}
        stages.append(seg)
        plan.append({"node_id": node["node_id"], **seg})
        cur += cnt

    # Everyone else replicates, filling the thinnest stage first so depth stays even. A machine
    # with no segment earns nothing and serves nothing, so the cap must never mean "idle".
    if stages:
        depth = [1] * len(stages)
        for node in extra_nodes:
            i = min(range(len(stages)), key=lambda k: (depth[k], k))
            depth[i] += 1
            plan.append({"node_id": node["node_id"], **stages[i]})
    return plan


def _assignment_key(plan):
    """Identity of a plan for change detection: who serves what, not just who is involved."""
    return {(a["node_id"], a["layer_start"], a["layer_end"]) for a in plan}


class MigrationController:
    """Orchestrates one migration at a time. All timing is caller-supplied (`now`)."""

    def __init__(self):
        self.phase = "steady"
        self.target = None       # {model_id, layers} being migrated to, or None
        self.plan = []           # [{node_id, layer_start, layer_end}] for the target
        self.ready = set()       # node_ids that reported the target slice downloaded
        self.blocked = None      # why a warranted migration is NOT being attempted, or None
        # Self-heal state -- entirely separate from the tier-migration fields above.
        self.heal_plan = []      # [{node_id, layer_start, layer_end}] closing the current gap
        self.heal_ready = set()  # node_ids that reported the heal slice downloaded
        self.heal_target = None  # {model_id, layers} -- always the CURRENT serving model
        self.heal_mode = None    # "surplus" (idle node fills the gap) | "resplit" ([R6])
        self.heal_shortfall = 0  # layers the survivors cannot hold, on a [R6] re-split

    def update(self, nodes, target, serving, now, apply_serving):
        """Advance the machine.

        target / serving : {model_id, layers, gb_per_layer(optional)}.
        `apply_serving(model_id, layers)` performs the cutover (flips the coordinator's serving
        model). Returns status().
        """
        # Nothing to migrate (or a just-completed cutover): the target is what we serve.
        if target["model_id"] == serving["model_id"]:
            if self.phase != "steady":
                self._reset()
            self.blocked = None
            return self.status()

        # MEMORY GATE. The tier gate upstream qualifies the network on AGGREGATE capacity
        # ("3 nodes, 20 GB"), which says nothing about whether any INDIVIDUAL node can hold a
        # slice -- that is how three machines with 8/8/12 GB were told to serve Qwen2.5-7B on
        # 2026-08-07. Refuse rather than prepare: preparing means every planned node downloads
        # several GB it cannot load, and cutover would then move the whole network onto a model
        # that OOM-kills its own pipeline. Serving stays exactly where it is.
        #
        # Deliberately NOT a new phase. self_heal() runs only while phase == "steady", so a
        # "blocked" phase would silently disable gap healing for as long as an unservable tier
        # stayed qualified -- the network would be both unable to grow AND unable to repair.
        # Blocked is a fact ABOUT a steady network, so it rides in status() instead.
        shortfall = partition_shortfall(nodes, int(target["layers"]),
                                        target.get("gb_per_layer"))
        if shortfall:
            if self.phase != "steady":
                self._reset()
            self.blocked = {"model_id": target["model_id"], "layers": int(target["layers"]),
                            "reason": "capacity", "capacity_shortfall": shortfall}
            return self.status()
        self.blocked = None

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
            self.target = {"model_id": target["model_id"], "layers": int(target["layers"]),
                           "gb_per_layer": target.get("gb_per_layer")}
            self.plan = plan_migration(nodes, self.target["layers"],
                                       gb_per_layer=self.target["gb_per_layer"])
            self.ready = set()
            self.phase = "preparing"
            # A real tier migration always wins -- abandon any in-flight self-heal rather than
            # let a stale heal assignment linger through a cutover it was never part of.
            self._clear_heal()

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

    def _clear_heal(self):
        self.heal_plan, self.heal_ready, self.heal_target, self.heal_mode = [], set(), None, None
        self.heal_shortfall = 0

    def status(self):
        planned = {a["node_id"] for a in self.plan}
        return {
            "phase": self.phase,
            "target": self.target,
            "plan": [{"node_id": a["node_id"], "layers": [a["layer_start"], a["layer_end"]],
                      "ready": a["node_id"] in self.ready} for a in self.plan],
            "ready_count": len(planned & self.ready),
            "plan_size": len(self.plan),
            # Not None means: a bigger model is qualified, and we are NOT going for it. The
            # network is healthy and serving; it just cannot hold what it qualifies for.
            "blocked": self.blocked,
        }

    # ----------------------------------------------------------------------- #
    # Self-heal: close a coverage gap -- idle surplus first, full re-split as the fallback.
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
                self._clear_heal()
            return self.heal_status()

        elig = [n for n in nodes if n.get("status") == "online" and n.get("eligible")]
        elig_ids = {n["node_id"] for n in elig}

        # Movable capacity, in order of how little it disturbs:
        #   IDLE     -- covering nothing at all. Free to move, always was.
        #   SPARE    -- a REDUNDANT REPLICA: another node serves the byte-identical segment, so
        #               moving this one costs no coverage either. Not previously counted, and
        #               that omission is what made the live 2026-08-07 heal fragile. Three
        #               machines sat on 0-13 and one on 14-20 with 21-27 empty; no node was
        #               idle, so the only option left was a full re-split that moved ALL FOUR --
        #               including the two that were serving correctly. Every planned node must
        #               report ready before anything cuts over, so a four-node plan waits on the
        #               slowest of four, and when one work PC dropped mid-download the plan was
        #               invalidated and every partial download thrown away. Observed: 13 minutes,
        #               0/4 ready, then back to square one.
        #
        # Two of those three were pure duplicates. Moving them closes the gap while leaving the
        # nodes that are already serving completely alone -- fewer downloads, fewer machines to
        # stay up, and no disturbance to a working stage.
        by_seg = collections.defaultdict(list)
        for n in elig:
            by_seg[(n["layer_start"], n["layer_end"])].append(n)
        spare_ids = set()
        for members in by_seg.values():
            if len(members) > 1:
                # keep one; sort by node_id so the same node is kept on every tick and the plan
                # does not churn between equivalent choices
                spare_ids.update(m["node_id"]
                                 for m in sorted(members, key=lambda m: m["node_id"])[1:])
        surplus_ids = (elig_ids - covering_ids) | spare_ids
        surplus = [n for n in nodes if n["node_id"] in surplus_ids]

        # Try each gap in order; heal the first one a proposal can actually cover so an
        # unhealable earlier gap never starves a later, healable one.
        gb_per_layer = serving.get("gb_per_layer")
        proposal, target_gap = [], None
        for gap_start, gap_end in missing:
            candidate = plan_migration(surplus, gap_end - gap_start + 1, start=gap_start,
                                       gb_per_layer=gb_per_layer)
            if candidate:
                proposal, target_gap = candidate, (gap_start, gap_end)
                break

        # [R6] Nothing movable can close the gap -- no idle node, no redundant replica. That is
        # not the rare case -- it is the NORMAL
        # case when a node LEAVES, because a shrunken network has no surplus by definition, so
        # the one situation where healing matters most was the one this skipped. The survivors
        # are still holding the ranges they were given when the network was bigger (0-13 and
        # 14-20 of a 28-layer model, observed live 2026-08-04), so the departed node's layers
        # belong to nobody and not one request can complete. Re-split the whole serving model
        # across whoever is actually here instead.
        #
        # Reassigning nodes that ARE currently covering is safe precisely because `missing` is
        # non-empty: the pipeline is already broken, so there is no working service to protect.
        # The plan still goes through the same download-then-report-ready handshake as any other
        # heal -- nothing cuts over until every planned node holds its new slice.
        #
        # Unlike a tier migration this is NOT gated on capacity. A gap means not one request can
        # complete, so there is no working service to protect and no smaller model to fall back
        # to from here -- refusing to re-split would leave the network dark AND unrepaired. The
        # honest answer to "the survivors cannot hold this model" is to serve a smaller one, and
        # that is the TierController's demotion, not this machine's. So heal on the best split
        # the survivors can take and REPORT the shortfall (heal_status) rather than hide it.
        resplit = False
        if not proposal:
            proposal = plan_migration(nodes, serving["layers"], gb_per_layer=gb_per_layer)
            resplit = bool(proposal)
        shortfall = (partition_shortfall(nodes, serving["layers"], gb_per_layer)
                     if resplit else 0)

        if not proposal:
            if self.heal_plan and not ({a["node_id"] for a in self.heal_plan} <= elig_ids):
                # what we were healing with is no longer available and nothing else fits either
                self._clear_heal()
            return self.heal_status()

        # Compare the full assignments, not just the node ids. The same node can be re-planned
        # onto a DIFFERENT range from one tick to the next (the gap moved, or a surplus heal was
        # superseded by a re-split); id-only comparison called that "unchanged" and kept serving
        # the stale range out of assignment_for(), so a node would download and cut over to a
        # slice the coordinator had already stopped intending.
        planned_ids = {a["node_id"] for a in self.heal_plan}
        node_dropped = self.heal_plan and not planned_ids.issubset(elig_ids)
        changed = _assignment_key(proposal) != _assignment_key(self.heal_plan) or node_dropped
        if changed:
            self.heal_plan = proposal
            self.heal_ready = set()
            self.heal_target = {"model_id": serving["model_id"], "layers": serving["layers"]}
            self.heal_mode = "resplit" if resplit else "surplus"
        self.heal_shortfall = shortfall

        planned = {a["node_id"] for a in self.heal_plan}
        if planned and planned <= self.heal_ready:
            apply_layers(self.heal_plan)
            self._clear_heal()
        return self.heal_status()

    def heal_status(self):
        planned = {a["node_id"] for a in self.heal_plan}
        return {
            "healing": bool(self.heal_plan),
            # "surplus" = an idle node is being moved onto the gap; "resplit" = there was no
            # idle node, so the whole model is being re-divided across the survivors ([R6]).
            # Worth distinguishing in the log and on /network/gap-heal: a resplit means the
            # network SHRANK, which is a different operational story from a newcomer landing.
            "mode": self.heal_mode,
            "target": self.heal_target,
            "plan": [{"node_id": a["node_id"], "layers": [a["layer_start"], a["layer_end"]],
                      "ready": a["node_id"] in self.heal_ready} for a in self.heal_plan],
            "ready_count": len(planned & self.heal_ready),
            "plan_size": len(self.heal_plan),
            # Non-zero on a re-split the survivors cannot actually hold: the heal is still the
            # best available move, but the real fix is a tier demotion onto a smaller model.
            "capacity_shortfall": self.heal_shortfall,
        }
