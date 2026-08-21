"""coordinator/test_repair_standdown_is_bounded.py — run:
       python -m coordinator.test_repair_standdown_is_bounded

**Auto-repair stands down for a migration, and the stand-down had no bound.**

The stand-down is right. A repair that rewrote ranges mid-cutover would leave half the network
on each model's partition, which is unrecoverable rather than merely unroutable. But `phase`
only leaves `"preparing"` when EVERY planned node reports ready — so one node that never
reports keeps repair switched off for as long as it stays away.

Live on 2026-08-21: the chain collapsed to `[[0,27]]` at 05:46 and sat unroutable until a human
ran `pin_layers.sh` hours later. `/status` said `routable: false` and nothing anywhere said
why; the reason was in the coordinator's stdout, on a VM whose SSH key needs a person to
unlock it.

**The argument for the bound is already in the codebase, about a different field.**
`migration.py` refuses to make "blocked" a phase, and says why: *"a 'blocked' phase would
silently disable gap healing ... the network would be both unable to grow AND unable to
repair."* That is exactly as true of a migration that is preparing and not progressing. This
file is that sentence, applied to `preparing`, and made a check instead of a comment.

What is pinned:

  * a migration that enters `preparing` records WHEN, so a caller can tell one that is working
    from one that is wedged;
  * a completed or abandoned migration clears it, so a stale timestamp cannot make a healthy
    network look stalled;
  * repair stands down inside the bound and proceeds past it — the property that turns a
    permanent outage back into a delay;
  * `/status` publishes why a repair is not happening, because an operator who can see
    `routable: false` should be able to see the next question's answer in the same response.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from coordinator import config
from coordinator.migration import MigrationController

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def _nodes(n=2, layers=28):
    """A roster big enough to plan a migration across."""
    return [{"node_id": f"n{i}", "status": "online", "eligible": True,
             "layer_start": i * (layers // n), "layer_end": (i + 1) * (layers // n) - 1,
             "ram_gb": 64, "ms_per_layer": 10.0, "head_ms": 1.0 if i == 0 else 0}
            for i in range(n)]


def stood_down(migrating, preparing_since, now):
    """The sweep's decision, as `coordinator/main.py` makes it. Kept as one expression here so
    the rule is stated once and read the same way by the test and the reader."""
    stalled = migrating and (preparing_since is None
                             or (now - preparing_since) > config.REPAIR_STANDDOWN_S)
    return migrating and not stalled


def main():
    print("-- entering 'preparing' records when")
    c = MigrationController()
    check("a fresh controller is steady with no timestamp",
          c.phase == "steady" and c.preparing_since is None)

    applied = []
    c.update(_nodes(), {"model_id": "big", "layers": 28, "gb_per_layer": 0.1},
             {"model_id": "small", "layers": 28}, 1000.0,
             lambda m, l: applied.append((m, l)))
    check("a warranted migration enters preparing", c.phase == "preparing", c.phase)
    check("...and stamps the clock it was given",
          c.preparing_since == 1000.0, repr(c.preparing_since))
    check("...and publishes it, so the sweep does not have to reach inside",
          c.status().get("preparing_since") == 1000.0, str(c.status().get("preparing_since")))

    print("\n-- a migration that ENDS clears the stamp")
    # Target == serving is the "nothing to migrate, or a just-completed cutover" path.
    c.update(_nodes(), {"model_id": "small", "layers": 28},
             {"model_id": "small", "layers": 28}, 2000.0, lambda m, l: None)
    check("back to steady", c.phase == "steady")
    check("...and the timestamp is gone, so nothing looks stalled later",
          c.preparing_since is None,
          "a stale stamp would make the next healthy migration look instantly wedged")

    print("\n-- the stand-down, at the boundary")
    t0 = 1000.0
    within = t0 + config.REPAIR_STANDDOWN_S - 1
    past = t0 + config.REPAIR_STANDDOWN_S + 1
    check("a healthy network is never stood down for", not stood_down(False, None, past))
    check("repair waits while a migration is genuinely preparing",
          stood_down(True, t0, within),
          "repairing mid-cutover leaves half the network on each partition — unrecoverable")
    check("repair PROCEEDS once the bound is passed",
          not stood_down(True, t0, past),
          "an unroutable network serves nobody, and a migration that has not converged is "
          "not a reason to keep it that way")

    print("\n-- a migration with no timestamp is not waited on forever either")
    check("missing preparing_since does not grant an unbounded stand-down",
          not stood_down(True, None, past),
          "an older controller, or a phase set by some path that forgot to stamp, must not "
          "be able to disable repair by omission")

    print("\n-- the bound is long enough that it cannot break a healthy migration")
    check("the stand-down exceeds a slow slice download",
          config.REPAIR_STANDDOWN_S >= 600,
          f"{config.REPAIR_STANDDOWN_S}s — repairing UNDER a working migration is the failure "
          f"this bound must not cause")

    print("\n-- /status answers the next question an operator asks")
    src = open(os.path.join(HERE, "main.py"), encoding="utf-8").read()
    check("the sweep records WHY a repair was skipped",
          '_repair_blocked = {"reason": "migration"' in src)
    check("...including when the roster is simply too small",
          '_repair_blocked = {"reason": "roster"' in src)
    check("...and /status publishes it next to `routable`",
          '"repair_blocked":' in src,
          "routable: false with no reason is what made the outage take hours")
    check("the stalled case says so out loud rather than repairing silently",
          "repairing anyway" in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
