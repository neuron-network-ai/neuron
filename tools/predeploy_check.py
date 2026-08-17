"""
tools/predeploy_check.py — would deploying the coordinator break anything?

    python tools/predeploy_check.py
    python tools/predeploy_check.py --offline        # skip the live-roster checks

Run it before `coordinator/deploy.sh`. Exit 0 means the checks passed; non-zero means stop.

**WHY THIS EXISTS.** A coordinator deploy is a hand-run step onto a 1 GB VM that the whole
network depends on, and Session 58 is the record of what that costs when it goes wrong: three
deploys reported success while the old build kept serving, because `systemctl enable --now`
does not restart a running unit and the script printed status codes and then said `done`
regardless. The lesson written down at the time was **a check that reports instead of failing
is not a check** — so every check here fails the exit code, and the ones that cannot be
verified say so rather than passing quietly.

The four ways a deploy of THIS coordinator can go wrong, in the order they hurt:

  1. **A migration that throws.** `init_db()` runs at import; a failed `ALTER TABLE` means the
     unit does not start at all, and the network has no coordinator. Tested against a real
     database copy, not a fresh one — a fresh DB is created with every column already present
     and exercises none of the migration path.
  2. **An import the deploy does not ship.** `deploy.sh` sends `coordinator/`, `relay_auth.py`,
     `common.py`, `logtail.py` and excludes `test_*.py`. A module that imports anything else
     imports fine here and crashes on the VM.
  3. **A placement change nobody asked for.** The health sweep applies `canonical_assignment`,
     so a deploy that changes how ranges are computed re-splits the live chain — every node
     re-downloads, and chat stops until they finish. The check compares the new code's plan
     against what the roster holds RIGHT NOW.
  4. **A version published with no hash.** `AGENT_SHA256` empty means no node installs
     anything, which is the correct failure direction and also means every node tries daily and
     declines. Fine if you know; alarming if you do not.
"""
import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Point every import that follows at a throwaway DB. Nothing here may touch a real ledger.
_TMP = tempfile.mkdtemp(prefix="neuron_predeploy_")
os.environ.setdefault("NEURON_DB", os.path.join(_TMP, "scratch.db"))

ok = fail = warn = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def note(label, detail=""):
    """Something true and worth knowing that is not a failure. Counted, never silent."""
    global warn
    warn += 1
    print(f"  NOTE  {label}" + (f"\n        {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
def check_migration():
    """Run the real init_db() against a real database, and prove nothing was lost."""
    print("\n  --- 1. the migration, against a real database ---")
    src = os.path.join(ROOT, "coordinator", "neuron.db")
    if not os.path.exists(src):
        note("no coordinator/neuron.db to migrate against",
             "a fresh DB is created with every column already present and exercises none of "
             "the migration path, so this check cannot be faked with one")
        return
    db = os.path.join(_TMP, "migrate.db")
    shutil.copy(src, db)
    os.environ["NEURON_DB"] = db

    con = sqlite3.connect(db)
    before = [r[1] for r in con.execute("PRAGMA table_info(nodes)")]
    rows = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    con.close()

    import importlib
    from coordinator import models
    importlib.reload(models)
    try:
        models.init_db()
        check("init_db() completes on an existing database", True)
    except Exception:
        check("init_db() completes on an existing database", False, traceback.format_exc())
        return

    con = sqlite3.connect(db)
    after = [r[1] for r in con.execute("PRAGMA table_info(nodes)")]
    rows_after = con.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
    con.close()
    check(f"no rows lost ({rows} nodes before and after)", rows == rows_after)
    added = sorted(set(after) - set(before))
    print(f"        adds {len(added)} column(s): {', '.join(added) or 'none'}")

    try:
        models.init_db(); models.init_db()
        check("init_db() is idempotent (the unit restarts, so it runs every start)", True)
    except Exception:
        check("init_db() is idempotent (the unit restarts, so it runs every start)", False,
              traceback.format_exc())

    # A node from before the migration must still be sized, pessimistically.
    from coordinator import balancer
    live = models.list_nodes()
    if live:
        check("a pre-migration node still sizes, at the pessimistic default",
              balancer.weight_bytes_for(live[0]) == balancer.ASSUMED_WEIGHT_BYTES)


def check_imports():
    """Everything a shipped coordinator module imports must be in the deploy set."""
    print("\n  --- 2. the deploy ships everything the coordinator imports ---")
    shipped = {"coordinator", "relay_auth", "common", "logtail"}
    offenders = []
    cdir = os.path.join(ROOT, "coordinator")
    for fn in sorted(os.listdir(cdir)):
        if not fn.endswith(".py") or fn.startswith("test_"):
            continue                       # deploy.sh excludes test_*.py
        src = open(os.path.join(cdir, fn), encoding="utf-8").read()
        for line in src.split("\n"):
            # MODULE SCOPE ONLY — an indented import is deferred on purpose, and deferring is
            # the correct remedy for exactly this problem. `register_nodes.py` imports
            # `security.proof_of_compute` inside the one branch that needs it, because that
            # module pulls in torch and the VM is deliberately torch-free; the script therefore
            # runs there for everything else. Flagging that would fail a deploy gate on
            # correct code, and a gate that cries wolf is one people learn to skip.
            if line[:1] in (" ", "\t") or not line.strip():
                continue
            s = line.strip()
            mod = None
            if s.startswith("import ") and " " in s:
                mod = s.split()[1].split(".")[0]
            elif s.startswith("from ") and " import " in s:
                mod = s.split()[1].split(".")[0]
            if not mod or mod in shipped:
                continue
            path = os.path.join(ROOT, mod)
            if os.path.exists(path) or os.path.exists(path + ".py"):
                offenders.append(f"{fn}: {s}")
    check("no shipped coordinator module imports a repo module the deploy leaves behind",
          not offenders, "\n        ".join(offenders))


def check_starts():
    """The unit has to come up, with the endpoints nodes depend on."""
    print("\n  --- 3. the coordinator starts ---")
    os.environ["NEURON_DB"] = os.path.join(_TMP, "start.db")
    try:
        from coordinator import main
        check("coordinator.main imports", True)
    except Exception:
        check("coordinator.main imports", False, traceback.format_exc())
        return None
    routes = {getattr(r, "path", None) for r in main.app.routes}
    for want in ("/node/register", "/node/{node_id}/slice-info", "/status", "/network/model"):
        check(f"route {want} is served", want in routes)
    return main


def check_release():
    """A published version with no hash is safe and confusing. Say which one this is."""
    print("\n  --- 4. what this coordinator would publish to nodes ---")
    from coordinator import config
    print(f"        AGENT_VERSION  {config.AGENT_VERSION}")
    if not config.AGENT_SHA256:
        note(f"AGENT_SHA256 is EMPTY while AGENT_VERSION is {config.AGENT_VERSION}",
             "no node will install anything — the correct failure direction, but every node "
             "will see a new version, find nothing to verify it against, and decline DAILY. "
             "Set NEURON_AGENT_SHA256 in the unit, or expect update_check failures fleet-wide.")
    else:
        check("AGENT_SHA256 is set, so a published version can actually be installed", True)


def check_placement(main):
    """The one that decides whether users notice the deploy."""
    print("\n  --- 5. would deploying move the live network? ---")
    from coordinator import config, router
    try:
        url = os.environ.get("NEURON_COORDINATOR", "https://neuronnet.duckdns.org")
        live = json.load(urllib.request.urlopen(f"{url}/node/list", timeout=20))["nodes"]
    except Exception as e:
        note(f"could not read the live roster ({e})",
             "this check is the one that says whether users notice the deploy; it has NOT run")
        return
    online = [n for n in live if n.get("status") == "online"]
    print(f"        {len(online)} online of {len(live)}: "
          + ", ".join(f"{n['node_id']}{n['assigned_layers']}" for n in online))

    sm = main.serving_model()
    plan = router.canonical_assignment(live, sm["layers"], serving_model_id=sm["model_id"])
    shape = router.chain_shape(live, sm["layers"], serving_model_id=sm["model_id"])
    check("the live chain is routable under the NEW code", shape["routable"],
          f"ranges {shape['ranges']}, expected stage 1 {shape['expected_stage1']}")
    if not plan:
        note("auto-repair would produce no plan", "nothing to compare; the chain is untouched")
        return
    cur = {n["node_id"]: tuple(n["assigned_layers"]) for n in live}
    moved = [(a["node_id"], cur.get(a["node_id"]), (a["layer_start"], a["layer_end"]))
             for a in plan if cur.get(a["node_id"]) != (a["layer_start"], a["layer_end"])]
    check("deploying moves NO node's layer range", not moved,
          "these nodes would be re-split and would re-download: "
          + "; ".join(f"{n} {was} -> {now}" for n, was, now in moved))
    over = router.assignment_overflow(live, plan, sm["model_id"])
    check("no node would be given more layers than it can hold", not over, str(over))


def main_():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--offline", action="store_true", help="skip the live-roster checks")
    args = ap.parse_args()

    print("\n  Pre-deploy check — coordinator")
    check_migration()
    check_imports()
    app = check_starts()
    check_release()
    if args.offline:
        note("--offline: the live placement check was SKIPPED, not passed")
    elif app is not None:
        check_placement(app)

    print(f"\n  {ok} passed, {fail} failed, {warn} to be aware of")
    if fail:
        print("  DO NOT DEPLOY until the failures above are understood.")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main_())
