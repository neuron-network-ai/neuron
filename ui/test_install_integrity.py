"""ui/test_install_integrity.py — the half of [P46] a build-time guard cannot cover.

    python -m ui.test_install_integrity

`ui/test_app_assets_resolve.py` proves the SOURCE tree is coherent before it is packaged. It
cannot prove the INSTALL is, and the install is what broke: on 2026-08-17 `/next` served an
index.html asking for `assets/index-DBnmnt4a.js` (404) while `index-eBBdaPpV.js` from a build
six days earlier sat beside it returning 200. The page loaded, rendered nothing, and every other
signal was green — `/` fine, `/status` fine, the node serving and earning.

So these run `app_assets.verify` over directories built to be broken in each of the ways an
install actually breaks, plus the three guards that now stand between a bad copy and a blank
page:

  * the app REFUSES to serve a broken /next, and says which files are missing;
  * it says so at startup too, where somebody is watching;
  * `neuron.iss` clears the hashed bundles before copying and stops the app first, so the copy
    is a replace rather than a merge.

Nothing here needs Node, a build, or an installer: the states are constructed on disk.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui import app_assets                                       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def _build(index_html, files=()):
    """An app directory with the given index.html and the given asset files present."""
    d = tempfile.mkdtemp(prefix="neuron-install-")
    os.makedirs(os.path.join(d, "assets"), exist_ok=True)
    with open(os.path.join(d, "index.html"), "w", encoding="utf-8") as f:
        f.write(index_html)
    for rel in files:
        p = os.path.join(d, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").close()
    return d


PAGE = ('<!doctype html><html><head>'
        '<script type="module" crossorigin src="./assets/index-NEW.js"></script>'
        '<link rel="stylesheet" href="./assets/index-NEW.css">'
        '</head><body><div id="root"></div></body></html>')


def main():
    print("\n-- the failure exactly as it happened live")
    # index.html from THIS build, bundle from the LAST one. Both files exist and look healthy;
    # they simply do not refer to each other.
    d = _build(PAGE, ["assets/index-OLD.js", "assets/index-NEW.css"])
    r = app_assets.verify(d)
    check("a merged install is reported as broken", r["built"] and not r["ok"])
    check("...naming the file that is missing", r["missing"] == ["assets/index-NEW.js"],
          str(r["missing"]))
    check("...and the stale bundle that is the fingerprint of the merge",
          r["orphans"] == ["assets/index-OLD.js"], str(r["orphans"]))

    print("\n-- a healthy install is silent")
    d = _build(PAGE, ["assets/index-NEW.js", "assets/index-NEW.css"])
    r = app_assets.verify(d)
    check("a complete install passes", r["ok"] and not r["missing"])
    check("...with nothing left over", not r["orphans"])

    print("\n-- an orphan ALONE is not a failure")
    # The page still loads and every asset it names resolves. Reporting this as broken would
    # cry wolf on a working install, and the check would stop being read.
    d = _build(PAGE, ["assets/index-NEW.js", "assets/index-NEW.css", "assets/index-ANCIENT.js"])
    r = app_assets.verify(d)
    check("an orphan does not make a working install 'broken'", r["ok"])
    check("...but it is still reported", r["orphans"] == ["assets/index-ANCIENT.js"],
          str(r["orphans"]))

    print("\n-- NOT BUILT is a legitimate state, never a failure")
    # A checkout with no Node has never run `npm run build`. ui/app.py serves a "Not built" page
    # for it on purpose, and reporting it as a broken install would fail the suite for every
    # contributor without a JavaScript toolchain.
    d = tempfile.mkdtemp(prefix="neuron-install-")
    r = app_assets.verify(d)
    check("an unbuilt directory reports built=False", r["built"] is False)
    check("...and is NOT reported as broken", r["ok"] is True)

    print("\n-- external and inline references are not asset paths")
    d = _build('<html><head><script src="https://cdn.example/x.js"></script>'
               '<link href="data:text/css,body{}"><a href="#top">t</a>'
               '<script src="./assets/index-NEW.js"></script></head></html>',
               ["assets/index-NEW.js"])
    r = app_assets.verify(d)
    check("a CDN url, a data: uri and a fragment are all ignored", r["ok"], str(r))
    check("...while the local bundle is still checked", "assets/index-NEW.js" in r["refs"])

    print("\n-- the diagnostic never raises, whatever it is pointed at")
    for label, path in [("a path that does not exist", os.path.join(HERE, "no-such-dir")),
                        ("a file where a directory should be", os.path.abspath(__file__))]:
        try:
            app_assets.verify(path)
            check(f"{label} is handled", True)
        except Exception as e:                                   # noqa: BLE001
            check(f"{label} is handled", False, f"{e.__class__.__name__}: {e}")

    print("\n-- the app refuses to serve a blank page, and says why")
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    check("/next verifies before serving", 'state = app_assets.verify(str(APP_DIR))' in src)
    check("...returns 503 rather than a page that renders nothing",
          'state["ok"]' in src and "status_code=503" in src)
    check("...lists the missing files to the person looking at it",
          'for m in state["missing"]' in src)
    check("...and points at the remedy rather than describing the fault",
          "app_assets.REMEDY" in src)
    check("the remedy says to run the installer, not to copy files",
          "run the installer again" in app_assets.REMEDY
          and "by hand" in app_assets.REMEDY)
    check("startup logs it loudly, because silence looks identical to health",
          "INSTALL IS INCOMPLETE" in src)
    check("...and an orphan is a warning, not an error",
          'log.warning("%d orphaned bundle' in src)

    print("\n-- the installer replaces the bundles rather than merging them")
    iss = open(os.path.join(os.path.dirname(HERE), "packaging", "neuron.iss"),
               encoding="utf-8").read()
    check("[InstallDelete] clears the hashed assets before the copy",
          "[InstallDelete]" in iss
          and r"{app}\_internal\ui\static\app\assets" in iss)
    # EVERY delete must end in `\assets`, rather than there being exactly ONE delete. Counting
    # was a proxy for "scoped", and it stopped being one the moment a second bundle needed the
    # same treatment: the workspace UI shipped its own hashed assets and did not inherit the
    # rule, which is how three builds' bundles ended up in one directory on 2026-08-21. A test
    # that forbids the fix for the bug it exists to prevent is worse than no test.
    deletes = [ln.split("Name:", 1)[1].strip().strip('"')
               for ln in iss.splitlines() if ln.startswith("Type: filesandordirs")]
    check("...scoped to `assets`, not the whole install",
          bool(deletes) and all(d.endswith(r"\assets") for d in deletes),
          f"a broader delete would throw away files this installer does not put back: {deletes}")
    check("...and every hashed bundle directory is covered, not just the first one",
          any("static\\app\\assets" in d for d in deletes)
          and any("static\\workspace\\assets" in d for d in deletes),
          f"{deletes} — a bundle that is not purged merges with the next build")
    check("the running app is closed first ([P24])", "CloseApplications=yes" in iss)
    check("...and NOT silently restarted, since the user chose when to run it",
          "RestartApplications=no" in iss)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
