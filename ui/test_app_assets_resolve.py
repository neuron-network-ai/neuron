"""ui/test_app_assets_resolve.py - run: python -m ui.test_app_assets_resolve

Does the built React page reference files that actually exist?

**A blank page is the failure mode, and nothing detected it.** Vite content-hashes every
bundle, so `index.html` names a DIFFERENT file on every build — `index-DBnmnt4a.js` today,
`index-eBBdaPpV.js` last week. The installer (`neuron.iss`) copies with `ignoreversion`, which
overwrites same-named files and never prunes ones that vanished. So a partial copy, an
interrupted upgrade, or a file locked by the running agent leaves a directory holding LAST
build's bundle and THIS build's index.html, and the two do not refer to each other.

Observed on the founder's machine 2026-08-17: `/next` served an index.html requesting
`index-DBnmnt4a.js` (404) while `index-eBBdaPpV.js` sat beside it serving 200. The page loaded,
rendered nothing, and every other endpoint was healthy — so the app looked fine from the
outside and `/status` had nothing to say about it.

The check is trivial and the reason it is worth a file is that it is the only thing standing
between a routine rebuild and a silently blank UI. It runs against the SOURCE tree
(`ui/static/app`), which is what the installer copies from, so it catches a bad build before it
is ever packaged.

It is deliberately NOT a test that the app is built at all: a source checkout that has never run
`npm run build` has no `ui/static/app`, `ui/app.py` serves a "Not built" page for that case on
purpose, and failing here would break the suite for every contributor who does not have Node.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "static", "app")

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


# Imported, not re-implemented. The build-time and runtime checks are the same question asked
# at different times ([P46]), and two copies of "which files does this page reference" would
# drift exactly where it matters: the installed app would be checked by a rule the build never
# had to satisfy.
from ui.app_assets import referenced_assets, verify   # noqa: E402


def main():
    index = os.path.join(APP, "index.html")
    if not os.path.exists(index):
        # Not built. That is a legitimate state -- ui/app.py serves a "Not built" page for it --
        # and a contributor without Node must not see a red suite because of it.
        print("  SKIP  ui/static/app/index.html is absent (run `npm run build` in ui/web/)")
        print("\n0 passed, 0 failed")
        return 0

    html = open(index, encoding="utf-8").read()
    refs = referenced_assets(html)
    check("index.html references at least one local asset", bool(refs), html[:200])

    missing = [r for r in sorted(refs) if not os.path.exists(os.path.join(APP, r))]
    check("every asset index.html references exists on disk", not missing,
          "MISSING: " + ", ".join(missing) + "\n        This is a blank page for the user: the "
          "page loads, the bundle 404s, and nothing else looks wrong.")

    # The other half of the same fault: files that no longer belong. Harmless on their own, but
    # they are the fingerprint of an install that was merged rather than replaced, and the
    # reason a stale bundle was sitting next to a new index.html.
    assets_dir = os.path.join(APP, "assets")
    if os.path.isdir(assets_dir):
        on_disk = {f"assets/{f}" for f in os.listdir(assets_dir)}
        orphans = sorted(on_disk - refs)
        check("no orphaned bundle from an earlier build is left beside it", not orphans,
              "ORPHANS: " + ", ".join(orphans) + "\n        Not fatal, but this is what a "
              "merged-rather-than-replaced copy looks like -- the state that produced the "
              "missing-asset failure above.")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
