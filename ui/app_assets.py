"""ui/app_assets.py — does the built React page reference files that actually exist? ([P46])

**One definition, two callers, because the two halves of [P46] are the same question asked at
different times.** `ui/test_app_assets_resolve.py` asks it of the SOURCE tree before a build is
packaged; `ui/app.py` asks it of the INSTALL at startup, on the machine, where the answer is
about what was really copied. The build-time guard was written first and it cannot see the case
that actually happened: the build was coherent and the install was not.

The mechanism is a property of the toolchain rather than a one-off. Vite content-hashes every
bundle, so `index.html` names a DIFFERENT file each build — `index-DBnmnt4a.js` today,
`index-eBBdaPpV.js` last week. `neuron.iss` copies with `ignoreversion`, which overwrites
same-named files and **never prunes** ones that vanished. So anything copying a SUBSET — an
interrupted install, a file locked by the running agent, a hand-copy — leaves last build's
bundle beside this build's index.html, and the two do not refer to each other. Files whose
names did not change (`react-*.js`) update fine and disguise it.

Live 2026-08-17: `/next` served an index.html requesting a 404, rendered nothing, and every
other signal was green.
"""
import os
import re

# A blank page is what this prevents, so the messages are written for the person looking at one.
REMEDY = ("Quit NEURON and run the installer again — do not copy files in by hand. A partial "
          "copy leaves the previous build's bundle beside this build's index.html and the two "
          "do not refer to each other.")


def referenced_assets(html):
    """Every local asset the page asks for, however it is spelled.

    Vite emits `./assets/name-HASH.js`; a leading `./` or `/` is stripped so the result is
    always a path relative to the app root, which is how it sits on disk.
    """
    out = set()
    for m in re.finditer(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']', html):
        ref = m.group(1)
        if ref.startswith(("http://", "https://", "//", "data:", "#")):
            continue
        out.add(ref.lstrip("./").lstrip("/"))
    return out


def verify(app_dir):
    """Inspect a built app directory. Never raises — a diagnostic that can crash its caller is
    worse than the fault it reports.

    Returns {built, ok, missing, orphans, refs, error}:
      * `built` False means there is no index.html at all. That is a LEGITIMATE state — a
        checkout with no Node has never run `npm run build`, and `ui/app.py` serves a "Not
        built" page for it on purpose. It is not a failure and must never be reported as one.
      * `missing` is the fault that renders a blank page: index.html asks for a file that is
        not there.
      * `orphans` is the fingerprint of a merged-rather-than-replaced copy — a bundle from an
        earlier build nobody references. Harmless alone, and it is the state that produces
        `missing`, so it is worth saying out loud before it does.
    """
    index = os.path.join(app_dir, "index.html")
    if not os.path.exists(index):
        return {"built": False, "ok": True, "missing": [], "orphans": [], "refs": [],
                "error": None}
    try:
        html = open(index, encoding="utf-8").read()
    except OSError as e:
        return {"built": True, "ok": False, "missing": [], "orphans": [], "refs": [],
                "error": f"could not read index.html: {e}"}

    refs = referenced_assets(html)
    missing = sorted(r for r in refs if not os.path.exists(os.path.join(app_dir, r)))

    orphans = []
    assets_dir = os.path.join(app_dir, "assets")
    if os.path.isdir(assets_dir):
        try:
            on_disk = {f"assets/{f}" for f in os.listdir(assets_dir)}
            orphans = sorted(on_disk - refs)
        except OSError:
            pass                      # unreadable directory is not this check's business

    # `ok` keys on `missing` only. An orphan is reported and does not make the app broken,
    # because it is not: the page still loads and every asset it names resolves.
    return {"built": True, "ok": not missing, "missing": missing, "orphans": orphans,
            "refs": sorted(refs), "error": None}
