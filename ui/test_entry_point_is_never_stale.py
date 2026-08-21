"""ui/test_entry_point_is_never_stale.py — run:
       python -m ui.test_entry_point_is_never_stale

**A page that is genuinely installed, genuinely served, and verifiable with `curl` can still be
invisible to the person sitting in front of it.** The `/` route has carried a paragraph about
this since 2026-08-19, and `no-store` to go with it. The workspace UI, mounted later through
`StaticFiles`, inherited neither: `StaticFiles` sends an ETag and **no `Cache-Control` at all**.

Live 2026-08-21. The founder's browser showed a blank `/workspace/` while the server was serving
a correct page — `curl` returned the right HTML and both assets 200'd, and the same URL rendered
perfectly in another browser. Two things had to be true at once, and both were:

  * the browser was free to reuse a cached `index.html` from an earlier build, because nothing
    told it not to;
  * that older `index.html` names a content-hashed bundle which was **still on disk**, because
    `[InstallDelete]` purged `static/app/assets` (the rule [P46] added) and never purged
    `static/workspace/assets` (added later). Three builds' bundles were sitting there at once.

So instead of a loud 404 the operator got a silent, working, two-builds-old app. **A stale page
that 404s is a bug report; a stale page that works is a mystery.**

Both halves are pinned here, because either one alone still leaves a way to serve yesterday's
app: no-store without the purge merely makes it rarer, and the purge without no-store turns a
silent old app into a blank page.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

os.environ.setdefault("NEURON_SESSION_SECRET", "test-secret-not-a-real-one")

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    from ui import app as uiapp

    print("-- the HTML entry point is never cached")
    check("the workspace mount uses the no-store StaticFiles, not the plain one",
          "_EntryPointNotCached(directory=str(STATIC_DIR / \"workspace\")" in
          open(os.path.join(HERE, "app.py"), encoding="utf-8").read(),
          "a plain StaticFiles sends an ETag and no Cache-Control, which is what let a cached "
          "index.html from an earlier build keep being used")

    cls = uiapp._EntryPointNotCached
    src = open(os.path.join(HERE, "app.py"), encoding="utf-8").read()
    body = src[src.index("class _EntryPointNotCached"):src.index("if (STATIC_DIR / \"workspace\"")]
    check("it sets no-store on .html", '"Cache-Control"' in body and "no-store" in body)
    check("...and must-revalidate, the same pair `/` uses", "must-revalidate" in body)
    check("...only for .html, so hashed assets stay cacheable",
          '.endswith(".html")' in body,
          "Vite content-hashes the assets, so a changed asset is a changed URL — caching those "
          "is correct and free, and the `/` route draws the same distinction")
    check("it overrides file_response, which is what StaticFiles actually calls",
          "def file_response(" in body)
    check("...and returns the response rather than dropping it",
          body.rstrip().endswith("return resp"))

    print("\n-- the behaviour, not just the source text")
    class _Stat:
        st_mode = 0o100644
        st_size = 10
        st_mtime = 0
        st_ino = 1
        st_dev = 1

    captured = {}

    class _Fake(cls):
        def __init__(self):
            pass

        def _sup(self, path):
            class R:
                headers = {}
            captured["r"] = R()
            return captured["r"]

    # Exercise the real method with a stubbed super(), so the header logic is what is tested.
    import starlette.staticfiles as sf
    real = sf.StaticFiles.file_response
    try:
        sf.StaticFiles.file_response = lambda self, p, s, scope, status_code=200: _Fake()._sup(p)
        inst = cls.__new__(cls)
        html = cls.file_response(inst, "/x/index.html", _Stat(), {}, 200)
        check("an .html response carries no-store",
              html.headers.get("Cache-Control") == "no-store, must-revalidate",
              str(html.headers))
        asset = cls.file_response(inst, "/x/assets/index-ABC123.js", _Stat(), {}, 200)
        check("a hashed asset is left alone", "Cache-Control" not in asset.headers,
              str(asset.headers))
        upper = cls.file_response(inst, "/x/INDEX.HTML", _Stat(), {}, 200)
        check("...and the extension check is case-insensitive",
              upper.headers.get("Cache-Control") == "no-store, must-revalidate")
    finally:
        sf.StaticFiles.file_response = real

    print("\n-- the installer purges the bundles it cannot prune")
    iss = open(os.path.join(ROOT, "packaging", "neuron.iss"), encoding="utf-8").read()
    dele = iss[iss.index("[InstallDelete]"):iss.index("[Files]")]
    for d in ("app", "workspace"):
        check(f"static\\{d}\\assets is cleared before copying",
              f"ui\\static\\{d}\\assets" in dele,
              "ignoreversion overwrites same-named files and NEVER prunes ones that vanished, "
              "so an upgrade merges two builds unless the directory is deleted first")
    check("both are `filesandordirs`, so the directory really goes",
          dele.count("Type: filesandordirs") >= 2)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
