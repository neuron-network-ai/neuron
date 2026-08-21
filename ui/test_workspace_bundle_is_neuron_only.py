"""ui/test_workspace_bundle_is_neuron_only.py — run:
       python -m ui.test_workspace_bundle_is_neuron_only

**The workspace bundle is only correct if it was built with the right flags, and nothing
checked.** `ui/static/workspace/` is a compiled artefact from a separate repo, so the mistake it
can carry is not a syntax error or a failing type — it is a build that succeeded with the wrong
environment and produced a DIFFERENT PRODUCT.

Two flags decide what comes out:

    NEURON_ONLY=1        strips Gemini/Ollama/KoboldCPP, and gates <NeuronBar /> — the wallet,
                         the balance, the sign-in link, and the server-conversation import
    NEURON_BASE=/workspace/   makes the page ask for /workspace/assets/... instead of /assets/...,
                         which is already mounted for the OLDER React app's chunks

Both went wrong on 2026-08-21, in the same afternoon:

  * a rebuild WITHOUT `NEURON_ONLY=1` shipped in 0.20.19 and 0.20.20. The app still worked, so
    every other check passed — and the founder's wallet, balance and sign-in link had silently
    vanished from the header. Reported as *"and in chrome wallet is gone too"*. The bundle was
    not broken; it was the standalone product wearing NEURON's URL.
  * `NEURON_BASE=/workspace/` under Git Bash became `/Program Files/Git/workspace/` through MSYS
    path translation. Caught only by reading the emitted `index.html`.

A compiled artefact from another repo cannot be reviewed in a diff, so it is checked here by
what it CONTAINS. Every assertion below is a string that only survives the correct build.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

WORKSPACE = os.path.join(HERE, "static", "workspace")

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
    index = os.path.join(WORKSPACE, "index.html")
    if not os.path.isfile(index):
        print("  SKIP  no workspace build present (a source checkout that has never run "
              "`npm run build` is a legitimate state — ui/app.py only mounts it when it exists)")
        print("\n0 passed, 0 failed")
        return 0

    html = open(index, encoding="utf-8").read()
    refs = re.findall(r'(?:src|href)="([^"]+)"', html)

    print("-- NEURON_BASE: the page asks for its own assets, not the other app's")
    check("every reference is under /workspace/",
          refs and all(r.startswith("/workspace/assets/") for r in refs),
          f"{refs} — /assets/ is already mounted for static/app, so a root-based build is "
          f"handed the OLD React app's chunks and silently loads the wrong JavaScript")
    check("...and none carries a translated Windows path",
          not any("Program Files" in r or ":" in r for r in refs),
          f"{refs} — MSYS rewrote NEURON_BASE once; use MSYS_NO_PATHCONV=1")

    print("\n-- every referenced asset is actually on disk")
    for r in refs:
        p = os.path.join(WORKSPACE, r[len("/workspace/"):].replace("/", os.sep))
        check(f"{os.path.basename(r)} exists", os.path.isfile(p), p)

    print("\n-- exactly one bundle of each kind, so nothing stale can be served")
    assets = os.listdir(os.path.join(WORKSPACE, "assets"))
    for ext in (".js", ".css"):
        n = [a for a in assets if a.endswith(ext)]
        check(f"one {ext} bundle, not several", len(n) == 1,
              f"{n} — content-hashed names accumulate, and a cached index.html naming an old "
              f"one will happily load it ([P61])")

    js = os.path.join(WORKSPACE, "assets", [a for a in assets if a.endswith(".js")][0])
    src = open(js, encoding="utf-8", errors="replace").read()

    print("\n-- NEURON_ONLY: the parts that only exist in NEURON's build")
    check("the wallet/session bar is compiled in",
          "Answers run here" in src,
          "built without NEURON_ONLY=1, `{__NEURON_ONLY__ && <NeuronBar />}` compiles away and "
          "the wallet, the balance and the SIGN-IN LINK disappear — a working app that has "
          "quietly lost the account surface (shipped this way in 0.20.19 and 0.20.20)")
    check("...and the server-conversation import with it",
          "/conversations" in src,
          "the same flag gates fetchServerThreads, so history stops following the account")

    print("\n-- and the fixes this bundle exists to carry")
    check("citations tolerate {title, href} as well as a url string",
          "href||" in src.replace(" ", ""),
          "[P62]: an object reached React as a child and blanked the whole workspace")
    check("a broken message is contained rather than fatal",
          "could not be displayed" in src,
          "[P62]: without the boundary one bad message unwinds the entire tree")
    check("deleting a conversation deletes it on the server too",
          'method:"DELETE"' in src.replace(" ", "") or "method:'DELETE'" in src.replace(" ", ""),
          "otherwise fetchServerThreads re-imports it on the next load and the chat comes back")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
