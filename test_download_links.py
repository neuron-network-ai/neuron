"""test_download_links.py — every "download NEURON" link points at one released build

    python test_download_links.py

The download page sat on **v0.20.2 while v0.20.3 was the published release** and nothing said
so: the version is hardcoded in `docs/index.html` (twice), `README.md`, and the install docs,
and a release bumps some of them. Nobody notices, because a stale link still WORKS — it serves
an older installer perfectly happily, so the failure is silent and every new volunteer gets the
old build.

The opposite mistake is worse and is one keystroke away: pointing at a version that was built
but never released (0.20.4 is exactly that today) 404s for every visitor.

So two invariants, both checkable offline:
  1. every download link names the SAME version;
  2. that version has release notes in the repo — the cheapest available proxy for "this was
     actually released", since a build nobody wrote notes for is not one to send strangers to.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["docs/index.html", "README.md", "STRANGER_INSTALL.md", "INSTALL.md", "PACKAGING.md"]
LINK = re.compile(r"releases/download/v([0-9]+(?:\.[0-9]+)+)/NEURON-Setup-([0-9]+(?:\.[0-9]+)+)\.exe")

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
    found = {}
    for rel in FILES:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        for tag, exe in LINK.findall(text):
            found.setdefault(rel, set()).add((tag, exe))

    check("at least one download link exists to check", bool(found), str(found))
    if not found:
        print(f"\n{ok} passed, {fail} failed")
        return fail == 0

    pairs = {p for pairs in found.values() for p in pairs}
    mismatched = {(t, e) for t, e in pairs if t != e}
    check("each link's tag and filename agree", not mismatched,
          f"a v{list(mismatched)[0][0]} tag serving a {list(mismatched)[0][1]} exe would 404"
          if mismatched else "")

    versions = {t for t, _ in pairs}
    check("every file names the same version", len(versions) == 1,
          "; ".join(f"{f}: {sorted(t for t, _ in ps)}" for f, ps in sorted(found.items())))

    for v in sorted(versions):
        notes = os.path.join(HERE, f"RELEASE_NOTES_v{v}.md")
        check(f"v{v} has release notes, so it was really released", os.path.exists(notes),
              f"no {os.path.basename(notes)} — is v{v} built but unreleased? that link 404s")

    # THE SAME QUESTION, ASKED OF THE VERSION THAT REACHES THE FLEET.
    #
    # A stale link on a web page serves an older installer and nothing breaks. What
    # `coordinator/config.AGENT_VERSION` names is different in kind: every agent asks
    # `/agent/version` daily, `AGENT_DOWNLOAD_URL` is DERIVED from it, and the personal Chat UI
    # now shows "vX is available — Get it" pointing at that url. Naming a version that was never
    # published turns all three into a 404 for every volunteer at once.
    #
    # It is 0.20.4 in the file today, and 0.20.4 has no release notes: production is correct only
    # because a systemd drop-in pins 0.20.3 on the VM. That is a real configuration holding back
    # a wrong default, and it is exactly the shape of [P48] — the repo self-consistent, the thing
    # a person is actually served wrong. Empty AGENT_SHA256 means nobody would INSTALL the 404
    # (the correct failure direction), but they are still told to go and get it.
    #
    # Read as text rather than imported: importing `coordinator.config` picks up the environment
    # of whoever runs the suite, so on the VM this check would read the pin and pass while the
    # committed default stayed wrong.
    cfg = open(os.path.join(HERE, "coordinator", "config.py"), encoding="utf-8").read()
    m = re.search(r'AGENT_VERSION\s*=\s*os\.environ\.get\(\s*"NEURON_AGENT_VERSION"\s*,\s*'
                  r'"([0-9][0-9.]*)"\s*\)', cfg)
    check("coordinator/config.py declares a default AGENT_VERSION", bool(m), cfg[:200])
    if m:
        av = m.group(1)
        notes = os.path.join(HERE, f"RELEASE_NOTES_v{av}.md")
        check(f"the coordinator's default AGENT_VERSION (v{av}) was really released",
              os.path.exists(notes),
              f"no RELEASE_NOTES_v{av}.md. AGENT_DOWNLOAD_URL is derived from this, so every "
              f"agent's daily update check and the Chat UI's update notice both point at a "
              f"release that does not exist. Production is only correct while an env pin "
              f"overrides it — remove the pin and the whole fleet is sent to a 404.")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
