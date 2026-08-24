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

**Both held on 2026-08-21 while the public site sent every visitor to v0.19.0.** They are
invariants about the repo AGREEING WITH ITSELF, and the repo agreed with itself perfectly at
0.20.3 for nineteen releases. Two more, added the day that was found:

  3. the version those links name is the version the COORDINATOR publishes at /agent/version —
     the one every running node is told to install. A page nineteen releases behind the fleet
     is not inconsistent with anything the first two checks can see;
  4. any SHA-256 printed beside a download link is the hash of the file that link points at
     (`config.AGENT_SHA256`). The installer is unsigned, so those pages tell people to verify
     the hash by hand — and `docs/index.html` said `80eb83a2...` while `README.md` said
     `b9d486b1...` for the SAME v0.20.3, whose published asset actually hashes to `01867f43...`.
     Neither was ever right. The only person harmed by that is the careful one: they check, find
     a mismatch on a completely legitimate download, and learn that checking is noise.
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

        # 3. THE PAGE MUST NOT BE BEHIND THE FLEET.
        #
        # This is the check the first three could not make, because nothing above compares the
        # repo to anything outside itself. Every file listed here agreed on 0.20.3 while the
        # coordinator published 0.20.22 and the live site served 0.19.0 — three answers to "what
        # do I install", all of them silent, because an old release link still downloads and the
        # installer it fetches then auto-updates itself. The visitor is never told, and the only
        # trace is that their first run is on a build from before the correctness work.
        stale = sorted({(f, t) for f, ps in found.items() for t, _ in ps if t != av})
        check(f"every download link names the version the coordinator publishes (v{av})",
              not stale,
              "; ".join(f"{f} sends people to v{t}" for f, t in stale)
              + f" while /agent/version tells every node to run v{av}" if stale else "")

    # 4. AND A HASH BESIDE A DOWNLOAD LINK MUST BE THAT DOWNLOAD'S HASH.
    #
    # Printed so a stranger can verify an unsigned installer by hand, which makes a wrong one
    # worse than none: it fails on a good download and teaches the careful reader to skip the
    # check. Both public pages carried a hash that matched nothing at all.
    ms = re.search(r'AGENT_SHA256\s*=\s*os\.environ\.get\(\s*"NEURON_AGENT_SHA256"\s*,'
                   r'\s*"([0-9a-f]{64})"\s*\)', cfg)
    check("coordinator/config.py declares a default AGENT_SHA256", bool(ms))
    if ms:
        published = ms.group(1)
        wrong = []
        for rel in FILES:
            path = os.path.join(HERE, rel)
            if not os.path.exists(path):
                continue
            text = open(path, encoding="utf-8").read()
            if "NEURON-Setup-" not in text:
                continue          # not a page that offers the download; nothing to verify
            for h in set(re.findall(r'\b[0-9a-f]{64}\b', text)):
                if h != published:
                    wrong.append((rel, h))
        check("every SHA-256 shown beside a download link is the published installer's",
              not wrong,
              "; ".join(f"{f} prints {h[:8]}..." for f, h in sorted(wrong))
              + f" but the published installer hashes to {published[:8]}..." if wrong else "")

    # 5. AND A FILENAME WRITTEN OUT ON ITS OWN IS A LINK WITH THE HREF FILED OFF.
    #
    # Everything above matches `releases/download/vX/NEURON-Setup-X.exe` — a full URL. But the
    # blocked-download instructions added to README.md tell somebody to run
    # `certutil -hashfile NEURON-Setup-0.20.25.exe SHA256`, and that bare filename goes stale on
    # exactly the same release the links do, in exactly the same silence. Worse than the links,
    # in fact: a stale LINK still downloads something, while a stale filename in a verify command
    # names a file that is not in the folder, so the one reader careful enough to check the hash
    # of an unsigned installer is the one who gets `The system cannot find the file specified`.
    #
    # PACKAGING.md carried `NEURON-Setup-0.12.0.exe` as its build-output example for thirteen
    # releases, which is how long this goes unnoticed when nothing looks.
    #
    # Version-less forms (`NEURON-Setup-<ver>.exe`) are the correct way to write it where no
    # specific build is meant, and do not match this pattern at all.
    NAMED = re.compile(r"NEURON-Setup-([0-9]+(?:\.[0-9]+)+)\.exe")
    if versions:
        want = sorted(versions)[0]
        stale_names = []
        for rel in FILES:
            path = os.path.join(HERE, rel)
            if not os.path.exists(path):
                continue
            for v in set(NAMED.findall(open(path, encoding="utf-8").read())):
                if v != want:
                    stale_names.append((rel, v))
        check("every installer filename spelled out names that same version",
              not stale_names,
              "; ".join(f"{f} writes NEURON-Setup-{v}.exe" for f, v in sorted(stale_names))
              + f" while the download links serve v{want}; write NEURON-Setup-<ver>.exe where "
                f"no particular build is meant" if stale_names else "")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
