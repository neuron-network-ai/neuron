"""
packaging/publish_coordinator.py — cut a coordinator release the VM will install by itself.

    python packaging/publish_coordinator.py                 # build + write the manifest
    python packaging/publish_coordinator.py --dry-run       # show what it would do

Publishing is the ONLY thing that moves the live coordinator once `neuron-coordinator-update`
is installed (see coordinator/selfupdate.py). Nothing polls git, nothing installs on a commit;
the VM acts on this manifest and only when its version is higher than what it is running.

This builds the archive, hashes it, and writes `packaging/coordinator-latest.json`. It does NOT
push anything — the two publishing steps are yours, deliberately, because they are the moment
the change becomes live everywhere:

    gh release upload coordinator-v<version> dist/neuron-coordinator-<version>.tgz
    git add packaging/coordinator-latest.json && git commit && git push

Order matters: upload the archive FIRST. The manifest is the trigger, so a manifest pointing at
a URL that 404s means the VM tries, fails the download, and stays on the current build — safe,
but it will retry every hour until you fix it.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "packaging" / "coordinator-latest.json"
DIST = ROOT / "dist"

# Exactly what the coordinator runs, and nothing else — same set as coordinator/deploy.sh.
# The VM has no torch and no business holding the agent, the model slices or the installer.
INCLUDE = ("coordinator", "relay_auth.py", "common.py")
EXCLUDE_SUFFIX = (".pyc", ".sh")


def _skip(rel: str) -> bool:
    """Tests and shell scripts don't run on the VM; .pyc is noise."""
    name = rel.rsplit("/", 1)[-1]
    if name.startswith("test_") and name.endswith(".py"):
        return True
    return rel.endswith(EXCLUDE_SUFFIX)


def tracked_files(include=INCLUDE):
    """The files git actually tracks under `include`.

    This is the whole defence against publishing a secret. `coordinator/` on a working machine
    also holds `node_tokens.json` (live per-node auth tokens) and `nodes.local.json` (the
    tailnet addresses) — both gitignored precisely because they must never leave the machine.
    `deploy.sh` scp's them to the VM, which is private; a GitHub release asset is public and
    permanent, and a leaked node token lets anyone impersonate that node's dashboard and payout
    address.

    Walking the directory and denying known-bad names would put the burden on remembering to
    extend the list every time a new runtime file appears. Asking git inverts it: nothing
    untracked can ever be shipped, so a new secret is excluded by default rather than by
    vigilance. No git, no release — refusing is the only safe failure here.
    """
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *include],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(
            f"cannot ask git which files are tracked ({e.__class__.__name__}).\n"
            f"Refusing to build a public release by walking the directory: coordinator/ holds "
            f"gitignored secrets (node_tokens.json, nodes.local.json) that must never ship.")
    return sorted(f for f in out.split("\0") if f and not _skip(f))


def untracked_modules(include=INCLUDE):
    """Python files under `include` that git does not track yet.

    The flip side of shipping only tracked files: a module written but not committed is silently
    left out, and the coordinator dies on `ImportError` the moment it restarts — after the
    archive has already been published. The updater's health gate would catch it and roll back,
    which is the system working, but the release would still be dead on arrival for no reason
    other than a forgotten `git add`.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--others", "--exclude-standard",
             "--", *include],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(f for f in out.split("\0") if f.endswith(".py") and not _skip(f))


def build(version, outdir=DIST):
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"neuron-coordinator-{version}.tgz"
    added = tracked_files()
    if not added:
        raise SystemExit("git reports no tracked files under coordinator/ — wrong directory?")
    missing = untracked_modules()
    if missing:
        raise SystemExit(
            "these modules are not committed, so they would NOT be in the release:\n"
            + "".join(f"  {m}\n" for m in missing)
            + "The coordinator would fail to import on restart. Commit them first:\n"
            + f"  git add {' '.join(missing)}")
    # Belt and braces. `ls-files` already excludes ignored paths, but this artifact is public
    # and permanent, so the assertion is worth its two lines.
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "--", *added],
                             capture_output=True, text=True).stdout.split()
    if ignored:
        raise SystemExit(f"refusing to ship gitignored file(s): {', '.join(ignored)}")
    with tarfile.open(out, "w:gz") as tf:
        for rel in added:
            tf.add(ROOT / rel, arcname=rel)
    return out, added


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--version", default=None,
                   help="version to publish (default: coordinator.config.COORDINATOR_VERSION)")
    p.add_argument("--repo", default="neuron-network-ai/neuron")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    from coordinator import config                              # noqa: PLC0415
    version = args.version or config.COORDINATOR_VERSION

    # The single most likely mistake: build a release whose code still says the old version, so
    # the VM installs it, sees /status report the OLD number, calls the update unhealthy and
    # rolls back a build that was actually fine.
    if args.version and args.version != config.COORDINATOR_VERSION:
        raise SystemExit(
            f"--version {args.version} does not match COORDINATOR_VERSION "
            f"{config.COORDINATOR_VERSION} in coordinator/config.py.\n"
            f"The updater verifies the running version after restart, so a mismatch here "
            f"gets a perfectly good build rolled back. Bump config.py first.")

    # `coordinator-v0.1.0`, not `v0.1.0`. The plain `v<x>` tags are the installer/agent releases
    # (v0.18.0 and back); a coordinator tag in that namespace would sort in among them and read
    # as an app release. The prefix keeps the two release lines legible in one list.
    tag = f"coordinator-v{version}"
    url = (f"https://github.com/{args.repo}/releases/download/"
           f"{tag}/neuron-coordinator-{version}.tgz")

    if args.dry_run:
        print(f"version      : {version}")
        print(f"download_url : {url}")
        print(f"would write  : {MANIFEST}")
        added = tracked_files()
        print(f"would ship   : {len(added)} tracked file(s)")
        for m in untracked_modules():
            print(f"   !! NOT COMMITTED, would be missing from the release: {m}")
        for a in added[:12]:
            print(f"   {a}")
        if len(added) > 12:
            print(f"   ... and {len(added) - 12} more")
        return 0

    archive, added = build(version)
    digest = sha256_file(archive)
    MANIFEST.write_text(json.dumps(
        {"version": version, "download_url": url, "sha256": digest}, indent=2) + "\n")

    print(f"built    {archive}  ({archive.stat().st_size // 1024} KB, {len(added)} files)")
    print(f"sha256   {digest}")
    print(f"manifest {MANIFEST}")
    print("\nNow, in this order:")
    print(f"  gh release create {tag} {archive} --generate-notes   # or `gh release upload`")
    print(f"  git add {MANIFEST.relative_to(ROOT)} && git commit -m 'coordinator {version}' "
          f"&& git push")
    print("\nThe VM installs it within the hour, verifies it, and rolls back by itself if it "
          "does not come up healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
