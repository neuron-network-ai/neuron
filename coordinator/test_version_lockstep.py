"""coordinator/test_version_lockstep.py - run: python -m coordinator.test_version_lockstep

The shipped version is written in THREE files and nothing checked that they agree.

    agent/updater.py     LOCAL_VERSION      what a running node reports as `agent_version`
    packaging/neuron.iss AppVersion         what the installer stamps on the build
    coordinator/config.py AGENT_VERSION     what the coordinator PUBLISHES as current

`updater.py` carried the comment "bump together with packaging/neuron.iss" — an instruction to
a human, enforced by nobody, and it named only two of the three.

**Each disagreement has its own failure, and none of them look like a version problem.**

  * `updater` != `iss`: the installer stamps a build that reports a different number, so
    `agent_version` on the dashboard describes a build nobody shipped. 0.20.2 exists precisely
    so a rollback can be CONFIRMED by reading that field — a wrong value there does not degrade
    the feature, it inverts it.
  * `config.AGENT_VERSION` != `updater.LOCAL_VERSION`: the coordinator publishes a version and
    every node compares its own against it. Publish an older one and a current fleet is told it
    is ahead; publish a newer one that no installer produces and every node downloads, verifies
    and installs its way to the same version forever, on a daily timer, on somebody's home
    connection.

The last one is why this file is in `coordinator/` rather than `agent/`: the coordinator is the
half that can turn a typo into a fleet-wide download loop.

It does NOT assert a particular version. It asserts they are the same one, and that whatever
they say parses as a version at all — a test that needs editing on every release is a test
people learn to edit without reading.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agent import updater
from coordinator import config

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def iss_version():
    """The version the Windows installer stamps, read from the Inno Setup script.

    Parsed rather than imported because .iss is not Python and there is no other way to reach
    it — which is exactly why it was the file most likely to be left behind.
    """
    src = open(os.path.join(ROOT, "packaging", "neuron.iss"), encoding="utf-8-sig").read()
    m = re.search(r'#define\s+AppVersion\s+"([^"]+)"', src)
    return m.group(1) if m else None


def main():
    local, published, installer = updater.LOCAL_VERSION, config.AGENT_VERSION, iss_version()
    print(f"        updater.LOCAL_VERSION   {local}")
    print(f"        config.AGENT_VERSION    {published}")
    print(f"        neuron.iss AppVersion   {installer}")

    check("the installer's version could be read at all", installer is not None,
          "packaging/neuron.iss has no `#define AppVersion \"...\"` this test can find")
    check("the running build and the installer agree", local == installer,
          f"a node would report {local} for a build stamped {installer}; `agent_version` is "
          f"the field a rollback is confirmed by")
    check("the coordinator publishes the version that actually exists", published == local,
          f"published {published}, shipped {local}. Publishing a version no installer produces "
          f"makes every node download and install its way to the same version, daily, forever")

    for name, v in (("updater.LOCAL_VERSION", local), ("config.AGENT_VERSION", published),
                    ("neuron.iss AppVersion", installer)):
        check(f"{name} looks like a version ({v})",
              bool(v) and re.fullmatch(r"\d+\.\d+\.\d+", str(v)) is not None)

    # The comment that used to be the only enforcement should now point at this file, so the
    # next person bumping a version finds the check rather than rediscovering the coupling.
    src = open(os.path.join(ROOT, "agent", "updater.py"), encoding="utf-8").read()
    check("updater.py names all three files, not two",
          "neuron.iss" in src and "AGENT_VERSION" in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
