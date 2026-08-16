"""agent/test_cpu_floor.py - run: python -m agent.test_cpu_floor

[P41]: nothing anywhere checked what instruction set a volunteer's CPU supports, and the torch
wheel we ship bundles MKL on x86. The documented failure is an invalid-opcode trap inside an
MKL kernel on a pre-AVX2 machine -- on Windows, `0xC000001D` with no message at all. A stranger
double-clicks an installer and the app dies silently, with no way to know it is their processor
rather than our software.

The fix is a probe, and a probe that refuses too eagerly is worse than the crash: it takes
working machines off a network that needs them, to guard against something NOBODY HERE HAS
REPRODUCED. The report is Linux, a specific torch build and an embedded MKL path; we own no
machine old enough to test the Windows wheel on.

So the asymmetry is the specification, and it is what this file mostly tests:

  refuse  ONLY on a positive determination of x86-without-AVX2
  proceed on anything undetermined -- unreadable probe, unknown OS, non-x86 machine
  proceed on an operator override, which the refusal names in its own text
  never  raise out of the probe itself: a capability check that can kill the agent has
         replaced one silent death with another
"""
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import cpu_check

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def P(supported, arch="x86_64", why="fixture", flags=()):
    return {"arch": arch, "supported": supported, "why": why, "flags": list(flags)}


def without_override(fn):
    old = os.environ.pop(cpu_check.OVERRIDE_ENV, None)
    try:
        return fn()
    finally:
        if old is not None:
            os.environ[cpu_check.OVERRIDE_ENV] = old


def main():
    # ---------- 1. the three-valued answer, which is the whole design --------
    check("a supported CPU is allowed through",
          without_override(lambda: cpu_check.refusal(P(True))) is None)
    check("an UNSUPPORTED x86 CPU is refused",
          without_override(lambda: cpu_check.refusal(P(False))) is not None)
    check("an UNDETERMINED probe is allowed through, not refused",
          without_override(lambda: cpu_check.refusal(P(None))) is None,
          "'we could not check' must never behave like 'we checked and it failed' ([P24])")

    # ---------- 2. never fail an architecture we have no opinion about -------
    for arch in ("aarch64", "arm64", "armv7l", "riscv64", "ppc64le", "s390x"):
        p = _probe_as(arch)
        check(f"{arch} is undetermined rather than unsupported (the agent targets ARM)",
              p["supported"] is None and without_override(
                  lambda: cpu_check.refusal(p)) is None,
              "the agent is pure Python + psutil + requests precisely so a phone or a Pi "
              "can run it; AVX is not a question there")

    # ---------- 3. the override the refusal advertises actually works --------
    msg = without_override(lambda: cpu_check.refusal(P(False)))
    check("the refusal names the override variable, so it is discoverable from the message",
          cpu_check.OVERRIDE_ENV in msg)
    old = os.environ.get(cpu_check.OVERRIDE_ENV)
    try:
        os.environ[cpu_check.OVERRIDE_ENV] = "1"
        check("...and setting it lets an unsupported machine try anyway",
              cpu_check.refusal(P(False)) is None,
              "this risk is documented elsewhere and unreproduced here; the person who owns "
              "the machine is better placed to find out than we are")
    finally:
        os.environ.pop(cpu_check.OVERRIDE_ENV, None)
        if old is not None:
            os.environ[cpu_check.OVERRIDE_ENV] = old
    check("an EMPTY override is not an override",
          _with_env(cpu_check.OVERRIDE_ENV, "", lambda: cpu_check.refusal(P(False))) is not None)

    # ---------- 4. the message is for a person, not for us -------------------
    for want, why in (("0xC000001D", "the code they will actually see on Windows"),
                      ("AVX2", "the thing their CPU lacks"),
                      ("not broken", "it is neither their machine nor a bad download")):
        check(f"the refusal mentions {want} - {why}", want in msg)
    check("...and does not claim a reproduction we have not made",
          "NOT been reproduced" in msg)

    # ---------- 5. the probe on THIS machine, whatever it is -----------------
    p = cpu_check.probe()
    check("probe() returns the full shape on the real machine",
          set(p) == {"arch", "supported", "why", "flags"} and p["supported"] in (True, False, None),
          repr(p))
    check(f"...and explains itself: {p['why']!r}", bool(p["why"]))
    print(f"        this machine: {cpu_check.summary(p)}")
    if _x86_here():
        check("an x86 machine running these tests reaches a definite answer",
              p["supported"] is not None,
              "if the probe cannot read a CPU it is standing on, it will not read a "
              "volunteer's either")
        check("...and that answer is True, since this machine is running the suite",
              p["supported"] is True)

    # ---------- 6. a probe must not be what kills the agent ------------------
    for broken in ("machine", "system"):
        old_fn = getattr(platform, broken)
        try:
            setattr(platform, broken, _boom)
            try:
                cpu_check.probe()
                survived = True
            except Exception:
                survived = False
        finally:
            setattr(platform, broken, old_fn)
        check(f"probe() survives platform.{broken}() raising", survived,
              "replacing a silent crash with a different silent crash is not progress")

    # ---------- 7. it is wired in BEFORE torch ------------------------------
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py"),
               encoding="utf-8").read()
    # By LINE, not by substring: agent.py discusses `from agent.node_server import NodeServer`
    # in a comment 60 lines above the real import, so a naive `src.index` compares against
    # prose and passes whatever the code does. (It did, until this test failed on the true
    # ordering being correct.)
    lines = [ln.split("#")[0].strip() for ln in src.split("\n")]
    at_check = lines.index("from agent import cpu_check")
    at_torch = lines.index("from agent.node_server import NodeServer")
    check("agent.py probes the CPU before importing the module that pulls in torch",
          at_check < at_torch,
          "it is torch's bundled MKL that faults; probing after importing it puts the check "
          "downstream of what it guards")
    check("...and refuses through crash_log, which works before logging exists",
          "crash_log(_CPU_REFUSAL)" in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


def _boom(*a, **k):
    raise RuntimeError("probe hostile environment")


def _x86_here():
    m = (platform.machine() or "").lower()
    return m in ("x86_64", "amd64", "i386", "i686", "x86")


def _probe_as(arch):
    old = platform.machine
    try:
        platform.machine = lambda: arch
        return cpu_check.probe()
    finally:
        platform.machine = old


def _with_env(name, value, fn):
    old = os.environ.get(name)
    try:
        os.environ[name] = value
        return fn()
    finally:
        os.environ.pop(name, None)
        if old is not None:
            os.environ[name] = old


if __name__ == "__main__":
    sys.exit(main())
