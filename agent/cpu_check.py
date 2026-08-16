"""
agent/cpu_check.py — does this machine's CPU support the instructions our torch build emits?

**The problem this exists for is a crash with no message.** NEURON's pitch is "ordinary
computers" and "spare hardware", which is precisely the population with the oldest CPUs, and
`agent/requirements.txt` pins `torch==2.4.1` — the standard wheel, which bundles MKL on x86.
`Uraroga/spikingbrain-cpu-cluster` documents that combination taking an invalid-opcode trap on
an Ivy Bridge i3-3240 (AVX and F16C, no AVX2): exit 132, symbolised to
`mkl_vml_kernel_sExp_Z0HAynn`, disassembled to an **EVEX/ZMM AVX-512 instruction on a CPU with
no AVX-512**. `ATEN_CPU_CAPABILITY`, oneDNN ISA and MKL ISA settings all failed to make that
runtime stable; their fix was a custom PyTorch build.

They are two engineers who could read a kernel trap. Ours lands on somebody who double-clicked
an installer, and on Windows the symptom is `0xC000001D` (STATUS_ILLEGAL_INSTRUCTION) —
typically with no message at all. The app simply dies. That is the first-run experience for the
exact person the installer was made for, and nothing tells them it is their processor rather
than our software.

**What this module does and does not claim.** It does not fix the crash and it is not a
verified reproduction — the report is Linux, a specific torch build and an embedded MKL path,
and whether the Windows wheel dispatches the same way on a pre-AVX2 CPU has not been tested
here, because we own no machine old enough. So the design rule is: turn a silent illegal
instruction into a sentence, and never turn a working machine away.

That asymmetry is the whole shape of the file:

  * it refuses ONLY when it positively determined an x86-64 CPU without AVX2;
  * anything it cannot determine — an unreadable probe, an unknown OS, a non-x86 machine —
    proceeds exactly as before, because "we could not check" must never behave like "we
    checked and it failed" ([P24]);
  * and the refusal names an override, because the machine it locks out may well be one that
    would have worked, and the operator is better placed to find out than we are.

ARM is not merely tolerated, it is a supported target: the agent is deliberately
"pure Python + psutil + requests" so a phone or a Pi can run it. AVX is meaningless there and
this module says so rather than failing an architecture it has no opinion about.
"""
import ctypes
import os
import platform
import re
import subprocess

# The instruction set the shipped x86 wheel is assumed to need. AVX2 rather than AVX-512: the
# faulting kernel in the report was AVX-512, but a CPU with AVX2 and no AVX-512 is the ordinary
# modern case (every Ryzen, most Intel desktop parts) and demonstrably works — the machines this
# network runs on today are exactly that. AVX2 is the line the failing report sits below.
REQUIRED = "avx2"

# The operator's escape hatch, named in the refusal itself. This exists because the refusal is
# based on an UNVERIFIED risk: if a pre-AVX2 machine turns out to run fine, the person who owns
# it should not have to edit our source to find out.
OVERRIDE_ENV = "NEURON_SKIP_CPU_CHECK"

# Windows: PF_AVX2_INSTRUCTIONS_AVAILABLE, from winnt.h. `IsProcessorFeaturePresent` is the
# documented way to ask, needs no admin rights and no third-party module -- which matters,
# because this check has to run before anything heavy is imported.
_PF_AVX2 = 40


def _x86():
    """Is this an x86/x86-64 machine at all? AVX is a question that only makes sense here."""
    m = (platform.machine() or "").lower()
    return m in ("x86_64", "amd64", "i386", "i686", "x86") or m.startswith("i") and m.endswith("86")


def _flags_linux():
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.lower().startswith("flags") and ":" in line:
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return None


def _flags_macos():
    out = set()
    for key in ("machdep.cpu.features", "machdep.cpu.leaf7_features"):
        try:
            r = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
            if r.returncode == 0:
                out |= {t.lower() for t in r.stdout.split()}
        except (OSError, subprocess.SubprocessError):
            pass
    return out or None


def _has_avx2_windows():
    try:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(_PF_AVX2))
    except (AttributeError, OSError):
        return None


def probe():
    """What we know about this CPU. Never raises — a probe that can crash the agent is worse
    than the crash it is trying to describe, because it fails on EVERY machine rather than on
    the old ones. The guard is real rather than defensive decoration: this runs at import, in a
    frozen app, on hardware nobody here has seen, and `platform` shells out on some platforms.

    Returns {"arch", "supported": True|False|None, "why", "flags"}. `supported` is
    deliberately three-valued: None means UNDETERMINED, and every caller must treat it as
    permission to continue rather than as a failure — which is also what any failure in here
    degrades to.
    """
    try:
        return _probe()
    except Exception as e:
        return {"arch": "unknown", "supported": None, "flags": [],
                "why": f"the capability probe itself failed ({type(e).__name__})"}


def _probe():
    arch = platform.machine() or "unknown"
    if not _x86():
        return {"arch": arch, "supported": None, "flags": [],
                "why": f"{arch} is not x86, so {REQUIRED.upper()} does not apply"}

    system = platform.system()
    if system == "Windows":
        has = _has_avx2_windows()
        return {"arch": arch, "supported": has, "flags": [],
                "why": ("IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE) "
                        f"= {has}" if has is not None
                        else "IsProcessorFeaturePresent was not callable")}

    flags = _flags_linux() if system == "Linux" else (_flags_macos() if system == "Darwin"
                                                     else None)
    if flags is None:
        return {"arch": arch, "supported": None, "flags": [],
                "why": f"no way to read CPU flags on {system or 'this platform'}"}
    # macOS reports leaf7 features capitalised (AVX2 vs avx2); both are lowercased on read.
    return {"arch": arch, "supported": REQUIRED in flags, "flags": sorted(flags),
            "why": f"{REQUIRED} {'present in' if REQUIRED in flags else 'absent from'} "
                   f"the CPU flags"}


def summary(p=None):
    """A one-line capability string for the log and, later, for the registration payload."""
    p = p or probe()
    state = {True: "ok", False: "MISSING", None: "unknown"}[p["supported"]]
    return f"cpu {p['arch']}: {REQUIRED} {state} ({p['why']})"


def refusal(p=None):
    """The sentence to die with, or None if this machine may proceed.

    None for `supported is None` is the load-bearing line in this file. An unreadable
    `/proc/cpuinfo`, an OS nobody here has tried, a `platform.machine()` we do not recognise --
    all of those must let the node start. Refusing on an inconclusive probe would take working
    machines off the network to guard against a risk that has never been reproduced here, which
    is a strictly worse trade than the crash.
    """
    p = p or probe()
    if p["supported"] is not False:
        return None
    if os.environ.get(OVERRIDE_ENV, "").strip():
        return None
    return (
        f"This processor ({p['arch']}) does not support {REQUIRED.upper()}, which the PyTorch "
        f"build NEURON ships is assumed to need.\n"
        f"Without it the node is expected to die with an illegal-instruction crash "
        f"(0xC000001D on Windows, exit 132 on Linux) during its first forward pass, usually "
        f"with no message at all -- so it is being stopped here, where it can say why.\n"
        f"Your machine is not broken and neither is the download; this build simply targets a "
        f"newer instruction set.\n"
        f"This has NOT been reproduced on hardware like yours -- it is a documented report "
        f"against a different OS and torch build. If you would like to find out, set "
        f"{OVERRIDE_ENV}=1 and run again; the worst case is the crash described above.")
