"""
agent/gpu.py — does this machine have a usable NVIDIA GPU, and is its owner using it?

Two separate questions, deliberately answered by two different mechanisms:

  detect_gpu()       what the node REPORTS at registration (has_gpu / VRAM / name).
                     Prefers torch.cuda, because that is the thing that would actually
                     run a layer; falls back to nvidia-smi so a machine with a driver but
                     no CUDA-enabled torch build still reports its hardware honestly.

  gpu_utilization()  whether the owner is using the GPU RIGHT NOW (gaming, rendering,
                     training). nvidia-smi only — torch cannot see load from other
                     processes, which is exactly the load we must yield to.

**Scope, stated plainly because it is easy to over-read.** Nothing here makes inference run
on the GPU. `common.py` builds every shard on CPU and there is no device selection anywhere
in the pipeline, so a GPU node today computes on its CPU like every other node. What this
module provides is honest capability *reporting* (so the coordinator knows the hardware
exists and how much VRAM it has) and a *yield signal* (so a node backs off when its owner
starts a game). Actual GPU execution is a `common.py` change and is not implemented.

ARM-compatible: pure Python, no x86 assumptions, no compiled extension. Every entry point
is failure-tolerant — a machine with no GPU, no driver, a broken driver, or an nvidia-smi
that hangs must behave exactly like a normal CPU-only node, never crash the agent.
"""
import logging
import shutil
import subprocess

log = logging.getLogger("neuron.gpu")

# nvidia-smi can block for seconds on a wedged driver. The agent calls this from its
# heartbeat loop, so a hang here stops the node reporting availability at all.
_SMI_TIMEOUT_S = 4

_detected = None          # cache: detection is hardware, it does not change while we run


def _run_smi(query):
    """`nvidia-smi --query-gpu=<query>` -> list of per-GPU field lists, or None.

    None means "could not ask" (no binary, no driver, timeout, non-zero exit) and is always
    distinguished from an empty list ("asked, no GPUs"), because the two justify different
    behaviour: unknown must never be treated as busy, or a machine without nvidia-smi would
    pause forever.
    """
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    rows = []
    for line in out.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append([f.strip() for f in line.split(",")])
    return rows


def _from_torch():
    """(name, vram_gb) from torch.cuda, or None. Guarded hard: importing torch is expensive
    and on a CPU-only wheel `torch.cuda` exists but reports False, which is the common case."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return None
        props = torch.cuda.get_device_properties(0)
        return props.name, round(props.total_memory / 1024 ** 3, 1)
    except Exception as e:
        # A driver/runtime mismatch raises here rather than returning False. That is a
        # CPU-only node as far as we are concerned, not a fatal error.
        log.debug("torch.cuda probe failed: %s", e)
        return None


def _from_smi():
    """(name, vram_gb) from nvidia-smi, or None. Reached when torch is CPU-only but the
    machine really does have a card — worth reporting, since it tells the operator (and a
    future CUDA-enabled build) that the hardware is there."""
    rows = _run_smi("name,memory.total")
    if not rows:
        return None
    name = rows[0][0]
    try:
        vram_gb = round(float(rows[0][1]) / 1024, 1)     # nounits gives MiB
    except (ValueError, IndexError):
        return None
    return name, vram_gb


def detect_gpu(refresh=False):
    """{"has_gpu": bool, "gpu_vram_gb": float|None, "gpu_name": str|None}. Never raises.

    `torch_cuda` says whether the GPU is reachable by the thing that would actually compute
    on it, as opposed to merely present. Today nothing computes on it either way (see the
    module docstring), but the distinction is what a future CUDA build has to key off, and
    recording it now costs nothing.
    """
    global _detected
    if _detected is not None and not refresh:
        return dict(_detected)
    # Belt and braces around both probes. Each guards itself internally, but this function
    # promises never to raise and registration depends on that promise — a node must be able
    # to join as CPU-only no matter how creatively a GPU stack fails.
    found, via_torch = None, True
    try:
        found = _from_torch()
    except Exception as e:
        log.debug("torch GPU probe raised: %s", e)
    if found is None:
        via_torch = False
        try:
            found = _from_smi()
        except Exception as e:
            log.debug("nvidia-smi GPU probe raised: %s", e)
    if found is None:
        _detected = {"has_gpu": False, "gpu_vram_gb": None, "gpu_name": None,
                     "torch_cuda": False}
    else:
        name, vram = found
        _detected = {"has_gpu": True, "gpu_vram_gb": vram, "gpu_name": name,
                     "torch_cuda": via_torch}
    return dict(_detected)


def gpu_utilization():
    """Busiest GPU's utilisation percent, or None if it cannot be determined.

    Returns the MAX across cards rather than an average: one saturated GPU in a two-card
    machine is still an owner who wants their machine back.
    """
    rows = _run_smi("utilization.gpu")
    if not rows:
        return None
    vals = []
    for r in rows:
        try:
            vals.append(float(r[0]))
        except (ValueError, IndexError):
            continue
    return max(vals) if vals else None


def gpu_busy(ceiling_pct):
    """(busy, reason). Unknown utilisation is NOT busy — see _run_smi.

    The failure direction matters: treating "cannot tell" as busy would silently stop every
    node that has no nvidia-smi, which is most of them, and it would look like the network
    had gone quiet rather than like a bug.
    """
    util = gpu_utilization()
    if util is None:
        return False, None
    if util > ceiling_pct:
        return True, f"gpu {util:.0f}% > gpu ceiling {ceiling_pct:.0f}%"
    return False, None


if __name__ == "__main__":       # quick manual check
    info = detect_gpu()
    print("detected:", info)
    print("utilisation:", gpu_utilization())
    print("busy(50):", gpu_busy(50.0))
    if info["has_gpu"] and not info["torch_cuda"]:
        print("note: GPU present but torch has no CUDA — reported, not usable by torch")
    print("note: NEURON does not run inference on the GPU yet (CPU-only pipeline)")
