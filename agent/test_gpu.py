"""GPU detection + yield tests — run: python -m agent.test_gpu

Every branch is driven through stubs rather than real hardware, because the machine these
were written on has no NVIDIA GPU at all (torch 2.4.1+cpu, no nvidia-smi) and the paths that
matter most are the failure ones: no binary, a wedged driver, a timeout, garbage output. A
node that cannot answer "is the GPU busy?" must behave exactly like a CPU-only node.
"""
import subprocess
import types

import agent.gpu as gpu
import agent.resource_guard as rg

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def stub_smi(stdout="", returncode=0, exe="/usr/bin/nvidia-smi", raises=None):
    """Point gpu._run_smi at a fake nvidia-smi."""
    gpu.shutil.which = lambda n: exe
    def _run(*a, **k):
        if raises:
            raise raises
        return types.SimpleNamespace(stdout=stdout, returncode=returncode)
    gpu.subprocess.run = _run


def main():
    real_which, real_run = gpu.shutil.which, gpu.subprocess.run
    try:
        # ---- detect_gpu: no GPU anywhere ----
        gpu._detected = None
        gpu.shutil.which = lambda n: None            # no nvidia-smi
        gpu._from_torch = lambda: None               # no CUDA torch
        info = gpu.detect_gpu(refresh=True)
        check("no GPU -> has_gpu False, fields None",
              info["has_gpu"] is False and info["gpu_vram_gb"] is None
              and info["gpu_name"] is None and info["torch_cuda"] is False)

        # ---- detect_gpu: torch is the preferred source ----
        gpu._from_torch = lambda: ("NVIDIA GeForce RTX 4070", 12.0)
        info = gpu.detect_gpu(refresh=True)
        check("torch.cuda GPU reported with vram + torch_cuda True",
              info["has_gpu"] and info["gpu_vram_gb"] == 12.0
              and info["gpu_name"] == "NVIDIA GeForce RTX 4070" and info["torch_cuda"])

        # ---- detect_gpu: falls back to nvidia-smi when torch is CPU-only ----
        gpu._from_torch = lambda: None
        stub_smi("NVIDIA GeForce GTX 1660, 6144\n")
        info = gpu.detect_gpu(refresh=True)
        check("CPU-only torch + real card -> reported via smi, torch_cuda False",
              info["has_gpu"] and info["gpu_vram_gb"] == 6.0 and not info["torch_cuda"])

        # ---- detect_gpu never raises ----
        def _boom():
            raise RuntimeError("driver/runtime mismatch")
        gpu._from_torch = _boom
        gpu.shutil.which = lambda n: None
        try:
            info = gpu.detect_gpu(refresh=True)
            crashed = False
        except Exception:
            info, crashed = None, True
        check("a raising torch probe does not crash detection", not crashed)

        # ---- caching ----
        gpu._from_torch = lambda: None
        gpu.shutil.which = lambda n: None
        gpu.detect_gpu(refresh=True)
        gpu._from_torch = lambda: ("Later Card", 24.0)
        check("detection is cached (hardware does not change mid-run)",
              gpu.detect_gpu()["has_gpu"] is False)

        # ---- gpu_utilization ----
        stub_smi("17\n")
        check("utilisation parsed", gpu.gpu_utilization() == 17.0)
        stub_smi("17\n95\n")
        check("multi-GPU takes the MAX, not the mean", gpu.gpu_utilization() == 95.0)
        gpu.shutil.which = lambda n: None
        check("no nvidia-smi -> utilisation unknown (None)", gpu.gpu_utilization() is None)
        stub_smi("", returncode=9)
        check("non-zero exit -> unknown, not 0%", gpu.gpu_utilization() is None)
        stub_smi(raises=subprocess.TimeoutExpired("nvidia-smi", 4))
        check("a hung nvidia-smi times out to unknown, never propagates",
              gpu.gpu_utilization() is None)
        stub_smi("not-a-number\n")
        check("garbage output -> unknown, not a crash", gpu.gpu_utilization() is None)

        # ---- gpu_busy: the failure direction is the load-bearing property ----
        stub_smi("95\n")
        busy, why = gpu.gpu_busy(15.0)
        check("GPU over ceiling -> busy with a reason", busy and "gpu 95%" in why)
        stub_smi("3\n")
        check("GPU under ceiling -> not busy", gpu.gpu_busy(15.0)[0] is False)
        gpu.shutil.which = lambda n: None
        check("UNKNOWN utilisation is NOT busy (else every CPU-only node pauses forever)",
              gpu.gpu_busy(15.0)[0] is False)

        # ---- resource_guard integration ----
        for mode, expect in (("idle", 15.0), ("balanced", 50.0), ("generous", 85.0)):
            check(f"{mode}: gpu ceiling {expect:.0f}%",
                  rg.ResourceGuard(mode).gpu_ceiling == expect)
        check("max never yields on GPU", rg.ResourceGuard("max").gpu_ceiling > 100)

        rg.psutil.cpu_percent = lambda interval=None: 1.0
        rg.seconds_since_input = lambda: 9999
        rg.on_battery = lambda: False
        rg.psutil.virtual_memory = lambda: types.SimpleNamespace(available=8 * 1024**3)

        rg._gpu.gpu_busy = lambda ceiling: (True, f"gpu 99% > gpu ceiling {ceiling:.0f}%")
        reasons = rg.ResourceGuard("idle").reasons_to_pause()
        check("an idle CPU with a busy GPU still pauses (the owner is gaming)",
              any("gpu" in r for r in reasons))

        rg._gpu.gpu_busy = lambda ceiling: (False, None)
        check("idle CPU + idle GPU -> no reasons at all",
              rg.ResourceGuard("idle").reasons_to_pause() == [])

        def _explode(ceiling):
            raise OSError("nvidia-smi vanished mid-run")
        rg._gpu.gpu_busy = _explode
        check("a raising GPU probe never pauses the node and never escapes the guard",
              rg.ResourceGuard("idle").reasons_to_pause() == [])

        # back-compat: an old config with only max_cpu_pct must still construct
        g = rg.ResourceGuard("idle", overrides={"cpu_ceiling": 40.0})
        check("legacy max_cpu_pct override still yields a usable gpu_ceiling",
              g.cpu_ceiling == 40.0 and g.gpu_ceiling == 15.0)
    finally:
        gpu.shutil.which, gpu.subprocess.run = real_which, real_run
        gpu._detected = None

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
