"""Donation-mode guard tests — run: python -m agent.test_resource_guard

The resource guard is now a DONATION LEVEL (ceiling) + automatic YIELD FLOOR. Verifies each
mode's policy and that reasons_to_pause() yields correctly under simulated CPU/idle/battery/RAM,
by monkeypatching the sensor helpers (no real hardware state needed).
"""
import types

import agent.resource_guard as rg

_real_seconds_since_input = rg.seconds_since_input  # reasons() below permanently overwrites
                                                     # rg.seconds_since_input with a lambda,
                                                     # so later tests need this saved reference
                                                     # to exercise the REAL dispatch logic.
# Same trap, one level deeper: `rg._gpu` IS the `agent.gpu` module, so stubbing
# `rg._gpu.gpu_busy` mutates it for every importer. Saved here so the cases that need the real
# implementation can put it back -- without this they silently assert against the stub and pass
# or fail for reasons that have nothing to do with the code under test.
_real_gpu_busy = rg._gpu.gpu_busy
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def reasons(mode, cpu, idle_secs, battery, avail_mb, overrides=None, gpu=None):
    """`gpu` is (busy, why) as `agent.gpu.gpu_busy` returns it; None = card idle/unreadable.

    Stubbed rather than left alone deliberately. Without this the guard shelled out to the REAL
    nvidia-smi on every one of these cases -- slow, and dependent on whatever the developer's
    machine happened to be doing -- and the `gpu_ceiling` branch had no coverage at all, on a
    machine where `gpu_busy` can only ever answer "no card".
    """
    rg.psutil.cpu_percent = lambda interval=None: cpu
    rg.seconds_since_input = lambda: idle_secs
    rg.on_battery = lambda: battery
    rg.psutil.virtual_memory = lambda: types.SimpleNamespace(available=avail_mb * 1024 * 1024)
    rg._gpu.gpu_busy = lambda ceiling: (gpu if gpu is not None else (False, None))
    return rg.ResourceGuard(mode, overrides=overrides).reasons_to_pause()


def main():
    # ---- mode -> policy mapping ----
    g = rg.ResourceGuard("balanced")
    check("balanced: ceiling 50, ignore user, AC-only",
          g.cpu_ceiling == 50 and not g.honor_user and g.honor_battery)
    g = rg.ResourceGuard("max")
    check("max: never yields to user/battery",
          not g.honor_user and not g.honor_battery and g.cpu_ceiling > 100)
    check("unknown mode falls back to idle", rg.ResourceGuard("bogus").donation_mode == "idle")

    # ---- idle = green default: strict ----
    r = reasons("idle", cpu=30, idle_secs=1, battery=True, avail_mb=8000)
    check("idle pauses on cpu+user+battery",
          any("cpu" in x for x in r) and any("user" in x for x in r) and any("battery" in x for x in r))
    r = reasons("idle", cpu=5, idle_secs=1000, battery=False, avail_mb=8000)
    check("idle runs when truly idle (low cpu, away, AC)", r == [])

    # ---- balanced = fill headroom while you work, AC only, yield above 50% ----
    check("balanced runs while you work (AC, cpu<50)",
          reasons("balanced", cpu=30, idle_secs=1, battery=False, avail_mb=8000) == [])
    check("balanced yields above 50% cpu",
          any("ceiling" in x for x in reasons("balanced", cpu=60, idle_secs=999, battery=False, avail_mb=8000)))
    check("balanced still AC-only",
          any("battery" in x for x in reasons("balanced", cpu=10, idle_secs=999, battery=True, avail_mb=8000)))

    # ---- generous = ignore user + battery, yield only near max ----
    check("generous ignores user+battery",
          reasons("generous", cpu=30, idle_secs=1, battery=True, avail_mb=8000) == [])
    check("generous yields near max cpu",
          any("ceiling" in x for x in reasons("generous", cpu=90, idle_secs=999, battery=False, avail_mb=8000)))

    # ---- max = server / always-on ----
    check("max never yields on cpu/user/battery",
          reasons("max", cpu=99, idle_secs=1, battery=True, avail_mb=8000) == [])

    # ---- low RAM is a safety rail in EVERY mode ----
    for m in ("idle", "balanced", "generous", "max"):
        check(f"{m}: low RAM always pauses",
              any("RAM" in x for x in reasons(m, cpu=1, idle_secs=999, battery=False, avail_mb=100)))

    # ---- the GPU yield floor: a machine can be CPU-idle while its card is flat out ----
    # The owner is gaming or rendering and the CPU meter cannot see it. Until gpu_busy was
    # stubbed this whole branch was uncovered: the real probe on this machine can only ever
    # answer "no card", so every case above took the False path and nothing tested the True one.
    BUSY = (True, "gpu 95% > donation ceiling 50%")
    check("a busy GPU pauses a node whose CPU looks idle",
          any("gpu" in x for x in reasons("balanced", cpu=5, idle_secs=999, battery=False,
                                          avail_mb=8000, gpu=BUSY)))
    check("...and the reason reaches the caller verbatim, so a log says why",
          BUSY[1] in reasons("balanced", cpu=5, idle_secs=999, battery=False,
                             avail_mb=8000, gpu=BUSY))
    check("an idle GPU adds no reason",
          reasons("balanced", cpu=5, idle_secs=999, battery=False, avail_mb=8000,
                  gpu=(False, None)) == [])
    # Unreadable utilisation must yield NO reason rather than a pause -- a node that stops
    # because a probe failed is a node that earns nothing for a reason nobody can see.
    check("a busy verdict with no reason string is not turned into a pause",
          reasons("balanced", cpu=5, idle_secs=999, battery=False, avail_mb=8000,
                  gpu=(True, None)) == [])

    # The guard must never be the thing that breaks the node: a probe that RAISES is swallowed.
    def _boom_gpu(ceiling):
        raise RuntimeError("nvidia-smi exploded")
    rg.psutil.cpu_percent = lambda interval=None: 5
    rg.seconds_since_input = lambda: 999
    rg.on_battery = lambda: False
    rg.psutil.virtual_memory = lambda: types.SimpleNamespace(available=8000 * 1024 * 1024)
    rg._gpu.gpu_busy = _boom_gpu
    check("a GPU probe that raises does not pause the node or crash the guard",
          rg.ResourceGuard("balanced").reasons_to_pause() == [])

    # The mode's ceiling is what decides, and it is `gpu_busy` that applies it -- the guard
    # only passes it through. So the property to pin here is that each mode hands over ITS OWN
    # gpu_ceiling; asserting a verdict instead would just be asserting the stub.
    seen = []
    rg._gpu.gpu_busy = lambda ceiling: seen.append(ceiling) or (False, None)
    for m in ("idle", "balanced", "generous", "max"):
        rg.ResourceGuard(m).reasons_to_pause()
    check("each mode passes its own gpu_ceiling to the probe",
          seen == [rg.DONATION_MODES[m]["gpu_ceiling"] for m in ("idle", "balanced", "generous", "max")])
    # And `max` hands over a ceiling no real card can exceed. Checked against the REAL gpu_busy
    # with utilisation pinned at 100%, not against the stub above.
    import agent.gpu as _real_gpu
    _saved_util = _real_gpu.gpu_utilization
    _real_gpu.gpu_busy = _real_gpu_busy          # undo the stubs above; test the real one
    try:
        _real_gpu.gpu_utilization = lambda: 100.0
        check("a card at 100% pauses a `balanced` node",
              _real_gpu.gpu_busy(rg.DONATION_MODES["balanced"]["gpu_ceiling"])[0] is True)
        check("...but never a `max` one: its ceiling is above anything a card can report",
              _real_gpu.gpu_busy(rg.DONATION_MODES["max"]["gpu_ceiling"]) == (False, None))
        _real_gpu.gpu_utilization = lambda: None
        check("unreadable utilisation is not busy, in every mode",
              all(_real_gpu.gpu_busy(rg.DONATION_MODES[m]["gpu_ceiling"]) == (False, None)
                  for m in rg.DONATION_MODES))
    finally:
        _real_gpu.gpu_utilization = _saved_util

    # ---- back-compat: explicit ceiling override (old max_cpu_pct) ----
    check("override raises the ceiling", rg.ResourceGuard("idle", overrides={"cpu_ceiling": 90.0}).cpu_ceiling == 90)

    # ---- macOS idle detection dispatch (this dev machine is Windows, so exercise the
    # branch by flipping the platform flags rather than actually being on a Mac) ----
    real_windows, real_macos, real_macos_fn = rg._IS_WINDOWS, rg._IS_MACOS, rg._macos_idle_seconds
    try:
        rg._IS_WINDOWS, rg._IS_MACOS = False, True
        rg._macos_idle_seconds = lambda: 42.0
        check("macOS: dispatches to CoreGraphics idle seconds, not the headless fallback",
              _real_seconds_since_input() == 42.0)

        def _boom():
            raise OSError("framework not loadable")
        rg._macos_idle_seconds = _boom
        check("macOS: framework load failure fails safe to always-idle (1e9), not a crash",
              _real_seconds_since_input() == 1e9)
    finally:
        rg._IS_WINDOWS, rg._IS_MACOS, rg._macos_idle_seconds = real_windows, real_macos, real_macos_fn

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
