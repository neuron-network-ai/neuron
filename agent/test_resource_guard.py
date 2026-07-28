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
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def reasons(mode, cpu, idle_secs, battery, avail_mb, overrides=None):
    rg.psutil.cpu_percent = lambda interval=None: cpu
    rg.seconds_since_input = lambda: idle_secs
    rg.on_battery = lambda: battery
    rg.psutil.virtual_memory = lambda: types.SimpleNamespace(available=avail_mb * 1024 * 1024)
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
