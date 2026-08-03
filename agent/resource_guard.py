"""
agent/resource_guard.py — donate as much (or as little) as the owner chooses.

NEURON must never make the machine feel slow to its owner, but *how much* spare
capacity a user donates is their call — phones want "only while charging", a
laptop user might fill their spare headroom while they work, and a server wants to
run flat out. So the guard is a DONATION LEVEL (a ceiling) plus an automatic YIELD
FLOOR (back off the instant the owner needs the CPU). More donation -> more requests
served -> more NRN. It never throttles compute directly (inference is bursty); it
gates whether the node advertises availability, sampled every few seconds.

Donation modes (config `donation_mode`):
  - "idle"     : only truly-spare compute — low CPU, owner away, on AC (the green default)
  - "balanced" : fill spare headroom while you work; yield above ~50% CPU; AC only
  - "generous" : donate aggressively; battery OK; yield only near-max CPU
  - "max"      : server / always-on; never yield on CPU/owner/battery (only low-RAM)

`low memory` (< 500 MB) always pauses, in every mode — that's a safety rail, not a
donation choice. (Thermal throttling is a future rail; psutil temps aren't portable.)

**GPU yielding.** Each mode also carries a `gpu_ceiling`. A machine can be CPU-idle while
its GPU is saturated by a game, a render or a training run — the owner is very much using
that machine, and the CPU meter cannot see it. When `agent/gpu.py` can read utilisation
(nvidia-smi present), exceeding the ceiling pauses the node the same way CPU does. When it
cannot read utilisation — no card, no driver, AMD/Intel, a timeout — the check contributes
nothing, because "cannot tell" must never mean "pause": most machines have no nvidia-smi
and treating them as busy would silently empty the network.

ARM-compatible: pure Python + psutil + ctypes only (no x86-specific code, no pyobjc).
Idle detection: Windows GetLastInputInfo; macOS CGEventSourceSecondsSinceLastEventType
(CoreGraphics via ctypes); Linux xprintidle if present; else a headless server (no
interactive session) is treated as always-idle.
"""
import ctypes
import ctypes.util
import platform
import shutil
import subprocess

import psutil

from agent import gpu as _gpu

_SYSTEM = platform.system()
_IS_WINDOWS = _SYSTEM == "Windows"
_IS_MACOS = _SYSTEM == "Darwin"
MIN_FREE_RAM_BYTES = 500 * 1024 * 1024

# mode -> policy. cpu_ceiling = pause (yield) if system CPU exceeds this; gpu_ceiling = the
# same for GPU utilisation, when it can be read at all; honor_user = yield while the owner is
# actively typing; honor_battery = don't run on battery.
#
# gpu_ceiling tracks cpu_ceiling per mode rather than being stricter. A gaming GPU sits near
# 100% while in use and near 0% when not, so the exact threshold barely matters — what matters
# is that `max` never yields, because that mode exists for machines with no interactive owner.
DONATION_MODES = {
    "idle":     dict(cpu_ceiling=15.0,  gpu_ceiling=15.0,  honor_user=True,  honor_battery=True),
    "balanced": dict(cpu_ceiling=50.0,  gpu_ceiling=50.0,  honor_user=False, honor_battery=True),
    "generous": dict(cpu_ceiling=85.0,  gpu_ceiling=85.0,  honor_user=False, honor_battery=False),
    "max":      dict(cpu_ceiling=100.1, gpu_ceiling=100.1, honor_user=False, honor_battery=False),
}
DEFAULT_MODE = "idle"


def _macos_idle_seconds():
    """Seconds since the last HID (mouse/keyboard) event, via CoreGraphics. Pure ctypes
    against the system framework — no pyobjc dependency, consistent with the rest of this
    module. Needs no special OS permission (unlike some other input-monitoring APIs)."""
    lib = ctypes.util.find_library("CoreGraphics") \
        or "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
    cg = ctypes.CDLL(lib)
    cg.CGEventSourceSecondsSinceLastEventType.restype = ctypes.c_double
    cg.CGEventSourceSecondsSinceLastEventType.argtypes = [ctypes.c_int, ctypes.c_uint32]
    kCGEventSourceStateHIDSystemState = 1
    kCGAnyInputEventType = 0xFFFFFFFF
    return cg.CGEventSourceSecondsSinceLastEventType(kCGEventSourceStateHIDSystemState,
                                                      kCGAnyInputEventType)


def seconds_since_input():
    """Seconds since last mouse/keyboard input; a large number if unknown/headless."""
    if _IS_WINDOWS:
        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            elapsed_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime
            return elapsed_ms / 1000.0
        return 1e9
    if _IS_MACOS:
        # previously fell straight through to the "headless -> always idle" default below,
        # which meant the idle donation mode never yielded to an actively-used Mac — fixed.
        try:
            return _macos_idle_seconds()
        except OSError:
            return 1e9   # framework not loadable (unexpected on real macOS) -> fail safe
    if shutil.which("xprintidle"):
        try:
            out = subprocess.run(["xprintidle"], capture_output=True, text=True, timeout=3)
            if out.returncode == 0:
                return int(out.stdout.strip()) / 1000.0
        except Exception:
            pass
    return 1e9   # headless server: no interactive user -> always idle


def on_battery():
    b = psutil.sensors_battery()
    return bool(b) and not b.power_plugged   # None (desktop/server) -> not on battery


class ResourceGuard:
    def __init__(self, donation_mode=DEFAULT_MODE, idle_threshold_s=60, overrides=None):
        policy = dict(DONATION_MODES.get(donation_mode, DONATION_MODES[DEFAULT_MODE]))
        if overrides:                     # advanced/back-compat explicit tuning
            policy.update(overrides)
        self.donation_mode = donation_mode if donation_mode in DONATION_MODES else DEFAULT_MODE
        self.cpu_ceiling = float(policy["cpu_ceiling"])
        # An old config carrying only `max_cpu_pct` (the pre-donation-mode override) reaches
        # here through `overrides` and sets cpu_ceiling alone, so default the GPU ceiling to
        # the mode's own value rather than assuming the key is present.
        self.gpu_ceiling = float(policy.get("gpu_ceiling", self.cpu_ceiling))
        self.honor_user = bool(policy["honor_user"])
        self.honor_battery = bool(policy["honor_battery"])
        self.idle_threshold_s = float(idle_threshold_s)
        psutil.cpu_percent(interval=None)   # prime the CPU meter

    def reasons_to_pause(self):
        reasons = []
        cpu = psutil.cpu_percent(interval=0.3)
        if cpu > self.cpu_ceiling:                      # yield floor: back off for the owner
            reasons.append(f"cpu {cpu:.0f}% > donation ceiling {self.cpu_ceiling:.0f}%")
        # The owner may be gaming or rendering on a machine whose CPU looks idle. Unreadable
        # utilisation yields no reason at all, never a pause — see the module docstring.
        try:
            busy, why = _gpu.gpu_busy(self.gpu_ceiling)
        except Exception:
            busy, why = False, None     # the guard must never be the thing that breaks
        if busy and why:
            reasons.append(why)
        if self.honor_user:
            idle = seconds_since_input()
            if idle < self.idle_threshold_s:
                reasons.append(f"user active ({idle:.0f}s ago)")
        if self.honor_battery and on_battery():
            reasons.append("on battery")
        avail = psutil.virtual_memory().available
        if avail < MIN_FREE_RAM_BYTES:                  # safety rail, every mode
            reasons.append(f"low RAM ({avail // (1024*1024)} MB)")
        return reasons

    def should_pause(self):
        return len(self.reasons_to_pause()) > 0


if __name__ == "__main__":   # quick manual check
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE
    g = ResourceGuard(mode)
    r = g.reasons_to_pause()
    print(f"mode={g.donation_mode} ceiling={g.cpu_ceiling:.0f}% gpu={g.gpu_ceiling:.0f}% "
          f"honor_user={g.honor_user} honor_battery={g.honor_battery}")
    print("gpu:", _gpu.detect_gpu(), "| util:", _gpu.gpu_utilization())
    print("idle secs:", round(seconds_since_input(), 1), "| on battery:", on_battery())
    print("PAUSE" if r else "ACTIVE", "-", ", ".join(r) if r else "all clear")
