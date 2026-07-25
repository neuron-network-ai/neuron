"""
agent/resource_guard.py — only ever use TRULY idle capacity.

NEURON must never compete with the machine's owner. The guard pauses the node
when the machine is in use and resumes automatically when it clears. Checked
every few seconds. Pause if ANY of:
  - system CPU busy        (> max_cpu_pct)
  - user active            (last input < idle_threshold seconds ago)
  - on battery             (laptop unplugged)
  - low memory             (< 500 MB available)

ARM-compatible: pure Python + psutil + ctypes only (no x86-specific code).
Idle detection: Windows GetLastInputInfo; Linux xprintidle if present, else a
headless server (no interactive session) is treated as always-idle.
"""
import ctypes
import platform
import shutil
import subprocess

import psutil

_IS_WINDOWS = platform.system() == "Windows"
MIN_FREE_RAM_BYTES = 500 * 1024 * 1024


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
    def __init__(self, max_cpu_pct=2.0, idle_threshold_s=60):
        self.max_cpu_pct = float(max_cpu_pct)
        self.idle_threshold_s = float(idle_threshold_s)
        psutil.cpu_percent(interval=None)   # prime the CPU meter

    def reasons_to_pause(self):
        reasons = []
        cpu = psutil.cpu_percent(interval=0.3)
        if cpu > self.max_cpu_pct:
            reasons.append(f"cpu {cpu:.0f}% > {self.max_cpu_pct:.0f}%")
        idle = seconds_since_input()
        if idle < self.idle_threshold_s:
            reasons.append(f"user active ({idle:.0f}s ago)")
        if on_battery():
            reasons.append("on battery")
        avail = psutil.virtual_memory().available
        if avail < MIN_FREE_RAM_BYTES:
            reasons.append(f"low RAM ({avail // (1024*1024)} MB)")
        return reasons

    def should_pause(self):
        return len(self.reasons_to_pause()) > 0


if __name__ == "__main__":   # quick manual check
    g = ResourceGuard()
    r = g.reasons_to_pause()
    print("idle secs:", round(seconds_since_input(), 1), "| on battery:", on_battery())
    print("PAUSE" if r else "ACTIVE", "-", ", ".join(r) if r else "all clear")
