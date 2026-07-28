"""macOS LaunchAgent support tests — run: python -m agent.test_install_macos

This dev machine is Windows, so these tests call the macOS-specific functions directly
(they don't branch on IS_MACOS internally) and monkeypatch subprocess.run/Popen so no real
`launchctl` binary is needed, plus a temp file in place of the real
~/Library/LaunchAgents/ path. The uninstall.py dispatch functions (_stop_agent,
_remove_startup) DO branch on IS_MACOS/IS_WINDOWS internally, so those flags are flipped
for the duration of the relevant checks.
"""
import os
import tempfile

import agent.install as install
import agent.uninstall as uninstall

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        class R:
            returncode = 0
        return R()

    tmp_plist = os.path.join(tempfile.mkdtemp(prefix="neuron_launchd_"), "com.neuron.agent.plist")

    # ---- install.py: add_to_startup_macos() writes a correct plist + calls launchctl ---- #
    real_path, real_run = install.LAUNCHD_PLIST_PATH, install.subprocess.run
    install.LAUNCHD_PLIST_PATH, install.subprocess.run = tmp_plist, fake_run
    try:
        install.add_to_startup_macos()
        check("plist file was written", os.path.exists(tmp_plist))
        content = open(tmp_plist).read()
        check("plist has the correct Label", "<string>com.neuron.agent</string>" in content)
        check("plist points at agent.py", "agent.py" in content)
        check("plist has RunAtLoad true", "<key>RunAtLoad</key><true/>" in content)
        check("plist has KeepAlive true", "<key>KeepAlive</key><true/>" in content)
        check("launchctl unload called first (clears a stale prior load)",
              calls[0] == ["launchctl", "unload", "-w", tmp_plist])
        check("launchctl load -w called with the plist path",
              calls[1] == ["launchctl", "load", "-w", tmp_plist])
    finally:
        install.LAUNCHD_PLIST_PATH, install.subprocess.run = real_path, real_run

    # ---- start_background() on macOS spawns directly (works even without a LaunchAgent,
    # e.g. --no-startup) rather than assuming one was already registered ---- #
    popen_calls = []
    real_windows, real_macos, real_popen = install.IS_WINDOWS, install.IS_MACOS, install.subprocess.Popen
    install.IS_WINDOWS, install.IS_MACOS = False, True
    install.subprocess.Popen = lambda *a, **k: popen_calls.append((a, k))
    try:
        install.start_background()
        check("macOS start_background spawns agent.py directly",
              popen_calls and popen_calls[0][0][0][1].endswith("agent.py"))
        check("macOS start_background detaches (start_new_session)",
              popen_calls[0][1].get("start_new_session") is True)
    finally:
        install.IS_WINDOWS, install.IS_MACOS, install.subprocess.Popen = \
            real_windows, real_macos, real_popen

    # ---- uninstall.py: _stop_agent() / _remove_startup() macOS branches ---- #
    real_uwin, real_umac = uninstall.IS_WINDOWS, uninstall.IS_MACOS
    real_uplist, real_urun = uninstall.LAUNCHD_PLIST_PATH, uninstall.subprocess.run
    uninstall.IS_WINDOWS, uninstall.IS_MACOS = False, True
    uninstall.LAUNCHD_PLIST_PATH = tmp_plist
    ucalls = []
    uninstall.subprocess.run = lambda cmd, **kw: (ucalls.append(cmd), fake_run(cmd, **kw))[1]
    try:
        uninstall._stop_agent()
        check("uninstall stop_agent calls launchctl unload",
              ucalls and ucalls[0] == ["launchctl", "unload", "-w", tmp_plist])

        check("plist still exists before removal (sanity)", os.path.exists(tmp_plist))
        uninstall._remove_startup()
        check("uninstall removes the plist file", not os.path.exists(tmp_plist))

        uninstall._remove_startup()  # must not raise if already removed
        check("removing a non-existent plist is a safe no-op", not os.path.exists(tmp_plist))
    finally:
        uninstall.IS_WINDOWS, uninstall.IS_MACOS = real_uwin, real_umac
        uninstall.LAUNCHD_PLIST_PATH, uninstall.subprocess.run = real_uplist, real_urun

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
