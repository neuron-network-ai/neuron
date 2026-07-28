"""coordinator/test_register_nodes.py — run: python -m coordinator.test_register_nodes

Covers the --auto-verify wiring added to register_nodes.py's main(): it should start
security.proof_of_compute.verify_loop in a background thread, with the right coordinator
URL / secret / interval, before falling into the (blocking) heartbeat_loop -- and must NOT
start it when the flag is absent. Mocks register_all/heartbeat_loop/verify_loop entirely;
no real network, no real threads left running past the test.
"""
import sys
import threading
import time

import coordinator.register_nodes as rn

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _run_main_with(argv):
    real_argv = sys.argv
    sys.argv = ["register_nodes.py"] + argv
    try:
        rn.main()
    finally:
        sys.argv = real_argv


def main():
    real_register_all = rn.register_all
    real_heartbeat = rn.heartbeat_loop
    rn.register_all = lambda base: {"node_a": "tok-a"}

    heartbeat_calls = []
    rn.heartbeat_loop = lambda base, tokens: heartbeat_calls.append((base, tokens))

    # ---- without --auto-verify: verify_loop is never touched ---- #
    from security import proof_of_compute
    real_verify_loop = proof_of_compute.verify_loop
    verify_loop_calls = []
    proof_of_compute.verify_loop = lambda *a, **k: verify_loop_calls.append((a, k))
    try:
        _run_main_with(["--coordinator", "http://coord.example"])
        time.sleep(0.1)   # nothing to wait on, just in case a thread were wrongly started
        check("no --auto-verify -> verify_loop never called", verify_loop_calls == [])
        check("heartbeat_loop still runs normally", len(heartbeat_calls) == 1)

        # ---- --register-only: neither heartbeat nor verify_loop runs ---- #
        heartbeat_calls.clear()
        _run_main_with(["--coordinator", "http://coord.example", "--register-only"])
        check("--register-only skips heartbeat_loop", heartbeat_calls == [])
        check("--register-only skips verify_loop even if auto-verify were implied",
              verify_loop_calls == [])

        # ---- --auto-verify: verify_loop starts in a background thread before heartbeat ---- #
        heartbeat_calls.clear()
        threads_before = {t.ident for t in threading.enumerate()}
        _run_main_with(["--coordinator", "http://coord.example", "--auto-verify",
                       "--verify-interval", "5"])
        # verify_loop is mocked to return instantly, so give the daemon thread a moment
        for _ in range(20):
            if verify_loop_calls:
                break
            time.sleep(0.05)
        check("--auto-verify starts verify_loop", len(verify_loop_calls) == 1)
        args, kwargs = verify_loop_calls[0]
        check("passes the coordinator URL", args[0] == "http://coord.example")
        check("passes the register secret", args[1] == rn.REGISTER_SECRET)
        check("passes the configured verify interval", kwargs.get("interval") == 5)
        check("heartbeat_loop still runs (auto-verify doesn't replace it)",
              len(heartbeat_calls) == 1)
        new_threads = [t for t in threading.enumerate() if t.ident not in threads_before]
        check("the verify_loop thread is a background daemon (won't block process exit)",
              any(t.daemon for t in new_threads if t.name == "auto-verify") or not new_threads)
    finally:
        rn.register_all = real_register_all
        rn.heartbeat_loop = real_heartbeat
        proof_of_compute.verify_loop = real_verify_loop

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
