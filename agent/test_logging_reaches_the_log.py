"""agent/test_logging_reaches_the_log.py — run: python -m agent.test_logging_reaches_the_log

The tray tells a user "Chat UI unavailable — see agent.log". That instruction has to be true.

It was not. `_setup_logging` attached its handlers to `neuron.agent`, but the two things that
actually fail while the Chat UI starts up log under `neuron.engine.local_gguf` (fetching
weights) and `neuron.driver` (loading the model). Neither is a child of `neuron.agent`, so
their records propagated to a root logger with no handlers and vanished — and in windowed tray
mode there is no console to catch them either. The user was sent to a file that could not
contain the answer.

So these tests are about one property: **if a NEURON component logs a failure, it lands in
agent.log.** Not "the logger is configured" -- that was true before and useless.
"""
import importlib
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def _fresh_logging(tmp):
    """Reset the `neuron` logger tree and point agent.LOG_PATH at a temp file."""
    parent = logging.getLogger("neuron")
    for h in list(parent.handlers):
        parent.removeHandler(h)
        h.close()
    from agent import agent as agent_mod
    path = os.path.join(tmp, "agent.log")
    agent_mod.LOG_PATH = path
    agent_mod._setup_logging("INFO")
    return agent_mod, path


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    from agent import agent as agent_mod       # noqa: F401  (import once up front)
    tmp = tempfile.mkdtemp(prefix="neuron-log-")

    print("\n-- every NEURON component reaches the file")
    mod, path = _fresh_logging(tmp)
    # The exact loggers used by the code paths behind "Chat UI unavailable".
    for name in ("neuron.agent", "neuron.agent.local_chat", "neuron.driver",
                 "neuron.engine.local_gguf"):
        logging.getLogger(name).error("marker-from-%s", name)
    body = read(path)
    for name in ("neuron.agent", "neuron.agent.local_chat", "neuron.driver",
                 "neuron.engine.local_gguf"):
        check(f"{name} lands in agent.log", f"marker-from-{name}" in body,
              f"log body was:\n{body}")

    print("\n-- the record says which component spoke")
    check("the logger name is in the line", "neuron.driver" in body, body)

    print("\n-- non-ASCII does not corrupt the file")
    logging.getLogger("neuron.agent").info("heartbeat ok · active — em dash")
    body = read(path)
    check("a '·' round-trips instead of becoming mojibake", "· active — em dash" in body, body)

    print("\n-- calling it twice does not double every line")
    before = read(path).count("marker-from-neuron.driver")
    mod._setup_logging("INFO")            # tray mode and main() both call this
    logging.getLogger("neuron.driver").error("marker-from-neuron.driver")
    after = read(path).count("marker-from-neuron.driver")
    check("a second setup adds one line, not two", after == before + 1,
          f"{before} -> {after}")

    print("\n-- an unrelated logger is not swept up")
    logging.getLogger("urllib3.connectionpool").warning("noisy-third-party")
    check("third-party logs stay out of agent.log", "noisy-third-party" not in read(path))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
