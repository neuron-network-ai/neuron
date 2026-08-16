"""agent/test_weight_dtype_report.py - run: python -m agent.test_weight_dtype_report

A node now tells the coordinator what it STORES weights at, and the coordinator sizes its slice
from that. Getting it wrong in the optimistic direction OOM-kills a volunteer's machine, so the
path from environment variable to layer count is worth pinning end to end.

The awkward part, and the reason for a file: `agent.py` must not import `common`. The agent is
the lightweight, ARM-compatible half; `common` imports torch at module scope. So the mapping
from NEURON_WEIGHT_DTYPE to a dtype is written TWICE, and a duplicated constant that nothing
compares is a constant that drifts. `common.WEIGHT_DTYPE` gaining a dtype the agent never
reports, or the agent's default moving while common's stays, both produce a network sized
against a precision it is not running -- which is the whole fault [P43] phase 1 corrected, one
layer down.

So: every value is resolved through BOTH implementations and compared. If they disagree this
file fails, whichever side moved.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agent import agent
from coordinator import balancer

ok = fail = 0

# Resolving `common.WEIGHT_DTYPE` needs torch, and it is read at IMPORT time, so each value has
# to be resolved in a fresh interpreter with the variable already set.
CHILD = r'''
import os, sys
sys.path.insert(0, r"{root}")
import common
print("DTYPE" + str(common.WEIGHT_DTYPE))
'''.format(root=ROOT)


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def common_dtype(value):
    """What `common.py` resolves NEURON_WEIGHT_DTYPE to, or None if it refuses to load."""
    env = dict(os.environ)
    if value is None:
        env.pop("NEURON_WEIGHT_DTYPE", None)
    else:
        env["NEURON_WEIGHT_DTYPE"] = value
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env)
    for line in p.stdout.splitlines():
        if line.startswith("DTYPE"):
            return line[5:].replace("torch.", "")
    return None


def _explains(value):
    """Does the runtime's refusal name the variable and the values it will accept?"""
    env = dict(os.environ, NEURON_WEIGHT_DTYPE=value)
    p = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, env=env)
    msg = (p.stdout + p.stderr).lower()
    return ("neuron_weight_dtype" in msg and "fp16" in msg and "traceback" not in msg)


def agent_dtype(value):
    old = os.environ.get("NEURON_WEIGHT_DTYPE")
    try:
        if value is None:
            os.environ.pop("NEURON_WEIGHT_DTYPE", None)
        else:
            os.environ["NEURON_WEIGHT_DTYPE"] = value
        return agent.weight_dtype()
    finally:
        os.environ.pop("NEURON_WEIGHT_DTYPE", None)
        if old is not None:
            os.environ["NEURON_WEIGHT_DTYPE"] = old


# agent's short name -> the torch dtype common resolves it to
EQUIVALENT = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def main():
    # ---------- 1. the two implementations agree, value by value ---------------
    for name in agent.WEIGHT_DTYPES:
        reported, actual = agent_dtype(name), common_dtype(name)
        check(f"NEURON_WEIGHT_DTYPE={name}: agent reports {reported}, common loads {actual}",
              reported == name and actual == EQUIVALENT[name],
              "the agent's report and the runtime's storage have diverged")

    check("unset resolves to the same default on both sides",
          agent_dtype(None) == agent.DEFAULT_WEIGHT_DTYPE
          and common_dtype(None) == EQUIVALENT[agent.DEFAULT_WEIGHT_DTYPE])
    check("the default really is fp32 (the premise every sizing figure rests on)",
          agent.DEFAULT_WEIGHT_DTYPE == "fp32")
    check("neither side knows a dtype the other does not",
          set(agent.WEIGHT_DTYPES) == set(EQUIVALENT),
          "add it to WEIGHT_DTYPES and EQUIVALENT together, or the agent will report a "
          "precision the coordinator cannot size, or size one the node does not run")

    # ---------- 2. a typo degrades to the pessimistic answer, never optimistic --
    for junk in ("int4", "", "  ", "float8_e4m3", "fp 16"):
        check(f"{junk!r} reports the default rather than a precision it is not running",
              agent_dtype(junk) == agent.DEFAULT_WEIGHT_DTYPE)

    # Case and surrounding whitespace are a VALUE, not a typo, and both sides must agree they
    # are. They did not: `common` lowercased without stripping, so `fp16 ` raised
    # `KeyError: 'fp16 '` at import -- before the node server's logging was up, giving the
    # operator a traceback naming a dict literal -- while the agent stripped it and told the
    # coordinator `fp16`. The machine was then sized for half the footprint of a process that
    # was not running. Found by this file.
    for spelled in ("  FP16  ", "BF16", "Fp32", "fp16\t"):
        want = spelled.strip().lower()
        check(f"{spelled!r} means {want} to the agent AND to the runtime",
              agent_dtype(spelled) == want and common_dtype(spelled) == EQUIVALENT[want],
              f"agent {agent_dtype(spelled)}, common {common_dtype(spelled)}")

    check("an unsupported dtype stops the runtime with a sentence, not a KeyError",
          common_dtype("int4") is None and _explains("int4"),
          "the node server would otherwise die at import with `KeyError: 'int4'`")

    # ---------- 3. what the coordinator does with it ---------------------------
    # The point of reporting it at all. Everything the agent can report must land on a byte
    # count the balancer believes, or the field is decorative.
    for name in agent.WEIGHT_DTYPES:
        check(f"the coordinator believes {name} and sizes from it",
              balancer.sane_weight_dtype(name) == name
              and balancer.weight_bytes_for({"weight_dtype": name})
              == (4.0 if name == "fp32" else 2.0))
    check("...and disbelieves a dtype no build of this software can store at",
          balancer.sane_weight_dtype("int8") is None
          and balancer.weight_bytes_for({"weight_dtype": "int8"})
          == balancer.ASSUMED_WEIGHT_BYTES,
          "int8 would QUARTER the divisor and quadruple the layers assigned; registration "
          "takes no credential under open join")
    check("an agent too old to report is sized pessimistically, not optimistically",
          balancer.weight_bytes_for({}) == balancer.ASSUMED_WEIGHT_BYTES == 4.0)

    # ---------- 4. it is actually in the registration body ---------------------
    src = open(os.path.join(HERE, "agent.py"), encoding="utf-8").read()
    check("the registration payload carries the field",
          '"weight_dtype": weight_dtype(),' in src)
    check("agent.py still does not import common (that is why this file exists)",
          "\nimport common" not in src and "\nfrom common import" not in src)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
