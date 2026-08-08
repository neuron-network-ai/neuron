"""agent/test_slice_range_guard.py — run: python -m agent.test_slice_range_guard

A node must refuse to serve layers its downloaded slice does not contain.

Observed live 2026-08-07 on agent-optinovate-67e4eb: the coordinator had re-split the model and
assigned it 0-27; the disk held only 19-27 from an earlier assignment. The agent started, logged
"serving layers 0-27", passed its own ms/layer benchmark, registered as verified and reported
healthy. Two thirds of the model it claimed to serve had never been downloaded.

Nothing catches that downstream. `load_slice_model` builds a FULL model skeleton from config.json
and fills in whatever the file holds (strict=False), so absent layers stay uninitialized meta
tensors and the forward pass runs on garbage. The node does not crash and its output does not
look wrong -- the user gets fluent nonsense, and the only later signal is a failed
proof-of-compute with no explanation attached.

So the check belongs at load time, on the BYTES, before the node can advertise anything.
"""
import json
import os
import struct
import sys
import tempfile

from agent import node_server

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def make_slice(tmp, name, lo, hi):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    header = {f"model.layers.{i}.self_attn.q_proj.weight":
              {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
              for i in range(lo, hi + 1)}
    header["model.norm.weight"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    blob = json.dumps(header).encode()
    with open(os.path.join(d, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * 16)
    return d


class _Server(node_server.NodeServer):
    """reload() without the torch load — the guard runs before it, which is the point."""

    def __init__(self):
        self.model = None
        self.lo = self.hi = self.n = None
        self._batchers = {}


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-guard-")
    loaded = []
    real_load = node_server.load_slice_model
    node_server.load_slice_model = lambda d: loaded.append(d) or object()

    try:
        # ---- reading the range off the bytes ------------------------------------ #
        d = make_slice(tmp, "s19-27", 19, 27)
        check("reads the layer range from the safetensors header",
              node_server._layers_in_slice(d) == (19, 27))
        check("an unreadable/absent slice returns None rather than raising",
              node_server._layers_in_slice(os.path.join(tmp, "nope")) is None)

        # ---- THE live failure: assigned 0-27, holding 19-27 --------------------- #
        loaded.clear()
        s = _Server()
        try:
            s.reload(d, 0, 27, 28)
            check("a node assigned MORE than it holds refuses to start", False)
        except RuntimeError as e:
            check("a node assigned MORE than it holds refuses to start", True)
            check("...and the error says what it holds and what it was asked for",
                  "19-27" in str(e) and "0-27" in str(e))
            check("...and it says what serving the gap would actually do",
                  "nonsense" in str(e) or "uninitialized" in str(e))
        check("...and nothing was loaded", loaded == [])
        check("...and the server did not start serving a range", s.lo is None)

        # ---- the legitimate cases still load ------------------------------------ #
        loaded.clear()
        s = _Server()
        s.reload(d, 19, 27, 28)
        check("an exact match loads", loaded == [d] and (s.lo, s.hi) == (19, 27))

        loaded.clear()
        big = make_slice(tmp, "s0-27", 0, 27)
        s = _Server()
        s.reload(big, 10, 18, 28)
        check("a slice that CONTAINS the assigned range loads (extra layers are fine)",
              loaded == [big] and (s.lo, s.hi) == (10, 18))

        # ---- partial overlap is still a refusal --------------------------------- #
        loaded.clear()
        try:
            _Server().reload(d, 14, 27, 28)          # holds 19-27, asked for 14-27
            check("a PARTIAL overlap refuses too", False)
        except RuntimeError:
            check("a PARTIAL overlap refuses too", loaded == [])

        # ---- an unreadable header must not block a working node ----------------- #
        loaded.clear()
        empty = os.path.join(tmp, "no-weights")
        os.makedirs(empty, exist_ok=True)
        s = _Server()
        s.reload(empty, 0, 27, 28)
        check("an unreadable header does not block startup — it is not the failure to guard",
              loaded == [empty])
    finally:
        node_server.load_slice_model = real_load

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
