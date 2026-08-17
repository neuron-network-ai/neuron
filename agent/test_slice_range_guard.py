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
    return make_slice_layers(tmp, name, range(lo, hi + 1))


def make_slice_layers(tmp, name, indices):
    """A slice holding an ARBITRARY set of layers, so a hole in the middle can be built."""
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    header = {f"model.layers.{i}.self_attn.q_proj.weight":
              {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
              for i in indices}
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

        # ---- [P42]: a HOLE IN THE MIDDLE, which the old bounds check waved through #
        # A slice holding 0-9 and 19-27 has min 0 and max 27, so `held[0] <= start and
        # end <= held[1]` passes for an assignment of 0-27 -- while layers 10-18 are absent.
        # Same ending as the 2026-08-07 incident (uninitialized meta tensors mid-forward),
        # reached by a path the extremes cannot see.
        loaded.clear()
        holed = make_slice_layers(tmp, "s-hole", list(range(0, 10)) + list(range(19, 28)))
        check("the extremes alone would not catch it",
              node_server._layers_in_slice(holed) == (0, 27))
        try:
            _Server().reload(holed, 0, 27, 28)
            check("a slice with a GAP in the middle refuses to start", False)
        except RuntimeError as e:
            check("a slice with a GAP in the middle refuses to start", loaded == [])
            check("...and the error names the missing layers", "10-18" in str(e))

        # a range that avoids the hole is still fine
        loaded.clear()
        s = _Server()
        s.reload(holed, 19, 27, 28)
        check("...but an assignment inside the held layers still loads",
              loaded == [holed] and (s.lo, s.hi) == (19, 27))

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

    # [P42] level 2. Defined BELOW main(); the call happens at runtime so the order
    # is fine, and keeping them here means one summary and one exit code.
    for fn in (test_level2_passes_when_the_assigned_range_is_fully_materialized,
               test_level2_catches_a_layer_that_never_arrived,
               test_level2_ignores_meta_layers_OUTSIDE_the_assigned_range,
               test_level2_does_not_block_an_architecture_it_does_not_recognise,
               test_level2_survives_a_range_wider_than_the_model,
               test_reload_refuses_and_says_what_to_do):
        fn()

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0




# --------------------------------------------------------------------------- #
# [P42] LEVEL 2: a layer the header PROMISED but the file did not deliver
#
# The range guard above reads the safetensors header and asks "is layer N named here". It
# cannot ask "did layer N actually arrive" -- `load_slice_model` fills a full skeleton with
# strict=False, so a truncated download, or a tensor mapped to a shard it is absent from,
# leaves the layer listed and its weights on the meta device. Nothing raises. The node serves
# uninitialized weights and returns fluent nonsense: the same ending as 2026-08-07, reached by
# the one path the header check does not inspect.
#
# Scoped to the ASSIGNED range, which is the whole difficulty -- most of a sliced node's model
# is legitimately meta, so the global version of this assertion would refuse every node.
# --------------------------------------------------------------------------- #
class _P:
    def __init__(self, meta):
        self.is_meta = meta


class _Layer:
    def __init__(self, meta):
        self._meta = meta

    def named_parameters(self):
        return [("self_attn.q_proj.weight", _P(self._meta)),
                ("mlp.down_proj.weight", _P(self._meta))]


class _Model:
    """A skeleton shaped like the real one: `model.model.layers`, some materialized, some not."""
    def __init__(self, n, materialized):
        inner = type("Inner", (), {})()
        inner.layers = [_Layer(i not in materialized) for i in range(n)]
        self.model = inner


def test_level2_passes_when_the_assigned_range_is_fully_materialized():
    m = _Model(28, materialized=set(range(10, 19)))
    check("a slice whose assigned layers all arrived loads",
          node_server.unmaterialized_layers(m, 10, 18) == [])


def test_level2_catches_a_layer_that_never_arrived():
    # 10-18 assigned, but 14 came back empty -- the case the header check waves through,
    # because the header NAMED layer 14.
    m = _Model(28, materialized=set(range(10, 19)) - {14})
    bad = node_server.unmaterialized_layers(m, 10, 18)
    check("a layer present in the header but unmaterialized is caught", len(bad) == 2)
    check("...and it is named, so the operator knows which one",
          all(b.startswith("layers.14.") for b in bad))


def test_level2_ignores_meta_layers_OUTSIDE_the_assigned_range():
    """The reason this cannot be a global assertion. `load_slice_model` builds the FULL model
    and this node holds a slice of it, so layers 0-9 and 19-27 being meta is correct."""
    m = _Model(28, materialized=set(range(10, 19)))
    check("layers this node does not serve are allowed to be meta",
          node_server.unmaterialized_layers(m, 10, 18) == [])
    check("...and the same model FAILS if it is assigned a range it did not download",
          node_server.unmaterialized_layers(m, 0, 27) != [])


def test_level2_does_not_block_an_architecture_it_does_not_recognise():
    """Unknown shape is 'do not block', the same answer an unreadable header gets: refusing to
    start on a check meant to catch a mismatch would take working nodes down."""
    check("a model with no .model.layers returns None rather than raising",
          node_server.unmaterialized_layers(object(), 0, 27) is None)


def test_level2_survives_a_range_wider_than_the_model():
    m = _Model(28, materialized=set(range(0, 28)))
    check("an end past the last layer does not IndexError",
          node_server.unmaterialized_layers(m, 20, 99) == [])


def test_reload_refuses_and_says_what_to_do():
    """The message has to name the file to delete: this is a stranger's PC, and the remedy is
    a re-download rather than anything they could debug."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_server.py"),
               encoding="utf-8").read()
    # Anchored on the CALL, not the name: `def unmaterialized_layers(model, layer_start` also
    # matches, so a looser anchor greps the function's docstring and grades the comment.
    call = "unfilled = unmaterialized_layers(model, layer_start, layer_end)"
    check("reload() raises on unmaterialized assigned layers", call in src)
    check("...before the pointer swap, so a failing node serves nothing rather than garbage",
          src.index(call) < src.index("self.model = model", src.index(call)))
    check("...and the refusal tells the operator to delete the slice",
          "re-download" in src.split(call)[1][:900])


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
