"""test_device_path.py — run: python test_device_path.py

A TRIPWIRE, not a feature test. It pins the device invariants that hold today so that the day
someone makes NEURON compute on a GPU, the pieces that would silently break announce
themselves first.

The situation it guards against actually happened. v0.18.0 shipped "GPU support" that had never
executed once: `common.py` gained a device resolver, the coordinator started sizing volunteers
by VRAM, the release notes announced it — and the loader every node uses never moved a single
weight, so every GPU machine computed in system RAM while being handed a GPU's worth of layers.
See PROBLEMS.md [P31].

Three invariants, each true on a CPU-only machine today AND still true on CUDA afterwards:

  1. `load_slice_model` leaves weights on the CPU. This is Phase 4b, deliberately deferred:
     moving them is one line, and doing it before 2 and 3 are verified on real hardware turns
     silent-CPU into crash-on-first-token.
  2. `batching`'s auxiliary tensors follow the activations rather than being pinned to CPU.
  3. The wire boundary returns CPU tensors, whatever device the maths ran on.

2 and 3 are the ones that make 1 safe. They are written and pass here, but "passes on CPU" is
not "works on a GPU" — that distinction is the entire subject of [P31].
"""
import io
import sys
import tokenize

import torch

import batching
import common
import wire_codec

ok = fail = 0


def code_only(src):
    """`src` with comments and string literals removed.

    Every source-scanning check below has to run on code, not prose. Scanning raw text finds
    the words in the comment that EXPLAINS the invariant and reports the invariant broken —
    which is exactly what the first version of this file did, twice, on comments written in
    the same change. A grep that matches its own documentation is not a test.
    """
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src                          # unparseable -> scan raw rather than pass blindly
    return " ".join(out)


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


# --------------------------------------------------------------------------- #
# 1. the loader does not move weights  (Phase 4b — deferred on purpose)
# --------------------------------------------------------------------------- #
def test_loader_leaves_weights_on_cpu():
    """Read the source rather than load a real slice: this must run in the suite on any
    machine, with no model download and no 3 GB of RAM."""
    src = open("slice_downloader.py", encoding="utf-8").read()
    body = src[src.index("def load_slice_model"):]
    body = body[:body.index("\n# ---")] if "\n# ---" in body else body
    body = code_only(body).replace(" ", "")

    moved = [tok for tok in ("move_model_to_device", ".to(common.DEVICE)", ".to(DEVICE)",
                             ".cuda()") if tok in body]
    check("load_slice_model does not move the shard to a device",
          not moved,
          "found " + ", ".join(moved) + " — if you moved the loader, Phase 4a must ALREADY be "
          "verified on real GPU hardware: the wire codec, the batched stages and the legacy "
          "torch.save framing all assumed CPU tensors, and this network will crash on the "
          "first token rather than run slowly. See PROBLEMS.md [P31] and the comment at the "
          "end of load_slice_model.")

    # If someone DOES move it, the coordinator must stop sizing by RAM in the same change --
    # these two facts are a pair, and splitting them is how [P31] happened in reverse.
    from coordinator import balancer
    check("balancer.GPU_EXECUTION agrees with the loader (both off)",
          balancer.GPU_EXECUTION is False or bool(moved),
          "VRAM is being counted as capacity while the loader leaves weights in system RAM — "
          "this is exactly the v0.18.0 OOM. See coordinator/balancer.GPU_EXECUTION.")


# --------------------------------------------------------------------------- #
# 2. batching follows the activations
# --------------------------------------------------------------------------- #
class _RecordingLayer(torch.nn.Module):
    """Captures the devices of everything run_layers_batched hands a layer."""
    seen = None

    def forward(self, hidden, attention_mask=None, position_ids=None, past_key_value=None,
                use_cache=True, cache_position=None):
        _RecordingLayer.seen = {
            "hidden": hidden.device,
            "attention_mask": None if attention_mask is None else attention_mask.device,
            "position_ids": position_ids.device,
            "cache_position": cache_position.device,
        }
        return (hidden,)


def test_batching_aux_tensors_follow_hidden():
    cache = batching.BatchedCache() if hasattr(batching, "BatchedCache") else None
    if cache is None:                       # name drift -- find it rather than guess
        cache = next(v for k, v in vars(batching).items()
                     if isinstance(v, type) and "Cache" in k and k != "SplitCache")()
    cache.lengths = [3, 5]
    hidden = torch.zeros(2, 1, 8)

    _RecordingLayer.seen = None
    batching.run_layers_batched(object(), [_RecordingLayer()], hidden, cache, [3, 5])
    seen = _RecordingLayer.seen
    check("run_layers_batched reaches the layer at all", seen is not None)
    if seen:
        check("position_ids follows hidden's device",
              seen["position_ids"] == seen["hidden"], seen)
        check("cache_position follows hidden's device",
              seen["cache_position"] == seen["hidden"], seen)
        check("the attention mask follows hidden's device",
              seen["attention_mask"] in (None, seen["hidden"]), seen)

    # build_mask takes an explicit device and honours it -- the mechanism behind the above.
    m = batching.build_mask([3, 5], 5, 1, torch.float32, device=torch.device("cpu"))
    check("build_mask accepts a device and builds there", m.device.type == "cpu")

    # The KV padding is created next to the cache entry it is concatenated onto. torch.cat
    # refuses to join a CPU pad to a CUDA cache, so this is load-bearing on a GPU node.
    src = open("batching.py", encoding="utf-8").read()
    pad = src[src.index("zk = torch.zeros"):src.index("zv = torch.zeros")]
    check("KV left-padding is allocated on the cache's own device", "device=k.device" in pad,
          pad.strip())


# --------------------------------------------------------------------------- #
# 3. the wire boundary is CPU
# --------------------------------------------------------------------------- #
def test_wire_boundary_returns_cpu():
    for codec in ("f32", "f16", "i8h"):
        t = torch.randn(2, 4, 256)
        blob = wire_codec.encode({"type": "act", "hidden": t}, codec)
        back = wire_codec.decode(blob)["hidden"]
        check(f"{codec}: round-trips and comes back on CPU",
              back.device.type == "cpu" and back.shape == t.shape)

    # `.numpy()` raises on a CUDA tensor, so every encoder has to reach CPU first. Asserted on
    # the source because this machine cannot make a CUDA tensor to prove it with.
    src = code_only(open("wire_codec.py", encoding="utf-8").read()).replace(" ", "")
    bare = src.count(".numpy()") - src.count(".cpu().numpy()")
    check("no encoder calls .numpy() without reaching CPU first", bare == 0,
          f"{bare} bare .numpy() call(s) — these raise TypeError on a CUDA tensor")

    # The Hadamard matrix is cached once, on CPU, and used against the caller's tensor. It has
    # to follow that tensor or the matmul raises on a GPU node -- on the DEFAULT codec.
    check("the Hadamard matrix follows the operand's device", "device=x.device" in src)

    # And the batched stages return CPU, like their unbatched twins in common.py. That
    # divergence -- unbatched returning _to_cpu while batched did not -- is the bug, because a
    # real chain serves through the batcher and never touches the unbatched path.
    bsrc = open("batching.py", encoding="utf-8").read()
    for fn in ("mid_stage_batched", "last_stage_batched", "apply_lm_head_batched",
               "first_stage_batched"):
        body = bsrc[bsrc.index(f"def {fn}("):]
        body = body[:body.index("\n@") if "\n@" in body else len(body)]
        check(f"{fn} returns a CPU tensor for the wire", "_to_cpu" in body)


def main():
    check("this machine has no CUDA device (so 'passes' here means 'passes on CPU')",
          not torch.cuda.is_available() or True)        # informational, never a failure
    print(f"  ..  common.DEVICE = {common.DEVICE}, torch {torch.__version__}")

    test_loader_leaves_weights_on_cpu()
    test_batching_aux_tensors_follow_hidden()
    test_wire_boundary_returns_cpu()

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
