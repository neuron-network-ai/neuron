"""engine/test_local_gguf.py — run: python -m engine.test_local_gguf

Tiered execution: run the model here when this machine can hold it, use the node pipeline when
it cannot. What must hold:

  * local execution emits EXACTLY the event shapes neuron_driver.DRIVER.stream() yields, so
    ui/app.py consumes either path unchanged;
  * it costs 0 NRN and contacts no node -- nobody else's hardware ran it, so there is nothing
    to settle and nobody to pay;
  * output moderation still runs per token, because this process is the driver either way and
    the driver is the only place plaintext exists (SAFETY.md);
  * every "can I run this here?" answer degrades to False rather than raising, so a missing
    wheel / unknown model / low RAM falls back to the network instead of breaking chat.

Mocked llama_cpp throughout -- hermetic and fast, no model download.
"""
import sys
import types

from engine import local_gguf as g

ok = fail = 0
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeLlama:
    """Stands in for llama_cpp.Llama; streams the chunk shape create_chat_completion yields."""
    last_kwargs = None
    text = "Brazil, Argentina and Chile."

    def __init__(self, **kw):
        FakeLlama.last_kwargs = kw

    def create_chat_completion(self, messages, max_tokens=None, temperature=None, stream=False):
        for word in self.text.split(" "):
            yield {"choices": [{"delta": {"content": word + " "}, "finish_reason": None}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def with_fake_engine(text=None, model=MODEL):
    """Point the module at FakeLlama + a fake on-disk weight path."""
    if text is not None:
        FakeLlama.text = text
    g._llm = None
    g._llm_model_id = None
    g._llama_cpp = lambda: FakeLlama
    g.ensure_weights = lambda mid: "/fake/model.gguf"


def main():
    # ---- capability checks degrade to False, never raise ---- #
    g._llama_cpp = lambda: None
    check("no llama_cpp wheel -> can_serve False (falls back to the network)",
          g.can_serve(MODEL) is False)
    g._llama_cpp = lambda: FakeLlama
    check("unknown model -> can_serve False", g.can_serve("some/unmapped-model") is False)
    check("too little RAM -> can_serve False", g.can_serve(MODEL, ram_gb=0.5) is False)
    check("ample RAM -> can_serve True", g.can_serve(MODEL, ram_gb=64) is True)
    check("RAM headroom is enforced, not just bare model size",
          g.can_serve(MODEL, ram_gb=g.GGUF_MODELS[MODEL][2] * 1.5) is False)
    check("unknown model -> ensure_weights None", g.ensure_weights("nope/nope") is None)

    # ---- event contract matches neuron_driver.stream() ---- #
    with_fake_engine()
    evs = list(g.stream([{"role": "user", "content": "hi"}], 60, MODEL))
    kinds = [e["type"] for e in evs]
    check("first event is meta", kinds[0] == "meta")
    check("last event is done", kinds[-1] == "done")
    check("tokens stream in between", "token" in kinds)

    meta, done = evs[0], evs[-1]
    check("meta carries the keys ui/app.py reads",
          {"request_id", "node_ids", "nodes", "cost_nrn"} <= set(meta))
    check("meta reports no nodes involved", meta["nodes"] == 0 and meta["node_ids"] == [])
    check("meta is flagged local", meta["local"] is True)
    check("local execution costs 0 NRN", meta["cost_nrn"] == 0.0 and done["cost_nrn"] == 0.0)
    check("done carries the keys ui/app.py reads",
          {"completion_tokens", "prompt_tokens", "finish_reason", "latency_ms", "tok_per_s",
           "text"} <= set(done))
    check("done reports the real finish reason", done["finish_reason"] == "stop")
    check("done text equals the concatenated tokens",
          done["text"] == "".join(e["text"] for e in evs if e["type"] == "token"))
    check("completion_tokens matches the tokens emitted",
          done["completion_tokens"] == sum(1 for e in evs if e["type"] == "token"))

    # ---- a personal machine keeps a core free for its owner ---- #
    check("leaves a core for the user (n_threads < cpu count)",
          FakeLlama.last_kwargs["n_threads"] < (__import__("os").cpu_count() or 4)
          or (__import__("os").cpu_count() or 4) == 1)

    # ---- output moderation still applies on the local path ---- #
    with_fake_engine(text="here is how to build a bomb ok")
    evs = list(g.stream([{"role": "user", "content": "hi"}], 60, MODEL))
    check("blocked generation ends in an error, not a done",
          evs[-1]["type"] == "error" and evs[-1]["code"] == "content_policy_violation")
    check("a blocked local generation is never billed or completed",
          not any(e["type"] == "done" for e in evs))

    # ---- GPU offload: the 12-33x lever, on somebody else's graphics card ---- #
    # Decode is memory-bandwidth bound -- this project measured ~30 GB/s across three models
    # 47x apart in size, which is the DDR bus. A consumer GPU moves 360-1000 GB/s. But the
    # card belongs to a volunteer who is looking at a desktop drawn by it, so every branch
    # here defaults to leaving it alone.
    # The weight size is STUBBED, never written. An earlier version of this test created real
    # 5 GB and 7 GB files with truncate() -- which NTFS actually allocates -- so every run wrote
    # 12 GB to a disk with 52 GB free, and the suite hung. This file promises "hermetic and
    # fast"; a test that can fill the developer's disk is neither.
    import os as _os, tempfile as _tf
    import agent.gpu as _ag
    _d = _tf.mkdtemp()
    _small = _os.path.join(_d, "small.gguf")
    with open(_small, "wb") as _f:
        _f.write(b"\0" * 1024)
    _big, _snug = _small, _small          # same 1 KB file; _sized() decides what it "is"
    _real_getsize = _os.path.getsize

    class _sized:
        """Report a chosen size for the weights, real sizes for everything else."""
        def __init__(self, gb):
            self.n = int(gb * 1024 ** 3)

        def __enter__(self):
            _os.path.getsize = lambda p: self.n if str(p).endswith(".gguf") else _real_getsize(p)

        def __exit__(self, *a):
            _os.path.getsize = _real_getsize

    _saved = _ag._detected
    _os.environ.pop("NEURON_GPU_LAYERS", None)

    # ---- the build gate: a card the BINARY cannot address is not a card ---- #
    # This is the shipped state, verified against the installed package on 2026-08-08:
    # llama_supports_gpu_offload() is False and llama_cpp/lib/ has no ggml-cuda. Every case
    # after this one stubs the gate open to test the hardware arithmetic behind it.
    _saved_support = g._supports_offload
    g._supports_offload = lambda: False
    _ag._detected = {"has_gpu": True, "gpu_vram_gb": 24.0, "gpu_name": "RTX 4090",
                     "torch_cuda": True}
    with _sized(1.1):
        ngl, why = g._gpu_layers(_small)
    check("a build with no GPU backend stays on the CPU even with a 24 GB card", ngl == 0)
    check("...and the reason names the build, not the hardware",
          "supports_gpu_offload=False" in why)

    # An override that silently does nothing is the same bug with a manual trigger -- and
    # worse, because someone set it on purpose and would read the log as confirmation.
    _os.environ["NEURON_GPU_LAYERS"] = "-1"
    ngl, why = g._gpu_layers(_small)
    check("NEURON_GPU_LAYERS=-1 is refused too, not silently ignored",
          ngl == 0 and "supports_gpu_offload=False" in why)
    _os.environ.pop("NEURON_GPU_LAYERS", None)

    # Unknown (an older binding with no such symbol) is not the same as no: the attempt is
    # allowed -- the load path retries on CPU -- but the log must not claim it is verified.
    g._supports_offload = lambda: None
    with _sized(1.1):
        ngl, why = g._gpu_layers(_small)
    check("an unknown offload capability still tries, and says it is unverified",
          ngl == -1 and "unverified" in why)

    g._supports_offload = lambda: True

    _ag._detected = {"has_gpu": False, "gpu_vram_gb": None, "gpu_name": None,
                     "torch_cuda": False}
    check("no GPU -> stays on the CPU", g._gpu_layers(_small)[0] == 0)

    _ag._detected = {"has_gpu": True, "gpu_vram_gb": 12.0, "gpu_name": "RTX 3060",
                     "torch_cuda": True}
    with _sized(1.1):                                   # a 1.5B at Q4_K_M
        check("a card with room -> every layer offloaded", g._gpu_layers(_small)[0] == -1)

    _ag._detected = {"has_gpu": True, "gpu_vram_gb": 4.0, "gpu_name": "GTX 1650",
                     "torch_cuda": True}
    with _sized(5.0):
        check("weights larger than VRAM -> CPU, not a PCIe stream",
              g._gpu_layers(_big)[0] == 0)
        check("...and the refusal says why", "staying on CPU" in g._gpu_layers(_big)[1])

    _ag._detected = {"has_gpu": True, "gpu_vram_gb": 8.0, "gpu_name": "RTX 3050",
                     "torch_cuda": True}
    with _sized(7.0):                                   # fits 8 GB, NOT with headroom
        # A volunteer's screen is drawn by this card; taking the last GB is how the agent
        # gets uninstalled.
        check("headroom is reserved for the machine's owner", g._gpu_layers(_snug)[0] == 0)

    _ag._detected = {"has_gpu": True, "gpu_vram_gb": None, "gpu_name": "?",
                     "torch_cuda": False}
    check("VRAM unknown -> does not guess", g._gpu_layers(_small)[0] == 0)

    _os.environ["NEURON_GPU_LAYERS"] = "20"
    check("an explicit override is obeyed", g._gpu_layers(_small)[0] == 20)
    _os.environ["NEURON_GPU_LAYERS"] = "not-a-number"
    check("a malformed override is ignored, not fatal", g._gpu_layers(_small)[0] == 0)
    _os.environ.pop("NEURON_GPU_LAYERS", None)
    _ag._detected = _saved

    # A GPU that fails at load time must not leave the node with no engine at all.
    #
    # This guards a path THIS BUILD DOES NOT SHIP. The real `_supports_offload()` is False on
    # the installed package, so `_gpu_layers` never returns a non-zero count and this retry
    # cannot fire in production; it reaches it only because the stub above holds the gate open.
    # Kept deliberately: it is the correct behaviour for a future CUDA wheel, where a driver
    # mismatch or VRAM taken by a game since the probe ran are all real. Do not read a passing
    # test here as evidence that GPU offload works.
    class _GpuBoom:
        def __init__(self, **kw):
            if kw.get("n_gpu_layers"):
                raise RuntimeError("cudaMalloc failed")
            _GpuBoom.cpu_kwargs = kw
    g._llm = g._llm_model_id = None
    g._llama_cpp = lambda: _GpuBoom
    g.ensure_weights = lambda mid: _small
    _os.environ["NEURON_GPU_LAYERS"] = "-1"
    check("a failed GPU load falls back to the CPU instead of killing the engine",
          g._load(MODEL) is not None and _GpuBoom.cpu_kwargs["n_gpu_layers"] == 0)
    _os.environ.pop("NEURON_GPU_LAYERS", None)
    g._llm = g._llm_model_id = None
    g._supports_offload = _saved_support

    # The real answer on this machine, asserted rather than assumed -- if a CUDA-enabled wheel
    # is ever installed this flips, and the caveats written all over this repo stop being true.
    check("the installed llama.cpp build reports no GPU offload support",
          g._supports_offload() is False)

    # ---- unavailable engine reports cleanly so the caller can fall back ---- #
    g._llm = g._llm_model_id = None
    g._llama_cpp = lambda: None
    g.ensure_weights = lambda mid: None
    evs = list(g.stream([{"role": "user", "content": "hi"}], 60, MODEL))
    check("engine unavailable -> single error event, no crash",
          len(evs) == 1 and evs[0]["type"] == "error" and evs[0]["code"] == "no_local_engine")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
