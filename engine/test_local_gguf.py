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
