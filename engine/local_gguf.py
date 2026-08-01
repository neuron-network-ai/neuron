"""engine/local_gguf.py — run the serving model locally, quantized, when this machine can hold it.

WHY THIS EXISTS (measured 2026-07-29, this machine, 8 threads, Qwen2.5-1.5B-Instruct):

    fp32 across the node pipeline (today)   240 ms/token   correct
    bf16                                   2002 ms/token   correct, 8x SLOWER (no HW bf16)
    dynamic int8, MLP only                  504 ms/token   degraded AND slower
    dynamic int8, all Linear                119 ms/token   BROKEN -- confirms [P9]
    GGUF Q4_K_M via llama.cpp                36 ms/token   correct (matched/beat fp32)

6.7x faster with quality intact, so a full answer lands in ~10s against TOKENOMICS.md
§11.6's "<30s answers" gate -- versus ~40 minutes before.

THE POINT IS NOT "STOP BEING A NETWORK." At 36 ms/token a 1.5B model (1.12 GB at Q4_K_M) fits
on essentially any machine, so splitting THAT across three PCs buys nothing and costs a network
hop, a bottleneck stage and a failure mode. NEURON's actual reason to exist is the opposite
case: models too big for the machine in front of you. So execution is tiered --

    fits locally      -> run it here. Fast, private, costs no NRN (it is your own CPU).
    does not fit      -> the node pipeline, which is the only way to run it at all.

That keeps the network aimed at what only it can do, and means a user always gets a working
answer instead of "Network degraded -- responses will fail" whenever the chain is incomplete.

Emits exactly the event shapes neuron_driver.DRIVER.stream() yields, so ui/app.py consumes
either path unchanged. Import and llama_cpp are both optional: if the wheel or the GGUF is
missing, available() is False and the caller falls back to the network.
"""
import logging
import os
import time
import uuid

from safety import moderation

log = logging.getLogger("neuron.engine.local_gguf")

# Quantized builds of the models the tier ladder serves (coordinator/model_registry.py).
# Q4_K_M is the k-quant sweet spot: ~1% quality loss, unlike the naive int8 that broke [P9].
GGUF_MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": ("Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                                   "qwen2.5-1.5b-instruct-q4_k_m.gguf", 1.2),
    "Qwen/Qwen2.5-0.5B-Instruct": ("Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                                   "qwen2.5-0.5b-instruct-q4_k_m.gguf", 0.5),
    # 7B is the tier ladder's next step. Its GGUF is SPLIT across two files upstream, unlike
    # the single-file smaller ones, so the filename below is part 1 of 2 -- llama.cpp follows
    # the split automatically once both parts are in the same cache directory.
    "Qwen/Qwen2.5-7B-Instruct": ("Qwen/Qwen2.5-7B-Instruct-GGUF",
                                 "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf", 4.7),
}

# Keep well clear of the RAM ceiling: the agent is also holding its own compute slice for the
# network, and this machine belongs to somebody who is using it for something else.
RAM_HEADROOM_FACTOR = 3.0

_llm = None          # loaded lazily; loading costs seconds and most turns never need a reload
_llm_model_id = None


def _llama_cpp():
    try:
        from llama_cpp import Llama
        return Llama
    except ImportError:
        return None


def can_serve(model_id, ram_gb=None):
    """True if this machine should run `model_id` itself rather than asking the network."""
    if _llama_cpp() is None or model_id not in GGUF_MODELS:
        return False
    need_gb = GGUF_MODELS[model_id][2]
    if ram_gb is None:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().available / 1e9
        except Exception:
            return False
    return ram_gb >= need_gb * RAM_HEADROOM_FACTOR


def best_local_model(preferred=None, require_cached=True):
    """The LARGEST model this machine can actually run itself, not just the network's model.

    The tier ladder picks what the *network* serves, which is bounded by its weakest members;
    a 16-core desktop with 64 GB was being handed 1.5B answers because that is all a 4-core
    laptop elsewhere can hold a slice of. Locally there is no such constraint — quality is
    free here, so take it. Measured on this machine, Q4_K_M: 1.5B = 27.9 tok/s, 7B = 7.8 tok/s.
    Both are comfortably readable, and 7B is a different class of answer.

    `require_cached` is what keeps a first run fast: it only returns weights already on disk,
    so the machine answers on whatever it has now and upgrades once the bigger download lands
    (started in the background by prefetch_best()). Set NEURON_LOCAL_MODEL to pin one.
    """
    pinned = os.environ.get("NEURON_LOCAL_MODEL")
    if pinned:
        return pinned if can_serve(pinned) else None
    # largest first, by the RAM figure in GGUF_MODELS
    for mid in sorted(GGUF_MODELS, key=lambda m: GGUF_MODELS[m][2], reverse=True):
        if not can_serve(mid):
            continue
        if require_cached and not available(mid):
            continue
        return mid
    if preferred and can_serve(preferred) and (not require_cached or available(preferred)):
        return preferred
    return None


def prefetch_best():
    """Fetch the biggest model this machine could run, if it isn't already here.

    Called in the background at startup: without it `best_local_model()` can never move up a
    tier, because it only ever returns weights that are already cached. Returns the model_id
    it fetched, or None.
    """
    target = best_local_model(require_cached=False)
    if not target or available(target):
        return None
    log.info("fetching a larger local model in the background: %s", target)
    return target if ensure_weights(target) else None


def available(model_id):
    """can_serve() plus the weights actually being on disk already -- used to decide whether to
    advertise local execution without triggering a download mid-request."""
    if not can_serve(model_id):
        return False
    repo, fname, _ = GGUF_MODELS[model_id]
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(repo, fname), str)
    except Exception:
        return False


def ensure_weights(model_id):
    """Download the quantized weights if absent (~1.1 GB for 1.5B). Returns a path or None."""
    if model_id not in GGUF_MODELS:
        return None
    repo, fname, _ = GGUF_MODELS[model_id]
    try:
        from huggingface_hub import hf_hub_download
        log.info("fetching quantized weights for local execution (%s)", fname)
        return hf_hub_download(repo, fname)
    except Exception as e:
        log.warning("could not fetch %s: %s", fname, e)
        return None


def _load(model_id, n_threads=None):
    global _llm, _llm_model_id
    if _llm is not None and _llm_model_id == model_id:
        return _llm
    Llama = _llama_cpp()
    path = ensure_weights(model_id)
    if Llama is None or path is None:
        return None
    # Leave a core free: this runs on somebody's personal machine while they use it.
    threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
    t0 = time.perf_counter()
    _llm = Llama(model_path=path, n_ctx=4096, n_threads=threads, verbose=False)
    _llm_model_id = model_id
    log.info("local engine ready (%s, %d threads) in %.1fs", model_id, threads,
             time.perf_counter() - t0)
    return _llm


def stream(messages, max_new, model_id, coordinator=None, wallet_id=None, request_id=None):
    """Generate locally, yielding neuron_driver.stream()'s event shapes.

    cost_nrn is 0.0 throughout and no /infer call is made: nobody else's hardware ran this, so
    there is nothing to settle and nobody to pay. Output moderation still runs on every token,
    exactly as the network path does -- this process is the driver either way, and the driver is
    the only place plaintext exists (SAFETY.md).
    """
    request_id = request_id or ("local-" + uuid.uuid4().hex[:12])
    llm = _load(model_id)
    if llm is None:
        yield {"type": "error", "detail": "local engine unavailable", "code": "no_local_engine"}
        return

    yield {"type": "meta", "request_id": request_id, "node_ids": [], "nodes": 0,
           "cost_nrn": 0.0, "local": True}

    t0 = time.time()
    full, completion, finish = "", 0, "length"
    try:
        for chunk in llm.create_chat_completion(messages, max_tokens=max_new,
                                                temperature=0.0, stream=True):
            choice = chunk["choices"][0]
            delta = (choice.get("delta") or {}).get("content")
            if choice.get("finish_reason") == "stop":
                finish = "stop"
            if not delta:
                continue
            full += delta
            completion += 1
            # Checked against the FULL accumulated text, not just the delta, so a blocked
            # phrase split across a token boundary is still caught -- same rule as the
            # network path in neuron_driver.
            verdict = moderation.check_text(full)
            if verdict.blocked:
                moderation.log_event("out", verdict.category, request_id, snippet=full)
                moderation.report_violation(coordinator, wallet_id, "out", verdict.category)
                yield {"type": "error",
                       "detail": "This response was blocked by NEURON's acceptable-use "
                                 "policy (see SAFETY.md).",
                       "code": "content_policy_violation"}
                return
            yield {"type": "token", "text": delta}
    except Exception as e:
        yield {"type": "error", "detail": f"{e.__class__.__name__}: {e}"}
        return

    elapsed = max(time.time() - t0, 1e-6)
    yield {"type": "done", "completion_tokens": completion, "prompt_tokens": 0,
           "finish_reason": finish, "latency_ms": int(elapsed * 1000),
           "tok_per_s": round(completion / elapsed, 2), "text": full, "cost_nrn": 0.0}
