"""stream_timing.py — the two rates a streaming reply has, because reporting one misleads.

**The number this exists to fix.** `neuron_driver` reported `tok_per_s` as
`completion / (now - t_start)` — output tokens divided by the WHOLE request. That denominator
carries the coordinator round trip, the connect and handshake, and the prefill of the entire
prompt, none of which get smaller when the answer is short. So a 119-token reply read 2.18 tok/s
and a 16-token reply over the same healthy chain read **1.16**, and the founder reasonably took
the second as a regression. Nothing had regressed: the hop measured 49.7 ms against 49.5 ms
either side of it. The fixed costs were simply divided by 16 instead of 119.

That is not a display quibble. It is the number a user judges the network by, it moves with
prompt length rather than with anything the network did, and it makes the product look worst
exactly when it answers most concisely.

**So report both, and name them.** `node_a.py` already learned this — *"report both rather than
picking whichever is flattering"* — and the chat footer picked the one that flatters in reverse.

  * `ttft_ms`      — time to the FIRST token: setup, handshake and prefill. Scales with the
                     PROMPT.
  * `decode_tok_per_s` — the steady rate after that, over the remaining tokens. Scales with the
                     network and the engine, and is the number that means "how fast is NEURON".
  * `tok_per_s`    — unchanged, still whole-request. Kept because callers and tests read it, and
                     because it is the honest answer to "how long did I wait per token of
                     output".

**Why `completion - 1` and not `completion`.** The first token's cost IS the prefill; charging
it to decode would drag the steady rate down by exactly the thing we are separating out. With
one token there is no steady state to report, so `decode_tok_per_s` is None rather than a number
computed from a single sample — an absent field is honest where a fabricated one is not.

No imports on purpose: `engine/local_gguf.py` is the torch-free local path and must stay that
way, so this cannot live in `common.py`.
"""


def fields(completion, t_start, t_first, t_end):
    """Timing fields for a `done` event.

    `t_first` is when the first token was yielded, or None if none ever was (a blocked or
    empty generation) — in which case there is no first-token moment to report and only the
    whole-request figures are returned.
    """
    total = max(t_end - t_start, 1e-6)
    out = {
        "latency_ms": int(total * 1000),
        "tok_per_s": round(completion / total, 2),
    }
    if t_first is None or completion <= 0:
        return out
    out["ttft_ms"] = int(max(t_first - t_start, 0.0) * 1000)
    if completion >= 2:
        decode = max(t_end - t_first, 1e-6)
        out["decode_tok_per_s"] = round((completion - 1) / decode, 2)
    return out
