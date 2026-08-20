"""test_stream_timing.py — run: python -m test_stream_timing

**The reading that caused this file.** Two replies over the same healthy chain, minutes apart:

    119 tokens  ->  2.18 tok/s
     16 tokens  ->  1.16 tok/s      (a web-grounded prompt, four sources)

which reads as the network halving in speed. It had not: `tools/bench_hop.py` measured 49.7 ms
against 49.5 ms either side of it. `tok_per_s` is `completion / (now - t_start)` — OUTPUT tokens
over the WHOLE request — so the coordinator round trip, the handshake and the prefill of the
entire prompt all land in the denominator, and dividing them by 16 instead of 119 is the entire
difference.

That number is what a user judges NEURON by, and it moves with prompt length rather than with
anything the network did. So the wait and the steady rate are now reported separately, which is
the conclusion `node_a.py` had already reached in another corner of the same codebase: *"report
both rather than picking whichever is flattering."*

This file pins the arithmetic, the honest-absence cases, and that every engine and the page
actually carry the fields — because a metric nobody plumbed through is [P54]'s shape again.
"""
import io
import sys

import stream_timing

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    # ---- 1. the split, on the two readings that prompted it ------------------- #
    # Constructed so BOTH decode at exactly 2.2 tok/s and reproduce the observed headline
    # figures: 16/13.79 = 1.16 and 119/54.59 = 2.18. The only difference between them is how
    # much prompt had to be read first -- 6.97 s for four web sources, 0.95 s for a bare
    # question -- which is the whole point being pinned.
    short = stream_timing.fields(completion=16, t_start=0.0, t_first=6.97, t_end=13.79)
    long_ = stream_timing.fields(completion=119, t_start=0.0, t_first=0.95, t_end=54.59)

    check("the SHORT reply still reports the misleading whole-request rate",
          short["tok_per_s"] == 1.16, short)
    check("...and the long one reports its own",
          long_["tok_per_s"] == 2.18, long_)
    check("but their DECODE rates are within 5% of each other — the network did not change",
          abs(short["decode_tok_per_s"] - long_["decode_tok_per_s"])
          / long_["decode_tok_per_s"] < 0.05,
          f"{short['decode_tok_per_s']} vs {long_['decode_tok_per_s']}")
    check("the short reply's wait is attributed to the PROMPT, where it belongs",
          short["ttft_ms"] == 6970, short)
    check("...and the long one's is far smaller despite its much longer total",
          long_["ttft_ms"] == 950 and long_["latency_ms"] > short["latency_ms"], long_)

    # ---- 2. the arithmetic ---------------------------------------------------- #
    f = stream_timing.fields(completion=11, t_start=100.0, t_first=101.0, t_end=111.0)
    check("latency_ms is the whole request", f["latency_ms"] == 11000, f)
    check("tok_per_s is completion over the whole request", f["tok_per_s"] == 1.0, f)
    check("ttft_ms is t_first - t_start", f["ttft_ms"] == 1000, f)
    check("decode uses completion-1 over the post-first-token window",
          f["decode_tok_per_s"] == 1.0, f)

    # ---- 3. absence is honest, never a fabricated number ---------------------- #
    one = stream_timing.fields(completion=1, t_start=0.0, t_first=5.0, t_end=5.2)
    check("ONE token reports no decode rate — a single sample is not a steady state",
          "decode_tok_per_s" not in one, one)
    check("...but still says how long the prompt took", one["ttft_ms"] == 5000, one)
    none = stream_timing.fields(completion=0, t_start=0.0, t_first=None, t_end=2.0)
    check("a reply that never emitted a token reports no ttft and no decode rate",
          "ttft_ms" not in none and "decode_tok_per_s" not in none, none)
    check("...and still reports the whole-request figures callers rely on",
          none["latency_ms"] == 2000 and none["tok_per_s"] == 0.0, none)
    check("a blocked generation (tokens counted, none yielded) does not crash",
          "ttft_ms" not in stream_timing.fields(3, 0.0, None, 1.0))
    check("a zero-length window cannot divide by zero",
          stream_timing.fields(5, 1.0, 1.0, 1.0)["decode_tok_per_s"] > 0)

    # ---- 4. every engine reports it, and the page shows it -------------------- #
    for path in ("neuron_driver.py", "engine/local_gguf.py", "engine/ggml_pipeline.py"):
        src = io.open(path, encoding="utf-8").read()
        check(f"{path} imports stream_timing", "import stream_timing" in src)
        check(f"{path} builds its done event from it", "stream_timing.fields(" in src)
        check(f"{path} marks the first token", "t_first = time.time()" in src)
        check(f"{path} starts it as None", "t_first = None" in src)

    app = io.open("ui/app.py", encoding="utf-8").read()
    check("ui/app.py forwards ttft_ms to the browser", '"ttft_ms": ev.get("ttft_ms")' in app)
    check("ui/app.py forwards decode_tok_per_s",
          '"decode_tok_per_s": ev.get("decode_tok_per_s")' in app)

    page = io.open("ui/static/chat.html", encoding="utf-8").read()
    check("the chat page prefers the decode rate", "d.decode_tok_per_s" in page)
    check("...shows the total wait beside it", "d.latency_ms / 1000" in page)
    check("...and still falls back for an engine that sends neither",
          "d.tok_per_s ?" in page)

    spec = io.open("packaging/neuron-agent.spec", encoding="utf-8").read()
    check("the frozen app carries the module ([P54]: a new top-level module must be declared)",
          '"stream_timing"' in spec)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
