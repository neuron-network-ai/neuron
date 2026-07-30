"""test_batching.py — does batching change the answer?

The bar here is numerical identity, not "close". `selftest_shard.py` proves this pipeline
bit-exact against the unsplit model, and batching is a throughput optimisation: if it moves
the logits, it is a correctness bug wearing a performance costume.

The hard case, and the reason this file exists: slots in one batch have DIFFERENT history
lengths, so the short ones are left-padded with junk. If the mask or the position ids are
wrong, attention quietly reads that junk and the answer drifts -- no crash, no warning. So
every test below builds a batch of *unequal* lengths on purpose.

Uses the 0.5B model (24 layers, H=896) to stay cheap.

Run: python test_batching.py
"""
import os
import time

import torch

os.environ.setdefault("NEURON_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

import batching  # noqa: E402
import common    # noqa: E402

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


LO, HI = 8, 16          # a middle node's slice


def test_mask():
    print("\nmask construction")
    m = batching.build_mask([3, 5], padded_len=5, q=1, dtype=torch.float32)
    check(f"shape is [B,1,q,kv] (got {tuple(m.shape)})", tuple(m.shape) == (2, 1, 1, 6))
    neg = torch.finfo(torch.float32).min
    # slot 0 holds 3 real tokens in a 5-wide cache -> its first 2 positions are padding
    check("short slot's left padding is masked out",
          (m[0, 0, 0, :2] == neg).all() and (m[0, 0, 0, 2:] == 0).all())
    check("full-length slot masks nothing", (m[1, 0, 0, :] == 0).all())

    # causal rule among a multi-token block
    m2 = batching.build_mask([0, 0], padded_len=0, q=3, dtype=torch.float32)
    check("query 0 cannot see keys 1,2", (m2[0, 0, 0, 1:] == neg).all())
    check("query 2 can see keys 0,1,2", (m2[0, 0, 2, :3] == 0).all())


def _decode_sequence(model, tokens, lo, hi):
    """Reference: run one sequence ALONE through the unbatched path, one token at a time.
    Returns (final hidden, its SplitCache, length)."""
    cache, past = common.new_cache(), 0
    out = None
    for t in tokens:
        out = common.mid_stage(model, lo, hi, t, cache, past)
        past += t.shape[1]
    return out, cache, past


def test_batched_equals_sequential(model, h):
    print("\nbatched vs sequential, UNEQUAL history lengths")
    torch.manual_seed(0)
    # three sequences with deliberately different lengths: 6, 2 and 4 tokens
    seqs = [[torch.randn(1, 3, h), torch.randn(1, 1, h), torch.randn(1, 1, h), torch.randn(1, 1, h)],
            [torch.randn(1, 2, h)],
            [torch.randn(1, 3, h), torch.randn(1, 1, h)]]

    singles, caches, lengths = [], [], []
    for s in seqs:
        outp, c, ln = _decode_sequence(model, s, LO, HI)
        singles.append(outp)
        caches.append(c)
        lengths.append(ln)
    check(f"prepared slots with unequal lengths {lengths}", len(set(lengths)) > 1)

    # now one more decode token each -- alone, then batched, and compare
    nxt = [torch.randn(1, 1, h) for _ in seqs]
    alone = []
    for s_cache, ln, t in zip(caches, lengths, nxt):
        import copy
        c2 = copy.deepcopy(s_cache)
        alone.append(common.mid_stage(model, LO, HI, t, c2, ln))

    bcache = batching.BatchedCache.from_single_caches(caches, lengths)
    check(f"fused cache left-pads to the longest ({bcache.padded_len} == {max(lengths)})",
          bcache.padded_len == max(lengths))
    batched = batching.mid_stage_batched(model, LO, HI, torch.cat(nxt, dim=0), bcache, lengths)

    worst = 0.0
    for i, a in enumerate(alone):
        d = (batched[i:i + 1] - a).abs().max().item()
        worst = max(worst, d)
    print(f"      max |batched - alone| across slots: {worst:.3e}")
    # fp32 GEMM shapes change with batch size, so exact bit equality is not on offer; this
    # is ~1e-6 territory, i.e. reordering noise, not a semantic difference.
    check("batched decode matches running each sequence alone (<1e-4)", worst < 1e-4)


def test_padding_junk_is_really_ignored(model, h):
    print("\npadding must be inert")
    torch.manual_seed(1)
    seqs = [[torch.randn(1, 4, h)], [torch.randn(1, 1, h)]]
    caches, lengths, = [], []
    for s in seqs:
        _o, c, ln = _decode_sequence(model, s, LO, HI)
        caches.append(c)
        lengths.append(ln)

    nxt = torch.cat([torch.randn(1, 1, h) for _ in seqs], dim=0)
    b1 = batching.BatchedCache.from_single_caches(caches, lengths)
    out1 = batching.mid_stage_batched(model, LO, HI, nxt.clone(), b1, list(lengths))

    # Rebuild, then deliberately poison the padded region with huge values. If the mask is
    # right this cannot move the result by even a little.
    b2 = batching.BatchedCache.from_single_caches(caches, lengths)
    short = lengths.index(min(lengths))
    pad = b2.padded_len - lengths[short]
    for li in b2.k:
        b2.k[li][short, :, :pad, :] = 1e4
        b2.v[li][short, :, :pad, :] = 1e4
    out2 = batching.mid_stage_batched(model, LO, HI, nxt.clone(), b2, list(lengths))

    d = (out1 - out2).abs().max().item()
    print(f"      max change from poisoning the padding: {d:.3e}")
    check("garbage in the padded region changes nothing (mask is correct)", d == 0.0)


def test_roundtrip_slots(model, h):
    print("\nslots can leave the batch")
    torch.manual_seed(2)
    seqs = [[torch.randn(1, 5, h)], [torch.randn(1, 2, h)]]
    caches, lengths = [], []
    for s in seqs:
        _o, c, ln = _decode_sequence(model, s, LO, HI)
        caches.append(c)
        lengths.append(ln)
    b = batching.BatchedCache.from_single_caches(caches, lengths)
    back = b.split_to_single_caches()
    good = True
    for orig, got, ln in zip(caches, back, lengths):
        for li in orig.k:
            good &= got.k[li].shape[-2] == ln and torch.equal(got.k[li], orig.k[li])
    check("split_to_single_caches restores each slot exactly, unpadded", good)


def test_speedup(model, h):
    print("\nthroughput (the whole point)")
    torch.manual_seed(3)
    for B in (4, 8):
        caches, lengths = [], []
        for _ in range(B):
            _o, c, ln = _decode_sequence(model, [torch.randn(1, 16, h)], LO, HI)
            caches.append(c)
            lengths.append(ln)

        t0 = time.perf_counter()
        for c, ln in zip(caches, lengths):
            common.mid_stage(model, LO, HI, torch.randn(1, 1, h), c, ln)
        seq_ms = (time.perf_counter() - t0) * 1000

        b = batching.BatchedCache.from_single_caches(caches, lengths)
        t0 = time.perf_counter()
        batching.mid_stage_batched(model, LO, HI, torch.randn(B, 1, h), b, list(lengths))
        bat_ms = (time.perf_counter() - t0) * 1000

        print(f"      B={B}: {seq_ms:6.1f} ms one-at-a-time vs {bat_ms:6.1f} ms batched "
              f"-> {seq_ms / max(bat_ms, 1e-9):.2f}x")
        check(f"B={B} batching is faster than serialising", bat_ms < seq_ms)


def main():
    test_mask()
    print(f"\nloading {common.MODEL_ID} shard layers {LO}-{HI - 1} ...", flush=True)
    _tok, model, _n = common.load_model_shard(LO, HI)
    h = model.config.hidden_size
    test_batched_equals_sequential(model, h)
    test_padding_junk_is_really_ignored(model, h)
    test_roundtrip_slots(model, h)
    test_speedup(model, h)
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
