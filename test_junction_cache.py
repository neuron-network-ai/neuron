"""test_junction_cache.py — the cache that lets a request survive a node dropping, and the
throughput-weighted routing/placement that replaces random.choice.

These are Petals mechanisms 1, 2 and 3 (PETALS_NOTES.md). Deliberately model-free so it runs
in a second and costs nothing: it tests the recovery BOOKKEEPING. The recovery itself --
replaying into a genuinely fresh chain across machines -- has to be proved on the real three
nodes, and is listed as such in sessions.md.

Run: python test_junction_cache.py
"""
import random

import torch

import junction_cache
from coordinator import router

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


H = 1536


def test_accounting():
    print("\njunction cache — accounting")
    c = junction_cache.JunctionCache()
    check("empty cache is not recoverable (nothing to replay)", not c.recoverable())
    check("empty cache replays as None", c.replay_block() is None)

    c.add(torch.randn(1, 45, H))          # prefill
    for _ in range(9):
        c.add(torch.randn(1, 1, H))       # 9 decode tokens
    check(f"tracks token count across prefill+decode (got {c.tokens})", c.tokens == 54)
    # fp16 storage: 2 bytes per element, not 4
    check(f"stored fp16, so {c.nbytes} == 54*{H}*2", c.nbytes == 54 * H * 2)
    check("recoverable once something is cached", c.recoverable())


def test_replay_block():
    print("\njunction cache — replay block")
    c = junction_cache.JunctionCache()
    a = torch.randn(1, 3, H)
    b = torch.randn(1, 1, H)
    c.add(a)
    c.add(b)
    blk = c.replay_block()
    check(f"replay is one concatenated block (got {tuple(blk.shape)})",
          tuple(blk.shape) == (1, 4, H))
    check("replay is fp32, ready for the wire", blk.dtype is torch.float32)
    # order matters: the chain rebuilds its K/V positionally, so a shuffled replay would
    # silently produce a coherent-but-wrong continuation
    check("replay preserves order and values (within fp16 storage error)",
          torch.allclose(blk[:, :3], a, atol=5e-2) and torch.allclose(blk[:, 3:], b, atol=5e-2))


def test_overflow_fails_honestly():
    print("\njunction cache — overflow")
    # A cap small enough that the second add cannot fit.
    c = junction_cache.JunctionCache(max_bytes=4 * H * 2)
    c.add(torch.randn(1, 4, H))
    check("under the cap: recoverable", c.recoverable())
    c.add(torch.randn(1, 4, H))
    # The whole point: a truncated history must NOT be replayed. Replaying it would give the
    # replacement node a plausible but wrong context and the user a quietly corrupted answer.
    check("over the cap: no longer recoverable", not c.recoverable())
    check("over the cap: refuses to hand back a partial replay", c.replay_block() is None)
    check("still counts every token it saw", c.tokens == 8)

    c.clear()
    check("clear() resets the overflow flag", not c.overflowed and c.tokens == 0)


def _node(nid, lo, hi, ms):
    return {"node_id": nid, "layer_start": lo, "layer_end": hi, "ms_per_layer": ms}


def test_throughput_math():
    print("\nrouting — throughput scoring")
    fast = _node("fast", 0, 9, 10.0)      # 10 layers x 10 ms = 100 ms
    slow = _node("slow", 0, 9, 40.0)      # 10 layers x 40 ms = 400 ms
    check(f"stage_ms counts layers x ms_per_layer (got {router.stage_ms(fast)})",
          router.stage_ms(fast) == 100.0)
    check("a 4x slower node has 4x lower throughput",
          abs(router.throughput(fast) / router.throughput(slow) - 4.0) < 1e-6)
    check("segment throughput sums its replicas (Petals' block throughput)",
          abs(router.segment_throughput([fast, slow])
              - (router.throughput(fast) + router.throughput(slow))) < 1e-9)

    unmeasured = _node("new", 0, 9, None)
    check("an unmeasured node still gets a finite score, not zero or a crash",
          router.throughput(unmeasured) > 0)
    check("a node with a junk ms_per_layer falls back instead of raising",
          router.throughput(_node("junk", 0, 9, "banana")) > 0)


def test_pick_prefers_fast_but_still_spreads():
    print("\nrouting — replica choice")
    fast = _node("fast", 0, 9, 10.0)
    slow = _node("slow", 0, 9, 40.0)
    pick = router.fastest_pick(rng=random.Random(0))
    counts = {"fast": 0, "slow": 0}
    for _ in range(4000):
        counts[pick([fast, slow])["node_id"]] += 1
    ratio = counts["fast"] / max(counts["slow"], 1)
    # ~4:1 is the point of the change (random.choice gave 1:1)
    check(f"traffic follows throughput, ~4:1 (got {ratio:.2f}:1)", 3.0 < ratio < 5.5)
    # ...but the slow node is NOT starved. An argmin router would give it zero, serialise
    # everything behind one machine's compute_lock, and undo [P16].
    check(f"the slow node still gets real traffic (got {counts['slow']})", counts["slow"] > 200)

    check("a single replica is returned without consulting the rng",
          pick([slow])["node_id"] == "slow")

    same = [_node("a", 0, 9, None), _node("b", 0, 9, None)]
    got = {pick(same)["node_id"] for _ in range(200)}
    check("with no perf data anywhere, choice stays spread (old behaviour)", got == {"a", "b"})


def main():
    test_accounting()
    test_replay_block()
    test_overflow_fails_honestly()
    test_throughput_math()
    test_pick_prefers_fast_but_still_spreads()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
