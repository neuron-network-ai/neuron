"""
NEURON — batching.py

The single highest-leverage change available to this project, and the one structural reason
a datacenter GPU beats a volunteer network by more than its clock speed.

THE PROBLEM
-----------
`agent/node_server.py` holds a module-level `compute_lock`, so a machine runs **exactly one
request's forward pass at a time**. A datacenter GPU runs one forward pass for dozens of
users simultaneously -- the weights are read from memory once and amortised across the whole
batch. On a CPU the same trick applies: a `[8, 1, H] @ [H, H]` GEMM costs far less than eight
`[1, 1, H] @ [H, H]` GEMMs, because the weight matrix is loaded once instead of eight times.
Decode is memory-bandwidth-bound, not FLOP-bound, which is exactly the regime where batching
is close to free.

So batching multiplies a machine's serving capacity without adding hardware. TOKENOMICS.md
§12.5 ranks it first for that reason: nothing else changes the concurrency ceiling by an
order of magnitude.

WHAT THIS MODULE DOES, AND DELIBERATELY DOES NOT
------------------------------------------------
It batches the **decode** step -- the q=1 pass that every request repeats once per output
token. A 200-token answer is 1 prefill and 199 decodes, so decode is where essentially all
the time goes, and batching just that captures most of the available win.

It does NOT implement full continuous batching (ragged prefill/decode mixing, paged
attention). That is a vLLM-scale project, and this pipeline's core is proved *bit-exact*
against the unsplit model by `selftest_shard.py` -- so the bar for touching it is
numerical identity, not "close enough". Prefill runs unbatched, alone.

HOW SEQUENCES OF DIFFERENT LENGTHS SHARE ONE BATCH
--------------------------------------------------
Requests arrive at different times, so their histories differ in length. The cache is a
single `[B, heads, S, D]` tensor, so every slot must share one S. Slots are therefore
**left-padded**: a slot holding `L` real tokens occupies the last `L` positions, and the
first `S - L` are junk that an additive mask hides from attention.

Left-padding (rather than right-) is what keeps the newest token at the same index for every
slot, so one `q=1` append stays correct for the whole batch.

Rotary position embeddings are unaffected by the padding because `position_ids` carries each
slot's TRUE absolute position, not its index in the padded tensor -- padding moves where a
token sits in memory, never what position the model thinks it has.

`test_batching.py` asserts the output is identical to running each sequence alone.
"""
import os
import queue
import threading
import time

import torch

import common

# How many requests one forward pass may carry. Measured on this hardware (0.5B, 8-layer
# slice): B=4 -> 3.16x, B=8 -> 3.74x versus running them one at a time. The curve flattens
# because CPU decode stops being purely bandwidth-bound once the batch is wide enough, and
# because a bigger batch waits longer to fill.
MAX_BATCH = int(os.environ.get("NEURON_MAX_BATCH", "8"))

# How long a request waits for company before going alone. This is pure added latency for
# the first arrival, so it must stay well under one decode step (~30-100 ms here). At 8 ms a
# lone request pays ~10% latency; a busy node fills the batch long before the deadline.
BATCH_WINDOW_S = float(os.environ.get("NEURON_BATCH_WINDOW_MS", "8")) / 1000.0


class BatchedCache:
    """K/V for a batch of slots, `[B, heads, S, D]` per layer.

    Same contract as `common.SplitCache` (dict-keyed by the layer's native index, because a
    node owns layers 14..27 and HF's DynamicCache assumes they start at 0), but every entry
    carries a batch dimension and `lengths` records how many of the S positions are real for
    each slot.
    """

    def __init__(self, lengths=None):
        self.k = {}
        self.v = {}
        self.lengths = list(lengths or [])

    # -- the Cache interface transformers' attention calls -------------------- #
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.k:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], key_states], dim=-2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], value_states], dim=-2)
        else:
            self.k[layer_idx] = key_states
            self.v[layer_idx] = value_states
        return self.k[layer_idx], self.v[layer_idx]

    def get_seq_length(self, layer_idx=None):
        if layer_idx is None:
            return next(iter(self.k.values())).shape[-2] if self.k else 0
        kv = self.k.get(layer_idx)
        return kv.shape[-2] if kv is not None else 0

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_max_length(self):
        return None

    # -- slot management ------------------------------------------------------ #
    @property
    def batch_size(self):
        return len(self.lengths)

    @property
    def padded_len(self):
        return self.get_seq_length(next(iter(self.k), None)) if self.k else 0

    @classmethod
    def from_single_caches(cls, caches, lengths):
        """Fuse N per-connection `SplitCache`s into one batched cache, left-padding each to
        the longest. This is how a request already in flight joins a batch."""
        assert len(caches) == len(lengths)
        target = max(lengths) if lengths else 0
        out = cls(lengths=list(lengths))
        layer_ids = sorted({li for c in caches for li in c.k})
        for li in layer_ids:
            ks, vs = [], []
            for c, ln in zip(caches, lengths):
                k, v = c.k.get(li), c.v.get(li)
                if k is None:
                    raise ValueError(f"slot missing layer {li}; caches must cover the same layers")
                pad = target - k.shape[-2]
                if pad > 0:
                    # Left-pad with zeros. The values are irrelevant -- build_mask() puts
                    # -inf on these positions, so they contribute exactly nothing to the
                    # softmax rather than "almost nothing".
                    zk = torch.zeros(k.shape[0], k.shape[1], pad, k.shape[3], dtype=k.dtype)
                    zv = torch.zeros(v.shape[0], v.shape[1], pad, v.shape[3], dtype=v.dtype)
                    k = torch.cat([zk, k], dim=-2)
                    v = torch.cat([zv, v], dim=-2)
                ks.append(k)
                vs.append(v)
            out.k[li] = torch.cat(ks, dim=0)
            out.v[li] = torch.cat(vs, dim=0)
        return out

    def split_to_single_caches(self):
        """Inverse of from_single_caches: hand each slot back its own unpadded SplitCache,
        so a request can leave the batch (finished, or its peer died) without disturbing the
        others."""
        outs = []
        for b, ln in enumerate(self.lengths):
            c = common.SplitCache()
            for li in self.k:
                total = self.k[li].shape[-2]
                start = total - ln          # strip the left padding
                c.k[li] = self.k[li][b:b + 1, :, start:, :].clone()
                c.v[li] = self.v[li][b:b + 1, :, start:, :].clone()
            outs.append(c)
        return outs


def build_mask(lengths, padded_len, q, dtype):
    """Additive attention mask `[B, 1, q, padded_len + q]`.

    Two jobs at once:
      * hide each slot's left padding (the `padded_len - length` junk positions), and
      * keep the causal rule among the q new tokens.

    Returns None only when there is nothing to hide, i.e. a single unpadded sequence doing
    one token -- in which case the plain unbatched path is already correct.
    """
    b = len(lengths)
    kv = padded_len + q
    neg = torch.finfo(dtype).min
    mask = torch.zeros(b, 1, q, kv, dtype=dtype)
    for i, ln in enumerate(lengths):
        pad = padded_len - ln
        if pad > 0:
            mask[i, :, :, :pad] = neg           # left padding is not a real token
    if q > 1:
        # causal among the new block: query j may not see new key j' > j
        for j in range(q):
            mask[:, :, j, padded_len + j + 1:] = neg
    return mask


@torch.no_grad()
def run_layers_batched(model, layers, hidden, cache, lengths):
    """`common._run_layers` for a batch whose slots have DIFFERENT history lengths.

    `hidden`  : [B, q, H]
    `lengths` : true (unpadded) history length per slot, before this call
    """
    b, q, _ = hidden.shape
    assert b == len(lengths), f"hidden batch {b} != {len(lengths)} slot lengths"
    padded_len = cache.padded_len

    # Absolute positions, per slot. This is what makes left-padding invisible to RoPE: a
    # slot's token is at position `length`, whatever index it occupies in the padded tensor.
    position_ids = torch.stack([
        torch.arange(ln, ln + q, dtype=torch.long) for ln in lengths
    ])
    attn_mask = build_mask(lengths, padded_len, q, hidden.dtype)
    # cache_position indexes the padded tensor, which is shared across slots.
    cache_position = torch.arange(padded_len, padded_len + q)

    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )[0]
    cache.lengths = [ln + q for ln in lengths]
    return hidden


@torch.no_grad()
def mid_stage_batched(model, lo, hi, hidden, cache, lengths):
    """Batched equivalent of common.mid_stage (a middle node's own layers)."""
    return run_layers_batched(model, list(model.model.layers)[lo:hi], hidden, cache, lengths)


@torch.no_grad()
def last_stage_batched(model, lo, hidden, cache, lengths):
    """Batched equivalent of common.last_stage (final layers + norm)."""
    hidden = run_layers_batched(model, list(model.model.layers)[lo:], hidden, cache, lengths)
    return model.model.norm(hidden)


# --------------------------------------------------------------------------- #
# The scheduler that replaces the module-level compute_lock
# --------------------------------------------------------------------------- #
class _Job:
    __slots__ = ("hidden", "cache", "length", "done", "out", "err")

    def __init__(self, hidden, cache, length):
        self.hidden = hidden
        self.cache = cache
        self.length = length
        self.done = threading.Event()
        self.out = None
        self.err = None


class MicroBatcher:
    """One worker thread owns the model; connection threads hand it work and wait.

    This is the direct replacement for `agent/node_server.py`'s module-level `compute_lock`.
    The lock made concurrency *safe* by making it non-existent -- N connections queued and
    each got the whole machine in turn. Here they queue too, but a whole batch of them is
    served by ONE forward pass, so N concurrent requests cost far less than N times one.

    `run_batched(hidden[B,q,H], batched_cache, lengths) -> out[B,q,H]` is supplied by the
    caller so the same scheduler serves the middle, last and probe roles.

    Caches are fused into a batch and split back每 step rather than kept resident. That costs
    a copy proportional to (batch x history), roughly 10-20% of the step at realistic sizes --
    paid because it keeps each connection owning a plain `SplitCache`, so a peer dying or a
    request finishing needs no slot bookkeeping and cannot corrupt anyone else's history.
    Making the batch resident is the obvious next optimisation, and a much easier one to get
    wrong.
    """

    def __init__(self, run_batched, max_batch=None, window_s=None):
        self.run_batched = run_batched
        self.max_batch = max_batch or MAX_BATCH
        self.window_s = window_s if window_s is not None else BATCH_WINDOW_S
        self._q = queue.Queue()
        self._stop = threading.Event()
        self.stats = {"passes": 0, "requests": 0, "max_batch_seen": 0}
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, hidden, cache, length):
        """Called from a connection thread. Blocks until this request's slice is computed."""
        job = _Job(hidden, cache, length)
        self._q.put(job)
        job.done.wait()
        if job.err is not None:
            raise job.err
        return job.out

    def stop(self):
        self._stop.set()

    # -- worker --------------------------------------------------------------- #
    def _collect(self):
        try:
            first = self._q.get(timeout=0.25)
        except queue.Empty:
            return None
        jobs = [first]
        # Only q=1 decode steps share a pass. A prefill (q>1) has a different shape and its
        # own causal structure, and is one call per request anyway -- batching it buys little
        # and is where ragged-batching bugs live. It runs alone, deliberately.
        if first.hidden.shape[1] == 1:
            deadline = time.perf_counter() + self.window_s
            while len(jobs) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=remaining)
                except queue.Empty:
                    break
                if nxt.hidden.shape[1] == 1:
                    jobs.append(nxt)
                else:
                    self._q.put(nxt)      # put the prefill back; it gets its own pass
                    break
        return jobs

    def _loop(self):
        while not self._stop.is_set():
            jobs = self._collect()
            if not jobs:
                continue
            try:
                self._run(jobs)
            except Exception as e:                      # never let one bad batch kill the node
                for j in jobs:
                    j.err = e
                    j.done.set()

    def _run(self, jobs):
        lengths = [j.length for j in jobs]
        bcache = BatchedCache.from_single_caches([j.cache for j in jobs], lengths)
        hidden = torch.cat([j.hidden for j in jobs], dim=0)
        out = self.run_batched(hidden, bcache, lengths)
        # Hand each connection back its own advanced cache, unpadded.
        for j, c in zip(jobs, bcache.split_to_single_caches()):
            j.cache.k, j.cache.v = c.k, c.v
        self.stats["passes"] += 1
        self.stats["requests"] += len(jobs)
        self.stats["max_batch_seen"] = max(self.stats["max_batch_seen"], len(jobs))
        for i, j in enumerate(jobs):
            j.out = out[i:i + 1]
            j.done.set()
