"""
NEURON — junction_cache.py

Petals mechanism 3: **a node dropping out must not kill the request.**

This is the gap `PETALS_NOTES.md` calls essential-not-an-optimization, and the thing
`PROBLEMS.md` [P4] describes from the other end: the Pavilion suspends when idle, the chain
goes incomplete, and the request dies. On a volunteer network node churn is the normal case,
not the exception -- somebody closes their laptop lid mid-answer.

WHY A CACHE IS REQUIRED, and not just a reconnect
-------------------------------------------------
Every node keeps the attention K/V for its own layers. A REPLACEMENT node has an empty cache,
so it cannot simply pick up at token 40 -- it has no memory of tokens 0..39. Reconnecting
alone produces confident garbage. To recover you have to give the replacement the same
history the dead node had.

NEURON's pipeline shape makes this cheaper than in Petals. The driver sends one tensor into
the chain (the hidden after its own layers) and node_c relays it onward, so there is exactly
ONE junction to cache: the driver -> chain boundary. Replaying that one sequence rebuilds
every downstream node's cache, because each activation flows through all of them again.

Replay is also cheap because it goes as ONE concatenated block instead of N sequential
messages. For causal attention, running a block of n tokens with past_len=0 gives exactly the
same result as running n single tokens in order -- `common._run_layers` builds the causal mask
from past_len, so the arithmetic is identical. So recovery costs about one prefill of the
tokens so far, not N round trips.

Stored in fp16, which halves the memory for free: `bench_wire.py` measured fp16 activations
as producing token-identical output to fp32 across every prompt tested (max Δlogit 0.0069).
At H=1536 a 500-token answer costs ~1.5 MB; at H=8192 (a 70B model) ~8 MB. That is the whole
price of not losing generations.
"""
import torch


# A hard ceiling, because this grows with the answer length and lives on the user's machine.
# 256 MB is far more than any real chat turn (H=8192 x 16k tokens is 256 MB) and small enough
# that a runaway loop cannot eat a laptop. On overflow we stop caching rather than raise: a
# request that can no longer be recovered is still a request worth finishing.
DEFAULT_MAX_BYTES = 256 << 20


class JunctionCache:
    """The activations this driver has sent into the chain, in order, for one request."""

    def __init__(self, max_bytes=DEFAULT_MAX_BYTES):
        self._blocks = []
        self._bytes = 0
        self._tokens = 0
        self.max_bytes = max_bytes
        self.overflowed = False

    def add(self, hidden):
        """Record one activation ([1, q, H]) that is about to be sent into the chain."""
        self._tokens += int(hidden.shape[1])
        if self.overflowed:
            return
        blk = hidden.detach().to(torch.float16)
        nbytes = blk.numel() * 2
        if self._bytes + nbytes > self.max_bytes:
            # Keep what we have but stop growing, and remember that replay is no longer
            # complete -- recoverable() goes False so the driver fails honestly instead of
            # replaying a truncated history and producing quiet nonsense.
            self.overflowed = True
            return
        self._blocks.append(blk)
        self._bytes += nbytes

    @property
    def tokens(self):
        return self._tokens

    @property
    def nbytes(self):
        return self._bytes

    def recoverable(self):
        """True if a replacement chain can be brought fully up to date from this cache."""
        return bool(self._blocks) and not self.overflowed

    def replay_block(self):
        """Everything sent so far as one [1, n, H] fp32 tensor, ready to hand a fresh chain.

        Returns None if nothing has been sent yet (nothing to rebuild) or if the cache
        overflowed (see `recoverable`).
        """
        if not self.recoverable():
            return None
        return torch.cat([b.to(torch.float32) for b in self._blocks], dim=1)

    def clear(self):
        self._blocks.clear()
        self._bytes = 0
        self._tokens = 0
        self.overflowed = False
