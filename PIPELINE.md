# NEURON — The N-stage pipeline

**Status:** design. Nothing here is built yet.
**Why this file exists:** the model NEURON can serve is capped by how many machines a request
can be routed through, and that number is currently **3**. Everything below is about lifting it.

---

## The problem, stated plainly

`coordinator/model_tiers.py` declares a 70b tier: Qwen2.5-72B, 80 layers, `min_nodes: 20`.

That tier can never run. The inference path handles exactly 3 stages. So the tier ladder
promises something the pipeline cannot deliver, and the gap gets worse the bigger the model:

| model | ÷ 3 stages | ÷ N stages (int8) | machines needed at ~2.5 GB usable each |
|-------|-----------|-------------------|----------------------------------------|
| 1.5B  | 0.9 GB/node  | — | 3 is fine |
| 7B    | 5 GB/node    | — | 3 works on 8 GB machines |
| 72B   | **48 GB/node** | 0.88 GB/layer | ~20 machines, ~4 layers each |
| ~200B | **66 GB/node** | ~1.7 GB/layer | ~80 machines, 1–2 layers each |

The founder's question was: *"if the model grows to 200B, do users download 100 GB onto a 4 GB
machine?"* With 3 stages, effectively yes, and it is impossible. With N stages a 4 GB machine
holds **one layer**, downloads ~1.7 GB once, and never touches the rest. The machine was never
the constraint. The stage count is.

This is also the ROADMAP's own promise — *"models larger than anything a single machine could
run"*, on *"any laptop"*. A 3-stage cap means the model can only ever be as large as three
volunteers' machines, which is the opposite of the stated goal.

---

## Why it caps at 3 — and why you cannot just delete the cap

### The cap is one line

`agent/node_server.py:164`. When a node acts as a MIDDLE relay it configures its next hop with:

```python
common.send_msg(bconn, {"type": "config", "s2": s2, "n": msg.get("n", self.n), "wire": ...})
```

No `host_b`. And `node_server.py:160` decides a node's role with `if "host_b" in msg`. So the
next hop is structurally told *"you are the end"*. A middle cannot describe a further hop
because the wire has no field for one. `host_b` is a single scalar address, not a list.

### Three reasons deleting that line is not enough

**1. Failure is silent, not loud.** Role is *inferred*, never declared
(`node_server.py:154,160,172,176`). A node that gets no `host_b` becomes LAST only if
`self.hi == self.n - 1`. Any *other* node in that position falls through to the PROBE branch and
returns an **un-normed** hidden state. The driver reads only `resp["hidden"]`
(`neuron_driver.py:278`), applies `lm_head`, and streams **fluent nonsense**. No exception is
raised anywhere. A naive 4th node would not crash the network — it would quietly corrupt it.

> **This happened, and it did not need a 4th node** ([P55], 2026-08-19). It needed TWO: the
> driver holding 0-9 and one node holding 10-27. The driver's config carried an `s1`, the
> last-stage branch had been narrowed to `"s1" not in msg` to keep a full-model node from
> answering a verifier's probe ([P49]), and so every real request fell through to PROBE and came
> back un-normed. The driver applied `lm_head` and streamed exactly the fluent nonsense
> described above, for a day, billed at ~0.13 NRN a request, while proof-of-compute — which
> only exercises the probe path — reported the node healthy on 5662 passed challenges. The
> paragraph above was written before it happened and describes it precisely. **Declaring the
> role is not a nicety for N-hop chains; the inference was already wrong at N=2.** `stage` and
> `probe` are now on the wire for that reason, which is a down payment on the `role` field this
> design proposes.

**2. `s2` means two different things.** In the driver's config it is *the last stage's start*
(`node_a.py:123` returns `last["layers"][0]`). Inside a middle it is *that middle's own
exclusive end* (`node_server.py:161`, `:206`). Those are the same number **only when there is
exactly one middle**. With two middles they diverge and stages silently run the wrong layers.

**3. Nodes run the caller's claimed range, not the range they downloaded.** `node_server.py:161`
and `:173` take `s1`/`s2` from the message. Only the PROBE role uses its own `self.lo/self.hi`.
Safe while the driver is the only thing that computes ranges; a correctness hazard the moment an
intermediate node originates that field.

### What is already fine

- **`coordinator/router.py`** — `build_chain`'s cursor walk already returns chains of any length,
  and `/infer` applies no cap. The cap is entirely client-side, in `node_a.py:112`.
- **Crash recovery survives any N.** This was the big unknown and it came back clean. The chain
  is a feed-forward composition with exactly one external input — the driver's own stage output.
  Replaying it into a fresh chain rebuilds *every* downstream KV cache regardless of hop count.
  `junction_cache` memory is O(tokens × H) with **no N term**.
- **Teardown (`bye`) is already recursive** and works for any N unchanged.
- **Framing** (8-byte length + payload) is N-agnostic.
- **The relay** splices raw bytes and is protocol-agnostic, so NAT'd nodes can already be dialled
  by *any* peer — hops 2..N-1 need no relay change.
- **MicroBatcher** keys per `(role, lo, hi)`, so one machine serving several ranges already works.

---

## The two costs that actually bound chain length

Memory is not the limit. These are.

### Latency: 2×(N−1) traversals per token

The reply walks back **hop by hop, synchronously** (`node_server.py:209-212`). A middle blocks
on its downstream socket while holding the upstream connection open, then forwards. So one token
costs `2(N-1)` network traversals, not `N`.

At N=80 on home internet (~30 ms/hop) that is ~5 s **per token**. Unusable.

**Fix:** have the last stage reply **directly to the driver**. The driver still sends one tensor
into the chain, so the one-junction recovery property is untouched. Cost drops to ~N traversals.

### Error: compounds twice per hop

Every relay **decodes with its downstream codec and re-encodes with its upstream codec** on the
way back, because codecs are negotiated independently per hop (`node_server.py:158,168,211`).
That is ~`2(N-1)` lossy round-trips per token.

`wire_codec.py:44` measures i8h at `rel_l2 0.0037` per hop, and `:46-48` already rejects int4
because *"~9% relative error per hop ... is not fine for the 20-node chain that a 70B model
implies"*. 198 i8h steps in quadrature lands around 5% — the same neighbourhood.

**Fix:** negotiate **one codec for the whole chain**, and let relays forward the return payload's
**raw bytes untouched** instead of round-tripping the tensor. Then error is paid once per
direction, not once per hop. Longer term, codec choice should be a function of chain depth.

### Realistic ceiling

With both fixes, ~N traversals and error paid once: **80 hops ≈ 2.4 s/token**. Slow, but real.
Without them, the practical ceiling is ~10 hops.

---

## Design

### Wire protocol

Replace the single `host_b`/`port_b` scalar with an ordered remainder list, and make role and
range **explicit** rather than inferred:

```python
{
  "type": "config",
  "lo": 10, "hi": 19,          # THIS node's own half-open range. Never derived from a peer's.
  "n": 80,                     # total layers, propagated from the driver
  "role": "middle",            # "middle" | "last" | "probe" — declared, never inferred
  "reply_to": {"host": ..., "port": ...},   # the driver, for the direct return
  "wire": "i8h",               # ONE codec for the whole chain, chosen by the driver
  "downstream": [              # the rest of the chain, verbatim
    {"host": ..., "port": ..., "lo": 20, "hi": 29},
    ...
  ]
}
```

A node takes `downstream[0]` as its next hop and forwards `downstream[1:]` unchanged. An empty
list means it is the tail. Each node **cross-checks `lo`/`hi` against its own loaded
`self.lo`/`self.hi+1` and refuses with `{"ok": false}` on mismatch** — the ack already carries
those fields (`node_server.py:192`) and `proof_of_compute.py:93-95` already checks them.

Keep `if "host_b" in msg` as a **legacy branch first**, so a half-upgraded fleet still runs
3-stage chains during rollout. Same compatibility discipline `wire_codec.negotiate` already uses.

### Driver

`coord_get_chain` returns the **full ordered hop list** instead of the flattened 9-tuple:

```python
[{"node_id": ..., "host": ..., "port": ..., "lo": ..., "hi": ...}, ...]
```

It validates only that `chain[0]` matches this driver's own shard, and that ranges are contiguous
and cover `0..n-1`. Callers to update: `neuron_driver.py:153-155`, the chain dict at `:171-174`,
`_reroute` at `:239-244`, and `node_a.py:195-196`.

### Coordinator

`config.PIPELINE_STAGES` (added as a stop-gap) goes away, replaced by a **memory-derived** stage
count: given the model's `gb_per_layer` and each node's usable RAM, compute how many layers each
node can hold and make the chain exactly as long as it needs to be.

---

## Build order

Each step is independently useful and testable. The live network keeps working throughout.

| # | Step | Proves | Risk to live network |
|---|------|--------|----------------------|
| 1 | Explicit `role` + `lo`/`hi` in config; node refuses on range mismatch. Legacy branch retained. | Kills the silent-corruption failure mode **before** anything gets longer | None — 3-stage path unchanged, new fields ignored by old nodes |
| 2 | `downstream` list; middles forward the remainder | 4+ stages route correctly. Test with 4 local processes | None until a 4th node exists |
| 3 | Direct reply to driver | Halves per-token latency | Medium — touches the hot path; needs `test_node_death.py` re-run |
| 4 | One chain-wide codec, pass-through relay | Stops error compounding; makes 20+ hops viable | Medium |
| 5 | Memory-derived stage count in the coordinator | 72B tier becomes reachable with 20 machines | Low |

**Step 1 first, and on its own.** It is the one that converts a whole class of silent wrong
answers into a loud refusal. Everything after it is safer because of it.

---

## Honest limits

- **200B needs ~80 machines.** NEURON has 4. No architecture fixes that — the network has to
  grow. What this design does is stop *the code* from being the thing that blocks it.
- **~2.4 s/token at 80 hops**, best case, after both latency fixes. Fine for a chat answer,
  not fine for anything interactive.
- **Error at 80 hops is unmeasured.** The i8h figure is per-hop; the accumulation estimate is
  arithmetic, not experiment. Needs a real measurement before trusting a long chain.
- **`MAX_REROUTES = 3` does not scale with N** (`neuron_driver.py:56`). Going 3 → 20 stages
  multiplies the chance of at least one node dying mid-request by roughly 7×, while the retry
  budget stays at 3. Separate fix, tracked in RESILIENCE.md [R4].
- **A reroute costs O(N) re-prefills.** At N=20, one laptop lid closing re-prefills 20 machines,
  19 of which were healthy. Partial recovery is provably impossible with one junction — no node
  stores the activations it forwarded, only K/V.
- **`test_node_death.py` proves recovery only for the 3-stage shape** with a constant split
  (`S1, S2 = 10, 19`). It cannot detect any N-stage hazard. Needs widening alongside step 2.
- **`test_short_chain.py:90-95` asserts the 4-stage rejection as intended behaviour.** Widening
  the chain is a deliberate contract change, not a bug fix, and that test must be rewritten.

---

## Prior art

Petals (arXiv:2209.01188), already cited in `router.py` and `wire_codec.py`, solves this at
scale: servers hold blocks, clients build routes per request, heterogeneous machines and
mid-chain failures are normal. The main divergence: NEURON routes **centrally** through the
coordinator; Petals is decentralised. The remaining design work — what to adopt and what not to —
was cut short by a session limit and is the obvious next investigation.
