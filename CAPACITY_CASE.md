# The capacity case — running a model no single machine here can hold

The claim NEURON exists to make, reduced to something runnable on two computers in one house:
**Qwen3-4B across the 12 GB Pavilion and the 8 GB node, with the 64 GiB OptiPlex switched out
of it.** Neither machine can hold the model. Together they can. That is a different product
from "a 1.5B model, slower than your laptop", and it is the reason someone would join.

Everything below is arithmetic and placement that has been run and tested. **No forward pass of
a 4B model has happened yet** — that is what this document is for.

---

## What was measured, not assumed

`tools/measure_model.py` reads the model's published safetensors header (~40 KB, no auth):

```bash
python tools/measure_model.py Qwen/Qwen3-4B-Instruct-2507 --against pavilion:12,node-b:8 --dtype fp16
```

| | |
|---|---|
| Layers | 36 × 100,930,816 params |
| Embedding | 388,956,160 params, **tied** (so `lm_head` is the embedding, counted once) |
| Total | 4,022,468,096 params |
| **At fp32** | **16.09 GB** |
| **At fp16 storage** | **8.04 GB** |
| Licence | Apache-2.0, ungated |

## fp32 does not fit, and that is arithmetic

The two machines hold 20 GB between them. After the 3 GB OS reserve and the 25% headroom the
coordinator budgets 10.5 GB — and even at zero reserve it is 16.09 GB into 20 GB with nothing
left for two OSes, two Python processes, the KV cache, or CastLinear's transient fp32 copy.

The coordinator refuses it, and the refusal is correct:

```
fp32: REFUSED - no node can hold stage 1
```

**At fp16 storage it fits with room.** Compute stays fp32 throughout — `cast_linears` keeps
every GEMM in fp32 because these CPUs have no half-precision GEMM ([P2]) — so this halves
resident bytes without halving the arithmetic. `test_weight_dtype.py` has already measured it:
2 B/param resident, checksum drift under 1e-2, ~2.9× slower at batch 1 falling to ~1.6× at
batch 8.

| node | layers | weights | budget |
|---|---|---|---|
| pavilion (driver) | 0–17 | 3.63 GB + 0.78 GB head = **4.41 GB** | 6.75 GB |
| node-b (tail) | 18–35 | **3.63 GB** | 3.75 GB |

Neither machine can hold all 36 layers alone: the Pavilion tops out at 29, node-b at 18.

## Downloads

Per-tensor byte ranges off the bf16 checkpoint, so the wire cost is the same at either storage
dtype:

| node | range | download |
|---|---|---|
| pavilion | 0–17 + embedding + tokenizer | **4.41 GB** |
| node-b | 18–35 + final norm | **3.63 GB** |

(8.04 GB total — the whole model, split, because at two nodes there is no redundancy to pay
for.)

---

## Running it

**Nothing here needs `NEURON_S1`.** It used to: stage-1 width was a global constant read from
the environment at import in both the coordinator and every driver, so changing it meant a
coordinated restart across machines. The 4b tier now declares `stage1_layers: 18` and the
driver derives its width from the shard the coordinator told it to fetch ([P44]).

### 1. Deploy the coordinator

The tier, the per-model stage-1 width, the `weight_dtype` column and the head-cost sizing all
live in the coordinator. None of it is on the VM yet.

```bash
bash coordinator/deploy.sh
```

### 2. Put both nodes on fp16 storage

On the Pavilion and on node-b, before the agent starts:

```bash
export NEURON_WEIGHT_DTYPE=fp16
```

This is what the coordinator sizes them by. Without it they report `fp32`, are budgeted at
4 bytes/param, and the model is refused — correctly, because at fp32 it genuinely does not fit.

### 3. Take the OptiPlex out of the roster

With a 64 GiB machine present there is no capacity case, only a big node: it can hold all 36
layers by itself even at fp32. Stop its agent, or the coordinator will place the model on it.

### 4. Pin the model

The 4b tier is `manual_only` — the capacity ladder will never promote to it, because at
`min_nodes` 2 the live 3-node network clears the promote margin and would migrate production
onto an experiment on a health sweep. Reach it deliberately:

```bash
curl -X POST "$COORDINATOR/network/model" -H "Content-Type: application/json" -H "X-Register-Secret: $NEURON_REGISTER_SECRET" -d '{"model_id":"Qwen/Qwen3-4B-Instruct-2507"}'
```

`POST /network/model` (guarded by `require_register_secret`) stores `pinned_model_id` and the
nodes migrate on the next health sweep — download the slice, report ready, cut over together.
**No node has to be touched**, which is the point: a volunteer's PC is behind a NAT and
"restart the agent" is not an instruction this product can give.

To hand the decision back to the capacity ladder afterwards, post `{"model_id": null}`.

### 5. Watch it place

```bash
curl -s "$COORDINATOR/status" | python -m json.tool
```

Expect `chain_ranges [[0,17],[18,35]]`, `expected_stage1 [0,17]`, `stage1_ok true`,
`routable true`. The agents will each download their slice and report ready; cutover follows
the usual prepare → ready → cutover handshake.

### 6. Then measure it

The thing none of the above establishes:

- tokens/sec end to end, against the 1.5B baseline on the same two machines
- resident RSS on each node, against the 4.41 / 3.63 GB predicted here
- whether the answers are coherent

---

## What can still go wrong

- **A driver shard that predates a width change is stale, and nothing refreshes it while the
  agent runs.** The migration handshake covers a node's compute slice, not the driver shard —
  that is a separate download loaded once per process. The symptom is every request refused
  with *"this driver holds 0..N, restart the agent to re-fetch it"*. Restarting the agent
  fixes it. ([P44], still open.)
- **The tail is assigned even when it does not fit.** `canonical_assignment` covers the last
  range whatever the node's cap says, because a gap means not one request completes. It is no
  longer silent — the repair log names the node and how far over it is — but it is still
  assigned. Watch for `[repair] WARNING`.
- **`total_earned` only grows and emission has never been reconciled** ([P40]). Running a new
  model does not change that.

## Sources

- `PROBLEMS.md` [P43] (sizing), [P44] (placement and stage-1 width)
- `sessions.md` Session 60
- `tools/measure_model.py` for the measurement, `coordinator/model_tiers.py` for the tier
