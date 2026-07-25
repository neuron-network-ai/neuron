"""
Verify the 3-STAGE sharded chain matches the full model:
  bit-exact prefill + correct KV-cache generation.
Chain: node_a (embed+layers[0:s1]+head) -> node_c (layers[s1:s2])
       -> node_b (layers[s2:n]+norm) -> node_a lm_head.
Loads full(reference) + 3 shards in one process (needs some RAM).
"""
import torch
import common
from common import new_cache

S1, S2 = 10, 19

print("loading full reference model ...")
tok, full = common.load_model()
print("loading shard A (embed + layers[0:s1] + head) ...")
_, ma, n = common.load_model_shard(0, S1, embed=True, head=True)
print("loading shard C (layers[s1:s2]) ...")
_, mc, _ = common.load_model_shard(S1, S2)
print("loading shard B (layers[s2:n] + norm) ...")
_, mb, _ = common.load_model_shard(S2, n, norm=True)
print(f"model={common.MODEL_ID} layers={n} | A=0..{S1-1} C={S1}..{S2-1} B={S2}..{n-1}")


def chain(block, ca, cc, cb, past):
    """Run the full 3-stage chain (all stages share one position counter)."""
    h1 = common.first_stage(ma, S1, block, ca, past)
    h2 = common.mid_stage(mc, S1, S2, h1, cc, past)
    h3 = common.last_stage(mb, S2, h2, cb, past)
    return common.apply_lm_head(ma, h3)


ids = tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True, return_tensors="pt",
)

# 1) bit-exact prefill
with torch.no_grad():
    logits_full = full(input_ids=ids, use_cache=False).logits[:, -1, :]
ca, cc, cb = new_cache(), new_cache(), new_cache()
logits_split = chain(ids, ca, cc, cb, 0)
diff = (logits_full - logits_split).abs().max().item()
print(f"[1 prefill] max|delta|={diff:.3e} -> {'MATCH' if diff == 0 else 'MISMATCH'}")

# 2) cache correctness: 8-token chain vs brute-force full no-cache greedy
STEPS = 8
ca, cc, cb = new_cache(), new_cache(), new_cache()
past = 0
logits = chain(ids, ca, cc, cb, past)
past += ids.shape[1]
t = int(logits.argmax(-1))
cached = [t]
for _ in range(STEPS - 1):
    logits = chain(torch.tensor([[t]]), ca, cc, cb, past)
    past += 1
    t = int(logits.argmax(-1))
    cached.append(t)

seq = ids.clone()
ref = []
for _ in range(STEPS):
    with torch.no_grad():
        lg = full(input_ids=seq, use_cache=False).logits[:, -1, :]
    t = int(lg.argmax(-1))
    ref.append(t)
    seq = torch.cat([seq, torch.tensor([[t]])], dim=1)

print(f"[2 cache  ] cached={cached}")
print(f"[2 cache  ] ref   ={ref}  -> {'MATCH' if cached == ref else 'MISMATCH'}")
print(f"           decoded: {tok.decode(cached, skip_special_tokens=True)!r}")
print("RESULT:", "ALL PASS" if (diff == 0 and cached == ref) else "FAILED")
