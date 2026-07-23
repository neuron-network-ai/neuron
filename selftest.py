"""
Local correctness checks (Session 2), run on one machine with one model load:

  1. PREFILL SPLIT is bit-exact: stage_a -> stage_b == full model forward.
  2. KV CACHE is correct: cached A->B greedy generation produces the SAME tokens
     as brute-force no-cache generation (recomputing the whole sequence each step).
"""
import torch
import common
from common import new_cache

tok, model = common.load_model()
n = common.num_layers(model)
split = n // 2
ids = tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True, return_tensors="pt",
)
print(f"model={common.MODEL_ID}  layers={n}  split={split}  prompt_len={ids.shape[1]}")

# --- 1) bit-exact prefill split -------------------------------------------- #
with torch.no_grad():
    logits_full = model(input_ids=ids, use_cache=False).logits[:, -1, :]
ca, cb = new_cache(), new_cache()
h = common.first_stage(model, split, ids, ca, 0)
logits_split = common.apply_lm_head(model, common.last_stage(model, split, h, cb, 0))
diff = (logits_full - logits_split).abs().max().item()
print(f"[1 prefill] hidden={tuple(h.shape)} "
      f"top_full={int(logits_full.argmax(-1))} top_split={int(logits_split.argmax(-1))} "
      f"max|delta|={diff:.3e} -> {'MATCH' if diff == 0 else 'MISMATCH'}")

# --- 2) cache correctness --------------------------------------------------- #
STEPS = 8

# cached A->B path (both stages share one position counter; they stay in lockstep)
ca, cb = new_cache(), new_cache()
past = 0
h = common.first_stage(model, split, ids, ca, past)
logits = common.apply_lm_head(model, common.last_stage(model, split, h, cb, past))
past += ids.shape[1]
t = int(logits.argmax(-1))
cached = [t]
for _ in range(STEPS - 1):
    h = common.first_stage(model, split, torch.tensor([[t]]), ca, past)
    logits = common.apply_lm_head(model, common.last_stage(model, split, h, cb, past))
    past += 1
    t = int(logits.argmax(-1))
    cached.append(t)

# reference: brute-force no-cache greedy on the full model
seq = ids.clone()
ref = []
for _ in range(STEPS):
    with torch.no_grad():
        lg = model(input_ids=seq, use_cache=False).logits[:, -1, :]
    t = int(lg.argmax(-1))
    ref.append(t)
    seq = torch.cat([seq, torch.tensor([[t]])], dim=1)

print(f"[2 cache  ] cached={cached}")
print(f"[2 cache  ] ref   ={ref}")
print(f"[2 cache  ] -> {'MATCH' if cached == ref else 'MISMATCH'}")
print(f"           decoded: {tok.decode(cached, skip_special_tokens=True)!r}")

ok = (diff == 0) and (cached == ref)
print("RESULT:", "ALL PASS" if ok else "FAILED")
