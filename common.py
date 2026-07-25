"""
NEURON — shared code.

Model split across N machines by transformer layer. Session 5 generalises the
split to an arbitrary chain of contiguous layer ranges:
    first  (node_a): embed_tokens + layers[0:s1] + lm_head
    middle (node_c): layers[s1:s2]                        (0+ middle nodes)
    last   (node_b): layers[s2:n] + final norm
Data flows first -> middle(s) -> last, then the last node's normed hidden comes
back to node_a, which applies lm_head and picks the token.

Session 3: the output head (lm_head over the ~152k vocab) lives on node_a. The
last node returns its normed hidden state; node_a applies lm_head. No token echo
back is needed — each node's KV cache is updated purely by running its layers on
the incoming hidden.

Session 2 change: each node keeps its OWN KV cache and processes ONE token at a
time during decode, so attention is not recomputed from scratch every step.

We no longer lean on the HF base-model forward (its cache/length bookkeeping
keys off layer 0, which node_b doesn't own). Instead we drive the decoder
layers directly:
  - compute rotary embeddings from absolute positions we track ourselves,
  - build a causal mask only for the multi-token prefill (a single decode token
    attends to all cached keys, so it needs no mask),
  - hand each layer a per-node DynamicCache keyed by the layer's native index.

Both nodes track their position counter in lockstep, so their rotary/positions
always agree — which is what keeps the split numerically identical to the whole
model (proved bit-exact by selftest.py).
"""

import io
import struct

import torch

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
# Session 3 goal 2: bf16 halves RAM but these CPUs have no bf16 GEMM (no AVX512-BF16/
# AMX), so it was several-x SLOWER in testing. fp32 stays the default on CPU.
DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        attn_implementation="eager",   # predictable, explicit mask handling
        low_cpu_mem_usage=True,        # keep the load peak down (matters on the desktop)
    )
    model.eval()
    return tok, model


def num_layers(model):
    return len(model.model.layers)


def load_model_shard(lo, hi, embed=False, norm=False, head=False):
    """Load ONLY layers[lo:hi] (+ optional embed/norm/head) — the 'light node' idea.

    Roles in the pipeline chain:
      first  (node_a): lo=0, embed=True, head=True   (+ its layers)
      middle (node_c): just its layers[lo:hi]
      last   (node_b): hi=n, norm=True               (+ its layers)
    `head=True` also loads the embedding weight (lm_head is tied to embed_tokens).

    The model is built on the meta device (no memory); only the wanted tensors are
    materialized from the safetensors shards. Unused layers stay on `meta` and are
    never executed. Returns (tok, model, n_layers).
    """
    import glob
    import os

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    config._attn_implementation = "eager"
    n = config.num_hidden_layers
    keep = set(range(lo, hi))

    # buffers (e.g. rotary inv_freq) are created real on CPU; only params -> meta
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config)
    model.eval()

    def want(key):
        if key.startswith("model.layers."):
            return int(key.split(".")[2]) in keep
        if key == "model.embed_tokens.weight":
            return embed or head          # lm_head is tied to the embedding weight
        if key == "model.norm.weight":
            return norm
        if key == "lm_head.weight":
            return head
        return False

    local_dir = snapshot_download(MODEL_ID, allow_patterns=["*.safetensors", "*.json"])
    sd = {}
    for fp in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        with safe_open(fp, framework="pt") as f:
            for key in f.keys():
                if want(key):
                    sd[key] = f.get_tensor(key).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    if head or embed:
        model.tie_weights()   # points lm_head at the (real) embedding weight
    return tok, model, n


class SplitCache:
    """A KV cache keyed by the layer's native index.

    HF's DynamicCache assumes layers fill sequentially from 0, which breaks on
    node_b (its layers are 14..27). This dict-keyed cache accepts any layer_idx,
    so each node caches exactly its own layers. The attention module calls
    update() and expects the full (past+current) K/V back for that layer.
    """

    def __init__(self):
        self.k = {}
        self.v = {}

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
            if not self.k:
                return 0
            return next(iter(self.k.values())).shape[-2]
        kv = self.k.get(layer_idx)
        return kv.shape[-2] if kv is not None else 0

    def get_usable_length(self, new_seq_length, layer_idx=0):
        # past length already cached for this specific layer (unbounded cache)
        return self.get_seq_length(layer_idx)

    def get_max_length(self):
        return None


def new_cache():
    return SplitCache()


# --------------------------------------------------------------------------- #
# Manual layer driver
# --------------------------------------------------------------------------- #
def _causal_mask(q, past_len, dtype, device):
    """Additive mask for a q-token block that already has `past_len` cached keys.
    Returns None for a single decode token (it may attend to everything)."""
    if q == 1:
        return None
    kv = past_len + q
    min_val = torch.finfo(dtype).min
    m = torch.full((q, kv), min_val, dtype=dtype, device=device)
    for i in range(q):
        m[i, : past_len + i + 1] = 0.0     # query i (abs pos past_len+i) sees keys 0..past_len+i
    return m[None, None, :, :]


@torch.no_grad()
def _run_layers(model, layers, hidden, cache, past_len):
    # transformers 4.44.2 Qwen2: each attention computes rotary itself from
    # position_ids, so we only supply position_ids + cache_position here.
    device = hidden.device
    q = hidden.shape[1]
    total = past_len + q
    position_ids = torch.arange(past_len, total, device=device).unsqueeze(0)
    attn_mask = _causal_mask(q, past_len, hidden.dtype, device)
    cache_position = torch.arange(past_len, total, device=device)
    for layer in layers:
        hidden = layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_value=cache,
            use_cache=True,
            cache_position=cache_position,
        )[0]
    return hidden


@torch.no_grad()
def first_stage(model, hi, token_ids, cache, past_len):
    """node_a: embed + layers[0:hi]  ->  hidden [1, q, H]."""
    hidden = model.model.embed_tokens(token_ids)
    return _run_layers(model, list(model.model.layers)[0:hi], hidden, cache, past_len)


@torch.no_grad()
def mid_stage(model, lo, hi, hidden, cache, past_len):
    """node_c (middle): layers[lo:hi]  ->  hidden [1, q, H]."""
    return _run_layers(model, list(model.model.layers)[lo:hi], hidden, cache, past_len)


@torch.no_grad()
def last_stage(model, lo, hidden, cache, past_len):
    """node_b (last): layers[lo:] + final norm  ->  normed hidden [1, q, H].
    No lm_head here — node_a owns the head (Session 3)."""
    hidden = _run_layers(model, list(model.model.layers)[lo:], hidden, cache, past_len)
    return model.model.norm(hidden)


@torch.no_grad()
def apply_lm_head(model, hidden):
    """normed hidden [1, q, H] -> next-token logits [1, vocab] (last position).

    Applies lm_head over the whole block then takes the last row, matching the
    full model's GEMM shape exactly (slicing to [1,1,H] first would change the
    matmul shape and perturb the logits by ~1e-5). Decode passes q=1 anyway, so
    the hot path is unaffected; only prefill does the (one-time) full-width head.
    """
    return model.lm_head(hidden)[:, -1, :]


# --------------------------------------------------------------------------- #
# Length-prefixed tensor framing over a raw TCP socket
# --------------------------------------------------------------------------- #
def send_msg(sock, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    sock.sendall(struct.pack(">Q", len(data)))
    sock.sendall(data)


def _recv_all(sock, n):
    chunks, got = [], 0
    while got < n:
        b = sock.recv(min(n - got, 1 << 20))
        if not b:
            raise ConnectionError("socket closed mid-message")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def recv_msg(sock):
    (n,) = struct.unpack(">Q", _recv_all(sock, 8))
    data = _recv_all(sock, n)
    return torch.load(io.BytesIO(data), weights_only=False)
