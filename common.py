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
import os
import struct

import torch

import wire_codec

# The model this node stack loads. Env-overridable so the whole stack (drivers, node_a/b/c,
# selftests) can be pointed at another model without code changes — matches how the
# coordinator's config.MODEL_ID works. Any Llama-family HF model runs through the same code.
MODEL_ID = os.environ.get("NEURON_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
# Session 3 goal 2: bf16 halves RAM but these CPUs have no bf16 GEMM (no AVX512-BF16/
# AMX), so it was several-x SLOWER in testing. fp32 stays the COMPUTE dtype on CPU.
DTYPE = torch.float32

# ...but compute dtype and STORAGE dtype do not have to match, and conflating them is what
# put the RAM wall where it is. HuggingFace ships these models as BF16 (2.00 bytes/param);
# `load_slice_model` then upcasts to fp32, doubling resident memory to 4.00 bytes/param for
# no quality gain — the extra bits are zeros. Measured consequence: Llama-3.1-8B needs 9.3 GB
# per node on a 3-way split, against 6 GB free on the Pavilion and 5 GB on the OptiPlex, so
# the model that proves NEURON's whole premise does not fit. At fp16 storage it is 4.7 GB and
# it fits on all three.
#
# `fp16` keeps weights half-precision in RAM and casts each Linear's weight to fp32 at
# forward time (see CastLinear). The cast is transient — one weight matrix at a time, and
# amortised across a whole batch — so peak memory stays near the fp16 figure while every
# GEMM still runs in fp32, which is the only dtype these CPUs are fast at.
WEIGHT_DTYPE = {"fp32": torch.float32, "fp16": torch.float16,
                "bf16": torch.bfloat16}[os.environ.get("NEURON_WEIGHT_DTYPE", "fp32").lower()]

# Socket timeouts for the pipeline's raw TCP hops. COLD_CONNECT_TIMEOUT_S covers connect
# + the config handshake, which blocks behind the peer's one-time model-shard load
# (measured ~35s cold on this hardware); every client socket used to default that phase's
# timeout to only 30s, so a legitimately-slow-but-working cold start would trip a timeout,
# and the peer's own uncaught TimeoutError (see node_c/node_b handle()) would slam the
# socket shut mid-handshake -- the far end then saw a plain "socket closed mid-message"
# with no clue it was actually a cold-start race. HOT_TIMEOUT_S is applied to the socket
# right after the config ack, once the peer is warm, so a genuinely dead/hung peer during
# real token generation still fails fast instead of hanging for two minutes.
COLD_CONNECT_TIMEOUT_S = 120
HOT_TIMEOUT_S = 30


# --------------------------------------------------------------------------- #
# Execution device (Session 42)
# --------------------------------------------------------------------------- #
# Where a node's layers actually run. CUDA when the machine has it, CPU otherwise, and
# `NEURON_DEVICE` overrides both (useful to force a GPU machine back onto CPU for an A/B).
#
# Two invariants hold the rest of the system together, and both are load-bearing:
#
#   1. **Every public interface stays CPU.** The stage functions take CPU tensors and return
#      CPU tensors, exactly as before. Moving to the device happens inside them and the
#      result is moved back. So `wire_codec`, `batching`, `junction_cache`, the relay and
#      `security/proof_of_compute` are all untouched by this — none of them can receive a
#      CUDA tensor and none of them had to learn about devices.
#   2. **On a CPU-only machine nothing changes at all.** `Tensor.to()` returns *self* when the
#      device and dtype already match, so every call added below is a no-op on CPU — not a
#      copy, not a new tensor. That is what lets `selftest_shard.py` still prove bit-exactness.
#
# TF32 is disabled deliberately. On Ampere and later, cuBLAS will silently run fp32 matmuls at
# ~10-bit mantissa precision, which drifts from the CPU result by ~1e-3. That is still inside
# proof-of-compute's atol=0.05, but it eats a fifth of the honest/cheat budget for nothing —
# and PoC's separation (honest ~1e-5, cheating ~25) is the mechanism that lets strangers earn.
def _resolve_device():
    override = os.environ.get("NEURON_DEVICE", "").strip()
    if override:
        try:
            return torch.device(override)
        except (RuntimeError, ValueError):
            pass                      # a typo must not stop the node starting
    try:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
    except Exception:
        pass                          # a broken driver is a CPU node, not a crash
    return torch.device("cpu")


DEVICE = _resolve_device()

if DEVICE.type == "cuda":
    try:
        torch.backends.cuda.matmul.allow_tf32 = False     # see note above
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def device_name():
    """Human-readable device string for logs, e.g. 'cuda:0 (NVIDIA GeForce RTX 4070)'."""
    if DEVICE.type != "cuda":
        return str(DEVICE)
    try:
        return f"{DEVICE} ({torch.cuda.get_device_name(DEVICE)})"
    except Exception:
        return str(DEVICE)


def _to_device(t):
    """Move a tensor onto the execution device. Identity on CPU-only machines."""
    return t if t is None or t.device == DEVICE else t.to(DEVICE)


def _to_cpu(t):
    """Bring a tensor back to CPU for the wire. Identity on CPU-only machines."""
    return t if t is None or t.device.type == "cpu" else t.cpu()


def move_model_to_device(model, device=None):
    """Move a *partially materialized* shard onto `device`, leaving meta tensors alone.

    `model.to(device)` cannot be used here and the reason is specific: `load_model_shard`
    builds the whole architecture on the `meta` device and materializes only this node's
    layers, so most parameters are still meta. `.to()` walks all of them and raises
    `NotImplementedError: Cannot copy out of meta tensor` on the first one it reaches.

    Tied weights are re-tied afterwards. Replacing parameters one at a time breaks the
    embedding/lm_head tie, and on a 152k-vocab model that silently doubles the largest single
    allocation on the node — 0.9 GB of VRAM, on the exact machines least likely to have it
    spare. `tie_weights()` restores the shared storage.
    """
    device = DEVICE if device is None else device
    if device.type == "cpu":
        return model                       # nothing to do, and no traversal cost
    for mod in model.modules():
        for name, p in list(mod.named_parameters(recurse=False)):
            if p is not None and p.device.type != "meta":
                setattr(mod, name, torch.nn.Parameter(p.data.to(device),
                                                      requires_grad=False))
        for name, b in list(mod.named_buffers(recurse=False)):
            if b is not None and b.device.type != "meta":
                mod.register_buffer(name, b.to(device), persistent=False)
    try:
        model.tie_weights()
    except Exception:
        pass                               # models without tied weights simply have none
    return model


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(model_id=MODEL_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        attn_implementation="eager",   # predictable, explicit mask handling
        low_cpu_mem_usage=True,        # keep the load peak down (matters on the OptiPlex)
    )
    model.eval()
    return tok, model


def num_layers(model):
    return len(model.model.layers)


def load_model_shard(lo, hi, embed=False, norm=False, head=False, model_id=MODEL_ID):
    """Load ONLY layers[lo:hi] (+ optional embed/norm/head) — the 'light node' idea.

    `model_id` selects which model's slice to load (defaults to the module MODEL_ID, so
    existing callers are unchanged). Any Llama-family HF model — Llama/Qwen/Mistral —
    shares the module layout the stage functions below use, so it loads through this same
    path; other architectures need an adapter (see the arch-adapter seam).

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

    try:
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception:
        tok = None   # middle/last nodes never decode — only the driver needs the tokenizer
    config = AutoConfig.from_pretrained(model_id)
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

    local_dir = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.json"])
    sd = {}
    for fp in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        with safe_open(fp, framework="pt") as f:
            for key in f.keys():
                if want(key):
                    sd[key] = f.get_tensor(key).to(WEIGHT_DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    if head or embed:
        model.tie_weights()   # points lm_head at the (real) embedding weight
    model = move_model_to_device(cast_linears(model))
    print(f"[neuron] shard layers {lo}-{hi - 1} loaded on device: {device_name()}",
          flush=True)
    return tok, model, n


class CastLinear(torch.nn.Module):
    """An nn.Linear whose weight lives in half precision and is cast to fp32 per forward.

    Why not just run the GEMM in fp16? Because [P2] measured bf16/fp16 compute at ~8x SLOWER
    on these CPUs — they have no half-precision GEMM, so PyTorch emulates it. The trick is to
    separate the two dtypes: pay 2 bytes/param in RAM, and still hand the CPU the fp32 matmul
    it is fast at.

    The fp32 copy is transient and covers ONE weight matrix at a time (the largest in an 8B
    Llama layer is ~235 MB), so peak memory sits near the fp16 total rather than the fp32
    one. It is also amortised across a batch: the cast happens once per forward regardless of
    how many requests ride in it, so the wider the batch the less it costs per request —
    which composes with batching.py rather than fighting it.
    """

    def __init__(self, linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = torch.nn.Parameter(linear.weight.data, requires_grad=False)
        b = getattr(linear, "bias", None)
        self.bias = None if b is None else torch.nn.Parameter(b.data.float(), requires_grad=False)

    def forward(self, x):
        # x is cast too, not just the weight. The embedding table is stored half-precision
        # as well, so the very first hidden state arriving from embed_tokens is fp16 and
        # F.linear refuses mixed dtypes. Activations are tiny next to a weight matrix
        # ([B, q, H] against [H, 4H]), so this cast is free -- and it makes the module
        # correct on any input rather than only on the paths that happen to hand it fp32.
        return torch.nn.functional.linear(x.float(), self.weight.float(), self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"stored={self.weight.dtype}, compute=torch.float32")


def cast_linears(model):
    """Swap every nn.Linear for a CastLinear, in place. No-op when weights are already fp32.

    Deliberately leaves embeddings, norms and rotary buffers alone: embeddings are a gather
    (no GEMM, so no dtype problem) and norms are tiny. Only the big weight matrices matter
    for RAM, and only they need the cast.
    """
    if WEIGHT_DTYPE is torch.float32:
        return model
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(mod, name, CastLinear(child))
    return model


def resident_bytes(model):
    """Actual bytes of unique parameter storage — what a node really costs its owner in RAM.
    Deduplicated by storage pointer so tied embed/lm_head weights are not double-counted."""
    seen, total = set(), 0
    for p in model.parameters():
        s = p.untyped_storage()
        if s.data_ptr() in seen:
            continue
        seen.add(s.data_ptr())
        total += s.nbytes()
    return total


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
    # The hidden state arrives from the wire on CPU; the weights live on DEVICE. Everything
    # below (mask, position_ids, the K/V cache) then follows hidden.device, so this one move
    # is what puts the whole layer stack on the GPU. No-op on a CPU-only machine.
    hidden = _to_device(hidden)
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
    # .to(DTYPE) so the hidden state is fp32 from the very first op even when the embedding
    # table is stored half-precision. Everything downstream -- the K/V cache, the wire, the
    # residual stream -- then stays fp32, and only the big weight matrices are half.
    hidden = model.model.embed_tokens(_to_device(token_ids)).to(DTYPE)
    out = _run_layers(model, list(model.model.layers)[0:hi], hidden, cache, past_len)
    return _to_cpu(out)


@torch.no_grad()
def mid_stage(model, lo, hi, hidden, cache, past_len):
    """node_c (middle): layers[lo:hi]  ->  hidden [1, q, H].

    Returns a CPU tensor whatever device it computed on — the caller's next move is
    `send_msg`, and the wire codec quantizes on CPU."""
    out = _run_layers(model, list(model.model.layers)[lo:hi], hidden, cache, past_len)
    return _to_cpu(out)


@torch.no_grad()
def last_stage(model, lo, hidden, cache, past_len):
    """node_b (last): layers[lo:] + final norm  ->  normed hidden [1, q, H].
    No lm_head here — node_a owns the head (Session 3)."""
    hidden = _run_layers(model, list(model.model.layers)[lo:], hidden, cache, past_len)
    return _to_cpu(model.model.norm(hidden))


@torch.no_grad()
def apply_lm_head(model, hidden):
    """normed hidden [1, q, H] -> next-token logits [1, vocab] (last position).

    Applies lm_head over the whole block then takes the last row, matching the
    full model's GEMM shape exactly (slicing to [1,1,H] first would change the
    matmul shape and perturb the logits by ~1e-5). Decode passes q=1 anyway, so
    the hot path is unaffected; only prefill does the (one-time) full-width head.

    `hidden` arrives from the last stage over the wire, i.e. on CPU, while lm_head lives on
    DEVICE — the largest GEMM in the pipeline (152k x H) and so the one most worth running on
    a GPU. Logits come back on CPU because the caller's next step is argmax + detokenize.
    """
    return _to_cpu(model.lm_head(_to_device(hidden))[:, -1, :])


# --------------------------------------------------------------------------- #
# Length-prefixed tensor framing over a raw TCP socket
# --------------------------------------------------------------------------- #
# A peer may be anywhere in the fleet's upgrade cycle, so the outer framing (8-byte
# big-endian length + payload) never changes and BOTH payload formats are accepted on read:
#   - `wire_codec` frames (magic NRNW) -- pure JSON header + raw tensor bytes, and 2-4x
#     smaller. What we send once the peer has said it understands them.
#   - legacy `torch.save` pickles -- what every node spoke before Session 21.
# `codec=None` means "send the legacy format", which is what the config handshake uses so an
# un-upgraded peer can still read it. See wire_codec.negotiate.
#
# A cap on the declared length, because these sockets are reachable from the open internet
# (relay public ports, Session 12): an unvalidated 8-byte length let a stray scanner make a
# node allocate an arbitrary buffer. 512 MB is far above any real activation (a 8192-wide
# prefill of 4k tokens in fp32 is 134 MB) and far below "kills a 1 GB VM".
MAX_MSG_BYTES = 512 << 20


def send_msg(sock, obj, codec=None):
    if codec:
        data = wire_codec.encode(obj, codec)
    else:
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
    if n > MAX_MSG_BYTES:
        raise ConnectionError(f"declared message size {n} exceeds {MAX_MSG_BYTES} -- refusing")
    data = _recv_all(sock, n)
    if wire_codec.is_frame(data):
        return wire_codec.decode(data)
    # weights_only=True is the security fix, not a tidy-up: this used to be False, which
    # hands the sender arbitrary code execution on this machine (pickle calls whatever
    # __reduce__ says to call). Every node's port is reachable by whoever is next in the
    # chain -- and since Session 12 that port is published on a public relay -- so on an
    # open-join network of strangers this was an unauthenticated RCE in both directions.
    # Verified against every message shape the pipeline actually sends: config, config-ack,
    # act, act-reply and bye all carry only dict/str/int/float/bool/Tensor, all of which
    # weights_only=True allows.
    return torch.load(io.BytesIO(data), weights_only=True)
