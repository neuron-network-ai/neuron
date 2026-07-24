"""NEURON coordinator — per-node slice info (Session 8).

Computes exactly which model weights a node must download, from the single
safetensors header. Qwen2.5-1.5B is one `model.safetensors` (not sharded), so the
"slice" is a set of byte-ranges within that one file. Uses only stdlib + requests
(no torch), and caches the header so it is fetched from HuggingFace just once.
"""
import json
import struct
import threading

import requests

_HF = "https://huggingface.co"
_cache = {}
_lock = threading.Lock()


def _fetch_header(model_id, revision="main"):
    url = f"{_HF}/{model_id}/resolve/{revision}/model.safetensors"
    r = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    n = struct.unpack("<Q", r.content)[0]
    r2 = requests.get(url, headers={"Range": f"bytes=8-{8+n-1}"}, timeout=60)
    r2.raise_for_status()
    h = json.loads(r2.content)
    h.pop("__metadata__", None)
    return h


def get_header(model_id, revision="main"):
    key = (model_id, revision)
    with _lock:
        if key not in _cache:
            _cache[key] = _fetch_header(model_id, revision)
        return _cache[key]


def _layer_of(name):
    return int(name.split(".")[2]) if name.startswith("model.layers.") else None


def slice_info(model_id, layer_start, layer_end, total_layers, revision="main"):
    header = get_header(model_id, revision)
    is_first = (layer_start == 0)
    is_last = (layer_end == total_layers - 1)
    sel, full, ntensors = 0, 0, 0
    for name, meta in header.items():
        b, e = meta["data_offsets"]
        size = e - b
        full += size
        L = _layer_of(name)
        keep = (
            (L is not None and layer_start <= L <= layer_end)
            or (name == "model.embed_tokens.weight" and is_first)   # tied lm_head
            or (name == "lm_head.weight" and is_last)
            or (name == "model.norm.weight" and is_last)
        )
        if keep:
            sel += size
            ntensors += 1
    return {
        "model_id": model_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "shards_needed": ["model.safetensors"],   # single-file model, sliced by byte-range
        "tensors_needed": ntensors,
        "tokenizer_needed": is_first,
        "lm_head_needed": is_first,                # NEURON: head on the first node (tied to embedding)
        "norm_needed": is_last,
        "is_first_node": is_first,
        "is_last_node": is_last,
        "estimated_download_gb": round(sel / 1e9, 3),
        "full_model_gb": round(full / 1e9, 3),
    }
