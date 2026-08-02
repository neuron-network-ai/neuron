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


def _file_header(model_id, filename, revision="main"):
    """The safetensors header of ONE weight file: 8-byte LE length, then that much JSON."""
    url = f"{_HF}/{model_id}/resolve/{revision}/{filename}"
    r = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    n = struct.unpack("<Q", r.content)[0]
    r2 = requests.get(url, headers={"Range": f"bytes=8-{8+n-1}"}, timeout=60)
    r2.raise_for_status()
    h = json.loads(r2.content)
    h.pop("__metadata__", None)
    return h


def _weight_files(model_id, revision="main"):
    """Which safetensors file(s) hold this model's weights.

    Only Qwen2.5-1.5B was ever exercised here, and HF ships it as ONE model.safetensors -- so
    this module hardcoded that name. Every larger model is SHARDED
    (model-00001-of-0000N.safetensors plus a model.safetensors.index.json), and the hardcoded
    URL simply 404s. That surfaced the moment the tier ladder promoted the network to
    Qwen2.5-7B: `/node/{id}/slice-info` returned 502 for every node, so nobody could learn what
    to download, nobody could serve the new model, and the chain could not heal. A tier the
    download path cannot fetch is a tier that takes the network down when it is reached.
    `slice_downloader.shard_files()` already did this properly; the coordinator did not.
    """
    idx = requests.get(f"{_HF}/{model_id}/resolve/{revision}/model.safetensors.index.json",
                       timeout=30)
    if idx.status_code == 200:
        files = sorted(set(idx.json().get("weight_map", {}).values()))
        if files:
            return files
    return ["model.safetensors"]


def _fetch_header(model_id, revision="main"):
    """Merged header across every shard. Each tensor keeps its own file's offsets, which is
    all slice_info needs (it only sums sizes); the downloader resolves real byte ranges."""
    header = {}
    for fn in _weight_files(model_id, revision):
        for name, meta in _file_header(model_id, fn, revision).items():
            meta = dict(meta)
            meta["_file"] = fn
            header[name] = meta
    return header


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
    sel, full, ntensors, files = 0, 0, 0, set()
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
            files.add(meta.get("_file", "model.safetensors"))
    return {
        "model_id": model_id,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "total_layers": total_layers,
        # the file(s) this node's tensors actually live in — one entry for a single-file
        # model, several for a sharded one. Was hardcoded to the single-file name.
        "shards_needed": sorted(files) or ["model.safetensors"],
        "tensors_needed": ntensors,
        "tokenizer_needed": is_first,
        "lm_head_needed": is_first,                # NEURON: head on the first node (tied to embedding)
        "norm_needed": is_last,
        "is_first_node": is_first,
        "is_last_node": is_last,
        "estimated_download_gb": round(sel / 1e9, 3),
        "full_model_gb": round(full / 1e9, 3),
    }
