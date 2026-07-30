"""
NEURON — slice_downloader.py  (Session 8)

Download ONLY the model weights a node needs — its assigned transformer layers,
plus the embedding (first node) or final norm (last node) — instead of the whole
model. This is what makes the "1 MB agent" real: a node fetches ~0.8–1.4 GB, not
the full 3.09 GB.

WHY BYTE-RANGE, NOT SHARDS: Qwen2.5-1.5B-Instruct is a *single* `model.safetensors`
(3.09 GB) — it is NOT split into shard files, so there is no `index.json` and
`hf_hub_download` can only fetch the whole 3 GB. Instead we use the safetensors
format directly: the file starts with an 8-byte length + a JSON header listing
every tensor's exact byte range. We fetch just that header (~38 KB), pick the
tensors for this node's layers, and HTTP-Range download only those byte ranges
(HuggingFace serves `Accept-Ranges: bytes`). This is *more* granular than shards —
per tensor — and reassembles into a small, valid safetensors file the node loads.

Usage:
  # the command a fresh node runs (asks the coordinator what it owns):
  python slice_downloader.py --coordinator https://neuronnet.duckdns.org --node-id node_a --output-dir ./model_slice
  # or fully manual:
  python slice_downloader.py --model-id Qwen/Qwen2.5-1.5B-Instruct --layer-start 0 --layer-end 9 --first --output-dir ./slice_a
"""
import argparse
import json
import os
import struct
import sys

import requests

MODEL_ID_DEFAULT = "Qwen/Qwen2.5-1.5B-Instruct"
HF = "https://huggingface.co"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILES = ["config.json", "generation_config.json"]           # always
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]  # first node


# --------------------------------------------------------------------------- #
# safetensors header (Task 1: map layers -> tensors -> byte ranges)
# --------------------------------------------------------------------------- #
def resolve_url(model_id, filename, revision="main"):
    return f"{HF}/{model_id}/resolve/{revision}/{filename}"


def shard_files(model_id, revision="main"):
    """Which safetensors file(s) hold this model's weights.

    HuggingFace splits any checkpoint over ~5 GB into `model-0000k-of-0000N.safetensors` with
    a `model.safetensors.index.json` mapping tensor -> file. This module originally assumed a
    SINGLE `model.safetensors`, which is true of Qwen2.5-1.5B and false of every model big
    enough to actually need a distributed pipeline: Qwen2.5-7B 404s on the single-file name.
    So the byte-range slicing that makes the "1 MB agent" real did not work for any model
    NEURON exists to serve. Found while sizing the 7B test.

    Returns a list of filenames, single-element for an unsharded model.
    """
    idx = requests.get(resolve_url(model_id, "model.safetensors.index.json", revision),
                       timeout=30)
    if idx.status_code == 200:
        weight_map = idx.json().get("weight_map", {})
        # sorted() only for reproducible ordering; correctness comes from the map itself
        return sorted(set(weight_map.values()))
    return [WEIGHTS_FILE]


def fetch_file_header(model_id, filename, revision="main"):
    """Return (header_dict, data_start_offset, url) for ONE safetensors file, fetching only
    its ~40 KB header rather than the whole multi-GB file."""
    url = resolve_url(model_id, filename, revision)
    r = requests.get(url, headers={"Range": "bytes=0-7"}, timeout=30)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError(f"HF did not honour Range (status {r.status_code}); cannot slice")
    n = struct.unpack("<Q", r.content)[0]
    r2 = requests.get(url, headers={"Range": f"bytes=8-{8+n-1}"}, timeout=60)
    r2.raise_for_status()
    header = json.loads(r2.content)
    header.pop("__metadata__", None)
    return header, 8 + n, url


def fetch_header(model_id, revision="main"):
    """Whole-model view: (header_dict, data_start, url) merged across every shard.

    Each tensor's meta gains `_file` and `_data_start` so a caller can still resolve it to a
    byte range in the right shard. The returned `data_start`/`url` describe the FIRST shard
    and are kept only for backwards compatibility with single-file callers -- anything
    handling sharded models must use the per-tensor fields.
    """
    files = shard_files(model_id, revision)
    merged, first_start, first_url = {}, None, None
    for fn in files:
        header, start, url = fetch_file_header(model_id, fn, revision)
        if first_start is None:
            first_start, first_url = start, url
        for name, meta in header.items():
            meta = dict(meta)
            meta["_file"] = fn
            meta["_data_start"] = start
            merged[name] = meta
    return merged, first_start, first_url


def _layer_of(name):
    return int(name.split(".")[2]) if name.startswith("model.layers.") else None


def get_tensors_for_layers(header, layer_start, layer_end, is_first_node, is_last_node):
    """The single-file equivalent of 'get_shards_for_layers': which tensors this
    node needs. Returns {name: meta} (meta has dtype, shape, data_offsets)."""
    keep = {}
    for name, meta in header.items():
        L = _layer_of(name)
        want = (
            (L is not None and layer_start <= L <= layer_end)
            or (name == "model.embed_tokens.weight" and is_first_node)  # tied lm_head lives here
            # lm_head belongs to the FIRST node, not the last: since Session 3 the driver
            # runs the output head (common.apply_lm_head, load_model_shard(head=True)) and
            # the last node returns only its normed hidden. This said is_last_node, which was
            # invisible for every model served so far because Qwen2.5-1.5B has
            # tie_word_embeddings=True -- lm_head IS embed_tokens, which the first node
            # already gets. The first UNTIED model breaks it: Qwen2.5-7B ships a separate
            # 545M-param lm_head, so the driver would hold an uninitialized meta tensor and
            # every generation would fail. Caught while sizing the 7B test, not by any test.
            or (name == "lm_head.weight" and is_first_node)
            or (name == "model.norm.weight" and is_last_node)
        )
        if want:
            keep[name] = meta
    return keep


def _bytes(meta):
    b, e = meta["data_offsets"]
    return e - b


# --------------------------------------------------------------------------- #
# byte-range download + reassembly (Task 2)
# --------------------------------------------------------------------------- #
def _range_get(url, begin, end, progress=None):
    """GET bytes [begin, end). Streams; optionally reports bytes via progress()."""
    r = requests.get(url, headers={"Range": f"bytes={begin}-{end-1}"}, stream=True, timeout=600)
    r.raise_for_status()
    out = bytearray()
    for chunk in r.iter_content(chunk_size=8 << 20):
        out += chunk
        if progress:
            progress(len(chunk))
    return bytes(out)


def _merge_spans(items):
    """items: [(name, abs_begin, abs_end)] -> contiguous spans to minimise requests."""
    items = sorted(items, key=lambda x: x[1])
    spans = []
    for name, b, e in items:
        if spans and b == spans[-1][1]:
            spans[-1][1] = e
            spans[-1][2].append((name, b, e))
        else:
            spans.append([b, e, [(name, b, e)]])
    return spans


def _write_safetensors_slice(path, keep, raw):
    """Write a valid safetensors file containing only `keep` tensors, data from `raw`."""
    new_header, data_parts, offset = {}, [], 0
    for name in keep:                       # any consistent order; offsets stay contiguous
        blob = raw[name]
        new_header[name] = {
            "dtype": keep[name]["dtype"],
            "shape": keep[name]["shape"],
            "data_offsets": [offset, offset + len(blob)],
        }
        data_parts.append(blob)
        offset += len(blob)
    new_header["__metadata__"] = {"format": "pt", "neuron_slice": "true"}
    hjson = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for p in data_parts:
            f.write(p)


def _download_whole(model_id, filename, target_dir, revision="main"):
    r = requests.get(resolve_url(model_id, filename, revision), timeout=120)
    if r.status_code == 200:
        with open(os.path.join(target_dir, filename), "wb") as f:
            f.write(r.content)
        return len(r.content)
    return 0   # 404 etc. (e.g. a tokenizer file this model doesn't ship) -> skip


def download_slice(model_id, layer_start, layer_end, target_dir, is_first_node, is_last_node,
                   revision="main"):
    os.makedirs(target_dir, exist_ok=True)
    header, data_start, url = fetch_header(model_id, revision)
    keep = get_tensors_for_layers(header, layer_start, layer_end, is_first_node, is_last_node)

    total_t = len(header)
    sel_bytes = sum(_bytes(m) for m in keep.values())
    full_bytes = sum(_bytes(m) for m in header.values())
    print(f"Downloading {len(keep)} of {total_t} tensors (node layers {layer_start}-{layer_end})")
    print(f"Skipping {total_t - len(keep)} tensors (not needed for this node)")
    print(f"Total download: {sel_bytes/1e9:.2f} GB of {full_bytes/1e9:.2f} GB full model "
          f"({100*sel_bytes/full_bytes:.0f}%)")

    # Group by shard: a sharded model's tensors live in different files, so byte ranges are
    # only meaningful within one file. Unsharded models fall out as a single group.
    by_file = {}
    for n, m in keep.items():
        fn = m.get("_file", WEIGHTS_FILE)
        start = m.get("_data_start", data_start)
        by_file.setdefault(fn, []).append(
            (n, start + m["data_offsets"][0], start + m["data_offsets"][1]))

    total_spans = sum(len(_merge_spans(items)) for items in by_file.values())
    print(f"Fetching {total_spans} contiguous byte-span(s) across {len(by_file)} file(s):")

    raw, got = {}, [0]

    def prog(nb):
        got[0] += nb
        pct = 100 * got[0] / sel_bytes if sel_bytes else 100
        sys.stdout.write(f"\r  {got[0]/1e9:5.2f}/{sel_bytes/1e9:.2f} GB  ({pct:3.0f}%)")
        sys.stdout.flush()

    for fn, items in by_file.items():
        furl = resolve_url(model_id, fn, revision)
        for sb, se, members in _merge_spans(items):
            chunk = _range_get(furl, sb, se, progress=prog)
            for name, b, e in members:
                raw[name] = chunk[b - sb: e - sb]
    print()

    out_path = os.path.join(target_dir, WEIGHTS_FILE)
    _write_safetensors_slice(out_path, keep, raw)

    small = list(CONFIG_FILES) + (TOKENIZER_FILES if is_first_node else [])
    for fn in small:
        _download_whole(model_id, fn, target_dir, revision)
    print(f"Wrote slice -> {out_path}  (+ {', '.join(small)})")
    return {"path": out_path, "slice_bytes": sel_bytes, "full_bytes": full_bytes,
            "tensors": list(keep.keys()), "keep": keep}


# --------------------------------------------------------------------------- #
# verification (Task 3): every downloaded tensor is byte-identical to the full model
# --------------------------------------------------------------------------- #
def verify_slice(target_dir, model_id, keep):
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    full_path = hf_hub_download(model_id, WEIGHTS_FILE)   # already cached -> instant
    slice_path = os.path.join(target_dir, WEIGHTS_FILE)
    max_diff, checked = 0.0, 0
    with safe_open(slice_path, framework="pt") as fs, safe_open(full_path, framework="pt") as ff:
        got = set(fs.keys())
        missing = set(keep) - got
        for k in got:
            a, b = fs.get_tensor(k), ff.get_tensor(k)
            if a.shape != b.shape:
                max_diff = float("inf")
            elif a.numel():
                max_diff = max(max_diff, (a.float() - b.float()).abs().max().item())
            checked += 1
    ok = (not missing) and (max_diff == 0.0)
    print(f"[verify] {checked} tensors, all needed present: {not missing}, "
          f"max|delta| vs full model: {max_diff:.3e} -> {'IDENTICAL' if ok else 'MISMATCH'}")
    if missing:
        print(f"[verify] MISSING: {sorted(missing)[:5]}")
    return ok


def load_slice_model(target_dir):
    """Load the downloaded slice into a partial model (for a functional forward check)."""
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(target_dir)
    config._attn_implementation = "eager"
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(config)
    model.eval()
    # Storage dtype, NOT compute dtype -- common.WEIGHT_DTYPE defaults to fp32 (unchanged
    # behaviour) and can be set to fp16 to halve a node's resident RAM. cast_linears() then
    # makes every GEMM run in fp32 regardless, because these CPUs have no half-precision
    # GEMM ([P2]). This is what decides whether an 8B model fits on a 6 GB laptop.
    import common
    sd = {k: v.to(common.WEIGHT_DTYPE)
          for k, v in load_file(os.path.join(target_dir, WEIGHTS_FILE)).items()}
    model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()
    return common.cast_linears(model)


# --------------------------------------------------------------------------- #
# CLI (Task 5)
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Download only a node's model slice.")
    ap.add_argument("--coordinator", default=None, help="coordinator URL; asks it what this node owns")
    ap.add_argument("--node-id", default=None, help="node id to look up at the coordinator")
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--layer-start", type=int, default=None)
    ap.add_argument("--layer-end", type=int, default=None)
    ap.add_argument("--first", action="store_true", help="first node (also fetch embedding + tokenizer)")
    ap.add_argument("--last", action="store_true", help="last node (also fetch final norm)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    if args.coordinator and args.node_id:
        info = requests.get(f"{args.coordinator.rstrip('/')}/node/{args.node_id}/slice-info",
                            timeout=20).json()
        print("[coordinator slice-info]", json.dumps(info, indent=2))
        model_id = info["model_id"]
        ls, le = info["layer_start"], info["layer_end"]
        is_first = info.get("is_first_node", info.get("tokenizer_needed", ls == 0))
        is_last = info.get("is_last_node", info.get("norm_needed", False))
    else:
        if args.layer_start is None or args.layer_end is None:
            ap.error("provide --coordinator + --node-id, or --layer-start + --layer-end")
        model_id, ls, le = args.model_id, args.layer_start, args.layer_end
        is_first, is_last = args.first, args.last

    res = download_slice(model_id, ls, le, args.output_dir, is_first, is_last)
    if not args.no_verify:
        verify_slice(args.output_dir, model_id, res["keep"])


if __name__ == "__main__":
    main()
