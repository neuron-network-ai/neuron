"""agent/test_slice_reuse.py — run: python -m agent.test_slice_reuse

ensure_slice() used to skip the download whenever *a* model.safetensors existed, without
checking which layers were in it. So a node re-placed on a different segment (delete
config.json and re-register, and the coordinator hands you whichever gap needs filling — not
the range you had before) reused the previous segment's weights while claiming the new range.

Nothing catches that locally. The node answers confidently with wrong activations, fails
proof-of-compute, and is eventually flagged — with nothing anywhere saying why. The layer range
is now read out of the safetensors header on disk, so the bytes decide, not a config claim.
"""
import json
import os
import struct
import sys
import tempfile

from agent import agent as agentmod

ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def _fake_slice(path, lo, hi, embed=True, norm=True, tokenizer=True, model_id="m"):
    """A minimal safetensors file: 8-byte LE header length, then the JSON header.

    Writes the provenance marker too, because a real `download_slice` does. Reuse now requires
    knowing WHICH MODEL a slice is for -- layer numbers cannot say, and comparing only layer
    numbers is what let a node serve 7B weights as 1.5B after a migration ([P36]). A fixture
    that omitted the marker would be testing a slice no download ever produces.
    `model_id=None` simulates a slice downloaded before the marker existed.
    """
    header = {f"model.layers.{i}.self_attn.q_proj.weight":
              {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
              for i in range(lo, hi + 1)}
    if norm:
        header["model.norm.weight"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    if embed:
        header["model.embed_tokens.weight"] = {"dtype": "F32", "shape": [2, 2],
                                               "data_offsets": [0, 16]}
    blob = json.dumps(header).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * 16)
    if tokenizer:
        open(os.path.join(os.path.dirname(path), "tokenizer.json"), "w").write("{}")
    if model_id is not None:
        with open(os.path.join(os.path.dirname(path), "neuron_slice.json"), "w") as f:
            json.dump({"model_id": model_id, "layer_start": lo, "layer_end": hi}, f)


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-slice-")

    w = os.path.join(tmp, "s1", "model.safetensors")
    _fake_slice(w, 19, 27)
    check("reads the layer range out of the safetensors header",
          agentmod.Agent.slice_layers_on_disk(w) == (19, 27))
    check("a truncated/garbage file returns None rather than raising",
          agentmod.Agent.slice_layers_on_disk(os.path.join(tmp, "nope.safetensors")) is None)

    # --- ensure_slice: matching range keeps the slice, mismatch re-downloads --------- #
    def make_agent(slice_lo, slice_hi, dirname=None, **slice_kw):
        dirname = dirname or f"slice-{slice_lo}-{slice_hi}"
        cfgp = os.path.join(tmp, f"cfg-{dirname}.json")
        cfg = dict(agentmod.DEFAULT_CONFIG)
        cfg.update(node_id="n", node_token="t", slice_dir=f"./{dirname}/")
        json.dump(cfg, open(cfgp, "w"))
        a = agentmod.Agent(config_path=cfgp)
        _fake_slice(os.path.join(agentmod.HERE, dirname, "model.safetensors"),
                    slice_lo, slice_hi, **slice_kw)
        return a

    downloads = []
    real_dl = agentmod.slice_downloader.download_slice
    agentmod.slice_downloader.download_slice = \
        lambda mid, lo, hi, d, **k: downloads.append((lo, hi))

    try:
        a = make_agent(19, 27)
        a.ensure_slice({"model_id": "m", "layer_start": 19, "layer_end": 27,
                        "estimated_download_gb": 0.8, "is_first_node": False,
                        "is_last_node": True})
        check("a slice matching the assigned range is reused (no re-download)", downloads == [])

        b = make_agent(19, 27)
        b.cfg["slice_dir"] = "./slice-19-27/"
        b.ensure_slice({"model_id": "m", "layer_start": 0, "layer_end": 9,
                        "estimated_download_gb": 1.4, "is_first_node": True,
                        "is_last_node": False})
        check("a slice for the WRONG range is discarded and re-downloaded",
              downloads == [(0, 9)])

        # ---- a slice that CONTAINS the new range is reused, not re-fetched ------ #
        # This is the fix for the thing that made a stranger's 8 GB PC download ~20 GB in one
        # afternoon: after a re-split, a node has usually already got a superset of whatever it
        # is asked for next. The loader fills a full model skeleton and the node runs only
        # layers[lo:hi], so extra layers cost resident RAM and nothing else.
        downloads.clear()
        c = make_agent(0, 27)                       # holds the whole model
        c.ensure_slice({"model_id": "m", "layer_start": 14, "layer_end": 27,
                        "estimated_download_gb": 1.4, "is_first_node": False,
                        "is_last_node": True})
        check("a slice covering the new range is reused with NO download", downloads == [])

        downloads.clear()
        c.ensure_slice({"model_id": "m", "layer_start": 0, "layer_end": 13,
                        "estimated_download_gb": 1.4, "is_first_node": True,
                        "is_last_node": False})
        check("...including when the node becomes the FIRST stage", downloads == [])

        # ---- but never reuse one that is missing what the new role needs -------- #
        downloads.clear()
        d = make_agent(0, 27, dirname="slice-nonorm", norm=False)   # a middle's slice: no norm
        d.ensure_slice({"model_id": "m", "layer_start": 14, "layer_end": 27,
                        "estimated_download_gb": 1.4, "is_first_node": False,
                        "is_last_node": True})
        check("a slice without the final norm is NOT reused for the last stage",
              downloads == [(14, 27)])

        downloads.clear()
        e = make_agent(0, 27, dirname="slice-noembed", embed=False)  # no embedding
        e.ensure_slice({"model_id": "m", "layer_start": 0, "layer_end": 13,
                        "estimated_download_gb": 1.4, "is_first_node": True,
                        "is_last_node": False})
        check("a slice without the embedding is NOT reused for the first stage",
              downloads == [(0, 13)])

        downloads.clear()
        g = make_agent(0, 27, dirname="slice-notok", tokenizer=False)  # no tokenizer
        g.ensure_slice({"model_id": "m", "layer_start": 0, "layer_end": 13,
                        "estimated_download_gb": 1.4, "is_first_node": True,
                        "is_last_node": False})
        check("a slice without the tokenizer is NOT reused for the first stage",
              downloads == [(0, 13)])

        downloads.clear()
        h = make_agent(10, 18)                      # only partly covers what is wanted
        h.ensure_slice({"model_id": "m", "layer_start": 5, "layer_end": 20,
                        "estimated_download_gb": 1.4, "is_first_node": False,
                        "is_last_node": False})
        check("a slice that only PARTLY covers the new range is re-downloaded",
              downloads == [(5, 20)])
    finally:
        agentmod.slice_downloader.download_slice = real_dl
        for d in os.listdir(agentmod.HERE):
            if d.startswith("slice-"):
                import shutil
                shutil.rmtree(os.path.join(agentmod.HERE, d), ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
