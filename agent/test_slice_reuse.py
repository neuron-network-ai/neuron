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


def _fake_slice(path, lo, hi):
    """A minimal safetensors file: 8-byte LE header length, then the JSON header."""
    header = {f"model.layers.{i}.self_attn.q_proj.weight":
              {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
              for i in range(lo, hi + 1)}
    header["model.norm.weight"] = {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}
    blob = json.dumps(header).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * 16)


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-slice-")

    w = os.path.join(tmp, "s1", "model.safetensors")
    _fake_slice(w, 19, 27)
    check("reads the layer range out of the safetensors header",
          agentmod.Agent.slice_layers_on_disk(w) == (19, 27))
    check("a truncated/garbage file returns None rather than raising",
          agentmod.Agent.slice_layers_on_disk(os.path.join(tmp, "nope.safetensors")) is None)

    # --- ensure_slice: matching range keeps the slice, mismatch re-downloads --------- #
    def make_agent(slice_lo, slice_hi):
        cfgp = os.path.join(tmp, f"cfg-{slice_lo}-{slice_hi}.json")
        cfg = dict(agentmod.DEFAULT_CONFIG)
        cfg.update(node_id="n", node_token="t", slice_dir=f"./slice-{slice_lo}-{slice_hi}/")
        json.dump(cfg, open(cfgp, "w"))
        a = agentmod.Agent(config_path=cfgp)
        _fake_slice(os.path.join(agentmod.HERE, f"slice-{slice_lo}-{slice_hi}",
                                 "model.safetensors"), slice_lo, slice_hi)
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
