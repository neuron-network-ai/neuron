"""tools/test_measure_model.py - run: python tools/test_measure_model.py

Offline. Every network call is replaced by a fixture, deliberately: a measurement tool whose
tests need huggingface.co to be up is a tool nobody runs before committing a tier row.

The property that matters is a ROUND TRIP. `measure_model` reads a header and emits
`gb_per_layer` / `head_gb`; the balancer reads those and decides whether a machine can hold a
slice. If the two disagree about the basis those figures are on, the numbers still look
plausible and the network is sized wrong -- which is exactly what happened when the tier table
was written on an fp16 basis and every node stored fp32. So the test does not check the tool's
arithmetic against a second copy of the same arithmetic. It feeds the tool's own output back
into the coordinator's own solver and checks the answer against a footprint computed from the
fixture's raw bytes.

Also pinned: a restricted licence emits NO row. Printing one "for reference" is how a
restricted model gets pasted into the table by somebody who trusted the output over the
warning.
"""
import io
import json
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coordinator import balancer
from tools import measure_model as mm

ok = fail = 0

# A 4-layer toy in bf16, at REAL magnitudes: Qwen3-4B's own per-layer and embedding sizes,
# rounded. Round numbers keep it hand-checkable; the magnitudes keep the layer caps in the
# range a real machine produces, so an assertion like "29 layers on 12 GB" means something.
# A toy at 1000 params/layer passes the same equality with a cap of three million layers,
# which proves the arithmetic and demonstrates nothing about the sizes it will meet.
LAYERS, LP, EMBED, NORM, BPP = 4, 100_000_000, 389_000_000, 2_560, 2


def _fixture(tied=True, uniform=True):
    h, off = {}, 0

    def add(name, params):
        nonlocal off
        h[name] = {"dtype": "BF16", "shape": [params], "data_offsets": [off, off + params * BPP],
                   "_file": "model.safetensors", "_data_start": 8}
        off += params * BPP

    for i in range(LAYERS):
        # a fat last layer when uniform=False, so the "sized on the LARGEST" rule is exercised
        add(f"model.layers.{i}.mlp.down_proj.weight",
            LP if (uniform or i < LAYERS - 1) else LP * 2)
    add("model.embed_tokens.weight", EMBED)
    if not tied:
        add("lm_head.weight", EMBED)
    add("model.norm.weight", NORM)
    return h


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"HTTP {self.status_code}")


def install(header, license_id="apache-2.0", config_status=200):
    """Point the tool at the fixture instead of huggingface.co."""
    cfg = {"num_hidden_layers": LAYERS, "hidden_size": 8, "intermediate_size": 16,
           "vocab_size": EMBED, "tie_word_embeddings": "lm_head.weight" not in header}

    def fake_get(url, **kw):
        if "/api/models" in url:
            return _Resp({"cardData": {"license": license_id}} if license_id else {"tags": []})
        return _Resp(cfg, config_status)

    mm.requests.get = fake_get
    mm.sd.fetch_header = lambda model_id, revision="main": (header, 8, "x")


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    # ---------- 1. the header is read, not the config ---------------------------
    install(_fixture(tied=True))
    m = mm.measure("toy/tied")
    check("counts params per layer from the byte ranges", m["layer_params"] == LP)
    check("counts the embedding as the driver's head", m["head_params"] == EMBED)
    check("a tied model has no separate lm_head, so the head is counted once",
          m["tied_observed"] and m["head_tensors"] == ["model.embed_tokens.weight"])
    check("the final norm is reported apart from the head (it is the LAST node's)",
          m["norm_params"] == NORM)
    check("total params covers every tensor",
          m["total_params"] == LAYERS * LP + EMBED + NORM)
    check("the checkpoint's own bytes/param is measured, not assumed",
          abs(m["file_bytes_per_param"] - BPP) < 1e-9)

    install(_fixture(tied=False))
    u = mm.measure("toy/untied")
    check("an UNTIED model charges the driver for lm_head as well",
          u["head_params"] == EMBED * 2 and not u["tied_observed"],
          "Qwen2.5-7B ships a separate 545M-param lm_head; missing it under-charges the "
          "driver by the size of a vocab projection")

    install(_fixture(uniform=False))
    nu = mm.measure("toy/ragged")
    check("a ragged checkpoint is sized on its LARGEST layer, not its mean",
          nu["layer_params"] == LP * 2 and not nu["layer_params_uniform"])

    # ---------- 2. the round trip: tool figures -> coordinator's own solver ------
    # The one that matters. Computed from the fixture's RAW BYTES, then compared against what
    # the balancer concludes from the tool's emitted figures. A basis mismatch fails here.
    install(_fixture(tied=True))
    m = mm.measure("toy/tied")
    gpl, hgb = mm.gb(m["layer_params"]), mm.gb(m["head_params"])
    check("emitted figures are on the fp16 basis balancer.effective_gb scales from",
          gpl == m["layer_params"] * balancer.TIER_BASIS_BYTES / 1e9)

    for dtype, bpp in (("fp16", 2.0), ("fp32", 4.0)):
        n = {"node_id": "m", "ram_gb": 12.0, "ms_per_layer": 12.0,
             "status": "online", "eligible": True, "weight_dtype": dtype}
        budget = (12.0 - balancer.RAM_OS_RESERVE_GB) * 0.75
        # what the machine can hold, from the fixture's bytes, with no reference to the tool
        raw_layer_gb = m["layer_params"] * bpp / 1e9
        raw_head_gb = m["head_params"] * bpp / 1e9
        expect = int((budget - raw_head_gb) / raw_layer_gb)
        check(f"at {dtype} the balancer agrees with the fixture's raw bytes ({expect} layers)",
              balancer.max_layers_for(n, gpl, head_gb=hgb) == expect,
              f"balancer said {balancer.max_layers_for(n, gpl, head_gb=hgb)}")

    # ---------- 3. the capacity verdict is the coordinator's, not a copy ---------
    machines = mm.parse_machines("big:12,small:8")
    check("--against builds a roster the balancer accepts",
          [x["node_id"] for x in machines] == ["big", "small"]
          and machines[0]["ram_gb"] == 12.0)
    check("a malformed --against is refused rather than silently sized at 0 GB",
          _raises(lambda: mm.parse_machines("big")))
    check("an empty --against is refused too", _raises(lambda: mm.parse_machines(" ,")))

    out = io.StringIO()
    with redirect_stdout(out):
        mm.report(m, machines, "fp16")
    text = out.getvalue()
    short = balancer.capacity_shortfall(
        [dict(x, weight_dtype="fp16") for x in machines], LAYERS, gpl, hgb)
    check("the printed verdict is the one balancer.capacity_shortfall gives",
          ("VERDICT: YES" in text) == (short == 0), f"shortfall {short}")
    check("...and it names which machine is charged the head",
          "charged to big" in text, text)

    # ---------- 4. the licence gate is not advisory -----------------------------
    install(_fixture(), license_id="cc-by-nc-4.0")
    bad = mm.measure("toy/restricted")
    out = io.StringIO()
    with redirect_stdout(out):
        mm.report(bad)
    check("a restricted licence emits NO tier row", "Tier row" not in out.getvalue())
    check("...and says why, in the refusal's own words",
          "non-commercial" in out.getvalue())

    install(_fixture(), license_id=None)
    out = io.StringIO()
    with redirect_stdout(out):
        mm.report(mm.measure("toy/undeclared"))
    check("an UNDECLARED licence is refused exactly like a restricted one",
          "Tier row" not in out.getvalue() and "UNDECLARED" in out.getvalue(),
          "'we did not check' and 'we checked and it is fine' must not look the same")

    install(_fixture(), config_status=401)
    check("a GATED repo is refused at measurement, not discovered at download",
          _raises(lambda: mm.measure("meta-llama/whatever"), SystemExit))

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


def _raises(fn, exc=SystemExit):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


if __name__ == "__main__":
    sys.exit(main())
