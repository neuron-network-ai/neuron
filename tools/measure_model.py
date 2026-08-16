"""
tools/measure_model.py — what a model actually weighs, and whether these machines can hold it.

MEASUREMENT ONLY. It downloads no weights and it edits no tier: it reads a model's published
safetensors header (~40 KB over HTTP, no auth) and prints the numbers, plus a paste-ready row
for `coordinator/model_tiers.TIERS`. A human commits the row, which is the point — the licence
gate in `model_registry.py` exists so that adding a model is a diff somebody has to justify,
and a tool that edited the table itself would route around it.

WHY IT EXISTS. Every figure in the tier table was arrived at by hand, in a comment:

    1.5B  hidden 1536, ffn 8960,  28 layers ->  46.8M params/layer -> 0.094 GB

That is arithmetic from a config file, and it has been wrong in both directions. It was on an
fp16 basis while every node stored fp32 — half the real footprint, and the generous half. And
`head_gb` did not exist at all, because, in the table's own words, the figure was "recorded
here rather than guessed". Reading the header instead removes the guess: the byte ranges are
the bytes the node downloads, per tensor, in the file it downloads them from.

    python tools/measure_model.py Qwen/Qwen3-4B-Instruct-2507
    python tools/measure_model.py Qwen/Qwen3-4B-Instruct-2507 --against pavilion:12,node-b:8
    python tools/measure_model.py Qwen/Qwen3-4B-Instruct-2507 --against pav:12,nb:8 --dtype fp16

`--against` answers the capacity question — the one NEURON exists for — through the
COORDINATOR'S OWN functions (`balancer.capacity_shortfall`, `balancer.solve`), never a second
implementation of the same arithmetic. A sizing tool that agreed with itself and disagreed with
the placer would be worse than no tool: it would produce confident numbers for a plan the
network then refuses, or clears one that OOM-kills a volunteer's machine.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import slice_downloader as sd
from coordinator import balancer, model_registry

HF_API = "https://huggingface.co/api/models"

# The tier table's figures are all on a 2-bytes-per-param basis and `balancer.effective_gb`
# scales from it. Emitting on any other basis would be corrected twice.
TIER_BASIS_BYTES = balancer.TIER_BASIS_BYTES


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def measure(model_id, revision="main"):
    """Per-layer and head footprints, in PARAMS, read from the published header.

    Params rather than bytes because the checkpoint's own dtype is not the node's: these
    files ship bf16, the runtime defaults to fp32, and a node may store fp16. Counting params
    and multiplying by the consumer's bytes-per-param keeps exactly one conversion in play.
    Derived per tensor as (byte range / numel), so a mixed-precision checkpoint is measured
    rather than assumed uniform.
    """
    cfg = requests.get(sd.resolve_url(model_id, "config.json", revision), timeout=30)
    if cfg.status_code == 401:
        # The trap model_tiers documents: a gated repo 401s, and `slice_downloader` fetches
        # byte ranges with no token, so promoting to it would break every node at download
        # with nothing useful to say.
        raise SystemExit(f"{model_id} is GATED (401). slice_downloader sends no auth, so every "
                         f"node would fail at download. Not servable on this network.")
    cfg.raise_for_status()
    cfg = cfg.json()

    header, _, _ = sd.fetch_header(model_id, revision)

    per_layer, head, norm, other = {}, {}, {}, {}
    for name, meta in header.items():
        n_bytes = sd._bytes(meta)
        params = _prod(meta["shape"]) or 0
        L = sd._layer_of(name)
        rec = (n_bytes, params)
        if L is not None:
            b, p = per_layer.get(L, (0, 0))
            per_layer[L] = (b + n_bytes, p + params)
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            # Exactly what `get_tensors_for_layers` hands the FIRST node. Whether lm_head is
            # present at all is the truth about tying; config's `tie_word_embeddings` is only
            # the claim, and it is the claim that has been wrong here before (Qwen2.5-1.5B ties
            # and Qwen2.5-7B does not, which is how the driver nearly loaded a meta lm_head).
            head[name] = rec
        elif name == "model.norm.weight":
            norm[name] = rec          # goes to the LAST node, and is a few thousand params
        else:
            other[name] = rec

    if not per_layer:
        raise SystemExit(f"{model_id}: no `model.layers.N.*` tensors in the header - this is "
                         f"not a decoder checkpoint this pipeline can slice.")

    layer_params = {L: p for L, (_, p) in per_layer.items()}
    uniform = len(set(layer_params.values())) == 1
    head_params = sum(p for _, p in head.values())
    total_params = sum(_prod(m["shape"]) for m in header.values())

    return {
        "model_id": model_id,
        "revision": revision,
        "n_layers_header": len(per_layer),
        "n_layers_config": cfg.get("num_hidden_layers"),
        "hidden": cfg.get("hidden_size"),
        "ffn": cfg.get("intermediate_size"),
        "vocab": cfg.get("vocab_size"),
        "tied_declared": bool(cfg.get("tie_word_embeddings")),
        "tied_observed": "lm_head.weight" not in head,
        "layer_params_uniform": uniform,
        # The MAX, not the mean: a node is sized by the slice it is handed, and a plan that
        # fits on average still OOMs on the fattest layer.
        "layer_params": max(layer_params.values()),
        "layer_params_min": min(layer_params.values()),
        "head_tensors": sorted(head),
        "head_params": head_params,
        "norm_params": sum(p for _, p in norm.values()),
        "other_params": sum(p for _, p in other.values()),
        "total_params": total_params,
        "file_bytes_per_param": sum(b for b, _ in per_layer.values())
                                / max(sum(layer_params.values()), 1),
        "shards": len({m.get("_file") for m in header.values()}),
        "license": _license(model_id),
    }


def _license(model_id):
    """The declared licence, or None. Non-fatal: an unreadable API is not evidence of a
    permissive licence, and `report()` refuses to emit a row it cannot check."""
    try:
        r = requests.get(f"{HF_API}/{model_id}", timeout=30)
        if r.status_code != 200:
            return None
        j = r.json()
        lic = (j.get("cardData") or {}).get("license")
        if lic:
            return str(lic).strip().lower()
        for t in j.get("tags", []):
            if isinstance(t, str) and t.startswith("license:"):
                return t.split(":", 1)[1].strip().lower()
    except (requests.RequestException, ValueError):
        pass
    return None


def gb(params, bytes_per_param=TIER_BASIS_BYTES):
    """Decimal GB, matching the tier table and `psutil.virtual_memory().total // 10**9`."""
    return params * bytes_per_param / 1e9


def parse_machines(spec):
    """"pavilion:12,node-b:8" -> roster dicts the balancer accepts."""
    out = []
    for part in [p for p in (spec or "").split(",") if p.strip()]:
        name, _, ram = part.partition(":")
        try:
            ram_gb = float(ram)
        except ValueError:
            raise SystemExit(f"--against: '{part}' is not name:GB")
        out.append({"node_id": name.strip(), "ram_gb": ram_gb, "ms_per_layer": 12.0,
                    "status": "online", "eligible": True})
    if not out:
        raise SystemExit("--against needs at least one machine, as name:GB")
    return out


def report(m, machines=None, dtype=None):
    bpp = balancer._DTYPE_BYTES.get((dtype or "").lower(), balancer.ASSUMED_WEIGHT_BYTES)
    gpl, hgb = gb(m["layer_params"]), gb(m["head_params"])
    L = m["n_layers_header"]

    print(f"\n{m['model_id']}  (revision {m['revision']}, {m['shards']} shard file(s))")
    print(f"  {L} layers, hidden {m['hidden']}, ffn {m['ffn']}, vocab {m['vocab']}")
    print(f"  checkpoint stores {m['file_bytes_per_param']:.2f} bytes/param")
    if m["n_layers_config"] not in (None, L):
        print(f"  !! config says {m['n_layers_config']} layers, the header holds {L}. The "
              f"HEADER is what a node downloads; chain assembly uses the tier's `layers`.")
    if not m["layer_params_uniform"]:
        print(f"  !! layers are NOT uniform ({m['layer_params_min']:,} to "
              f"{m['layer_params']:,} params). Sized on the LARGEST; one figure per model "
              f"cannot describe this checkpoint exactly.")
    if m["tied_declared"] != m["tied_observed"]:
        print(f"  !! config says tie_word_embeddings={m['tied_declared']} but the header "
              f"{'has' if not m['tied_observed'] else 'has no'} a separate lm_head. Believing "
              f"the header.")

    print(f"\n  per layer   {m['layer_params']:>14,} params"
          f"   {gpl:.4f} GB @fp16   {gpl * 2:.4f} GB @fp32")
    print(f"  driver head {m['head_params']:>14,} params"
          f"   {hgb:.4f} GB @fp16   {hgb * 2:.4f} GB @fp32"
          f"   [{', '.join(m['head_tensors']) or 'none'}]")
    if m["tied_observed"]:
        print("              (embeddings are tied, so lm_head IS the embedding - counted once)")
    print(f"  final norm  {m['norm_params']:>14,} params   (last node; negligible, not charged)")
    print(f"  TOTAL       {m['total_params']:>14,} params"
          f"   {gb(m['total_params']):.2f} GB @fp16  {gb(m['total_params'], 4):.2f} GB @fp32")

    refusal = model_registry.license_refusal(m["license"])
    print(f"\n  licence: {m['license'] or 'UNDECLARED'}"
          + (f"  -> REFUSED: {refusal}" if refusal else "  -> permitted"))

    if refusal:
        # Deliberately no row. Emitting one "for reference" is how a restricted model ends up
        # pasted into the table by somebody who trusted the tool's output over its warning.
        print("\n  No tier row emitted. Serving weights is distribution to end users, so a "
              "restricted licence\n  binds the whole network - see coordinator/model_registry.py.")
    else:
        print("\n  Tier row (fp16 basis, which is what balancer.effective_gb scales from):\n")
        print(f'    {{"name": "?", "model_id": "{m["model_id"]}", "layers": {L},\n'
              f'     "min_nodes": 2, "min_ram_gb": '
              f'{round(gb(m["total_params"], 4) * 1.6, 1)}, "min_replicas": 1,\n'
              f'     "gb_per_layer": {round(gpl, 4)}, "head_gb": {round(hgb, 4)},\n'
              f'     "description": "?"}},')
        print("\n    `name`, `description` and `min_nodes` are POLICY, not measurement - set "
              "them yourself.\n    `min_ram_gb` above is fp32 total x1.6 (OS reserve + "
              "headroom), a starting point only:\n    it is the AGGREGATE gate, and "
              "`gb_per_layer`/`head_gb` are what decide if it can be placed.")

    if machines:
        _capacity_report(m, machines, dtype, bpp, gpl, hgb, L)


def _capacity_report(m, machines, dtype, bpp, gpl, hgb, L):
    """Can these machines hold it? Answered by the coordinator's own solver, never a copy."""
    nodes = [dict(n, **({"weight_dtype": dtype} if dtype else {})) for n in machines]
    print(f"\n  ---- capacity: {len(nodes)} machine(s), storing "
          f"{dtype or f'the assumed default ({bpp:g} B/param)'} ----")
    for n in nodes:
        usable = max(n["ram_gb"] - balancer.RAM_OS_RESERVE_GB, 0.0)
        budget = usable * 0.75
        alone = balancer.capacity_shortfall([n], L, gpl, hgb)
        print(f"    {n['node_id']:<14} {n['ram_gb']:>5.1f} GB total -> {budget:5.2f} GB budget"
              f"   (-{balancer.RAM_OS_RESERVE_GB:g} OS, x0.75 headroom)"
              f"   alone: {'holds it' if not alone else f'{alone} layer(s) short'}")

    hi = balancer.head_node_index(nodes)
    caps = balancer.layer_caps(nodes, gpl, L, hgb)
    short = balancer.capacity_shortfall(nodes, L, gpl, hgb)
    print(f"    head ({balancer.effective_gb(hgb, nodes[hi]):.2f} GB) charged to "
          f"{nodes[hi]['node_id']}, as router.canonical_assignment would place it")
    print(f"    layer capacity: {' + '.join(str(c) for c in caps)} = {sum(caps)} "
          f"of {L} needed")
    if short:
        print(f"\n    VERDICT: NO - {short} layer(s) short.")
    else:
        assign = balancer.solve(nodes, L, gpl, hgb)
        print(f"\n    VERDICT: YES - a split exists:")
        for a in assign:
            print(f"      {a['node_id']:<14} layers {a['layer_start']:>3}-{a['layer_end']:<3}"
                  f" ({a['layers']:>2})   {gb(m['layer_params'], bpp) * a['layers']:5.2f} GB"
                  + ("  + head" if a["node_id"] == nodes[0]["node_id"] else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--against", default=None,
                    help="machines to size against, e.g. 'pavilion:12,node-b:8' (name:total GB)")
    ap.add_argument("--dtype", default=None, choices=sorted(balancer._DTYPE_BYTES),
                    help="storage dtype those machines run at; default is the coordinator's "
                         "pessimistic assumption")
    ap.add_argument("--json", action="store_true", help="raw measurement, nothing else")
    args = ap.parse_args()

    m = measure(args.model_id, args.revision)
    if args.json:
        print(json.dumps(m, indent=2))
        return 0
    report(m, parse_machines(args.against) if args.against else None, args.dtype)
    return 0


if __name__ == "__main__":
    sys.exit(main())
