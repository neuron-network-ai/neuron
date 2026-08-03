"""Model weights are licensed separately from the code, and a restricted licence binds the
whole network rather than one machine. These tests hold the line that NEURON_EXTRA_MODELS
cannot put restricted weights into the catalog — it used to be an env var with no review.

Run:  python -m coordinator.test_model_license_gate      (from the repo root)
      pytest coordinator/test_model_license_gate.py
"""
import importlib
import json
import os

from coordinator import model_registry


def catalog_with(extra):
    """The MODELS catalog produced when NEURON_EXTRA_MODELS is `extra`.

    Returns a *snapshot*, not the module: importlib.reload mutates the one module object in
    place, so the restoring reload in `finally` would otherwise wipe the very result the
    caller is about to assert on.
    """
    old = os.environ.get("NEURON_EXTRA_MODELS")
    os.environ["NEURON_EXTRA_MODELS"] = json.dumps(extra)
    try:
        importlib.reload(model_registry)
        return dict(model_registry.MODELS)
    finally:
        if old is None:
            os.environ.pop("NEURON_EXTRA_MODELS", None)
        else:
            os.environ["NEURON_EXTRA_MODELS"] = old
        importlib.reload(model_registry)


def test_permitted_licence_is_admitted():
    cat = catalog_with([{"id": "org/open-model", "layers": 32, "license": "apache-2.0"}])
    assert "org/open-model" in cat
    assert cat["org/open-model"]["license"] == "apache-2.0"


def test_llama_community_licence_is_refused():
    cat = catalog_with([{"id": "meta-llama/Llama-3.3-70B-Instruct", "layers": 80,
                        "license": "llama3.3"}])
    assert "meta-llama/Llama-3.3-70B-Instruct" not in cat


def test_qwen_licence_is_refused_even_though_qwen_is_the_default_family():
    """Qwen2.5 is Apache 2.0 at 1.5B and NOT at 72B. Reaching for a bigger Qwen looks like
    a size change and is actually a licence change — the gate has to catch that."""
    cat = catalog_with([{"id": "Qwen/Qwen2.5-72B-Instruct", "layers": 80, "license": "qwen"}])
    assert "Qwen/Qwen2.5-72B-Instruct" not in cat


def test_missing_licence_is_refused_not_defaulted():
    cat = catalog_with([{"id": "org/undeclared", "layers": 32}])
    assert "org/undeclared" not in cat


def test_unknown_licence_is_refused_by_default():
    """'We did not check' must not look like 'we checked and it is fine'."""
    cat = catalog_with([{"id": "org/mystery", "layers": 32, "license": "some-new-licence"}])
    assert "org/mystery" not in cat


def test_a_refused_model_does_not_break_the_others():
    cat = catalog_with([
        {"id": "meta-llama/Llama-3.3-70B-Instruct", "layers": 80, "license": "llama3.3"},
        {"id": "org/open-model", "layers": 32, "license": "MIT"},
    ])
    assert "meta-llama/Llama-3.3-70B-Instruct" not in cat
    assert cat["org/open-model"]["license"] == "mit"   # normalised to lowercase


def test_default_model_declares_a_permitted_licence():
    assert model_registry.license_refusal(
        model_registry.MODELS[model_registry.DEFAULT_MODEL]["license"]) is None


def test_every_catalogued_model_declares_a_permitted_licence():
    for m in model_registry.list_models():
        assert model_registry.license_refusal(m.get("license")) is None, m["id"]


def test_known_restricted_licences_explain_themselves():
    """A refusal has to say why, or the next person just adds it to the allowlist."""
    for lic in ("llama3", "llama3.1", "llama3.2", "llama3.3",
                "qwen", "qwen-research", "gemma", "cc-by-nc-4.0"):
        reason = model_registry.license_refusal(lic)
        assert reason and len(reason) > len(lic) + 8, lic


# --------------------------------------------------------------------------- #
def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
