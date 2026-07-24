"""
coordinator/model_registry.py — the models NEURON can serve  [Session 15]

Tracks available models (id, layer count, description). Today the network serves one
model; this is the registry + selection surface so more can be added without code
surgery. Per-request model routing + nodes serving multiple models (RAM permitting) is
the extension; for now the coordinator exposes the catalog and resolves a default.

Add a model by editing MODELS or setting NEURON_EXTRA_MODELS (JSON list of
{"id","layers","description"}). `layers` must equal the model's transformer depth
(used for chain assembly).
"""
import json
import os

from coordinator import config

MODELS = {
    config.MODEL_ID: {
        "id": config.MODEL_ID,
        "layers": config.TOTAL_LAYERS,
        "description": "Qwen2.5-1.5B-Instruct — the default NEURON model.",
        "ready": True,
    },
}

# optional extra models from env (not yet deployed to nodes -> ready:false)
try:
    for m in json.loads(os.environ.get("NEURON_EXTRA_MODELS", "[]")):
        MODELS[m["id"]] = {"id": m["id"], "layers": int(m["layers"]),
                           "description": m.get("description", ""), "ready": False}
except Exception:
    pass

DEFAULT_MODEL = config.MODEL_ID
ALIASES = {"neuron", "default", "gpt-3.5-turbo"}   # map friendly names to the default


def list_models():
    return list(MODELS.values())


def get_model(model_id):
    return MODELS.get(model_id)


def resolve(model_id=None):
    """Pick the model for a request. Unknown ids / aliases fall back to the default."""
    if model_id and model_id in MODELS:
        return MODELS[model_id]
    return MODELS[DEFAULT_MODEL]
