"""
coordinator/model_registry.py — the models NEURON can serve  [Session 15]

Tracks available models (id, layer count, description). Today the network serves one
model; this is the registry + selection surface so more can be added without code
surgery. Per-request model routing + nodes serving multiple models (RAM permitting) is
the extension; for now the coordinator exposes the catalog and resolves a default.

Add a model by editing MODELS or setting NEURON_EXTRA_MODELS (JSON list of
{"id","layers","description","license"}). `layers` must equal the model's transformer
depth (used for chain assembly).

**EVERY MODEL MUST DECLARE A PERMITTED LICENCE, AND THIS FILE ENFORCES IT.**

Model weights are licensed separately from the code that runs them, and the difference is
not cosmetic. Serving weights is *distribution to end users*, so a restricted licence binds
the whole network, not one machine:

  * **Llama family** — the Llama Community Licence is not an open-source licence. It carries
    an acceptable-use policy, a 700M monthly-active-user ceiling, a required "Built with
    Llama" attribution, and a naming rule for derivatives.
  * **Qwen is not uniformly Apache 2.0 either.** Qwen2.5 at 0.5B/1.5B/7B/14B/32B is Apache
    2.0; **3B and 72B are under the separate Qwen licence.** Reaching for "a bigger Qwen"
    is exactly the move that looks like a size change and is actually a licence change.

Before this gate, `NEURON_EXTRA_MODELS` was an *environment variable* — the network could
begin serving restricted weights with no code change, no review and no record. Adding one
now requires editing PERMITTED_LICENSES here, which is a diff someone has to justify.
"""
import json
import logging
import os

from coordinator import config

log = logging.getLogger("neuron.coordinator.model_registry")

# SPDX-style ids we will serve without further thought: genuinely open, no field-of-use
# restriction, no user ceiling, no attribution obligation beyond keeping the notice.
PERMITTED_LICENSES = {"apache-2.0", "mit", "bsd-3-clause", "cc0-1.0"}

# Named so the refusal message can say *why*, rather than "unknown licence". Not exhaustive
# and not meant to be — anything absent from PERMITTED_LICENSES is refused regardless.
KNOWN_RESTRICTED = {
    "llama3": "Llama Community Licence — AUP, 700M MAU ceiling, attribution and naming rules",
    "llama3.1": "Llama Community Licence — AUP, 700M MAU ceiling, attribution and naming rules",
    "llama3.2": "Llama Community Licence — AUP, 700M MAU ceiling, attribution and naming rules",
    "llama3.3": "Llama Community Licence — AUP, 700M MAU ceiling, attribution and naming rules",
    "qwen": "Qwen Licence — not Apache 2.0; applies to Qwen2.5 3B and 72B",
    "qwen-research": "Qwen Research Licence — non-commercial use only",
    "gemma": "Gemma Terms of Use — prohibited-use policy, distribution conditions",
    "cc-by-nc-4.0": "non-commercial only",
}

MODELS = {
    config.MODEL_ID: {
        "id": config.MODEL_ID,
        "layers": config.TOTAL_LAYERS,
        "description": "Qwen2.5-1.5B-Instruct — the default NEURON model.",
        "license": "apache-2.0",
        "license_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct",
        "ready": True,
    },
}


def license_refusal(license_id):
    """Why this licence cannot be served, or None if it may be. Refusing is the default:
    an absent or unrecognised licence is refused exactly like a restricted one, because
    'we did not check' and 'we checked and it is fine' must not look the same here."""
    lic = (license_id or "").strip().lower()
    if not lic:
        return "no licence declared — every model must declare one"
    if lic in PERMITTED_LICENSES:
        return None
    if lic in KNOWN_RESTRICTED:
        return f"{license_id}: {KNOWN_RESTRICTED[lic]}"
    return (f"{license_id}: not in PERMITTED_LICENSES. If it is genuinely open, add it there "
            f"deliberately; if it restricts use, it does not belong on a public network")


# optional extra models from env (not yet deployed to nodes -> ready:false)
for m in json.loads(os.environ.get("NEURON_EXTRA_MODELS", "[]") or "[]"):
    try:
        model_id = m["id"]
        refusal = license_refusal(m.get("license"))
        if refusal:
            # Loud, and skipped. Silently degrading to ready:false would leave a restricted
            # model sitting in the catalog looking like a deployment problem.
            log.error("REFUSING model %s from NEURON_EXTRA_MODELS — %s", model_id, refusal)
            continue
        MODELS[model_id] = {"id": model_id, "layers": int(m["layers"]),
                            "description": m.get("description", ""),
                            "license": m["license"].strip().lower(),
                            "license_url": m.get("license_url", ""),
                            "ready": False}
    except (KeyError, TypeError, ValueError) as e:
        log.error("ignoring malformed NEURON_EXTRA_MODELS entry %r: %s", m, e)

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
