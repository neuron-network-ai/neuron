"""
safety/moderation.py — content-policy gate for NEURON's chat/API layer (Workstream A).

WHY THIS SHAPE: NEURON splits inference across machines, and only the DRIVER (the node
holding the embedding + lm_head — today ui/app.py and api/openai_compat.py, both running
neuron_driver.py) ever handles plaintext. Middle/last-stage compute nodes only ever see
opaque hidden-state tensors (common.py:98, mid_stage/last_stage) — they cannot read prompts
or completions. So moderation belongs ONLY at the driver's intake and output stream, never
distributed to compute nodes (they have nothing meaningful to moderate) and never at the
coordinator (torch-free, never sees plaintext either).

WHAT THIS IS: a cheap, fast, trivially-evadable v1 — a case-insensitive keyword/phrase
blocklist. It is NOT a promise of robustness (paraphrase, other languages, and leetspeak all
slip past it). check_text()'s signature is the deliberate seam for a future classifier-based
backend; callers (ui/app.py, api/openai_compat.py, neuron_driver.py) never need to change.
See SAFETY.md for the policy this enforces and its honest limits.
"""
import dataclasses
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKLIST_PATH = os.environ.get("NEURON_BLOCKLIST_PATH", os.path.join(HERE, "blocklist.json"))
LOG_PATH = os.environ.get("NEURON_MODERATION_LOG", os.path.join(HERE, "moderation.log"))

_cache = None


def _load_blocklist(path=None):
    """Compiled once and cached; pass `path` to force a reload (used by tests)."""
    global _cache
    p = path or BLOCKLIST_PATH
    with open(p) as f:
        raw = json.load(f)
    compiled = {
        category: [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms]
        for category, terms in raw.items()
    }
    if path is None:
        _cache = compiled
    return compiled


@dataclasses.dataclass
class ModerationResult:
    blocked: bool
    category: str = None
    matched_term: str = None


def check_text(text):
    """Scan `text` against every category's phrase patterns. Word-boundary + case-insensitive
    matching only — deliberately simple, see module docstring for why. Never raises on odd
    input (empty/None text is always allowed through)."""
    if not text:
        return ModerationResult(blocked=False)
    blocklist = _cache if _cache is not None else _load_blocklist()
    for category, patterns in blocklist.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                return ModerationResult(blocked=True, category=category, matched_term=m.group(0))
    return ModerationResult(blocked=False)


def log_event(direction, category, request_id, identity_hash=None, snippet=None):
    """Local-only audit line — NEVER sent to the coordinator. NEURON's one honest privacy
    property is that the coordinator and compute nodes never see plaintext; a moderation log
    is itself plaintext-adjacent, so it stays on the driver machine only, same as agent.log's
    own local-only convention. Snippet is truncated to 40 chars — enough to review a flagged
    event, not enough to reconstruct the full prompt/answer. Logging failures must never break
    a real request, so I/O errors are swallowed."""
    entry = {"ts": time.time(), "direction": direction, "category": category,
             "request_id": request_id, "identity_hash": identity_hash,
             "snippet": (snippet or "")[:40]}
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass
