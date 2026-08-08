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

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCKLIST_PATH = os.environ.get("NEURON_BLOCKLIST_PATH", os.path.join(HERE, "blocklist.json"))
LOG_PATH = os.environ.get("NEURON_MODERATION_LOG", os.path.join(HERE, "moderation.log"))
WALLET_LINK_SECRET = os.environ.get("NEURON_WALLET_LINK_SECRET", "neuron-wallet-link-dev-secret")

_cache = None


# Scripts that do not separate words with spaces. `\b` is defined against `\w`, and every Han
# character IS a `\w`, so inside 我想要儿童色情内容 there is no boundary between 要 and 儿 and a
# `\b`-anchored pattern never fires. A term in one of these scripts therefore has to be matched
# as a plain substring, or the blocklist would look correct, pass a unit test against the bare
# term, and match nothing whatsoever in a real sentence.
_NO_WORD_BOUNDARY = (
    (0x2E80, 0x9FFF),      # CJK radicals, kana, Han
    (0xA000, 0xA4CF),      # Yi
    (0xAC00, 0xD7AF),      # Hangul syllables
    (0xF900, 0xFAFF),      # CJK compatibility ideographs
    (0x0E00, 0x0E7F),      # Thai
    (0x1780, 0x17FF),      # Khmer
)


def _needs_no_boundary(term):
    return any(any(lo <= ord(ch) <= hi for lo, hi in _NO_WORD_BOUNDARY) for ch in term)


def _compile(term):
    """Compile one blocklist term, anchoring it only where anchoring is meaningful."""
    if _needs_no_boundary(term):
        return re.compile(re.escape(term), re.IGNORECASE)
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def _load_blocklist(path=None):
    """Compiled once and cached; pass `path` to force a reload (used by tests).

    `encoding="utf-8"` is load-bearing, not tidiness: Python's default here is the locale
    codepage (cp1252 on this machine), so the moment the blocklist contained a single non-ASCII
    character this raised UnicodeDecodeError and took the whole content gate down with it.
    """
    global _cache
    p = path or BLOCKLIST_PATH
    with open(p, encoding="utf-8") as f:
        raw = json.load(f)
    compiled = {category: [_compile(term) for term in terms] for category, terms in raw.items()}
    if path is None:
        _cache = compiled
    return compiled


@dataclasses.dataclass
class ModerationResult:
    blocked: bool
    category: str = None
    matched_term: str = None
    # Which script the text is predominantly written in, and whether this gate actually has
    # any patterns for it. `blocked=False` used to mean two very different things -- "scanned
    # and clean" and "we have nothing to scan this with" -- and reported them identically, so
    # the share of traffic passing through unexamined was not merely unmeasured, it was
    # unmeasurable. Existing callers read only `.blocked` and are unaffected.
    script: str = "latin"
    screened: bool = True


# Unicode ranges that decide the dominant script of a piece of text. Deliberately coarse: the
# question is only "do we hold patterns for this", not language identification.
_SCRIPTS = (
    ("han",      ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF))),
    ("kana",     ((0x3040, 0x30FF),)),
    ("hangul",   ((0xAC00, 0xD7AF), (0x1100, 0x11FF))),
    ("cyrillic", ((0x0400, 0x04FF),)),
    ("arabic",   ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("hebrew",   ((0x0590, 0x05FF),)),
    ("thai",     ((0x0E00, 0x0E7F),)),
    ("greek",    ((0x0370, 0x03FF),)),
)


def dominant_script(text):
    """The script most of `text`'s letters are written in. 'latin' when in doubt.

    Only letters count -- digits, spaces and punctuation are shared across scripts and would
    otherwise drag every sample toward latin.
    """
    counts, letters = {}, 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        letters += 1
        cp = ord(ch)
        name = "latin"
        for script, ranges in _SCRIPTS:
            if any(lo <= cp <= hi for lo, hi in ranges):
                name = script
                break
        counts[name] = counts.get(name, 0) + 1
    if not letters:
        return "latin"
    return max(counts, key=counts.get)


def covered_scripts(blocklist=None):
    """Scripts the loaded blocklist actually holds at least one term for."""
    bl = blocklist if blocklist is not None else (_cache or _load_blocklist())
    scripts = set()
    for patterns in bl.values():
        for pat in patterns:
            scripts.add(dominant_script(pat.pattern))
    return scripts


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
                return ModerationResult(blocked=True, category=category,
                                        matched_term=m.group(0),
                                        script=dominant_script(text), screened=True)
    script = dominant_script(text)
    return ModerationResult(blocked=False, script=script,
                            screened=script in covered_scripts(blocklist))


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


def report_violation(coordinator_base, wallet_id, direction, category):
    """Tell the coordinator a wallet's IDENTITY was behind a blocked request, so repeated
    attempts escalate (see coordinator/models.py's record_violation + MODERATION_BAN_
    THRESHOLD) even across separate requests -- a per-request block alone forgets who did it
    the moment the response is sent. Sends ONLY the category label (e.g. "weapons_cbrn"),
    NEVER the snippet/text -- the coordinator staying blind to plaintext is NEURON's one
    honest privacy property (see module docstring / log_event's own comment); a violation
    COUNT tied to a wallet is enough to enforce consequences without breaking that. Best-
    effort and fire-and-forget: an anonymous request (no wallet_id) or a network hiccup here
    must never block or crash the moderation response the user already got."""
    if not wallet_id:
        return
    try:
        requests.post(f"{coordinator_base.rstrip('/')}/wallet/{wallet_id}/violation",
                      json={"direction": direction, "category": category},
                      headers={"X-Wallet-Link-Secret": WALLET_LINK_SECRET}, timeout=5)
    except requests.RequestException:
        pass
