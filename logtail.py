"""
logtail.py — take the tail of a log and strip anything that isn't ours to send.

Shared deliberately by BOTH ends of log collection: `agent/agent.py` scrubs before uploading and
`coordinator/nodelogs.py` scrubs again on arrival. Two copies of these rules would drift, and the
half that drifted would be the half that leaked -- so there is one copy, at the repo root
alongside `common.py`, which is already shipped to the coordinator (`deploy.sh`) and bundled into
the frozen agent (`packaging/neuron-agent.spec`).

Why scrub at all: an agent log is a file on a volunteer's personal computer. Collecting it is the
difference between diagnosing a broken node and asking a stranger to paste a file, but it must
not become a way for credentials or somebody's name to leave their machine. The rules below are
deliberately over-eager. Blanking something harmless costs a round trip; missing a live node
token puts it in a database and then on a screen.
"""
import re

# A tail big enough for a traceback and the minutes around it, small enough that a handful of
# nodes cost nothing on a 1 GB VM.
MAX_BYTES = 64 * 1024

# Ordered most-specific first: a token inside a JSON blob has to be caught by its key before the
# generic long-hex rule rewrites it into something the key rule no longer matches.
_REDACTIONS = (
    # "node_token": "abc...",  X-Node-Token: abc...,  token=abc...
    (re.compile(r'((?:node_token|x-node-token|token|secret|ticket|api_key|apikey|password)'
                r'["\']?\s*[:=]\s*["\']?)([A-Za-z0-9_\-./+]{8,})',
                re.IGNORECASE), r'\1<redacted>'),
    (re.compile(r'(authorization\s*:\s*\w+\s+)(\S+)', re.IGNORECASE), r'\1<redacted>'),
    (re.compile(r'(x-register-secret\s*:\s*)(\S+)', re.IGNORECASE), r'\1<redacted>'),
    # a bare 0x private key or long hex blob
    (re.compile(r'\b(0x)?[0-9a-fA-F]{40,}\b'), '<redacted-hex>'),
    # C:\Users\firstname\... and /home/firstname/... name a real person
    (re.compile(r'([A-Za-z]:\\Users\\)[^\\\s"\']+', re.IGNORECASE), r'\1<user>'),
    (re.compile(r'(/(?:home|Users)/)[^/\s"\']+'), r'\1<user>'),
)


def redact(text):
    """Strip credentials and personal paths, leaving the log readable."""
    if not text:
        return ""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


def tail_bytes(text, limit=MAX_BYTES):
    """The last `limit` bytes, cut at a line boundary so the first line isn't half a line.

    The END of the file is the part worth having: a node that broke did so most recently, and
    the top of a long-running agent's log is a startup banner from days ago.
    """
    if text is None:
        return ""
    raw = text.encode("utf-8", "replace")
    if len(raw) <= limit:
        return raw.decode("utf-8", "replace")
    cut = raw[-limit:].decode("utf-8", "replace")
    nl = cut.find("\n")
    return cut[nl + 1:] if nl != -1 else cut


def clean_tail(text, limit=MAX_BYTES):
    """Both, in the order that matters: cut first, then scrub what survives."""
    return redact(tail_bytes(text, limit))
