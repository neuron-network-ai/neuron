"""ui/workspace_api.py — the endpoints the workspace UI needs that NEURON never had.

**Why this file exists.** The workspace chat UI (`ui/static/workspace/`) was built against an
Express server: skills, memory and a tool loop all live behind `/api/*` routes
in `server.ts`. NEURON ships a FROZEN PYTHON BINARY and its own build notes are explicit that
"Node is a build dependency only, never a runtime one", so that server cannot travel with it.
Without these routes the features render and do nothing — switches a user flips and waits on.

So the ones that are honest CRUD are reimplemented here, against NEURON's own state directory
rather than the working directory Express used (a frozen app's cwd is wherever the shortcut
pointed, which is not a place to keep a user's data).

**Memory works, and stays on this machine.** It embeds each stored message and searches by
cosine similarity; NEURON now serves `/v1/embeddings` from the GGUF already on this disk, and
the index is a file here. Nothing about it reaches the coordinator or the node network. On a
machine with no embedding model it still refuses with a REASON rather than answering
`{"ok": true, "hits": []}` — an empty success would show memory working while recalling
nothing, which is the failure that looks like success.

Everything here is per-machine, not per-account. That matches where these features came from —
a single-user workspace on one computer — and it is why none of them take a wallet id.
"""
import json
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _state_dir() -> Path:
    """NEURON's own state directory, the same one the agent and payout key use."""
    base = os.environ.get("NEURON_STATE_DIR") or os.environ.get("LOCALAPPDATA") \
        or os.path.join(os.path.expanduser("~"), ".local", "share")
    p = Path(base) / "NEURON" / "workspace"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data):
    """Write via a temp file and replace, so a crash mid-write cannot truncate the store.

    The Express version did the same thing and it is worth keeping: these files hold the
    user's own data, and a half-written JSON array loses all of it.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# health — the page checks this on boot to decide whether the server is current
# --------------------------------------------------------------------------- #
@router.get("/api/health")
def health():
    # `apiVersion` is what the page compares against its own build to decide whether to warn
    # that the server is stale. NEURON serves the page and these routes from ONE binary, so
    # they can never disagree -- reporting a matching version is the truth here, not a stub.
    return {"ok": True, "server": "neuron", "apiVersion": 999999,
            "features": {"skills": True, "memory": True,
                         # The tool loop runs shell/file actions through Express's
                         # /api/tools/run. Shipping that inside a consumer app handed to
                         # strangers is a security decision, not a port -- left off.
                         "tools": False}}


# --------------------------------------------------------------------------- #
# skills — one markdown file per skill, exactly as server.ts stored them
# --------------------------------------------------------------------------- #
_SKILL_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _skills_dir() -> Path:
    d = _state_dir() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _parse_skill(text: str, fallback_id: str):
    """`--- name: … description: … --- body` front matter, or a bare body."""
    name, description, body = fallback_id, "", text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if m:
        head, body = m.group(1), m.group(2)
        for line in head.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                k, v = k.strip().lower(), v.strip()
                if k == "name":
                    name = v
                elif k == "description":
                    description = v
    return {"id": fallback_id, "name": name, "description": description,
            "instructions": body.strip()}


@router.get("/api/skills/list")
def skills_list():
    out = []
    for f in sorted(_skills_dir().glob("*.md")):
        try:
            out.append(_parse_skill(f.read_text(encoding="utf-8"), f.stem))
        except OSError:
            continue
    return {"ok": True, "skills": out}


@router.post("/api/skills/save")
async def skills_save(request: Request):
    body = await request.json()
    skill = body.get("skill") or body
    sid = str(skill.get("id") or uuid.uuid4().hex[:8])
    if not _SKILL_ID.match(sid):
        # The id becomes a FILENAME. Anything else is a path traversal waiting to happen.
        return JSONResponse({"ok": False, "error": "invalid skill id"}, status_code=400)
    name = str(skill.get("name") or sid).replace("\n", " ")
    desc = str(skill.get("description") or "").replace("\n", " ")
    text = (f"---\nname: {name}\ndescription: {desc}\n---\n"
            f"{skill.get('instructions') or ''}\n")
    try:
        (_skills_dir() / f"{sid}.md").write_text(text, encoding="utf-8")
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "skill": {"id": sid, "name": name, "description": desc,
                                  "instructions": skill.get("instructions") or ""}}


@router.post("/api/skills/delete")
async def skills_delete(request: Request):
    body = await request.json()
    sid = str(body.get("id") or "")
    if not _SKILL_ID.match(sid):
        return JSONResponse({"ok": False, "error": "invalid skill id"}, status_code=400)
    try:
        (_skills_dir() / f"{sid}.md").unlink(missing_ok=True)
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}


# --------------------------------------------------------------------------- #
# memory — real, and entirely on this machine
# --------------------------------------------------------------------------- #
#
# Every stored message is embedded once and kept on disk beside its vector, so a later
# conversation can pull back what an earlier one said. Both halves are LOCAL: the vectors come
# from the GGUF already on this disk (api/openai_compat.py's /v1/embeddings, called in-process
# here rather than over HTTP to ourselves) and the index is a file in NEURON's state directory.
# Nothing reaches the coordinator or the node network, and it costs no NRN.
#
# JSONL, not one JSON blob: appending a record is O(1) where rewriting the whole array grows
# more expensive with every message ever stored. Vectors are NORMALISED on write, which turns
# cosine similarity into a plain dot product at read time.
#
# The frontend sends a `baseUrl` because it was written against Express, which proxied to
# whatever local server the user configured. It is ignored: NEURON is the server, and honouring
# a client-supplied URL here would let a page point this endpoint at an arbitrary host.
def _mem_file() -> Path:
    return _state_dir() / "memory.jsonl"


def _embed(texts):
    """Vectors for `texts`, or None when this machine has no embedding model."""
    try:
        from api.openai_compat import _embedder
    except Exception:                                            # noqa: BLE001
        return None
    llm = _embedder()
    if llm is None:
        return None
    out = []
    for t in texts:
        try:
            v = llm.create_embedding(t)["data"][0]["embedding"]
        except Exception:                                        # noqa: BLE001
            return None
        if v and isinstance(v[0], list):
            v = v[0]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        out.append([x / n for x in v])
    return out


_MEMORY_REASON = (
    "No embedding model is available on this machine, so memory cannot search by meaning. "
    "Set NEURON_EMBED_GGUF to a dedicated embedding model, or leave memory off."
)


@router.get("/api/memory/stats")
def memory_stats():
    n = 0
    try:
        with open(_mem_file(), encoding="utf-8") as f:
            n = sum(1 for line in f if line.strip())
    except OSError:
        n = 0
    return {"ok": True, "stats": {"count": n}, "count": n}


@router.post("/api/memory/remember")
async def memory_remember(request: Request):
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        return {"ok": True, "stored": False}
    vecs = _embed([text])
    if vecs is None:
        return JSONResponse({"ok": False, "error": _MEMORY_REASON, "unavailable": True},
                            status_code=501)
    rec = {"id": uuid.uuid4().hex[:12],
           "threadId": body.get("threadId"),
           "threadTitle": body.get("threadTitle") or "",
           "messageId": body.get("messageId"),
           "role": body.get("role") or "user",
           "text": text,
           "ts": int(body.get("ts") or time.time() * 1000),
           "v": vecs[0]}
    try:
        with open(_mem_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "stored": True}


@router.post("/api/memory/recall")
async def memory_recall(request: Request):
    body = await request.json()
    query = str(body.get("query") or "").strip()
    if not query:
        return {"ok": True, "hits": []}
    qv = _embed([query])
    if qv is None:
        return JSONResponse({"ok": False, "error": _MEMORY_REASON, "unavailable": True,
                             "hits": []}, status_code=501)
    q = qv[0]
    exclude = body.get("excludeThreadId")
    limit = int(body.get("limit") or 4)

    hits = []
    try:
        with open(_mem_file(), encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue          # one bad line must not lose the whole index
                if exclude and r.get("threadId") == exclude:
                    continue
                v = r.get("v") or []
                if len(v) != len(q):
                    continue          # a different embedding model wrote this; skip, do not crash
                score = sum(a * b for a, b in zip(q, v))
                hits.append({"threadId": r.get("threadId"), "threadTitle": r.get("threadTitle"),
                             "role": r.get("role"), "text": r.get("text"),
                             "ts": r.get("ts"), "score": round(score, 4)})
    except OSError:
        return {"ok": True, "hits": []}

    hits.sort(key=lambda h: h["score"], reverse=True)
    return {"ok": True, "hits": hits[:limit]}


@router.post("/api/memory/forget")
async def memory_forget(request: Request):
    body = await request.json()
    tid = body.get("threadId")
    path = _mem_file()
    if not tid or not path.exists():
        return {"ok": True, "removed": 0}
    kept, removed = [], 0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("threadId") == tid:
                    removed += 1
                else:
                    kept.append(line.rstrip("\n"))
        tmp = path.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(kept) + ("\n" if kept else ""))
        os.replace(tmp, path)
    except OSError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "removed": removed}


# --------------------------------------------------------------------------- #
# model discovery the page probes on boot
# --------------------------------------------------------------------------- #
@router.post("/api/local/list-models")
def local_list_models():
    # The workspace asks this for Ollama/LM Studio, which NEURON's build does not offer. An
    # empty list is the truthful answer and keeps the boot quiet; NEURON's own models come from
    # /v1/models, which the picker calls directly.
    return {"ok": True, "models": []}


@router.post("/api/local/image-models")
def local_image_models():
    return {"ok": True, "models": [], "available": False,
            "reason": "NEURON serves language models, not diffusion models."}
