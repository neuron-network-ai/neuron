"""ui/workspace_api.py — the endpoints the workspace UI needs that NEURON never had.

**Why this file exists.** The workspace chat UI (`ui/static/workspace/`) was built against an
Express server: skills, the secretary, memory and a tool loop all live behind `/api/*` routes
in `server.ts`. NEURON ships a FROZEN PYTHON BINARY and its own build notes are explicit that
"Node is a build dependency only, never a runtime one", so that server cannot travel with it.
Without these routes the features render and do nothing — switches a user flips and waits on.

So the ones that are honest CRUD are reimplemented here, against NEURON's own state directory
rather than the working directory Express used (a frozen app's cwd is wherever the shortcut
pointed, which is not a place to keep a user's data).

**What is deliberately NOT reimplemented, and why it returns a REASON rather than an empty
list.** Memory embeds every stored message through an `/v1/embeddings` endpoint and searches by
cosine similarity. NEURON serves no embeddings model — `api/openai_compat.py` has
`/v1/chat/completions`, `/v1/completions` and `/v1/models`, and nothing else. Answering
`{"ok": true, "memories": []}` would be a lie shaped like success: the UI would show memory
working and silently recall nothing. It returns `ok: false` with the reason instead, which is
the same discipline `/app/update` uses for "we could not check" versus "you are up to date".

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

    The Express version did the same thing for the secretary and it is worth keeping: this is
    the file holding somebody's reminders, and a half-written JSON array loses all of them.
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
            "features": {"skills": True, "secretary": True, "memory": False, "tools": False}}


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
# secretary — one JSON store, atomic writes
# --------------------------------------------------------------------------- #
def _sec_file() -> Path:
    return _state_dir() / "secretary.json"


def _sec_items():
    data = _read_json(_sec_file(), {"items": []})
    return data.get("items", []) if isinstance(data, dict) else []


def _sec_save(items):
    _write_json(_sec_file(), {"items": items})


def _sec_stats(items):
    return {"total": len(items),
            "open": sum(1 for i in items if not i.get("done")),
            "done": sum(1 for i in items if i.get("done"))}


@router.get("/api/secretary/list")
def secretary_list(open: str = "", kind: str = ""):
    items = _sec_items()
    if open == "1":
        items = [i for i in items if not i.get("done")]
    if kind:
        items = [i for i in items if i.get("kind") == kind]
    return {"ok": True, "items": items, "stats": _sec_stats(_sec_items())}


@router.get("/api/secretary/due")
def secretary_due():
    now = time.time() * 1000
    items = [i for i in _sec_items()
             if not i.get("done") and i.get("dueAt") and float(i["dueAt"]) <= now]
    return {"ok": True, "items": items}


@router.post("/api/secretary/add")
async def secretary_add(request: Request):
    body = await request.json()
    items = _sec_items()
    item = {"id": uuid.uuid4().hex[:10],
            "text": str(body.get("text") or "").strip(),
            "kind": body.get("kind") or "task",
            "dueAt": body.get("dueAt"),
            "done": False,
            "createdAt": int(time.time() * 1000)}
    if not item["text"]:
        return JSONResponse({"ok": False, "error": "text is required"}, status_code=400)
    items.append(item)
    _sec_save(items)
    return {"ok": True, "item": item, "items": items, "stats": _sec_stats(items)}


@router.post("/api/secretary/update")
async def secretary_update(request: Request):
    body = await request.json()
    sid = str(body.get("id") or "")
    items = _sec_items()
    for i in items:
        if i.get("id") == sid:
            for k in ("text", "kind", "dueAt", "done"):
                if k in body:
                    i[k] = body[k]
            _sec_save(items)
            return {"ok": True, "item": i, "items": items, "stats": _sec_stats(items)}
    return JSONResponse({"ok": False, "error": "not found"}, status_code=404)


@router.post("/api/secretary/remove")
async def secretary_remove(request: Request):
    body = await request.json()
    sid = str(body.get("id") or "")
    items = [i for i in _sec_items() if i.get("id") != sid]
    _sec_save(items)
    return {"ok": True, "items": items, "stats": _sec_stats(items)}


# --------------------------------------------------------------------------- #
# memory — refused with a REASON, never faked
# --------------------------------------------------------------------------- #
_MEMORY_REASON = (
    "Memory needs an embeddings model to search by meaning, and NEURON does not serve one "
    "(/v1 offers chat and completions only). Turning it on would store notes nothing could "
    "ever recall."
)


def _memory_unavailable():
    # ok:false, not an empty success. An empty list here would render as "memory is working and
    # remembers nothing", which is exactly the failure that looks like success.
    return JSONResponse({"ok": False, "error": _MEMORY_REASON,
                         "unavailable": True, "memories": [], "stats": {"count": 0}},
                        status_code=501)


@router.get("/api/memory/stats")
def memory_stats():
    return _memory_unavailable()


@router.post("/api/memory/remember")
def memory_remember():
    return _memory_unavailable()


@router.post("/api/memory/recall")
def memory_recall():
    return _memory_unavailable()


@router.post("/api/memory/forget")
def memory_forget():
    return _memory_unavailable()


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
