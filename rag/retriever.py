"""
rag/retriever.py — retrieval-augmented generation for NEURON  [Session 15]

Before inference, search the web (DuckDuckGo, no API key) for context relevant to the
user's prompt and inject it — so the model can answer CURRENT questions despite its
training cutoff, and a small model gets grounded facts instead of hallucinating.

Fails SOFT: if search is unavailable or returns nothing, the original prompt is used
unchanged (inference still runs, just without fresh context). Only extra dep: `ddgs`.

    from rag import retriever
    augmented, sources = retriever.retrieve_and_augment("latest AI news")
"""


def search(query, max_results=5, timeout=10):
    """Web search -> list of {title, href, body}. Returns [] on any failure."""
    try:
        from ddgs import DDGS
        with DDGS(timeout=timeout) as ddg:
            return list(ddg.text(query, max_results=max_results))
    except Exception as e:  # network down, rate-limited, lib change — never fatal
        print(f"[rag] search failed ({e.__class__.__name__}: {e}); continuing without context")
        return []


def build_context(results, max_chars=1200):
    """Compact the search snippets into a bounded context block."""
    out, used = [], 0
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        body = (r.get("body") or "").strip()
        snippet = f"[{i}] {title} — {body}".strip(" —")
        if not snippet:
            continue
        if used + len(snippet) > max_chars:
            snippet = snippet[: max(0, max_chars - used)]
        out.append(snippet)
        used += len(snippet)
        if used >= max_chars:
            break
    return "\n".join(out)


def augment(prompt, context):
    """Wrap the prompt with retrieved context (or return it unchanged if none)."""
    if not context:
        return prompt
    return (
        "Use the up-to-date search results below to answer the question. Prefer facts "
        "from them; if they don't cover it, say what you do know.\n\n"
        f"Search results:\n{context}\n\n"
        f"Question: {prompt}"
    )


def retrieve_and_augment(prompt, max_results=5):
    """Convenience: search -> build context -> augment. Returns (augmented_prompt, sources).
    sources is a list of {title, href} for attribution (empty if search failed)."""
    results = search(prompt, max_results=max_results)
    context = build_context(results)
    sources = [{"title": r.get("title"), "href": r.get("href")} for r in results]
    return augment(prompt, context), sources
