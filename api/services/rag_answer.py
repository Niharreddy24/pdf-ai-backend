import os
import re
import ollama


def build_context(results):
    """
    Convert chroma query output -> sorted list of {text,page,distance}
    """
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    items = []
    for doc, meta, dist in zip(docs, metas, dists):
        items.append(
            {
                "text": doc or "",
                "page": (meta or {}).get("page"),
                "distance": dist,
            }
        )

    items.sort(key=lambda x: x["distance"] if x["distance"] is not None else 999999)
    return items


def _important_tokens(question: str):
    q = question or ""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]+", q)
    extra = re.findall(r"[a-zA-Z]+", q.lower())

    stop = {
        "what", "is", "the", "a", "an", "of", "about", "does", "do", "in", "on", "to", "and",
        "how", "many", "pages", "pdf", "have", "has", "tell", "me", "explain", "define",
        "where", "configured", "which", "file",
    }
    extra = [w for w in extra if w not in stop and len(w) > 2]

    return list(dict.fromkeys(tokens + extra))[:12]


def _make_context(items, question: str, max_chars=900):
    tokens = _important_tokens(question)

    selected = []
    used = set()

    # Baseline: top similar chunks
    for i, it in enumerate(items[:6]):
        selected.append(it)
        used.add(i)

    # Add keyword-hit chunks
    if tokens:
        for i, it in enumerate(items):
            if i in used:
                continue
            t_low = (it.get("text") or "").lower()
            if any(tok.lower() in t_low for tok in tokens):
                selected.append(it)
                used.add(i)
                if len(selected) >= 10:
                    break

    blocks = []
    total = 0
    for it in selected:
        page = it.get("page")
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        block = f"[Page {page}]\n{txt}\n"
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    return "\n---\n".join(blocks).strip()


def answer_from_context(question: str, items):
    """
    Returns: (answer_text, sources_list)
    sources_list: [{page, snippet}]
    """
    if not items:
        return "I couldn't find that in the PDF.", []

    sources = [
        {
            "page": it.get("page"),
            "snippet": (it.get("text", "")[:250].replace("\n", " ").strip()),
        }
        for it in items[:3]
        if it.get("text")
    ]

    context = _make_context(items, question)
    if not context:
        return "I couldn't find that in the PDF.", sources

    # Use a small model for your RAM (tinyllama works on 2GB machines)
    model = (os.getenv("OLLAMA_MODEL") or "").strip() or "tinyllama"

    system = (
        "You are a PDF question-answering assistant.\n"
        "Answer ONLY using the provided PDF Context.\n"
        "If the answer is not explicitly in the context, reply exactly:\n"
        "I couldn't find that in the PDF.\n"
    )

    prompt = f"PDF Context:\n{context}\n\nQuestion: {question}"

    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        options={"num_predict": 120,
    "temperature": 0.1,
    "num_ctx": 1024},
    )

    answer = (resp.get("message") or {}).get("content", "").strip()
    if not answer:
        return "I couldn't find that in the PDF.", sources

    return answer, sources

