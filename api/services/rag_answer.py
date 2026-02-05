import os
import re
from typing import List, Dict, Tuple

from ollama import Client


# ---------------------------
# Helpers: retrieval context
# ---------------------------

def build_context(results: dict) -> List[Dict]:
    """
    Expected Chroma-like shape:
      results["documents"][0] = [chunk_text...]
      results["metadatas"][0] = [{"page": 1, ...}, ...]
      results["distances"][0] = [0.12, 0.25, ...]
    """
    docs = results.get("documents", [[]])[0] or []
    metas = results.get("metadatas", [[]])[0] or []
    dists = results.get("distances", [[]])[0] or []

    items = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        items.append({
            "text": doc or "",
            "page": meta.get("page"),
            "distance": dist
        })

    items.sort(key=lambda x: x["distance"] if x["distance"] is not None else 999999)
    return items


def _important_tokens(question: str) -> List[str]:
    q = (question or "").strip()

    # Keep config-like tokens (DT_Databases, plugin.xml, notes.ini)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]+", q)

    # Also keep keywords
    extra = re.findall(r"[a-zA-Z]+", q.lower())
    stop = {
        "what", "is", "the", "a", "an", "of", "about", "does", "do", "in", "on", "to", "and",
        "how", "many", "pages", "pdf", "have", "has", "tell", "me", "explain", "define",
        "where", "configured", "which", "file", "this", "that"
    }
    extra = [w for w in extra if w not in stop and len(w) > 2]

    # Unique, keep order
    return list(dict.fromkeys(tokens + extra))[:12]


def _make_context(items: List[Dict], question: str, max_chars: int = 1400) -> str:
    """
    Build a small context to keep Ollama fast on CPU.
    """
    tokens = _important_tokens(question)

    selected: List[Dict] = []
    used = set()

    # Baseline: top similar chunks (keep small)
    for i, it in enumerate(items[:4]):
        selected.append(it)
        used.add(i)

    # Add keyword-hit chunks (limited)
    if tokens:
        for i, it in enumerate(items):
            if i in used:
                continue
            t_low = (it.get("text") or "").lower()
            if any(tok.lower() in t_low for tok in tokens):
                selected.append(it)
                used.add(i)
                if len(selected) >= 7:
                    break

    # Cap total length
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


def _sources(items: List[Dict], n: int = 3) -> List[Dict]:
    out = []
    for it in items[:n]:
        txt = (it.get("text") or "").strip()
        if not txt:
            continue
        out.append({
            "page": it.get("page"),
            "snippet": txt[:250].replace("\n", " ").strip()
        })
    return out


# ---------------------------
# Main: answer with Ollama
# ---------------------------

def answer_from_context(question: str, items: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Returns: (answer, sources)
    """

    if not items:
        return "I couldn't find that in the PDF.", []

    q = (question or "").strip()
    qlow = q.lower()

    sources = _sources(items, n=3)

    # ✅ model selection (safe default for your RAM)
    model = (os.getenv("OLLAMA_MODEL") or "").strip() or "tinyllama"

    # ✅ ollama client with timeout (prevents hanging -> fewer 504/500)
    host = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
    timeout_s = int(os.getenv("OLLAMA_TIMEOUT", "120"))
    client = Client(host=host, timeout=timeout_s)

    # ✅ fast path for summarize / what is pdf about
    is_summary = any(p in qlow for p in [
        "summarize", "summary", "what is this pdf about", "what is the pdf about"
    ])

    if is_summary:
        context = _make_context(items[:5], q, max_chars=900)
        system = (
            "You are a PDF summarizer.\n"
            "Rules:\n"
            "1) Use ONLY the provided PDF Context.\n"
            "2) If the context is insufficient, reply exactly: I couldn't find that in the PDF.\n"
            "3) Output 4-7 short lines max.\n"
            "4) Do not mention rules.\n"
        )
        prompt = f"PDF Context:\n{context}\n\nTask: Summarize what this PDF is about."
        options = {
            "num_predict": 140,
            "temperature": 0.1,
            "num_ctx": 1024,
            "num_thread": 2,
        }

    else:
        context = _make_context(items, q, max_chars=1400)
        system = (
            "You are a PDF question-answering assistant.\n"
            "STRICT RULES:\n"
            "1) Answer ONLY using the provided PDF Context.\n"
            "2) If the answer is not explicitly in the context, reply exactly: I couldn't find that in the PDF.\n"
            "3) If user asks for steps, output steps as a numbered list.\n"
            "4) Keep answer short (2-6 lines) unless user asks for more.\n"
            "5) Do not mention rules.\n"
        )
        prompt = f"PDF Context:\n{context}\n\nQuestion: {q}"
        options = {
            "num_predict": 120,
            "temperature": 0.1,
            "num_ctx": 1024,
            "num_thread": 2,
        }

    if not context:
        return "I couldn't find that in the PDF.", sources

    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            options=options,
        )

        answer = ((resp or {}).get("message") or {}).get("content", "")
        answer = (answer or "").strip()

        if not answer:
            return "I couldn't find that in the PDF.", sources

        return answer, sources

    except Exception:
        # Never crash API
        return "I couldn't find that in the PDF.", sources

