import os
import json
import math
import re
from django.conf import settings

# Stores extracted PDF chunks on disk (no Chroma, no embeddings)
STORE_DIR = os.path.join(settings.BASE_DIR, "doc_store")


def _doc_path(doc_id: str) -> str:
    os.makedirs(STORE_DIR, exist_ok=True)
    return os.path.join(STORE_DIR, f"{doc_id}.json")


def upsert_doc_chunks(doc_id: str, chunks_with_meta: list[dict]):
    """
    Save chunks for a PDF document.
    chunks_with_meta: [{"text": "...", "page": 1}, ...]
    """
    if not chunks_with_meta:
        return

    data = {"doc_id": doc_id, "chunks": chunks_with_meta}
    with open(_doc_path(doc_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _tokenize(text: str):
    # Simple tokenizer for keyword scoring
    return re.findall(r"[a-zA-Z0-9_]+", (text or "").lower())


def query_doc(doc_id: str, question: str, top_k: int = 6):
    """
    Lightweight retrieval: keyword overlap scoring.
    Returns a dict shaped like Chroma results:
    {
      "documents": [[...]],
      "metadatas": [[{"page":...}, ...]],
      "distances": [[...]]
    }
    """
    path = _doc_path(doc_id)
    if not os.path.exists(path):
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    q_tokens = _tokenize(question)
    if not q_tokens:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    q_set = set(q_tokens)

    scored = []
    for ch in chunks:
        txt = (ch.get("text") or "").strip()
        if not txt:
            continue
        tokens = _tokenize(txt)
        if not tokens:
            continue

        overlap = sum(1 for t in tokens if t in q_set)
        if overlap == 0:
            continue

        # score scaled by chunk length so long chunks don't always win
        score = overlap / max(1.0, math.log(len(tokens) + 2))
        scored.append((score, ch))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[:top_k]]

    documents = [c.get("text", "") for c in top]
    metadatas = [{"page": c.get("page")} for c in top]

    # Convert to "distance": lower is better
    distances = []
    for i in range(len(top)):
        s = scored[i][0]
        distances.append(1.0 / (s + 1e-6))

    return {"documents": [documents], "metadatas": [metadatas], "distances": [distances]}

