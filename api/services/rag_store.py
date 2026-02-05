import os
import chromadb
from django.conf import settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Local embedding model (free, fast, stable)
embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

COLLECTION_NAME = "pdf_chunks"


def get_chroma():
    os.makedirs(settings.CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.CHROMA_DIR))


def get_collection():
    chroma = get_chroma()
    return chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}  # better similarity behavior
    )


def upsert_doc_chunks(doc_id: str, chunks_with_meta: list[dict]):
    if not chunks_with_meta:
        return

    texts = [c["text"].strip() for c in chunks_with_meta]
    ids = [f"{doc_id}_{i}" for i in range(len(texts))]
    metadatas = [{"doc_id": doc_id, "page": c["page"]} for c in chunks_with_meta]

    col = get_collection()
    col.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )


def query_doc(doc_id: str, question: str, top_k: int = 6):
    """
    Query only chunks belonging to this document.
    """
    if not question.strip():
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    col = get_collection()

    results = col.query(
        query_texts=[question.strip()],
        n_results=top_k,
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"],
    )

    return results
