from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError


def extract_pages(pdf_path: str):
    """
    Returns a list of dicts:
    [
      {"page": 1, "text": "..."},
      {"page": 2, "text": "..."},
    ]
    """
    try:
        reader = PdfReader(pdf_path, strict=False)
    except (PdfReadError, PdfStreamError, Exception) as e:
        raise ValueError(f"PDF read failed: {str(e)}")

    pages = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    """
    Simple character-based chunker.
    """
    text = (text or "").replace("\x00", " ").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks

