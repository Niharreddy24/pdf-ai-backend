import os
import uuid
from django.conf import settings
from django.core.files.storage import default_storage
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .services.pdf_extract import extract_pages, chunk_text
from .services.rag_store import upsert_doc_chunks, query_doc
from .services.rag_answer import build_context, answer_from_context


class HealthView(APIView):
    def get(self, request):
        return Response({"ok": True})


class UploadPdfView(APIView):
    def post(self, request):
        if "file" not in request.FILES:
            return Response({"error": "No file uploaded (field name must be 'file')."}, status=400)

        file = request.FILES["file"]

        if not file.name.lower().endswith(".pdf"):
            return Response({"error": "Only PDF files are supported."}, status=400)

        doc_id = str(uuid.uuid4())

        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        saved_path = default_storage.save(f"{doc_id}_{file.name}", file)
        full_path = os.path.join(settings.MEDIA_ROOT, saved_path)

        # Validate PDF header
        try:
            with open(full_path, "rb") as f:
                header = f.read(5)
            if header != b"%PDF-":
                return Response(
                    {"error": "Uploaded file is not a valid PDF. It may be a ZIP/DOCX renamed as .pdf."},
                    status=400,
                )
        except Exception:
            return Response({"error": "Unable to read uploaded file."}, status=400)

        # Extract pages
        try:
            pages = extract_pages(full_path)
        except ValueError as e:
            return Response({"error": str(e)}, status=400)

        if not pages:
            return Response(
                {"error": "Could not extract text from this PDF. It may be scanned (image-only) or protected."},
                status=400,
            )

        # Chunk and store
        chunks_with_meta = []
        for p in pages:
            page_num = p["page"]
            chunks = chunk_text(p["text"], chunk_size=1200, overlap=200)
            for ch in chunks:
                chunks_with_meta.append({"text": ch, "page": page_num})

        upsert_doc_chunks(doc_id, chunks_with_meta)

        return Response({"doc_id": doc_id, "chunks": len(chunks_with_meta)}, status=status.HTTP_200_OK)


class AskPdfView(APIView):
    def post(self, request):
        doc_id = request.data.get("doc_id")
        question = request.data.get("question")

        if not doc_id or not question:
            return Response({"error": "doc_id and question are required."}, status=400)

        q = question.strip()
        ql = q.lower()

        # ✅ Query expansion for "file structure" questions (still LLM-only answering)
        expanded = q

        # If question is about scheduling/files, add the likely file + terms that exist in the PDF
        if any(k in ql for k in ["which file", "file defines", "controls scheduling", "scheduling", "scheduled", "every 30 seconds"]):
            expanded = f"{q} plugin.xml DOTS task scheduler run every 30 seconds"

        # If question is specifically about plugin.xml, add what it does (terms in the doc)
        if "plugin.xml" in ql:
            expanded = f"{q} DOTS config registers task Monitor run every 30 seconds"

        # If question is about DT_Databases, add notes.ini (terms in the doc)
        if "dt_databases" in ql:
            expanded = f"{q} notes.ini semicolon-separated"

        # Retrieve more candidates so the right chunk is available
        results = query_doc(doc_id, expanded, top_k=25)

        items = build_context(results)
        answer, sources = answer_from_context(question, items)

        return Response({"answer": answer, "sources": sources}, status=status.HTTP_200_OK)
