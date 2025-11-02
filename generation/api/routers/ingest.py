# rag_app/generation/api/routers/ingest.py

import os
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from generation.api.schemas import IngestResponse
from ingestion.ingest_service import IngestService
from config.settings import settings

router = APIRouter()

@router.post("/test")
async def test_upload(
    file: UploadFile = File(...),
    backend: Optional[str] = Form(None)
):
    """Test endpoint to verify file upload works."""
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "backend": backend,
        "size": len(await file.read())
    }

@router.post("/", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    backend: Optional[str] = Form(None)
):
    """
    Upload and ingest a document.

    Args:
        file: Document file to ingest
        backend: "openai" or "ollama". If not specified, uses settings.embedding_backend
    """
    try:
        print(f"[Ingest API] Received file: {file.filename}, backend: {backend}")

        # Save uploaded file
        raw_folder = settings.data.raw_docs
        os.makedirs(raw_folder, exist_ok=True)
        filepath = os.path.join(raw_folder, file.filename)

        file_content = await file.read()
        print(f"[Ingest API] Read {len(file_content)} bytes")

        with open(filepath, "wb") as f:
            f.write(file_content)
        print(f"[Ingest API] Saved to: {filepath}")

        # Trigger ingestion with specified backend
        print(f"[Ingest API] Starting ingestion with backend: {backend}")
        service = IngestService(backend=backend)
        chunks_meta = service.ingest_single(filepath)

        backend_used = backend or settings.embedding_backend
        response = IngestResponse(
            success=True,
            message=f"Document ingested successfully using {backend_used}. Created {len(chunks_meta)} chunks."
        )
        print(f"[Ingest API] Success: {response.message}")
        return response
    except Exception as e:
        print(f"[Ingest API] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return IngestResponse(success=False, message=str(e))
