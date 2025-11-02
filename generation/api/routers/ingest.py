# rag_app/generation/api/routers/ingest.py

import os
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form
from generation.api.schemas import IngestResponse, BulkIngestResponse, FileIngestResult
from ingestion.ingest_service import IngestService
from config.settings import settings
from enum import Enum

router = APIRouter()

class LlmModels(str, Enum):
    openai = "openai"
    ollama = "ollama"

@router.post("/", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    backend: LlmModels = Form(...)
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

@router.post("/bulk", response_model=BulkIngestResponse)
async def upload_multiple_documents(
    files: List[UploadFile] = File(...),
    backend: LlmModels = Form(...)
):
    """
    Upload and ingest multiple documents at once.

    Args:
        files: List of document files to ingest
        backend: "openai" or "ollama". If not specified, uses settings.embedding_backend
    """
    print(f"[Ingest API Bulk] Received {len(files)} files, backend: {backend}")

    results = []
    successful_count = 0
    failed_count = 0
    raw_folder = settings.data.raw_docs
    os.makedirs(raw_folder, exist_ok=True)

    backend_used = backend or settings.embedding_backend

    for file in files:
        try:
            print(f"[Ingest API Bulk] Processing file: {file.filename}")

            # Save uploaded file
            filepath = os.path.join(raw_folder, file.filename)

            file_content = await file.read()
            print(f"[Ingest API Bulk] Read {len(file_content)} bytes from {file.filename}")

            with open(filepath, "wb") as f:
                f.write(file_content)
            print(f"[Ingest API Bulk] Saved to: {filepath}")

            # Trigger ingestion with specified backend
            service = IngestService(backend=backend)
            chunks_meta = service.ingest_single(filepath)

            results.append(FileIngestResult(
                filename=file.filename,
                success=True,
                message=f"Successfully ingested with {len(chunks_meta)} chunks",
                chunks_created=len(chunks_meta)
            ))
            successful_count += 1
            print(f"[Ingest API Bulk] Success: {file.filename} - {len(chunks_meta)} chunks")

        except Exception as e:
            print(f"[Ingest API Bulk] Error processing {file.filename}: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append(FileIngestResult(
                filename=file.filename,
                success=False,
                message=f"Failed: {str(e)}",
                chunks_created=None
            ))
            failed_count += 1

    response = BulkIngestResponse(
        total_files=len(files),
        successful=successful_count,
        failed=failed_count,
        results=results,
        backend_used=backend_used
    )

    print(f"[Ingest API Bulk] Completed: {successful_count} successful, {failed_count} failed")
    return response
