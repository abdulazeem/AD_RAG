# rag_app/generation/api/routers/admin.py

from fastapi import APIRouter, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, JSON
from pgvector.sqlalchemy import Vector
from config.settings import settings
from config.backend_config import BackendConfig
import uuid

router = APIRouter()

class BackendType(str, Enum):
    openai = "openai"
    ollama = "ollama"

class TableInfoResponse(BaseModel):
    table_name: str
    backend: str
    dimensions: int
    row_count: int
    exists: bool

class ResetResponse(BaseModel):
    success: bool
    message: str
    table_name: str
    backend: str

class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int

class DocumentsListResponse(BaseModel):
    backend: str
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]

@router.get("/health", tags=["admin"])
async def health_check():
    return {"status": "healthy"}

@router.get("/version", tags=["admin"])
async def version_check():
    return {"version": "0.1.0"}

@router.get("/tables", response_model=List[TableInfoResponse], tags=["admin"])
async def list_tables():
    """Get information about all vector tables."""
    try:
        engine = create_engine(settings.database.postgres_url)
        tables_info = []

        for backend in ["openai", "ollama"]:
            config = BackendConfig.get_config(backend)
            table_name = config["vector_table"]
            dimensions = config["embedding_dimensions"]

            with engine.connect() as conn:
                # Check if table exists
                check_query = text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = :table_name
                    )
                """)
                exists = conn.execute(check_query, {"table_name": table_name}).scalar()

                row_count = 0
                if exists:
                    # Get row count
                    count_query = text(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = conn.execute(count_query).scalar()

                tables_info.append(TableInfoResponse(
                    table_name=table_name,
                    backend=backend,
                    dimensions=dimensions,
                    row_count=row_count,
                    exists=exists
                ))

        engine.dispose()
        return tables_info

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tables: {str(e)}")

@router.post("/tables/reset", response_model=ResetResponse, tags=["admin"])
async def reset_table(
    backend: BackendType = Form(...),
    confirm: bool = Form(False)
):
    """
    Drop and recreate a vector table.

    Args:
        backend: "openai" or "ollama"
        confirm: Must be True to proceed (safety check)
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set 'confirm=true' to reset table. This will delete all data!"
        )

    try:
        config = BackendConfig.get_config(backend.value)
        table_name = config["vector_table"]
        dimensions = config["embedding_dimensions"]

        engine = create_engine(settings.database.postgres_url)

        with engine.connect() as conn:
            # Drop table if exists
            drop_query = text(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(drop_query)
            conn.commit()

            # Recreate table
            metadata = MetaData()
            Table(
                table_name,
                metadata,
                Column('id', String, primary_key=True, default=lambda: str(uuid.uuid4())),
                Column('text', String, nullable=False),
                Column('chunk_metadata', JSON, nullable=True),
                Column('embedding', Vector(dimensions), nullable=False),
            )
            metadata.create_all(engine)

        engine.dispose()

        return ResetResponse(
            success=True,
            message=f"Table '{table_name}' has been reset successfully",
            table_name=table_name,
            backend=backend.value
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting table: {str(e)}")

@router.delete("/tables/{backend}/clear", response_model=ResetResponse, tags=["admin"])
async def clear_table(backend: BackendType, confirm: bool = False):
    """
    Delete all rows from a vector table without dropping it.

    Args:
        backend: "openai" or "ollama"
        confirm: Must be True to proceed (safety check)
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set '?confirm=true' query parameter to clear table. This will delete all data!"
        )

    try:
        config = BackendConfig.get_config(backend.value)
        table_name = config["vector_table"]

        engine = create_engine(settings.database.postgres_url)

        with engine.connect() as conn:
            # Delete all rows
            delete_query = text(f"DELETE FROM {table_name}")
            result = conn.execute(delete_query)
            conn.commit()
            rows_deleted = result.rowcount

        engine.dispose()

        return ResetResponse(
            success=True,
            message=f"Deleted {rows_deleted} rows from '{table_name}'",
            table_name=table_name,
            backend=backend.value
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing table: {str(e)}")

@router.get("/documents/{backend}", response_model=DocumentsListResponse, tags=["admin"])
async def list_documents(backend: BackendType):
    """
    List all unique documents in the vector database for a given backend.

    Args:
        backend: "openai" or "ollama"

    Returns:
        List of documents with their chunk counts
    """
    try:
        config = BackendConfig.get_config(backend.value)
        table_name = config["vector_table"]

        engine = create_engine(settings.database.postgres_url)

        with engine.connect() as conn:
            # Check if table exists
            check_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = :table_name
                )
            """)
            exists = conn.execute(check_query, {"table_name": table_name}).scalar()

            if not exists:
                return DocumentsListResponse(
                    backend=backend.value,
                    total_documents=0,
                    total_chunks=0,
                    documents=[]
                )

            # Get unique documents with chunk counts
            query = text(f"""
                SELECT
                    chunk_metadata->>'source_file' as filename,
                    COUNT(*) as chunk_count
                FROM {table_name}
                WHERE chunk_metadata->>'source_file' IS NOT NULL
                GROUP BY chunk_metadata->>'source_file'
                ORDER BY filename
            """)

            result = conn.execute(query)
            documents = []
            total_chunks = 0

            for row in result:
                documents.append(DocumentInfo(
                    filename=row.filename,
                    chunk_count=row.chunk_count
                ))
                total_chunks += row.chunk_count

        engine.dispose()

        return DocumentsListResponse(
            backend=backend.value,
            total_documents=len(documents),
            total_chunks=total_chunks,
            documents=documents
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")
