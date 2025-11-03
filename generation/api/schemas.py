# rag_app/generation/api/schemas.py

from typing import List, Optional
from pydantic import BaseModel

class IngestRequest(BaseModel):
    file_path: str
    backend: Optional[str] = None  # "openai" or "ollama", defaults to settings.embedding_backend

class IngestResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class FileIngestResult(BaseModel):
    filename: str
    success: bool
    message: str
    chunks_created: Optional[int] = None

class BulkIngestResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    results: List[FileIngestResult]
    backend_used: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    llm_backend: Optional[str] = None  # "openai" or "ollama", defaults to settings.llm_backend
    chat_session_id: Optional[str] = None  # Chat session ID for maintaining conversation history
    selected_documents: Optional[List[str]] = None  # Filter results to specific documents (filenames)

class DocumentChunk(BaseModel):
    text: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    used_chunks: List[DocumentChunk]
    chat_session_id: str  # The chat session ID (new or existing)
    retrieved_count: Optional[int] = None  # Number of chunks retrieved before reranking
    reranked_count: Optional[int] = None  # Number of chunks after reranking

class ChatSessionResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    sources: Optional[str] = None
    cost: Optional[float] = None
    timestamp: str

class RerankRequest(BaseModel):
    query: str
    documents: List[str]  # List of document texts to rerank
    llm_backend: Optional[str] = None  # "openai" or "ollama", defaults to settings.llm_backend
    top_k: Optional[int] = None  # Return only top K documents

class RankedDocument(BaseModel):
    text: str
    score: float
    original_index: int  # Index in the original list

class RerankResponse(BaseModel):
    query: str
    ranked_documents: List[RankedDocument]
