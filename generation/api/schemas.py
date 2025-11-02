# rag_app/generation/api/schemas.py

from typing import List, Optional
from pydantic import BaseModel

class IngestRequest(BaseModel):
    file_path: str
    backend: Optional[str] = None  # "openai" or "ollama", defaults to settings.embedding_backend

class IngestResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    llm_backend: Optional[str] = None  # "openai" or "ollama", defaults to settings.llm_backend
    chat_session_id: Optional[str] = None  # Chat session ID for maintaining conversation history

class DocumentChunk(BaseModel):
    text: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    used_chunks: List[DocumentChunk]
    cost_usd: float
    chat_session_id: str  # The chat session ID (new or existing)

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
