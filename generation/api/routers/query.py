# rag_app/generation/api/routers/query.py

from fastapi import APIRouter
from generation.api.schemas import QueryRequest, QueryResponse, DocumentChunk
from retrieval.retrieval_pipeline import RetrievalPipeline
from generation.generator import Generator
from database.chat_manager import ChatManager
import json

router = APIRouter()
chat_manager = ChatManager()


@router.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query = request.query
    llm_backend = request.llm_backend
    chat_session_id = request.chat_session_id
    selected_documents = request.selected_documents

    # Create new chat session if not provided
    if not chat_session_id:
        chat_session_id = chat_manager.create_chat_session(query)

    # Retrieve conversation history
    conversation_history = chat_manager.get_chat_messages(chat_session_id)

    # Save user query
    chat_manager.add_message(
        chat_session_id=chat_session_id,
        role="user",
        content=query
    )

    # Step 1: Retrieval (and optional reranking)
    pipeline = RetrievalPipeline(backend=llm_backend)
    chunks, retrieval_stats = pipeline.run(query, document_filter=selected_documents)

    # Step 2: LLM Generation (handled by Phoenix tracing automatically)
    generator = Generator(backend=llm_backend)
    answer, usage = generator.generate_answer(query, chunks, conversation_history)

    # Step 3: Save assistant message and retrieved sources
    sources_json = json.dumps([{
        "text": chunk["text"],
        "metadata": chunk["metadata"]
    } for chunk in chunks])

    chat_manager.add_message(
        chat_session_id=chat_session_id,
        role="assistant",
        content=answer,
        sources=sources_json
    )

    used_chunks = [
        DocumentChunk(text=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    # Phoenix will handle cost, latency, and span traces automatically
    return QueryResponse(
        answer=answer,
        used_chunks=used_chunks,
        chat_session_id=chat_session_id,
        retrieved_count=retrieval_stats.get("retrieved_count"),
        reranked_count=retrieval_stats.get("reranked_count")
    )
