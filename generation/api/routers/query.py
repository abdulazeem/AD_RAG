# rag_app/generation/api/routers/query.py

from fastapi import APIRouter, Depends
from generation.api.schemas import QueryRequest, QueryResponse, DocumentChunk
from generation.api.dependencies import get_vector_store, get_llm_backend
from retrieval.retrieval_pipeline import RetrievalPipeline
from generation.generator import Generator
from observability.cost_tracker import track_llm_usage
from observability.prompt_tracking import track_prompt
from config.settings import settings
from database.chat_manager import ChatManager
import json

router = APIRouter()
chat_manager = ChatManager()

@router.post("/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    query = request.query
    llm_backend = request.llm_backend  # Use requested backend or default
    chat_session_id = request.chat_session_id
    selected_documents = request.selected_documents  # Optional document filter

    # Create new chat session if not provided
    if not chat_session_id:
        chat_session_id = chat_manager.create_chat_session(query)

    # Get previous conversation history (excluding current query)
    conversation_history = chat_manager.get_chat_messages(chat_session_id)

    # Save user message to chat history
    chat_manager.add_message(
        chat_session_id=chat_session_id,
        role="user",
        content=query
    )

    # Retrieval + reranking (use same backend as LLM for consistency)
    # Pass selected_documents to filter results
    pipeline = RetrievalPipeline(backend=llm_backend)
    chunks, retrieval_stats = pipeline.run(query, document_filter=selected_documents)

    # Build prompt with conversation history and track
    generator = Generator(backend=llm_backend)
    prompt = generator.build_prompt(query, chunks, conversation_history)
    track_prompt(
        template=prompt,
        variables={
            "query": query,
            "chunk_count": len(chunks),
            "llm_backend": generator.backend_name,
            "history_length": len(conversation_history)
        }
    )

    # Generate answer with conversation history and capture token usage
    answer, usage = generator.generate_answer(query, chunks, conversation_history)

    # Track actual token usage
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    track_llm_usage(
        model_name=generator.model_name,
        provider=generator.backend_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens
    )

    # Build response
    used_chunks = [
        DocumentChunk(text=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunks
    ]

    # Save assistant message to chat history
    sources_json = json.dumps([{
        "text": chunk["text"],
        "metadata": chunk["metadata"]
    } for chunk in chunks])

    chat_manager.add_message(
        chat_session_id=chat_session_id,
        role="assistant",
        content=answer,
        sources=sources_json,
        cost=0.0  # adjust cost if needed
    )

    return QueryResponse(
        answer=answer,
        used_chunks=used_chunks,
        cost_usd=0.0,
        chat_session_id=chat_session_id,
        retrieved_count=retrieval_stats.get("retrieved_count"),
        reranked_count=retrieval_stats.get("reranked_count")
    )
