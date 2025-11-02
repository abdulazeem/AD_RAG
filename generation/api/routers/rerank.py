# rag_app/generation/api/routers/rerank.py

from fastapi import APIRouter
from generation.api.schemas import RerankRequest, RerankResponse, RankedDocument
from retrieval.reranker_pointwise import RerankerPointwise
from config.settings import settings

router = APIRouter()

@router.post("/", response_model=RerankResponse)
async def rerank_endpoint(request: RerankRequest):
    """
    Rerank a list of documents based on their relevance to a query.

    Args:
        request: RerankRequest containing query, documents, and optional parameters

    Returns:
        RerankResponse with ranked documents sorted by relevance score
    """
    query = request.query
    documents = request.documents
    llm_backend = request.llm_backend or settings.llm_backend
    top_k = request.top_k

    # Initialize reranker with specified backend
    reranker = RerankerPointwise(backend=llm_backend)

    # Convert documents to the format expected by reranker
    docs_for_reranking = [
        {"text": doc, "metadata": {"original_index": idx}}
        for idx, doc in enumerate(documents)
    ]

    # Score and rank documents
    scored_docs = reranker.score_documents(query, docs_for_reranking)

    # Apply top_k filter if specified
    if top_k:
        scored_docs = scored_docs[:top_k]

    # Convert to response format
    ranked_documents = [
        RankedDocument(
            text=doc["text"],
            score=doc["score"],
            original_index=doc["metadata"]["original_index"]
        )
        for doc in scored_docs
    ]

    return RerankResponse(
        query=query,
        ranked_documents=ranked_documents
    )