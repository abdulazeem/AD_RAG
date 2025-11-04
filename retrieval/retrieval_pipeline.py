# rag_app/retrieval/retrieval_pipeline.py

from typing import List, Dict, Any, Optional
from config.settings import settings
from .retriever import Retriever
from .reranker_pointwise import RerankerPointwise
from dotenv import load_dotenv
load_dotenv()

class RetrievalPipeline:
    def __init__(self, backend: str = None):
        """
        Initialize RetrievalPipeline with dynamic backend support.

        Args:
            backend: "openai" or "ollama". If None, uses settings.embedding_backend
        """
        self.backend = backend or settings.embedding_backend
        self.retriever = Retriever(backend=self.backend)
        self.reranker = RerankerPointwise(backend=self.backend)
        self.rerank_top_m = settings.retrieval.rerank_top_m

    def run(
        self,
        query: str,
        document_filter: Optional[List[str]] = None
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Full pipeline:
          1. Retrieve top K candidate chunks via vector store.
          2. Rerank them point-wise with LLM.
          3. Return the top M reranked chunks along with retrieval stats.

        Args:
            query: Query string
            document_filter: Optional list of document filenames to filter by

        Returns:
            Tuple of (reranked chunks, retrieval stats dict)
        """
        # Step 1: retrieval with optional document filter
        candidates = self.retriever.retrieve(query, document_filter=document_filter)
        # Each candidate: { "text": ..., "metadata": ..., "distance": ... }

        # Step 2: reranking (point-wise)
        reranked = self.reranker.rerank_top_m(
            query=query,
            docs=candidates,
            top_m=self.rerank_top_m
        )

        # Step 3: prepare retrieval stats
        retrieval_stats = {
            "retrieved_count": len(candidates),
            "reranked_count": len(reranked)
        }

        return reranked, retrieval_stats
