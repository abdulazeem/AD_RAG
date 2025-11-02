# rag_app/retrieval/retriever.py

from typing import List, Dict, Any, Optional
from config.settings import settings
from embeddings.vector_store import VectorStore
from embeddings.embedder import Embedder
from dotenv import load_dotenv
load_dotenv()

class Retriever:
    def __init__(self, backend: str = None):
        """
        Initialize Retriever with dynamic backend support.

        Args:
            backend: "openai" or "ollama". If None, uses settings.embedding_backend
        """
        self.backend = backend or settings.embedding_backend
        self.vector_store = VectorStore(backend=self.backend)
        self.embedder = Embedder(backend=self.backend)
        # Number of chunks to retrieve before reranking
        self.top_k: int = settings.retrieval.top_k

    def retrieve(
        self,
        query: str,
        document_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Given a query string, embed it, retrieve the top_k most relevant chunks
        from the vector store, and return them as a list of dicts
        with text, metadata, and similarity score/distance.

        Args:
            query: Query string
            document_filter: Optional list of document filenames to filter by
        """
        # Step 1: embed the query
        query_vector = self.embedder.embed_query(query)

        # Step 2: perform vector similarity search in vector store with optional filter
        results = self.vector_store.query(
            vector=query_vector,
            top_k=self.top_k,
            document_filter=document_filter
        )

        # Step 3: return results (each entry includes text + metadata + distance/score)
        return results
