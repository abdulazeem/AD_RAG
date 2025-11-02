# rag_app/embeddings/embedder.py

import os
from typing import List, Optional
from config.settings import settings
from config.backend_config import BackendConfig

# Import LangChain embedding interface
from langchain_core.embeddings.embeddings import Embeddings  # base interface :contentReference[oaicite:4]{index=4}

# Import provider‐specific embedding classes
from langchain_openai.embeddings import OpenAIEmbeddings  # :contentReference[oaicite:5]{index=5}

# Assuming you have a class for Ollama embeddings (you may need to install or implement)
# from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

class Embedder:
    def __init__(self, backend: str = None):
        """
        Initialize Embedder with dynamic backend configuration.

        Args:
            backend: "openai" or "ollama". If None, uses settings.embedding_backend
        """
        # Use provided backend or fall back to settings
        backend = (backend or settings.embedding_backend).lower()

        # Get dynamic configuration for this backend
        config = BackendConfig.get_config(backend)
        self.backend = backend
        self.model_name = config["embedding_model"]
        self.dimensions = config["embedding_dimensions"]

        if backend == "openai":
            self.embeddings: Embeddings = OpenAIEmbeddings(model=self.model_name)
        elif backend == "ollama":
            # Example wrapper: replace with actual Ollama embeddings class or custom implementation
            from langchain_ollama.embeddings import OllamaEmbeddings
            self.embeddings: Embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=settings.ollama.host
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

        self.batch_size: int = settings.embedding.batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts.
        Returns a list of embedding vectors.
        """
        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self.embeddings.embed_documents(batch)
            all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        Returns one embedding vector.
        """
        return self.embeddings.embed_query(text)
