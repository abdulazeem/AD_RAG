"""
Dynamic backend configuration that automatically adjusts settings based on the selected backend.
"""

from typing import Dict, Any
from config.settings import settings


class BackendConfig:
    """Configuration manager that provides backend-specific settings."""

    # Configuration for each backend
    BACKEND_CONFIGS = {
        "openai": {
            "embedding_model": "text-embedding-ada-002",
            "embedding_dimensions": 1536,
            "vector_table": "chunk_embeddings",
            "embedding_class": "openai"
        },
        "ollama": {
            "embedding_model": "nomic-embed-text",
            "embedding_dimensions": 768,
            "vector_table": "chunk_embeddings_ollama",
            "embedding_class": "ollama"
        }
    }

    @classmethod
    def get_config(cls, backend: str) -> Dict[str, Any]:
        """
        Get configuration for a specific backend.

        Args:
            backend: "openai" or "ollama"

        Returns:
            Dict with backend-specific configuration
        """
        backend = backend.lower()
        if backend not in cls.BACKEND_CONFIGS:
            raise ValueError(f"Unsupported backend: {backend}. Choose 'openai' or 'ollama'")
        return cls.BACKEND_CONFIGS[backend]

    @classmethod
    def get_embedding_model(cls, backend: str) -> str:
        """Get the embedding model name for the backend."""
        return cls.get_config(backend)["embedding_model"]

    @classmethod
    def get_embedding_dimensions(cls, backend: str) -> int:
        """Get the embedding dimensions for the backend."""
        return cls.get_config(backend)["embedding_dimensions"]

    @classmethod
    def get_vector_table(cls, backend: str) -> str:
        """Get the vector table name for the backend."""
        return cls.get_config(backend)["vector_table"]

    @classmethod
    def get_embedding_class(cls, backend: str) -> str:
        """Get the embedding class type for the backend."""
        return cls.get_config(backend)["embedding_class"]
