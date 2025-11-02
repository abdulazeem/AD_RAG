# rag_app/generation/api/dependencies.py

from fastapi import Depends
from embeddings.vector_store import VectorStore
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from config.settings import settings

def get_vector_store() -> VectorStore:
    return VectorStore()

def get_llm_backend() -> LLMBackend:
    if settings.llm_backend.lower() == "openai":
        return OpenAILLM(api_key=settings.openai.api_key, model=settings.openai.model)
    elif settings.llm_backend.lower() == "ollama":
        return OllamaLLM(host=settings.ollama.host, model=settings.ollama.model)
    else:
        raise ValueError(f"Unsupported LLM backend: {settings.llm_backend}")

# You can add more dependencies (e.g., embeddings, retriever, etc.) here
