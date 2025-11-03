# rag_app/embeddings/embedder.py
from observability.phoenix_tracer import init_phoenix_tracing
tracer = init_phoenix_tracing()  # must come FIRST

import os
from typing import List
from config.settings import settings
from config.backend_config import BackendConfig
from langchain_core.embeddings.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from opentelemetry.trace import SpanKind
from dotenv import load_dotenv
load_dotenv()

tracer = init_phoenix_tracing()

class Embedder:
    def __init__(self, backend: str = None):
        backend = (backend or settings.embedding_backend).lower()
        config = BackendConfig.get_config(backend)
        self.backend = backend
        self.model_name = config["embedding_model"]
        self.dimensions = config["embedding_dimensions"]

        if backend == "openai":
            self.embeddings: Embeddings = OpenAIEmbeddings(model=self.model_name)
        elif backend == "ollama":
            self.embeddings: Embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=settings.ollama.host
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {backend}")

        self.batch_size: int = settings.embedding.batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        all_vectors: List[List[float]] = []
        with tracer.start_as_current_span("embed_documents", kind=SpanKind.CLIENT) as span:
            span.set_attribute("embedding.backend", self.backend)
            span.set_attribute("embedding.model", self.model_name)
            span.set_attribute("embedding.count", len(texts))
            if self.backend == "ollama":
                span.set_attribute("llm.total_cost_usd", 0.0)

            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                vectors = self.embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        with tracer.start_as_current_span("embed_query", kind=SpanKind.CLIENT) as span:
            span.set_attribute("embedding.backend", self.backend)
            span.set_attribute("embedding.model", self.model_name)
            if self.backend == "ollama":
                span.set_attribute("llm.total_cost_usd", 0.0)
            return self.embeddings.embed_query(text)
