
# rag_app/retrieval/reranker_pointwise.py

from typing import List, Dict, Any
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()

class RerankerPointwise:
    def __init__(self, backend: str = None):
        """
        Initialize Reranker with dynamic backend support.

        Args:
            backend: "openai" or "ollama". If None, uses settings.llm_backend
        """
        backend_name = (backend or settings.llm_backend).lower()
        if backend_name == "openai":
            self.llm: LLMBackend = OpenAILLM(
                api_key=settings.openai.api_key,
                model=settings.openai.model
            )
        elif backend_name == "ollama":
            self.llm: LLMBackend = OllamaLLM(
                host=settings.ollama.host,
                model=settings.ollama.model
            )
        else:
            raise ValueError(f"Unsupported LLM backend for reranking: {backend_name}")

    def score_documents(
        self,
        query: str,
        docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Given a user query and a list of document dicts (each with 'text' and 'metadata'),
        call the LLM to score each document individually, then return the list of docs
        enriched with a 'score' field, and sorted descending by score.
        """
        scored_docs: List[Dict[str, Any]] = []

        for doc in docs:
            text = doc["text"]
            metadata = doc.get("metadata", {})
            # format prompt
            prompt = (
                f"Rate the relevance of the following document to the query on a scale from 1 to 10:\n\n"
                f"Query: {query}\n"
                f"Document: {text}\n\n"
                f"Relevance Score (1-10):"
            )
            # call LLM (returns tuple of (text, usage))
            resp, usage = self.llm.generate(prompt)
            try:
                score = float(resp.strip())
            except ValueError:
                # fallback: if the model returns text, attempt to parse numeric part
                import re
                m = re.search(r"\d+(?:\.\d+)?", resp)
                score = float(m.group()) if m else 0.0

            # enrich doc
            enriched = {
                "text": text,
                "metadata": metadata,
                "score": score
            }
            scored_docs.append(enriched)

        # sort by score descending
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs

    def rerank_top_m(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_m: int
    ) -> List[Dict[str, Any]]:
        """
        Perform pointwise scoring then return only the top_m highest‐scored documents.
        """
        scored = self.score_documents(query, docs)
        return scored[:top_m]
