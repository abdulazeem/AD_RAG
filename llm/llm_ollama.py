# rag_app/llm/llm_ollama.py

from typing import Dict, Any, Tuple
from llm.llm_base import LLMBackend
from config.settings import settings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

load_dotenv()


class OllamaLLM(LLMBackend):
    """
    LLMBackend implementation using LangChain's ChatOllama wrapper.
    This enables Phoenix OpenInference auto-tracing.
    """

    def __init__(self, host: str = None, model: str = None):
        self.host = host or settings.ollama.host
        self.model = model or settings.ollama.model

        # LangChain ChatOllama automatically respects `base_url` and integrates with Phoenix
        self.llm = ChatOllama(
            model=self.model,
            base_url=self.host,
            temperature=0.7,
        )

    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate a response string for a given prompt via LangChain's ChatOllama.
        Phoenix will automatically trace this call.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)], **kwargs)
        text = response.content.strip() if hasattr(response, "content") else str(response)

        # LangChain Ollama doesn’t provide token counts yet; send placeholders
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return text, usage

    def score_document(self, query: str, document: str, **kwargs) -> float:
        """
        Score document relevance to query (1–10) using Ollama via LangChain.
        """
        prompt = (
            f"Rate the relevance of the following document to the query on a scale from 1 to 10:\n\n"
            f"Query: {query}\nDocument: {document}\n\n"
            "Relevance Score (1-10):"
        )
        response = self.llm.invoke([HumanMessage(content=prompt)], **kwargs)
        raw = response.content.strip()
        try:
            return float(raw)
        except ValueError:
            import re
            m = re.search(r"\d+(?:\.\d+)?", raw)
            return float(m.group()) if m else 0.0
