# rag_app/llm/llm_base.py

from typing import Dict, Any, Tuple
from dotenv import load_dotenv
load_dotenv()

class LLMBackend:
    """
    Abstract interface for an LLM backend.
    Implementations must provide:
      - generate(prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]
      - score_document(query: str, document: str, **kwargs) -> float
    """
    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate a text response for a given prompt.
        :param prompt: The input prompt to the LLM.
        :param kwargs: Additional keyword arguments for the generation, e.g. max_tokens, temperature.
        :return: Tuple of (generated text, usage dict with 'prompt_tokens' and 'completion_tokens')
        """
        raise NotImplementedError("Method generate() must be implemented by subclass")

    def score_document(self, query: str, document: str, **kwargs) -> float:
        """
        Score how relevant a document is to a query.
        :param query: The user’s query.
        :param document: The document text.
        :param kwargs: Additional keyword arguments (e.g., prompt template details).
        :return: A numeric relevance score.
        """
        raise NotImplementedError("Method score_document() must be implemented by subclass")
