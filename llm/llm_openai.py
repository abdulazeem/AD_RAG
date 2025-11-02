# rag_app/llm/llm_openai.py

import os
from openai import OpenAI
from typing import Dict, Any
from llm.llm_base import LLMBackend
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()

class OpenAILLM(LLMBackend):
    def __init__(self, api_key: str = None, model: str = None, timeout: int = None):
        # Use provided values or fall back to settings
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or settings.openai.api_key
        self.client = OpenAI(api_key=self.api_key)
        self.model = model or settings.openai.model
        self.timeout = timeout or settings.openai.timeout_seconds

    def generate(self, prompt: str, **kwargs):
        """
        Generate a response given a prompt string using the OpenAI ChatCompletion API.
        Returns tuple of (text, usage_dict with prompt_tokens and completion_tokens).
        """
        messages = [
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            **kwargs
        )
        text = response.choices[0].message.content.strip()
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens
        }
        return text, usage

    def score_document(self, query: str, document: str, **kwargs) -> float:
        """
        Score a document's relevance to a query via OpenAI model.
        Returns a float score (for example 1-10).
        """
        prompt = (
            f"Rate the relevance of the following document to the query on a scale from 1 to 10:\n\n"
            f"Query: {query}\n\n"
            f"Document: {document}\n\n"
            "Relevance Score (1-10):"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=self.timeout,
            **kwargs
        )
        raw_score = response.choices[0].message.content.strip()
        try:
            return float(raw_score)
        except ValueError:
            # Fallback parse
            import re
            m = re.search(r"\d+(?:\.\d+)?", raw_score)
            return float(m.group()) if m else 0.0
