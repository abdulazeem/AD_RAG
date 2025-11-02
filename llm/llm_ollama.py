# rag_app/llm/llm_ollama.py

import os
from typing import Dict, Any
from llm.llm_base import LLMBackend
from config.settings import settings

# Import the Ollama Python library client
import ollama
from dotenv import load_dotenv
load_dotenv()

class OllamaLLM(LLMBackend):
    def __init__(self, host: str = None, model: str = None):
        """
        Implementation of LLMBackend using Ollama local/self‐hosted service.
        :param host: Base URL of Ollama API (e.g., http://localhost:11434)
        :param model: Name of the loaded model in Ollama (e.g., "llama3.2", "gemma3")
        :param timeout: Optional timeout in seconds for calls
        """
        self.host = host or settings.ollama.host
        self.model = model or settings.ollama.model
        # self.timeout = timeout or settings.ollama.timeout_seconds
        # Optionally set environment variable or configure client
        self.client = ollama.Client(host=self.host)  # or default
        # note: depending on version of ollama library you may call ollama.chat or ollama.generate

    def generate(self, prompt: str, **kwargs):
        """
        Generate a response string for a given prompt via Ollama.
        Returns tuple of (text, usage_dict with prompt_tokens and completion_tokens).
        """
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # timeout=self.timeout,
            **kwargs
        )
        # The response object may have .message.content or ['message']['content'] depending on version
        try:
            text = response.message.content
        except AttributeError:
            text = response["message"]["content"]

        # Extract token usage from Ollama response
        usage = {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0)
        }
        return text, usage

    def score_document(self, query: str, document: str, **kwargs) -> float:
        """
        Score how relevant the document is to the query using Ollama.
        Returns a float (e.g., 1-10 relevance).
        """
        # Example prompt for scoring
        prompt = (
            f"Rate the relevance of the following document to the query on a scale from 1 to 10:\n\n"
            f"Query: {query}\nDocument: {document}\n\n"
            "Relevance Score (1-10):"
        )
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # timeout=self.timeout,
            **kwargs
        )
        try:
            raw = response.message.content.strip()
        except AttributeError:
            raw = response["message"]["content"].strip()
        try:
            return float(raw)
        except ValueError:
            # fallback parse numeric
            import re
            m = re.search(r"\d+(?:\.\d+)?", raw)
            return float(m.group()) if m else 0.0
