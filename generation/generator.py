# rag_app/generation/generator.py

from typing import List, Dict, Any
import os
from config.settings import settings
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate  # support for prompt templates :contentReference[oaicite:0]{index=0}
from dotenv import load_dotenv
load_dotenv()

class Generator:
    def __init__(self, backend: str = None):
        """
        Initialize Generator with specified LLM backend.

        Args:
            backend: LLM backend to use ("openai" or "ollama").
                    If None, uses settings.llm_backend as default.
        """
        backend_name = (backend or settings.llm_backend).lower()

        if backend_name == "openai":
            self.llm: LLMBackend = OpenAILLM(
                api_key=settings.openai.api_key,
                model=settings.openai.model,
                timeout=settings.openai.timeout_seconds
            )
            self.backend_name = "openai"
            self.model_name = settings.openai.model
        elif backend_name == "ollama":
            self.llm: LLMBackend = OllamaLLM(
                host=settings.ollama.host,
                model=settings.ollama.model
            )
            self.backend_name = "ollama"
            self.model_name = settings.ollama.model
        else:
            raise ValueError(f"Unsupported LLM backend: {backend_name}")

        # Load prompt template from file or use default
        template_path = getattr(settings, 'prompt_template_path', None)
        if template_path and os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                template_str = f.read()
        else:
            # Default prompt template
            template_str = """You are a helpful AI assistant. Answer the user's question based on the provided context.

Context:
{context}

Question: {query}

Answer: Based on the context provided above, """

        self.prompt_template: PromptTemplate = PromptTemplate.from_template(template_str)

    def build_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Builds the prompt string given a user query and list of chunk dicts.
        Each chunk dict has 'text' and 'metadata'.
        """
        # combine chunk texts into one context string
        context = "\n\n".join([chunk["text"] for chunk in chunks])
        prompt = self.prompt_template.format(query=query, context=context)
        return prompt

    def generate_answer(self, query: str, chunks: List[Dict[str, Any]], **kwargs):
        """
        Generate an answer to the query using provided context chunks.
        Returns tuple of (answer text, usage dict with prompt_tokens and completion_tokens).
        """
        prompt = self.build_prompt(query, chunks)
        answer, usage = self.llm.generate(prompt, **kwargs)
        return answer, usage
