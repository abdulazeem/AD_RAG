# rag_app/generation/generator.py

from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from phoenix.client import Client
from config.settings import settings
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from observability.phoenix_tracer import init_phoenix_tracing

load_dotenv()
init_phoenix_tracing(project_name="rag-llm-app")

class Generator:
    def __init__(self, backend: str = None):
        backend_name = (backend or settings.llm_backend).lower()

        if backend_name == "openai":
            self.llm: LLMBackend = OpenAILLM(
                api_key=settings.openai.api_key,
                model=settings.openai.model,
                timeout=settings.openai.timeout_seconds
            )
            self.prompt_identifier = settings.prompts.openai_prompt  # use identifier
        elif backend_name == "ollama":
            self.llm: LLMBackend = OllamaLLM(
                host=settings.ollama.host,
                model=settings.ollama.model
            )
            self.prompt_identifier = settings.prompts.ollama_prompt  # use identifier
        else:
            raise ValueError(f"Unsupported LLM backend: {backend_name}")

        # Load prompt version from Phoenix
        client = Client()
        prompt_obj = client.prompts.get(prompt_identifier=self.prompt_identifier)
        self.prompt_version = prompt_obj

    def build_prompt_variables(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        context = "\n\n".join([c["text"] for c in chunks])
        history_text = ""
        if conversation_history:
            for msg in conversation_history[-10:]:
                role = msg.get("role")
                content = msg.get("content")
                history_text += f"{role.capitalize()}: {content}\n"
        return {
            "query": query,
            "context": context,
            "conversation_history": history_text.strip(),
        }

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ):
        # Format the prompt using Phoenix prompt version
        variables = self.build_prompt_variables(query, chunks, conversation_history)
        formatted = self.prompt_version.format(variables=variables)

        # Invoke the correct LLM
        if isinstance(self.llm, OpenAILLM):
            response = self.llm.invoke_with_formatted_prompt(formatted, **kwargs)
        else:  # OllamaLLM
            response, usage = self.llm.generate(formatted["prompt"], **kwargs)

        answer = response.choices[0].message.content if hasattr(response, "choices") else response
        usage = getattr(response, "usage", {})
        return answer, usage
