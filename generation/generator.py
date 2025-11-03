# rag_app/generation/generator.py

from typing import List, Dict, Any
import os
from config.settings import settings
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate  # support for prompt templates :contentReference[oaicite:0]{index=0}
from observability.phoenix_prompt_manager import get_prompt_manager
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

        # Load prompt template from Phoenix, file, or use default fallback
        template_str = None
        prompt_version = None

        # Try to load from Phoenix first
        try:
            prompt_manager = get_prompt_manager()
            prompt_data = prompt_manager.get_prompt("rag_generation")
            template_str = prompt_data["template"]
            prompt_version = prompt_data["version"]
            print(f"[Generator] Loaded prompt 'rag_generation' version {prompt_version} from Phoenix")
        except Exception as e:
            print(f"[Generator] Warning: Could not load prompt from Phoenix: {e}")
            print(f"[Generator] Falling back to file or default prompt")

        # Fallback to file if Phoenix failed
        if template_str is None:
            template_path = getattr(settings, 'prompt_template_path', None)
            if template_path and os.path.isfile(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    template_str = f.read()
            else:
                # Default prompt template with conversation history
                template_str = """You are a helpful AI assistant. Answer the user's question based on the provided context{history_note}.

{conversation_history}Context:
{context}

Question: {query}

Answer:"""

        self.prompt_template: PromptTemplate = PromptTemplate.from_template(template_str)
        self.prompt_version = prompt_version

    def build_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        """
        Builds the prompt string given a user query, list of chunk dicts, and conversation history.

        Args:
            query: Current user query
            chunks: List of retrieved context chunks with 'text' and 'metadata'
            conversation_history: List of previous messages with 'role' and 'content'
        """
        # Combine chunk texts into one context string
        context = "\n\n".join([chunk["text"] for chunk in chunks])

        # Format conversation history
        if conversation_history and len(conversation_history) > 0:
            # Limit to last 10 messages to avoid token overflow
            recent_history = conversation_history[-10:]

            history_text = "Previous Conversation:\n"
            for msg in recent_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    history_text += f"User: {content}\n"
                elif role == "assistant":
                    history_text += f"Assistant: {content}\n"
            history_text += "\n"
            history_note = " and our previous conversation"
        else:
            history_text = ""
            history_note = ""

        prompt = self.prompt_template.format(
            query=query,
            context=context,
            conversation_history=history_text,
            history_note=history_note
        )
        return prompt

    def generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Generate an answer to the query using provided context chunks and conversation history.

        Args:
            query: Current user query
            chunks: List of retrieved context chunks
            conversation_history: List of previous messages with 'role' and 'content'

        Returns:
            Tuple of (answer text, usage dict with prompt_tokens and completion_tokens)
        """
        prompt = self.build_prompt(query, chunks, conversation_history)
        answer, usage = self.llm.generate(prompt, **kwargs)
        return answer, usage
