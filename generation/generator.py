# rag_app/generation/generator.py

from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from config.settings import settings
from llm.llm_base import LLMBackend
from llm.llm_openai import OpenAILLM
from llm.llm_ollama import OllamaLLM
from observability.phoenix_tracer import init_phoenix_tracing

load_dotenv()
# Phoenix tracing is initialized in main.py startup
init_phoenix_tracing(project_name="rag-llm-app")  # Idempotent - safe to call multiple times

# Default RAG prompts
DEFAULT_RAG_PROMPT = """You are a helpful AI assistant. Use ONLY the given context and chat history to answer. If the answer is not in the documents, say "I don't know based on the provided documents."

--- RULES ---
- Answer clearly and concisely
- Cite context sources if provided (Chunk numbers)
- Never hallucinate unsupported facts
- Respect chat history
- If context is empty: politely say you don't have enough information

Context:
{context}

Chat History:
{conversation_history}

Question: {query}"""

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
            self.backend_type = "openai"
        elif backend_name == "ollama":
            self.llm: LLMBackend = OllamaLLM(
                host=settings.ollama.host,
                model=settings.ollama.model
            )
            self.prompt_identifier = settings.prompts.ollama_prompt  # use identifier
            self.backend_type = "ollama"
        else:
            raise ValueError(f"Unsupported LLM backend: {backend_name}")

        # Try to load prompt version from Phoenix, fallback to default
        self.prompt_version = None
        self.use_default_prompt = False
        try:
            from phoenix.client import Client
            client = Client()
            prompt_obj = client.prompts.get(prompt_identifier=self.prompt_identifier)
            self.prompt_version = prompt_obj
            print(f"[Generator] Loaded prompt '{self.prompt_identifier}' from Phoenix")
        except Exception as e:
            print(f"[Generator] Warning: Could not load prompt from Phoenix: {e}")
            print(f"[Generator] Using default RAG prompt for {backend_name}")
            self.use_default_prompt = True

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
        variables = self.build_prompt_variables(query, chunks, conversation_history)

        # Use default prompt if Phoenix is unavailable
        if self.use_default_prompt:
            # Format the default prompt with variables
            prompt_text = DEFAULT_RAG_PROMPT.format(**variables)

            if self.backend_type == "openai":
                # For OpenAI, wrap in messages format
                response = self.llm.client.chat.completions.create(
                    model=settings.openai.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant."},
                        {"role": "user", "content": prompt_text}
                    ]
                )
                answer = response.choices[0].message.content
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            else:  # Ollama
                response, usage = self.llm.generate(prompt_text, **kwargs)
                answer = response
        else:
            # Format the prompt using Phoenix prompt version
            formatted = self.prompt_version.format(variables=variables)

            # Invoke the correct LLM
            if isinstance(self.llm, OpenAILLM):
                # Phoenix returns {"messages": [...], "model": "...", "temperature": ...}
                # Send it directly to OpenAI chat endpoint
                response = self.llm.client.chat.completions.create(**formatted)
                answer = response.choices[0].message.content

                # Extract usage information from response
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            else:  # OllamaLLM
                # Phoenix returns messages format, need to convert to single prompt string
                # Check if formatted has "messages" key (chat format) or "prompt" key (completion format)
                if "messages" in formatted:
                    # Convert messages to a single prompt string
                    messages = formatted.get("messages", [])
                    ollama_prompt = ""
                    for msg in messages:
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        if role == "system":
                            ollama_prompt += f"System: {content}\n\n"
                        elif role == "user":
                            ollama_prompt += f"User: {content}\n\n"
                        elif role == "assistant":
                            ollama_prompt += f"Assistant: {content}\n\n"
                        else:
                            ollama_prompt += f"{content}\n\n"
                    ollama_prompt = ollama_prompt.strip()
                else:
                    # Fallback to prompt key if present
                    ollama_prompt = formatted.get("prompt", "")

                response, usage = self.llm.generate(ollama_prompt, **kwargs)
                answer = response

        return answer, usage

