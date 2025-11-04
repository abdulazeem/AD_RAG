# rag_app/llm/llm_openai.py

from typing import Dict, Any, Tuple
from llm.llm_base import LLMBackend
from config.settings import settings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from openai import OpenAI

load_dotenv()


class OpenAILLM(LLMBackend):
    """
    LLMBackend implementation using LangChain's ChatOpenAI wrapper.
    Compatible with Phoenix OpenInference auto-tracing.
    Also provides a raw OpenAI client for direct API access.
    """

    def __init__(self, api_key: str = None, model: str = None, timeout: int = None):
        self.api_key = api_key or settings.openai.api_key
        self.model = model or settings.openai.model
        self.timeout = timeout or settings.openai.timeout_seconds

        # LangChain ChatOpenAI automatically integrates with Phoenix tracing
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=0.7,
            openai_api_key=self.api_key,
            timeout=self.timeout,
        )

        # Raw OpenAI client for direct API access (e.g., with Phoenix prompts)
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)

    def generate(self, prompt: str, **kwargs) -> Tuple[str, Dict[str, int]]:
        """
        Generate text using OpenAI via LangChain. Phoenix automatically traces this call.
        """
        response = self.llm.invoke([HumanMessage(content=prompt)], **kwargs)
        text = response.content.strip() if hasattr(response, "content") else str(response)

        # Extract usage if available (LangChain may not expose detailed token counts yet)
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        return text, usage

    def score_document(self, query: str, document: str, **kwargs) -> float:
        """
        Score document relevance to query (1–10) using OpenAI model via LangChain.
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
