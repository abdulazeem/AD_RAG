# rag_app/ingestion/chunker.py

from observability.phoenix_tracer import init_phoenix_tracing
tracer = init_phoenix_tracing()

import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from opentelemetry.trace import SpanKind
from config.settings import settings
from config.backend_config import BackendConfig
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings

load_dotenv()


class Chunker:
    def __init__(self, backend: str = None):
        """Initialize Chunker with dynamic backend support.

        Args:
            backend: The embedding backend to use ('openai' or 'ollama').
                    If None, uses settings.embedding_backend.
        """
        self.backend = (backend or settings.embedding_backend).lower()

        # Get dynamic configuration for this backend
        config = BackendConfig.get_config(self.backend)
        model_name = config["embedding_model"]

        # Choose embedding backend
        if self.backend == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
        elif self.backend == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                base_url=settings.ollama.host
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

        # Semantic chunker uses chosen embedding model for semantic boundaries
        self.chunker = SemanticChunker(self.embeddings)

    def process_text(self, text: str, metadata: Dict[str, Any], pages: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        with tracer.start_as_current_span("chunking.process_text", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("chunking.backend", self.backend)
            span.set_attribute("chunking.model", self.embeddings.model)
            span.set_attribute("chunking.text_length", len(text))
            if self.backend == "ollama":
                span.set_attribute("llm.total_cost_usd", 0.0)

            try:
                docs = self.chunker.create_documents([text], metadatas=[metadata])
                chunks = []

                # Create page mapping if pages are provided
                page_mapping = self._create_page_mapping(text, pages) if pages else {}

                for idx, doc in enumerate(docs):
                    chunk_text = doc.page_content
                    chunk_metadata = {**metadata, "chunk_index": idx}

                    # Add page info if available
                    if page_mapping:
                        page_nums = self._find_chunk_pages(chunk_text, text, page_mapping)
                        if page_nums:
                            chunk_metadata["page_numbers"] = page_nums
                            chunk_metadata["page"] = page_nums[0]

                    chunks.append({"text": chunk_text, "metadata": chunk_metadata})

                span.set_attribute("num_chunks", len(chunks))
                return chunks

            except Exception as e:
                span.record_exception(e)
                raise e

    def _create_page_mapping(
        self, full_text: str, pages: List[Dict[str, Any]]
    ) -> Dict[int, int]:
        """Create mapping of character positions to page numbers."""
        page_mapping = {}
        current_pos = 0

        for page_info in pages:
            page_num = page_info["page_number"]
            page_content = page_info["content"]

            page_start = full_text.find(page_content, current_pos)
            if page_start != -1:
                page_end = page_start + len(page_content)
                for pos in range(page_start, page_end):
                    page_mapping[pos] = page_num
                current_pos = page_end

        return page_mapping

    def _find_chunk_pages(
        self, chunk_text: str, full_text: str, page_mapping: Dict[int, int]
    ) -> List[int]:
        """Find which pages a chunk spans."""
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return []

        chunk_end = chunk_start + len(chunk_text)
        page_nums = set()

        for pos in range(
            chunk_start,
            min(chunk_end, max(page_mapping.keys()) + 1) if page_mapping else chunk_end,
        ):
            if pos in page_mapping:
                page_nums.add(page_mapping[pos])

        return sorted(list(page_nums))

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """Process all text files in a folder and return chunks."""
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        all_chunks = []
        with tracer.start_as_current_span("chunking.process_folder") as span:
            span.set_attribute("folder_path", folder_path)

            for fname in os.listdir(folder_path):
                full_path = os.path.join(folder_path, fname)
                if os.path.isfile(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        metadata = {"source_file": fname}
                        file_chunks = self.process_text(content, metadata)
                        all_chunks.extend(file_chunks)
                        print(f"[Chunker] Processed {fname} → {len(file_chunks)} chunks")
                    except Exception as e:
                        span.record_exception(e)
                        print(f"[Chunker] Error processing {fname}: {e}")

            span.set_attribute("total_chunks", len(all_chunks))
        return all_chunks
