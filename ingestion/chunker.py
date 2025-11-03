# rag_app/ingestion/chunker.py

import os
from typing import List, Dict, Any
from config.settings import settings
from config.backend_config import BackendConfig
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()


class Chunker:
    def __init__(self, backend: str = None):
        """Initialize Chunker with dynamic backend support.

        Args:
            backend: The embedding backend to use ('openai' or 'ollama').
                    If None, uses settings.embedding_backend.
        """
        # Use provided backend or fall back to settings
        self.backend = (backend or settings.embedding_backend).lower()

        # Get dynamic configuration for this backend
        config = BackendConfig.get_config(self.backend)
        model_name = config["embedding_model"]

        # Choose embedding backend based on config
        if self.backend == "openai":
            self.embeddings = OpenAIEmbeddings(model=model_name)
        elif self.backend == "ollama":
            self.embeddings = OllamaEmbeddings(
                model=model_name,
                base_url=settings.ollama.host
            )
        else:
            raise ValueError(f"Unsupported embedding backend: {self.backend}")

        # Instantiate semantic chunker
        self.chunker = SemanticChunker(self.embeddings)

    def process_text(self, text: str, metadata: Dict[str, Any], pages: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process text and create chunks with metadata.

        Args:
            text: Full document text
            metadata: Base metadata for all chunks
            pages: Optional list of page-level content with page numbers
                  Format: [{"page_number": 1, "content": "..."}, ...]

        Returns:
            List of chunks with metadata including page numbers
        """
        docs = self.chunker.create_documents([text], metadatas=[metadata])
        chunks = []

        # Create page mapping if pages are provided
        page_mapping = self._create_page_mapping(text, pages) if pages else {}

        # Try position-based mapping if exact matching fails
        position_based = False
        if pages and not page_mapping:
            print(f"[Chunker] Text matching failed, using position-based page assignment")
            position_based = True

        for idx, doc in enumerate(docs):
            chunk_text = doc.page_content
            chunk_metadata = {**metadata, "chunk_index": idx}

            # Add page information if available
            if page_mapping:
                page_nums = self._find_chunk_pages(chunk_text, text, page_mapping)
                if page_nums:
                    chunk_metadata["page_numbers"] = page_nums
                    chunk_metadata["page"] = page_nums[0]  # Primary page for backward compatibility
            elif position_based and pages:
                # Fallback: estimate page based on chunk position in document
                page_num = self._estimate_page_by_position(idx, len(docs), len(pages))
                if page_num:
                    chunk_metadata["page_numbers"] = [page_num]
                    chunk_metadata["page"] = page_num

            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

        # Debug logging
        chunks_with_pages = sum(1 for c in chunks if c['metadata'].get('page_numbers'))
        print(f"[Chunker] {chunks_with_pages}/{len(chunks)} chunks assigned page numbers")

        return chunks

    def _create_page_mapping(self, full_text: str, pages: List[Dict[str, Any]]) -> Dict[int, int]:
        """
        Create a mapping of character positions to page numbers.

        Args:
            full_text: Complete document text
            pages: List of page-level content

        Returns:
            Dict mapping text position to page number
        """
        page_mapping = {}
        current_pos = 0

        for page_info in pages:
            page_num = page_info["page_number"]
            page_content = page_info["content"]

            # Find this page's content in the full text
            page_start = full_text.find(page_content, current_pos)
            if page_start != -1:
                page_end = page_start + len(page_content)
                # Map all positions in this range to this page number
                for pos in range(page_start, page_end):
                    page_mapping[pos] = page_num
                current_pos = page_end

        return page_mapping

    def _find_chunk_pages(self, chunk_text: str, full_text: str, page_mapping: Dict[int, int]) -> List[int]:
        """
        Find which pages a chunk spans.

        Args:
            chunk_text: The chunk's text
            full_text: Complete document text
            page_mapping: Mapping of positions to page numbers

        Returns:
            List of page numbers the chunk appears on
        """
        # Find chunk position in full text
        chunk_start = full_text.find(chunk_text)
        if chunk_start == -1:
            return []

        chunk_end = chunk_start + len(chunk_text)

        # Find all unique page numbers in this range
        page_nums = set()
        for pos in range(chunk_start, min(chunk_end, max(page_mapping.keys()) + 1) if page_mapping else chunk_end):
            if pos in page_mapping:
                page_nums.add(page_mapping[pos])

        return sorted(list(page_nums))

    def _estimate_page_by_position(self, chunk_idx: int, total_chunks: int, total_pages: int) -> int:
        """
        Estimate page number based on chunk position when exact matching fails.

        Args:
            chunk_idx: Index of current chunk
            total_chunks: Total number of chunks
            total_pages: Total number of pages in document

        Returns:
            Estimated page number
        """
        if total_chunks == 0 or total_pages == 0:
            return None

        # Calculate approximate position (0.0 to 1.0)
        position = chunk_idx / total_chunks

        # Map to page number (1-indexed)
        page_num = int(position * total_pages) + 1

        # Clamp to valid range
        page_num = max(1, min(page_num, total_pages))

        return page_num

    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        all_chunks = []
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
                    print(f"[Chunker] Error processing {fname}: {e}")
        return all_chunks
