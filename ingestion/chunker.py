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

    def process_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs = self.chunker.create_documents([text], metadatas=[metadata])
        chunks = []
        for idx, doc in enumerate(docs):
            chunks.append({
                "text": doc.page_content,
                "metadata": {**metadata, "chunk_index": idx}
            })
        return chunks

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
