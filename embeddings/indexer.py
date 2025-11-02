# rag_app/embeddings/indexer.py

import os
import json
from typing import List, Dict, Any
from config.settings import settings
from ingestion.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from dotenv import load_dotenv
load_dotenv()

class Indexer:
    def __init__(
        self,
        chunks_folder: str = None
    ):
        self.chunks_folder = chunks_folder or settings.data.chunks
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def load_chunk_files(self) -> List[Dict[str, Any]]:
        """
        Loads chunk-files (assumed to be JSON) from chunks_folder: each file contains
        { "text": str, "metadata": dict }
        Returns list of chunk dicts.
        """
        chunk_dicts: List[Dict[str, Any]] = []
        for fname in os.listdir(self.chunks_folder):
            full_path = os.path.join(self.chunks_folder, fname)
            if os.path.isfile(full_path) and fname.endswith(".json"):
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        chunk = json.load(f)
                        chunk_dicts.append(chunk)
                except Exception as e:
                    print(f"[Indexer] Error reading chunk file {fname}: {e}")
        print(f"[Indexer] Loaded {len(chunk_dicts)} chunk files from {self.chunks_folder}")
        return chunk_dicts

    def index_all(self) -> None:
        """
        Full indexing flow:
        - Load or re-process raw documents + chunk them (if necessary)
        - Convert chunk text to embeddings
        - Store embedding + metadata + text in vector store
        """
        # If you prefer fresh embedding of processed docs instead of pre-saved chunks:
        # chunk_dicts = self.chunker.process_folder(settings.data.processed_docs)
        # Otherwise load existing chunk files:
        chunk_dicts = self.load_chunk_files()

        texts = [c["text"] for c in chunk_dicts]
        meta_list = [c["metadata"] for c in chunk_dicts]

        print(f"[Indexer] Embedding {len(texts)} texts …")
        vectors = self.embedder.embed_documents(texts)

        for vec, meta, text in zip(vectors, meta_list, texts):
            self.vector_store.add_vector(vector=vec, metadata=meta, text=text)
        print(f"[Indexer] Indexed {len(vectors)} vectors into vector store")

    def index_single_document(self, file_path: str) -> None:
        """
        For a single new processed document (text), chunk, embed, store.
        """
        # chunk
        chunks = self.chunker.process_document(file_path)
        texts = [c["text"] for c in chunks]
        meta_list = [c["metadata"] for c in chunks]

        print(f"[Indexer] Embedding {len(texts)} new chunks for document {file_path}")
        vectors = self.embedder.embed_documents(texts)

        for vec, meta, text in zip(vectors, meta_list, texts):
            self.vector_store.add_vector(vector=vec, metadata=meta, text=text)
        print(f"[Indexer] Completed indexing for document {file_path}")
