# rag_app/embeddings/indexer.py

import os
import json
from typing import List, Dict, Any
from config.settings import settings
from ingestion.chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from observability.phoenix_tracer import init_phoenix_tracing
from dotenv import load_dotenv

load_dotenv()
tracer = init_phoenix_tracing()


class Indexer:
    def __init__(self, chunks_folder: str = None):
        """
        Initialize the Indexer with chunk folder, embedder, and vector store.
        """
        self.chunks_folder = chunks_folder or settings.data.chunks
        self.chunker = Chunker()
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def load_chunk_files(self) -> List[Dict[str, Any]]:
        """
        Loads chunk-files (assumed to be JSON) from chunks_folder.
        Each file contains {"text": str, "metadata": dict}.
        Returns list of chunk dicts.
        """
        with tracer.start_as_current_span("indexer.load_chunk_files") as span:
            chunk_dicts: List[Dict[str, Any]] = []
            span.set_attribute("chunks_folder", self.chunks_folder)

            try:
                for fname in os.listdir(self.chunks_folder):
                    full_path = os.path.join(self.chunks_folder, fname)
                    if os.path.isfile(full_path) and fname.endswith(".json"):
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                chunk = json.load(f)
                                chunk_dicts.append(chunk)
                        except Exception as e:
                            span.record_exception(e)
                            print(f"[Indexer] Error reading chunk file {fname}: {e}")

                span.set_attribute("num_chunks_loaded", len(chunk_dicts))
                print(f"[Indexer] Loaded {len(chunk_dicts)} chunk files from {self.chunks_folder}")
                return chunk_dicts

            except Exception as e:
                span.record_exception(e)
                raise e

    def index_all(self) -> None:
        """
        Full indexing flow:
        - Load pre-chunked data
        - Generate embeddings
        - Store in vector DB
        """
        with tracer.start_as_current_span("indexer.index_all") as span:
            try:
                chunk_dicts = self.load_chunk_files()
                texts = [c["text"] for c in chunk_dicts]
                meta_list = [c["metadata"] for c in chunk_dicts]

                span.set_attribute("num_texts", len(texts))
                span.set_attribute("backend", self.embedder.backend)
                print(f"[Indexer] Embedding {len(texts)} texts …")

                # Sub-span for embedding phase
                with tracer.start_as_current_span("indexer.embed_documents"):
                    vectors = self.embedder.embed_documents(texts)

                # Sub-span for storing vectors
                with tracer.start_as_current_span("indexer.store_vectors"):
                    for vec, meta, text in zip(vectors, meta_list, texts):
                        self.vector_store.add_vector(vector=vec, metadata=meta, text=text)

                span.set_attribute("indexed_vectors", len(vectors))
                print(f"[Indexer] Indexed {len(vectors)} vectors into vector store")

            except Exception as e:
                span.record_exception(e)
                raise e

    def index_single_document(self, file_path: str) -> None:
        """
        For a single new processed document (text), chunk, embed, and store.
        """
        with tracer.start_as_current_span("indexer.index_single_document") as span:
            span.set_attribute("file_path", file_path)

            try:
                chunks = self.chunker.process_document(file_path)
                texts = [c["text"] for c in chunks]
                meta_list = [c["metadata"] for c in chunks]

                span.set_attribute("num_chunks", len(chunks))
                print(f"[Indexer] Embedding {len(texts)} new chunks for document {file_path}")

                with tracer.start_as_current_span("indexer.embed_documents"):
                    vectors = self.embedder.embed_documents(texts)

                with tracer.start_as_current_span("indexer.store_vectors"):
                    for vec, meta, text in zip(vectors, meta_list, texts):
                        self.vector_store.add_vector(vector=vec, metadata=meta, text=text)

                span.set_attribute("indexed_vectors", len(vectors))
                print(f"[Indexer] Completed indexing for document {file_path}")

            except Exception as e:
                span.record_exception(e)
                raise e
