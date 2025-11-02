# rag_app/ingestion/ingest_service.py

import os
import json
from typing import List, Dict, Any
from config.settings import settings
from .docling_loader import convert_document, convert_folder
from .chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from dotenv import load_dotenv
load_dotenv()

class IngestService:
    def __init__(
        self,
        raw_folder: str = None,
        processed_folder: str = None,
        chunks_folder: str = None,
        backend: str = None
    ):
        """
        Initialize IngestService with dynamic backend support.

        Args:
            raw_folder: Path to raw documents
            processed_folder: Path to processed documents
            chunks_folder: Path to chunks
            backend: "openai" or "ollama". If None, uses settings.embedding_backend
        """
        self.raw_folder = raw_folder or settings.data.raw_docs
        self.processed_folder = processed_folder or settings.data.processed_docs
        self.chunks_folder = chunks_folder or settings.data.chunks
        self.backend = backend or settings.embedding_backend

        self.chunker = Chunker(backend=self.backend)
        self.embedder = Embedder(backend=self.backend)
        self.vector_store = VectorStore(backend=self.backend)

    def ingest_all(self) -> List[Dict[str, Any]]:
        """
        Process all documents: convert → chunk → embed → store in vector DB.
        Returns list of metadata for all chunks.
        """
        # Step 1: Convert all raw docs into processed format
        conversion_results = convert_folder(self.raw_folder, self.processed_folder)
        print(f"[IngestService] Converted {len(conversion_results)} documents")

        all_chunk_metadata: List[Dict[str, Any]] = []

        for doc_res in conversion_results:
            source = doc_res["metadata"]["source"]
            processed_filename = os.path.splitext(os.path.basename(source))[0] + ".md"
            processed_path = os.path.join(self.processed_folder, processed_filename)

            try:
                chunks = self.chunker.process_text(
                    text=open(processed_path, "r", encoding="utf-8").read(),
                    metadata={"source_file": processed_filename}
                )
                print(f"[IngestService] {source} → {len(chunks)} chunks")

                # Step 2: build embedding vectors and store chunks
                texts = [c["text"] for c in chunks]
                metadata_list = [c["metadata"] for c in chunks]
                vectors = self.embedder.embed_documents(texts)

                # Store into vector DB
                for vec, meta, text in zip(vectors, metadata_list, texts):
                    self.vector_store.add_vector(vector=vec, metadata=meta, text=text)
                    all_chunk_metadata.append(meta)

                # Optionally: save chunks locally for inspection
                os.makedirs(self.chunks_folder, exist_ok=True)
                for idx, chunk in enumerate(chunks):
                    base_name = os.path.splitext(os.path.basename(source))[0]
                    fname = f"{base_name}_chunk_{idx}.json"
                    filepath = os.path.join(self.chunks_folder, fname)
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"[IngestService] Error processing {source}: {e}")

        print(f"[IngestService] Indexed {len(all_chunk_metadata)} chunks in vector store")
        return all_chunk_metadata

    def ingest_single(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single document (upload path): convert → chunk → embed → store.
        Returns list of chunk metadata for that document.
        """
        doc_res = convert_document(file_path)
        print(f"[IngestService] Converted single document: {file_path}")

        processed_filename = os.path.splitext(os.path.basename(file_path))[0] + ".md"
        processed_path = os.path.join(self.processed_folder, processed_filename)

        # Write the converted content to disk
        os.makedirs(self.processed_folder, exist_ok=True)
        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(doc_res["content"])
        print(f"[IngestService] Saved converted document to: {processed_path}")

        chunks = self.chunker.process_text(
            text=doc_res["content"],
            metadata={"source_file": processed_filename}
        )
        print(f"[IngestService] {file_path} → {len(chunks)} chunks")

        texts = [c["text"] for c in chunks]
        metadata_list = [c["metadata"] for c in chunks]
        vectors = self.embedder.embed_documents(texts)

        for vec, meta, text in zip(vectors, metadata_list, texts):
            self.vector_store.add_vector(vector=vec, metadata=meta, text=text)

        return metadata_list
