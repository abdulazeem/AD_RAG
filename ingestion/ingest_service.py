# rag_app/ingestion/ingest_service.py
from observability.phoenix_tracer import init_phoenix_tracing
tracer = init_phoenix_tracing()

import os
import json
from typing import List, Dict, Any
from config.settings import settings
from .docling_loader import convert_document, convert_folder
from .chunker import Chunker
from embeddings.embedder import Embedder
from embeddings.vector_store import VectorStore
from opentelemetry.trace import SpanKind

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
        """Initialize IngestService with dynamic backend support."""
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
        with tracer.start_as_current_span("ingest.ingest_all") as span:
            span.set_attribute("backend", self.backend)
            span.set_attribute("raw_folder", self.raw_folder)

            all_chunk_metadata: List[Dict[str, Any]] = []

            try:
                # Step 1: Convert raw docs
                with tracer.start_as_current_span("ingest.convert_folder") as convert_span:
                    conversion_results = convert_folder(self.raw_folder, self.processed_folder)
                    convert_span.set_attribute("converted_docs", len(conversion_results))
                    print(f"[IngestService] Converted {len(conversion_results)} documents")

                # Step 2: Process each converted document
                for doc_res in conversion_results:
                    source = doc_res["metadata"]["source"]
                    processed_filename = os.path.splitext(os.path.basename(source))[0] + ".md"
                    processed_path = os.path.join(self.processed_folder, processed_filename)

                    try:
                        # Chunking stage
                        with tracer.start_as_current_span("ingest.chunk_text") as chunk_span:
                            chunk_span.set_attribute("source_file", source)
                            with open(processed_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            chunks = self.chunker.process_text(
                                text=content,
                                metadata={"source_file": processed_filename}
                            )
                            chunk_span.set_attribute("num_chunks", len(chunks))
                            print(f"[IngestService] {source} → {len(chunks)} chunks")

                        # Embedding stage
                        with tracer.start_as_current_span("ingest.embed_chunks") as embed_span:
                            texts = [c["text"] for c in chunks]
                            metadata_list = [c["metadata"] for c in chunks]
                            vectors = self.embedder.embed_documents(texts)
                            embed_span.set_attribute("num_vectors", len(vectors))

                        # Vector store stage
                        with tracer.start_as_current_span("ingest.store_vectors") as store_span:
                            for vec, meta, text in zip(vectors, metadata_list, texts):
                                self.vector_store.add_vector(vector=vec, metadata=meta, text=text)
                                all_chunk_metadata.append(meta)
                            store_span.set_attribute("stored_vectors", len(vectors))

                        # Optional: Save chunks locally
                        os.makedirs(self.chunks_folder, exist_ok=True)
                        for idx, chunk in enumerate(chunks):
                            base_name = os.path.splitext(os.path.basename(source))[0]
                            fname = f"{base_name}_chunk_{idx}.json"
                            filepath = os.path.join(self.chunks_folder, fname)
                            with open(filepath, "w", encoding="utf-8") as f:
                                json.dump(chunk, f, ensure_ascii=False, indent=2)

                    except Exception as e:
                        span.record_exception(e)
                        print(f"[IngestService] Error processing {source}: {e}")

                span.set_attribute("total_chunks_indexed", len(all_chunk_metadata))
                print(f"[IngestService] Indexed {len(all_chunk_metadata)} chunks in vector store")
                return all_chunk_metadata

            except Exception as e:
                span.record_exception(e)
                raise e

    def ingest_single(self, file_path: str) -> List[Dict[str, Any]]:
        with tracer.start_as_current_span("ingest.ingest_single", kind=SpanKind.INTERNAL) as span:
            span.set_attribute("ingest.backend", self.backend)
            span.set_attribute("ingest.file", os.path.basename(file_path))
            if self.backend == "ollama":
                span.set_attribute("llm.total_cost_usd", 0.0)

            try:
                # Step 1: Convert document
                with tracer.start_as_current_span("ingest.convert_document"):
                    doc_res = convert_document(file_path)
                    print(f"[IngestService] Converted single document: {file_path}")

                processed_filename = os.path.splitext(os.path.basename(file_path))[0] + ".md"
                processed_path = os.path.join(self.processed_folder, processed_filename)

                # Save converted content
                os.makedirs(self.processed_folder, exist_ok=True)
                with open(processed_path, "w", encoding="utf-8") as f:
                    f.write(doc_res["content"])
                print(f"[IngestService] Saved converted document to: {processed_path}")

                # Step 2: Chunk + Embed + Store
                pages = doc_res.get("pages", None)

                with tracer.start_as_current_span("ingest.chunk_text") as chunk_span:
                    chunks = self.chunker.process_text(
                        text=doc_res["content"],
                        metadata={"source_file": processed_filename, **doc_res["metadata"]},
                        pages=pages
                    )
                    chunk_span.set_attribute("num_chunks", len(chunks))
                    print(f"[IngestService] {file_path} → {len(chunks)} chunks")

                with tracer.start_as_current_span("ingest.embed_chunks"):
                    texts = [c["text"] for c in chunks]
                    metadata_list = [c["metadata"] for c in chunks]
                    vectors = self.embedder.embed_documents(texts)

                with tracer.start_as_current_span("ingest.store_vectors"):
                    for vec, meta, text in zip(vectors, metadata_list, texts):
                        self.vector_store.add_vector(vector=vec, metadata=meta, text=text)

                span.set_attribute("total_chunks", len(chunks))
                return metadata_list

            except Exception as e:
                span.record_exception(e)
                raise e
