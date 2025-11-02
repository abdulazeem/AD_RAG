# rag_app/embeddings/vector_store.py

import os
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, Column, Integer, String, JSON, text, MetaData, Table
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError

from pgvector.sqlalchemy import Vector  # pgvector integration for SQLAlchemy
from config.settings import settings
from config.backend_config import BackendConfig
from dotenv import load_dotenv
load_dotenv()


class VectorStore:
    def __init__(self, backend: str = None):
        """
        Initialize VectorStore with dynamic backend configuration.

        Args:
            backend: "openai" or "ollama". If None, uses settings.embedding_backend
        """
        # Use provided backend or fall back to settings
        backend = (backend or settings.embedding_backend).lower()

        # Get dynamic configuration for this backend
        config = BackendConfig.get_config(backend)
        self.backend = backend
        self.table_name = config["vector_table"]
        self.dimensions = config["embedding_dimensions"]

        self.database_url = settings.database.postgres_url
        self.engine = create_engine(self.database_url)

        # Create table dynamically with the right dimensions
        self.metadata = MetaData()
        self.chunk_embeddings = Table(
            self.table_name,
            self.metadata,
            Column('id', String, primary_key=True, default=lambda: str(uuid.uuid4())),
            Column('text', String, nullable=False),
            Column('chunk_metadata', JSON, nullable=True),
            Column('embedding', Vector(self.dimensions), nullable=False),
            extend_existing=True
        )

        # Create table if it doesn't exist
        self.metadata.create_all(self.engine)

        self.Session = sessionmaker(bind=self.engine)

    def add_vector(self, vector: List[float], metadata: Dict[str, Any], text: str) -> None:
        """
        Add one embedding vector + metadata + text to the store.
        """
        with self.Session() as session:
            try:
                stmt = self.chunk_embeddings.insert().values(
                    id=str(uuid.uuid4()),
                    text=text,
                    chunk_metadata=metadata,
                    embedding=vector
                )
                session.execute(stmt)
                session.commit()
            except IntegrityError:
                session.rollback()
                # Possibly update existing record or skip
            finally:
                session.close()

    def query(self, vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query for nearest neighbors given a vector. Returns list of metadata + text.
        """
        with self.Session() as session:
            # Using cosine distance operator in pgvector: "<=>"
            # Build raw SQL query for vector similarity
            stmt = text(f"""
                SELECT id, text, chunk_metadata, embedding <=> :vector AS distance
                FROM {self.table_name}
                ORDER BY embedding <=> :vector
                LIMIT :top_k
            """)

            results = session.execute(stmt, {"vector": str(vector), "top_k": top_k}).fetchall()

            output = []
            for row in results:
                output.append({
                    "id": row[0],
                    "text": row[1],
                    "metadata": row[2],
                    "distance": float(row[3])
                })
            session.close()
        return output

