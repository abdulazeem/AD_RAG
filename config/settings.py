# config/settings.py

import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class AppSettings(BaseModel):
    name: str
    version: str


class OpenAISettings(BaseModel):
    api_key: str
    model: str
    reranker_model: str
    timeout_seconds: int


class OllamaSettings(BaseModel):
    host: str
    model: str
    reranker_model: str
    timeout_seconds: int


class RetrievalSettings(BaseModel):
    top_k: int
    rerank_top_m: int
    chunk_size: int
    chunk_overlap: int


class EmbeddingSettings(BaseModel):
    model_name: str
    batch_size: int
    dimensions: int


class DatabaseSettings(BaseModel):
    postgres_url: str
    pgvector_table: str


class ArizeSettings(BaseModel):
    api_key: Optional[str] = ""
    region: Optional[str] = "us"
    collector_endpoint: str


class ObservabilitySettings(BaseModel):
    arize: ArizeSettings
    enable_cost_tracking: bool
    enable_prompt_tracking: bool


class LoggingSettings(BaseModel):
    level: str


class DataSettings(BaseModel):
    base_dir: str
    raw_docs: str
    processed_docs: str
    processed_chunks: str
    chunks: str


class Settings(BaseModel):
    app: AppSettings
    llm_backend: str
    embedding_backend: str
    openai: OpenAISettings
    ollama: OllamaSettings
    retrieval: RetrievalSettings
    embedding: EmbeddingSettings
    database: DatabaseSettings
    observability: ObservabilitySettings
    logging: LoggingSettings
    data: DataSettings
    prompt_template_path: Optional[str] = None


def load_settings() -> Settings:
    """Load settings from environment variables and ensure data directories exist."""

    settings_obj = Settings(
        app=AppSettings(
            name=os.getenv("APP_NAME", "rag_app"),
            version=os.getenv("APP_VERSION", "0.1.0")
        ),
        llm_backend=os.getenv("LLM_BACKEND", "openai"),
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "openai"),
        openai=OpenAISettings(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            reranker_model=os.getenv("OPENAI_RERANKER_MODEL", "gpt-4o"),
            timeout_seconds=int(os.getenv("OPENAI_TIMEOUT", "120"))
        ),
        ollama=OllamaSettings(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
            reranker_model=os.getenv("OLLAMA_RERANKER_MODEL", "llama3.2:latest"),
            timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT", "120"))
        ),
        retrieval=RetrievalSettings(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
            rerank_top_m=int(os.getenv("RETRIEVAL_RERANK_TOP_M", "5")),
            chunk_size=int(os.getenv("RETRIEVAL_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200"))
        ),
        embedding=EmbeddingSettings(
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        ),
        database=DatabaseSettings(
            postgres_url=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_db"),
            pgvector_table=os.getenv("PGVECTOR_TABLE", "chunk_embeddings")
        ),
        observability=ObservabilitySettings(
            arize=ArizeSettings(
                api_key=os.getenv("ARIZE_API_KEY", ""),
                region=os.getenv("ARIZE_REGION", "us"),
                collector_endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
            ),
            enable_cost_tracking=os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true",
            enable_prompt_tracking=os.getenv("ENABLE_PROMPT_TRACKING", "true").lower() == "true"
        ),
        logging=LoggingSettings(
            level=os.getenv("LOG_LEVEL", "INFO")
        ),
        data=DataSettings(
            base_dir=os.getenv("DATA_BASE_DIR", "data"),
            raw_docs=os.getenv("DATA_RAW_DOCS", "data/raw_docs"),
            processed_docs=os.getenv("DATA_PROCESSED_DOCS", "data/processed_docs"),
            processed_chunks=os.getenv("DATA_PROCESSED_CHUNKS", "data/processed_chunks"),
            chunks=os.getenv("DATA_CHUNKS", "data/chunks")
        ),
        prompt_template_path=os.getenv("PROMPT_TEMPLATE_PATH")
    )

    # Automatically create data directories if they don't exist
    data_dirs = [
        settings_obj.data.raw_docs,
        settings_obj.data.processed_docs
    ]

    for directory in data_dirs:
        os.makedirs(directory, exist_ok=True)

    return settings_obj


# Global settings instance
settings = load_settings()
