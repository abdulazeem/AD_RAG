-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- OpenAI embeddings table (1536D)
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    chunk_metadata JSONB,
    embedding vector(1536) NOT NULL
);

-- Ollama embeddings table (768D)
CREATE TABLE IF NOT EXISTS chunk_embeddings_ollama (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    chunk_metadata JSONB,
    embedding vector(768) NOT NULL
);
