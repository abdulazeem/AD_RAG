# Use Python 3.11 as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Basic app metadata & non-secret defaults
ENV APP_NAME=rag_app \
    APP_VERSION=0.1.0 \
    LLM_BACKEND=openai \
    EMBEDDING_BACKEND=openai \
    OPENAI_MODEL=gpt-4o-mini \
    OPENAI_TIMEOUT=120 \
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=llama3.2:latest \
    OLLAMA_TIMEOUT=120 \
    RETRIEVAL_TOP_K=20 \
    RETRIEVAL_RERANK_TOP_M=5 \
    RETRIEVAL_CHUNK_SIZE=1000 \
    RETRIEVAL_CHUNK_OVERLAP=200 \
    EMBEDDING_MODEL_NAME=text-embedding-ada-002 \
    EMBEDDING_BATCH_SIZE=64 \
    EMBEDDING_DIMENSIONS=1536 \
    PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006 \
    ENABLE_COST_TRACKING=true \
    ENABLE_PROMPT_TRACKING=true \
    LOG_LEVEL=INFO \
    DATA_RAW_DOCS=data/raw_docs \
    DATA_PROCESSED_DOCS=data/processed_docs \
    DATA_PROCESSED_CHUNKS=data/processed_chunks \
    DATA_CHUNKS=data/chunks \
    API_BASE_URL=http://localhost:8000 \
    POSTGRES_USER=postgres \
    POSTGRES_DB=rag-db \
    PGVECTOR_TABLE=chunk_embeddings

# Note: Secrets such as POSTGRES_PASSWORD and OPENAI_API_KEY are NOT baked in.
# They should be supplied at runtime via -e or --env-file when running the container.

# Install system dependencies (PostgreSQL, pgvector dev libs, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    postgresql \
    postgresql-contrib \
    postgresql-server-dev-all \
    && rm -rf /var/lib/apt/lists/*

# Install pgvector extension
RUN git clone https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && cd /tmp/pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/raw_docs data/processed_docs data/processed_chunks data/chunks

# Expose ports
EXPOSE 5432 8000 8501 6006

# Copy entrypoint script and set permissions
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Define entrypoint in exec form
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Default command (could be overridden at runtime)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
