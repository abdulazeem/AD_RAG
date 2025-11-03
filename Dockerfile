# -----------------------------
# Dockerfile for FastAPI RAG App
# -----------------------------

# Base image
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install dependencies required for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the entire project into container
COPY . .

# Create data directories
RUN mkdir -p data/raw_docs data/processed_docs data/processed_chunks data/chunks

# Expose application ports
EXPOSE 8000 8501

# Run FastAPI server by default
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]

