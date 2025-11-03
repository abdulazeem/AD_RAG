<div align="center">

# 🤖 Agentic RAG System

### Production-Ready Retrieval-Augmented Generation with LangChain, FastAPI & PGVector

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://python.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#️-architecture) • [API Docs](#-api-endpoints) • [Integrations](#-integrations)

---

</div>

A production-ready, modular Retrieval-Augmented Generation (RAG) system featuring dual LLM support (OpenAI & Ollama), advanced document processing with Docling, semantic chunking, vector search with PostgreSQL+PGVector, and comprehensive observability through Arize Phoenix.

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│         Streamlit UI  +  OpenWebUI                          │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP/REST
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Backend Layer (FastAPI)                      │
│  • Document Processing (Docling + Semantic Chunking)        │
│  • Embeddings (OpenAI + Ollama)                             │
│  • Vector Search (PostgreSQL + PGVector)                    │
│  • LLM Generation (OpenAI GPT-4 + Ollama Llama3.2)         │
│  • Conversation Memory                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Layer (PostgreSQL + PGVector)              │
│  • Vector Embeddings Storage                                │
│  • Document Metadata                                        │
│  • Conversation History                                     │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Features

### Core Capabilities
- ✅ **Dual Embedding Support**: OpenAI (text-embedding-ada-002) OR Ollama (nomic-embed-text)
- ✅ **Dual LLM Support**: OpenAI (gpt-4o-mini) OR Ollama (llama3.2:latest)
- ✅ **Advanced Document Processing**: Multi-format support (PDF, DOCX, TXT, MD) via Docling
- ✅ **Semantic Chunking**: LangChain SemanticChunker with percentile-based breakpoints
- ✅ **Vector Search**: PostgreSQL with PGVector extension for efficient similarity search
- ✅ **Intelligent Reranking**: Pointwise reranking for improved result relevance

### User Interfaces
- ✅ **Streamlit UI**: Full-featured web interface for document management, chat, and evaluation
- ✅ **Open WebUI Integration**: Custom pipeline for seamless Open WebUI integration
- ✅ **RESTful API**: Comprehensive FastAPI with automatic OpenAPI documentation

### Advanced Features
- ✅ **Conversation Memory**: Multi-session chat with context-aware responses
- ✅ **Evaluation Framework**: Custom evaluation system with ground truth generation and metrics
- ✅ **Observability**: Arize Phoenix integration with prompt management and cost tracking
- ✅ **Admin Tools**: Document management, bulk operations, and system monitoring
- ✅ **Flexible Configuration**: Environment-based settings with backend switching

## 📁 Project Structure
```
RAG_v2/
├── config/
│   ├── settings.py                   # Environment-based configuration
│   └── backend_config.py             # Backend-specific settings
├── data/
│   ├── raw_docs/                     # Original uploaded documents
│   ├── processed_docs/               # Processed document data
│   ├── chunks/                       # Generated chunks
│   └── evaluation/                   # Evaluation datasets & results
├── database/
│   ├── init_db.py                    # Database initialization
│   └── models.py                     # SQLAlchemy models
├── ingestion/
│   ├── docling_loader.py             # Docling document loader
│   ├── chunker.py                    # Semantic chunking logic
│   └── ingest_service.py             # Document ingestion service
├── embeddings/
│   ├── embedder.py                   # Embedding generation
│   ├── vector_store.py               # PGVector operations
│   └── indexer.py                    # Document indexing
├── llm/
│   ├── llm_base.py                   # LLM base class
│   ├── llm_openai.py                 # OpenAI implementation
│   └── llm_ollama.py                 # Ollama implementation
├── retrieval/
│   ├── retriever.py                  # Vector retrieval
│   ├── reranker_pointwise.py         # Pointwise reranker
│   └── retrieval_pipeline.py         # Complete retrieval pipeline
├── generation/
│   ├── generator.py                  # Response generation
│   ├── prompt_templates/             # Prompt templates
│   └── api/
│       ├── main.py                   # FastAPI application
│       └── routers/
│           ├── query.py              # Query endpoints
│           ├── ingest.py             # Ingestion endpoints
│           ├── chat.py               # Chat endpoints
│           ├── rerank.py             # Reranking endpoints
│           ├── evaluation.py         # Evaluation endpoints
│           └── admin.py              # Admin endpoints
├── evaluation/
│   ├── ground_truth_generator.py     # Generate evaluation datasets
│   └── llm_evaluator.py              # LLM-based evaluation
├── observability/
│   ├── arize_setup.py                # Phoenix initialization
│   ├── phoenix_prompt_manager.py     # Prompt management
│   ├── cost_tracker.py               # Cost tracking
│   ├── prompt_tracking.py            # Prompt tracking
│   └── retrieval_tracking.py         # Retrieval tracking
├── open_webui_integration/
│   ├── rag_pipeline.py               # Open WebUI pipeline
│   └── README.md                     # Integration guide
├── streamlit_app.py                  # Streamlit web interface
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata (uv)
├── .env.example                      # Example environment variables
└── README.md                         # This file
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend Framework** | FastAPI 0.120+ |
| **Document Processing** | Docling 2.60+ |
| **Text Chunking** | LangChain SemanticChunker |
| **Embeddings** | OpenAI text-embedding-ada-002 / Ollama nomic-embed-text |
| **Vector Database** | PostgreSQL 16+ with PGVector |
| **LLMs** | OpenAI gpt-4o-mini / Ollama llama3.2:latest |
| **Orchestration** | LangChain 0.3+ |
| **Observability** | Arize Phoenix 12.9+ |
| **Evaluation** | Custom (RAGAs-like) Framework |
| **UI Framework** | Streamlit 1.50+ |
| **Open WebUI** | Custom Pipeline Integration |

## 📋 Prerequisites

- **Python 3.11+**
- **PostgreSQL 16+** with PGVector extension
- **OpenAI API Key** (if using OpenAI backend)
- **Ollama** (if using Ollama backend) - [Installation Guide](https://ollama.ai/)
- **Arize Phoenix** (optional, for observability) - `python -m phoenix.server.main serve`
- **UV package manager** (recommended) or pip

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd RAG_v2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# OR using uv (faster)
uv pip install -r requirements.txt
```

### 2. Setup PostgreSQL with PGVector

```bash
# Using Docker
docker run -d \
  --name postgres-pgvector \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# The database will be automatically initialized on first API start
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env  # or use your preferred editor
```

**Key configuration options:**

```bash
# Backend Selection
LLM_BACKEND=openai              # or "ollama"
EMBEDDING_BACKEND=openai        # or "ollama"

# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Ollama Configuration (if using Ollama)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag_db

# Observability
PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
ENABLE_COST_TRACKING=true
```

### 4. Start the Services

#### Option A: FastAPI Backend Only

```bash
# Start the FastAPI backend
uvicorn generation.api.main:app --reload --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# API docs at http://localhost:8000/docs
```

#### Option B: Streamlit UI

```bash
# Start Streamlit interface
streamlit run streamlit_app.py

# UI will be available at http://localhost:8501
```

#### Option C: Both Services

```bash
# Terminal 1: Start API
uvicorn generation.api.main:app --reload --port 8000

# Terminal 2: Start Streamlit
streamlit run streamlit_app.py
```

#### Option D: With Phoenix Observability

```bash
# Terminal 1: Start Phoenix
python -m phoenix.server.main serve

# Terminal 2: Start API
uvicorn generation.api.main:app --reload --port 8000

# Terminal 3: Start Streamlit
streamlit run streamlit_app.py

# Phoenix UI at http://localhost:6006
```

## 📚 API Endpoints

### Document Ingestion
- `POST /api/v1/ingest/` - Upload and process a document
- `POST /api/v1/ingest/bulk` - Bulk upload documents
- `DELETE /api/v1/ingest/{doc_id}` - Delete a document

### Query & Retrieval
- `POST /api/v1/query/` - Query the RAG system
- `POST /api/v1/rerank/` - Rerank retrieved documents

### Chat
- `POST /api/v1/chat/sessions` - Create chat session
- `GET /api/v1/chat/sessions` - List chat sessions
- `POST /api/v1/chat/sessions/{chat_id}/messages` - Send message
- `GET /api/v1/chat/sessions/{chat_id}/messages` - Get chat history
- `DELETE /api/v1/chat/sessions/{chat_id}` - Delete chat session

### Evaluation
- `POST /api/v1/evaluation/generate-ground-truth` - Generate evaluation dataset
- `POST /api/v1/evaluation/evaluate` - Run evaluation
- `GET /api/v1/evaluation/ground-truth-files` - List ground truth files
- `GET /api/v1/evaluation/evaluation-results` - List evaluation results

### Admin
- `GET /api/v1/admin/documents/{backend}` - List documents
- `DELETE /api/v1/admin/documents/{backend}` - Delete all documents
- `GET /api/v1/admin/stats/{backend}` - Get system statistics

**Full API documentation**: http://localhost:8000/docs


### Arize Phoenix Observability

Phoenix provides comprehensive observability for your RAG pipeline:

- **Trace LLM calls** with prompt and response tracking
- **Monitor costs** for OpenAI API usage
- **Track retrieval** performance and relevance
- **Manage prompts** with version control

**Access Phoenix UI**: http://localhost:6006

## 🎮 Usage Examples

### Via API

```bash
# 1. Upload a document
curl -X POST "http://localhost:8000/api/v1/ingest/" \
  -F "file=@document.pdf"

# 2. Query the document
curl -X POST "http://localhost:8000/api/v1/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic?",
    "backend": "openai",
    "top_k": 20,
    "rerank_top_m": 5
  }'

# 3. Start a chat session
curl -X POST "http://localhost:8000/api/v1/chat/sessions" \
  -H "Content-Type: application/json" \
  -d '{"session_name": "My Chat"}'

# 4. Send a message
curl -X POST "http://localhost:8000/api/v1/chat/sessions/{chat_id}/messages" \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain this concept"}'
```

### Via Python

```python
import requests

# Query endpoint
response = requests.post(
    "http://localhost:8000/api/v1/query/",
    json={
        "query": "What are the key features?",
        "backend": "openai",
        "top_k": 20
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Sources: {result['source_documents']}")
```

### Via Streamlit UI

1. **Document Management**: Upload, view, and delete documents
2. **Query Interface**: Ask questions with customizable retrieval settings
3. **Chat Interface**: Multi-turn conversations with context
4. **Evaluation**: Generate ground truth and evaluate system performance
5. **Reranking Test**: Test and compare different reranking strategies

## 🔬 Evaluation

The system includes a custom evaluation framework for assessing RAG performance:

### Generate Ground Truth

```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/evaluation/generate-ground-truth" \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "openai",
    "num_questions": 10,
    "output_filename": "eval_dataset.json"
  }'
```

### Run Evaluation

```bash
# Via API
curl -X POST "http://localhost:8000/api/v1/evaluation/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "ground_truth_file": "eval_dataset.json",
    "backend": "openai"
  }'
```



## 🔧 Configuration

All configuration is managed via environment variables in `.env`:

**Application Settings:**
- `APP_NAME`, `APP_VERSION`

**Backend Selection:**
- `LLM_BACKEND` - "openai" or "ollama"
- `EMBEDDING_BACKEND` - "openai" or "ollama"

**OpenAI Settings:**
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`

**Ollama Settings:**
- `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`

**Database Settings:**
- `DATABASE_URL`, `PGVECTOR_TABLE`

**Retrieval Settings:**
- `RETRIEVAL_TOP_K`, `RETRIEVAL_RERANK_TOP_M`
- `RETRIEVAL_CHUNK_SIZE`, `RETRIEVAL_CHUNK_OVERLAP`

**Observability:**
- `PHOENIX_COLLECTOR_ENDPOINT`
- `ENABLE_COST_TRACKING`, `ENABLE_PROMPT_TRACKING`

See `.env.example` for all available options.

## 📚 Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.ai/)
- [Docling Documentation](https://github.com/DS4SD/docling)
- [Arize Phoenix Documentation](https://docs.arize.com/phoenix/)
- [Streamlit Documentation](https://docs.streamlit.io/)



## 👥 Author

- Mohammed Abdul Azeem Siddiqui

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- OpenAI for embeddings and LLMs
- Ollama for local LLM support
- Docling team for document processing
- Arize team for Phoenix observability
