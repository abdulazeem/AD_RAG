<div align="center">

# 🤖 Agentic RAG System

### Production-Ready Retrieval-Augmented Generation with LangChain, FastAPI & PGVector

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)](https://python.langchain.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

[Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#️-architecture) • [API Docs](#-api-endpoints) • [Contributing](#-contributing)

---

</div>

A production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, FastAPI, PostgreSQL with PGVector, and support for both OpenAI and Ollama models.

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

- ✅ **Dual Embedding Support**: OpenAI (text-embedding-3-small) OR Ollama (llama3.2)
- ✅ **Dual LLM Support**: OpenAI (gpt-4o-mini) OR Ollama (llama3.2)
- ✅ **Semantic Chunking**: Intelligent document splitting using percentile-based breakpoints
- ✅ **Vector Search**: PostgreSQL with PGVector extension for similarity search
- ✅ **Document Processing**: Support for PDF, DOCX, TXT, MD using Docling
- ✅ **Conversation Memory**: Context-aware chat with history
- ✅ **RESTful API**: FastAPI with automatic OpenAPI documentation
- ✅ **Observability**: Arize Phoenix integration for tracing
- ✅ **Evaluation**: RAGAs for quality metrics

## 📁 Project Structure
```
RAG_LC/
├── backend/
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── route_embeddings.py       # Embeddings API
│   │   ├── embeddings.py             # Embeddings logic
│   │   ├── route_documents.py        # Documents API
│   │   ├── documents.py              # Document processing
│   │   ├── route_chat.py             # Chat API
│   │   └── chat.py                   # Chat logic
│   │
│   ├── config.py                     # Configuration
│   ├── database.py                   # PostgreSQL + PGVector
│   ├── models.py                     # Pydantic models
│   ├── main.py                       # FastAPI app
│   ├── requirements.txt
│   └── .env
│
├── uploads/                          # Document uploads
├── docker-compose.yml
└── README.md
```

## 🛠️ Tech Stack

| Component | Technology                           |
|-----------|--------------------------------------|
| **Backend Framework** | FastAPI                              |
| **Document Processing** | Docling                              |
| **Text Chunking** | LangChain SemanticChunker            |
| **Embeddings** | OpenAI + Ollama                      |
| **Vector Database** | PostgreSQL + PGVector                |
| **LLMs** | OpenAI GPT-4o-mini + Ollama Llama3.2 |
| **Orchestration** | LangChain + LangGraph                |
| **Observability** | Arize Phoenix                        |
| **Evaluation** | RAGAS Framework                      |
| **Deployment** | Dockerfile                           |

## 📋 Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API Key
- UV package manager (optional, recommended)


## 🔧 Configuration

All configuration is managed through environment variables in `.env` file:

- **OpenAI Settings**: API key, models
- **Ollama Settings**: Base URL, models
- **PostgreSQL Settings**: Connection details
- **Vector Store Settings**: Collection names, dimensions
- **RAG Settings**: Chunk size, top-k retrieval

## 🎯 Roadmap

- [x] Configuration management
- [x] Database setup with PGVector
- [x] Embeddings service (OpenAI + Ollama)
- [x] Document processing with Docling
- [ ] Vector store operations
- [ ] LLM service (OpenAI + Ollama)
- [ ] RAG pipeline
- [ ] Chat endpoints
- [ ] Conversation memory
- [ ] Streamlit frontend
- [ ] OpenWebUI integration
- [ ] Arize Phoenix observability
- [ ] RAGAs evaluation framework
- [ ] Document indexer CLI
- [ ] Production deployment

```

## 📚 Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PGVector Documentation](https://github.com/pgvector/pgvector)
- [Ollama Documentation](https://ollama.ai/)
- [Docling Documentation](https://github.com/DS4SD/docling)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

MIT License

## 👥 Author

- Mohammed Abdul Azeem Siddiqui

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- OpenAI for embeddings and LLMs
- Ollama for local LLM support





A modular Retrieval-Augmented Generation (RAG) application built with:
- Document ingestion via Docling
- Semantic chunking via LangChain’s `SemanticChunker`
- Vector storage using pgvector (PostgreSQL)
- Dual LLM support (OpenAI API & Ollama)
- Observability and cost tracking via Arize Phoenix

---

## ⚙️ Project Structure

```

rag_app/
├── config/
│   ├── settings.yaml
│   └── logging.yaml
├── data/
│   ├── raw_docs/
│   ├── processed_docs/
│   └── chunks/
├── ingestion/
│   ├── docling_loader.py
│   ├── chunker.py
│   └── ingest_service.py
├── embeddings/
│   ├── embedder.py
│   ├── vector_store.py
│   └── indexer.py
├── llm/
│   ├── llm_base.py
│   ├── llm_openai.py
│   └── llm_ollama.py
├── retrieval/
│   ├── retriever.py
│   ├── reranker_pointwise.py
│   └── retrieval_pipeline.py
├── generation/
│   ├── prompt_templates/
│   ├── generator.py
│   └── api/
│       ├── main.py
│       ├── dependencies.py
│       ├── schemas.py
│       └── routers/
│           ├── query.py
│           ├── ingest.py
│           └── admin.py
├── ui/
│   └── streamlit_app.py
├── observability/
│   ├── arize_setup.py
│   ├── prompt_tracking.py
│   ├── retrieval_tracking.py
│   └── cost_tracker.py
├── tests/
│   ├── unit/
│   └── integration/
├── requirements.txt
└── README.md

````

---

## 🚀 Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
````

2. Create and activate a virtual environment (Python 3.10+ recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Linux/Mac
   venv\Scripts\activate      # on Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure `config/settings.yaml` with your keys:

   ```yaml
   openai:
     api_key: "YOUR_OPENAI_API_KEY"
     model: "gpt-4o-turbo"
   ollama:
     host: "http://localhost:11434"
     model: "llama3-4b"
   database:
     postgres_url: "postgresql://user:password@localhost:5432/rag_db"
   ```

## 🧩 Usage

### Ingest a document

Either use the Streamlit UI or the API:

```bash
# via API
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
     -F "file=@path/to/doc.pdf"
```

Or open Streamlit (`ui/streamlit_app.py`) and upload a document.

### Query for an answer

```bash
# via API
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{"query":"What are the benefits of RAG?"}'
```

Or use the Streamlit UI: select **Query Documents** tab, type your query, and run.

---

## 🔍 Features

* **Dual LLM support**: Switch between OpenAI and Ollama via `settings.yaml`.
* **Semantic chunking**: Uses embedding-based chunk splitting for more meaningful chunks.
* **Persistent vector store**: All chunks and embeddings stored in PostgreSQL with pgvector.
* **Observability**: Track prompt usage, retrieval latency, token costs via Arize Phoenix.
* **Modular architecture**: Clear separation of loading, chunking, embedding, retrieval, reranking, generation, UI, and tests.

---

## 🧪 Running Tests

Run the test suite (unit + integration) with pytest:

```bash
pytest
```

---

## 📆 Roadmap

* Support additional embedding backends (e.g., SentenceTransformers).
* Introduce list-wise reranking (in addition to point-wise).
* Add Docker and Kubernetes deployment.
* Enhance Streamlit UI with analytics dashboards (token usage, cost over time).
* Expand documentation (developer guides, architecture diagrams).

---

## 🧑‍💻 Contributing

Contributions are welcome!
Please:

* Fork the repository
* Create a branch (`feature/my-feature`)
* Write tests and update documentation
* Submit a pull request

---

## 📝 License

This project is licensed under the MIT License – see `LICENSE` for more details.

---

## 📫 Contact

Maintained by *Your Name*.
Feel free to open issues, submit pull requests, or connect via [your-email@example.com](mailto:your-email@example.com).

```

You can modify the placeholders (GitHub URL, Your Name, contact email) as needed.

If you’d like, I can **generate a basic `LICENSE` file** next (MIT license template) for you.
::contentReference[oaicite:0]{index=0}
```
