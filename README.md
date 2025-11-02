```markdown
# RAG Pipeline with Docling, LangChain, PGVector & Arize Phoenix

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
