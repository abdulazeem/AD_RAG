# rag_app/generation/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generation.api.routers import query, ingest, admin, chat, rerank, prompts
from database.init_db import init_db
from observability.phoenix_tracer import init_phoenix_tracing

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Service API")

    # Enable CORS for Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins like ["http://localhost:8501"]
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["ingest"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["query"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
    app.include_router(rerank.router, prefix="/api/v1/rerank", tags=["rerank"])
    app.include_router(prompts.router)

    return app


app = create_app()


@app.on_event("startup")
def on_startup():
    # Initialize Phoenix tracing once at application startup
    init_phoenix_tracing(project_name="rag-llm-app")
    init_db()
