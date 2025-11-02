# rag_app/generation/api/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from generation.api.routers import query, ingest, admin, chat, rerank, evaluation
from observability.arize_setup import init_tracing
from database.init_db import init_db

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

    # Initialize observability/tracing
    init_tracing(service_name="rag_app_api")

    # Include routers
    app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
    app.include_router(query.router, prefix="/api/v1/query", tags=["Query"])
    app.include_router(admin.router, prefix="/api/v1/admin", tags=["Admin"])
    app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
    app.include_router(rerank.router, prefix="/api/v1/rerank", tags=["Rerank"])
    app.include_router(evaluation.router, prefix="/api/v1/evaluation", tags=["Evaluation"])

    return app

app = create_app()

@app.on_event("startup")
def on_startup():
    init_db()