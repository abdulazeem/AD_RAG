# rag_app/observability/phoenix_tracer.py
from phoenix.otel import register
from opentelemetry import trace


def init_phoenix_tracing(project_name: str = "rag-llm-app"):
    """
    Initialize Phoenix tracing for the RAG system.
    Auto-instruments LangChain (OpenAI, Ollama, etc.) and
    exposes a tracer for manual instrumentation of embeddings,
    chunking, and ingestion stages.
    """
    # Register Phoenix OpenTelemetry tracer provider
    tracer_provider = register(
        project_name=project_name,
        auto_instrument=True  # enables LangChain + LLM auto-tracing
    )

    # Retrieve a tracer for custom spans
    tracer = trace.get_tracer("rag.observability")

    return tracer
