# rag_app/observability/phoenix_tracer.py
from phoenix.otel import register
from opentelemetry import trace

# Global flag to track if tracing has been initialized
_tracing_initialized = False
_global_tracer = None


def init_phoenix_tracing(project_name: str = "rag-llm-app", force: bool = False):
    """
    Initialize Phoenix OpenTelemetry tracing. This function is idempotent -
    it will only initialize once unless force=True.

    Args:
        project_name: Name of the project for Phoenix
        force: Force re-initialization even if already initialized

    Returns:
        OpenTelemetry tracer instance
    """
    global _tracing_initialized, _global_tracer

    # Return existing tracer if already initialized
    if _tracing_initialized and not force:
        return _global_tracer

    # Register Phoenix OpenTelemetry tracer provider
    tracer_provider = register(
        project_name=project_name,
        auto_instrument=True  # enables LangChain + LLM auto-tracing
    )

    # Retrieve a tracer for custom spans
    _global_tracer = trace.get_tracer("rag.observability")
    _tracing_initialized = True

    return _global_tracer
