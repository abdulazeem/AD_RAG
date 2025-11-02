# rag_app/observability/arize_setup.py

import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HTTPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from config.settings import settings

from dotenv import load_dotenv
load_dotenv()

def init_tracing(service_name: str = None):
    """
    Initialize tracing for Arize Phoenix using OpenTelemetry.
    Should be called once at application startup.
    """
    service = service_name or settings.app.name

    collector_endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", settings.observability.arize.collector_endpoint)
    api_key = os.environ.get("PHOENIX_API_KEY", settings.observability.arize.api_key)

    # Set resource attributes
    resource = Resource.create({
        "service.name": service,
        "project.name": service
    })

    # Setup tracer provider
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    # Determine if using local or cloud Phoenix
    is_local = "localhost" in collector_endpoint or "127.0.0.1" in collector_endpoint

    # Setup OTLP exporter to Phoenix
    if is_local:
        # Local Phoenix - use HTTP endpoint on same port as UI
        exporter = HTTPSpanExporter(
            endpoint=f"{collector_endpoint}/v1/traces",
        )
    else:
        # Cloud Phoenix - use HTTP endpoint with API key
        exporter = HTTPSpanExporter(
            endpoint=f"{collector_endpoint}/v1/traces",
            headers={"api_key": api_key}
        )

    span_processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(span_processor)

    # Optional: console exporter for local debug
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Acquire tracer
    tracer = trace.get_tracer(__name__)

    # Return tracer for manual instrumentation or decorators
    return tracer
