# rag_app/observability/retrieval_tracking.py

import time
from typing import List, Dict, Any
from opentelemetry import trace
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()

tracer = trace.get_tracer(__name__)


def track_retrieval(
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        metadata: Dict[str, Any] = None
) -> None:
    """
    Instrument the retrieval step:
      - query text
      - number of retrieved docs (top K)
      - metadata for each retrieved doc (ids, scores or distances)
      - latency of retrieval
    """
    with tracer.start_as_current_span("retrieval") as span:
        start = time.time()
        # While this function shouldn't do the actual retrieval logic, we record timing:
        # (retrieval logic should be executed outside and passed in)
        # Set attributes
        span.set_attribute("retrieval.query", query)
        span.set_attribute("retrieval.top_k", len(retrieved_docs))

        # Build list of doc ids & distances if available
        doc_ids = [doc.get("id") for doc in retrieved_docs]
        distances = [doc.get("distance") for doc in retrieved_docs]
        span.set_attribute("retrieval.document_ids", str(doc_ids))
        span.set_attribute("retrieval.distances", str(distances))

        if metadata:
            # Generic metadata map if passed
            span.set_attribute("retrieval.metadata", str(metadata))

        # End timing
        latency = time.time() - start
        span.set_attribute("retrieval.latency_seconds", latency)
