# rag_app/observability/prompt_tracking.py

from typing import Dict, Any, Optional
from opentelemetry import trace, context
from openinference.instrumentation import using_prompt_template
# Note: `using_prompt_template` from OpenInference library as per Phoenix docs. :contentReference[oaicite:2]{index=2}
from dotenv import load_dotenv
load_dotenv()

tracer = trace.get_tracer(__name__)

def track_prompt(
    template: str,
    variables: Dict[str, Any],
    version: Optional[str] = None
):
    """
    Use this to wrap the LLM call so that prompt template metadata
    (template text, version, variables) are logged to Phoenix spans.
    """
    # version fallback
    version_str = version or "v1.0"
    # Use the context manager so that any spans inside will carry the prompt metadata
    with using_prompt_template(
        template=template,
        variables=variables,
        version=version_str
    ):
        # Additionally, you may create a span explicitly to capture prompt init
        with tracer.start_as_current_span("prompt.initialization") as span:
            span.set_attribute("prompt.template", template)
            span.set_attribute("prompt.version", version_str)
            span.set_attribute("prompt.variables", str(variables))
            # At this point the actual LLM call should happen after this call
            # Example: answer = llm.generate(...)
