# rag_app/observability/cost_tracker.py

import time
from typing import Dict, Any, Optional
from opentelemetry import trace
from config.settings import settings
from dotenv import load_dotenv
load_dotenv()

tracer = trace.get_tracer(__name__)

def log_llm_cost(
    model_name: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: Optional[float] = None
) -> None:
    """
    Log token counts and optionally cost into a Phoenix span.
    Phoenix will compute cost automatically if you only provide tokens,
    but you may also compute and log cost manually for your internal tracking.
    """
    with tracer.start_as_current_span("llm.cost_tracking") as span:
        span.set_attribute("llm.model_name", model_name)
        span.set_attribute("llm.provider", provider)
        span.set_attribute("llm.token_count.prompt", prompt_tokens)
        span.set_attribute("llm.token_count.completion", completion_tokens)
        # Optionally set total tokens
        span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)

        # If you computed a cost yourself, record it
        if cost is not None:
            span.set_attribute("llm.cost_usd", cost)

def compute_cost_usd(
    model_name: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int
) -> float:
    """
    Compute approximate cost for given tokens using rough per-token pricing.
    You can refine these rates in settings.yaml or pull from a rate table.
    """
    # example rates per million tokens (USD) — you should customise
    rates_per_million = {
        ("openai", "gpt-4-turbo"): {"prompt": 0.03, "completion": 0.06},
        ("openai", "gpt-3.5-turbo"): {"prompt": 0.002, "completion": 0.002},
        ("ollama", "llama3-4b"): {"prompt": 0.00, "completion": 0.00},  # assume local cost = 0 for cloud est
    }
    key = (provider.lower(), model_name)
    rate = rates_per_million.get(key, {"prompt": 0.0, "completion": 0.0})
    cost_usd = (prompt_tokens * rate["prompt"] + completion_tokens * rate["completion"]) / 1_000_000
    return cost_usd

def track_llm_usage(
    model_name: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int
) -> None:
    """
    High-level function: compute cost then log via Phoenix.
    """
    cost = compute_cost_usd(model_name, provider, prompt_tokens, completion_tokens)
    log_llm_cost(model_name=model_name, provider=provider,
                 prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                 cost=cost)
