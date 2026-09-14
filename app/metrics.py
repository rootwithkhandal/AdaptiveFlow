"""Prometheus metrics definitions."""
from prometheus_client import Counter, Histogram

REQUEST_COUNTER = Counter(
    "llm_router_requests_total",
    "Total number of routed requests",
    ["model", "task"],
)

LATENCY_HISTOGRAM = Histogram(
    "llm_router_latency_seconds",
    "Request latency in seconds",
    ["model"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

COST_COUNTER = Counter(
    "llm_router_cost_usd_total",
    "Total cost in USD",
    ["model"],
)

CACHE_HIT_COUNTER = Counter(
    "llm_router_cache_hits_total",
    "Total number of prompt cache hits",
    ["type"],
)

IMPLICIT_FEEDBACK_COUNTER = Counter(
    "llm_router_implicit_feedback_total",
    "Total number of implicit feedback signals inferred",
    ["model", "signal_type"],
)

CIRCUIT_BREAKER_TRIPPED_COUNTER = Counter(
    "llm_router_circuit_breaker_trips_total",
    "Total number of times a model circuit breaker has tripped to OPEN",
    ["model"],
)

RATE_LIMIT_EXCEEDED_COUNTER = Counter(
    "llm_router_rate_limit_exceeded_total",
    "Total number of requests rejected due to rate limiting",
    ["tier"],
)

