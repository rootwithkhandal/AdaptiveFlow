"""Unified model client: routes to LiteLLM (OpenAI/Gemini), OpenRouter, or Ollama.
Handles retries, circuit breaking, and fallback on failure.
"""
import asyncio
import time
import enum
from typing import Dict, Any, Optional, Set
import httpx

from app.models.registry import MODEL_REGISTRY
from app.config import settings
from app.logger import get_logger
from app.metrics import CIRCUIT_BREAKER_TRIPPED_COUNTER

def estimate_cost(model: str, total_tokens: int) -> float:
    meta = MODEL_REGISTRY.get(model, {})
    cost_per_1k = meta.get("cost_per_1k_tokens", 0.0)
    return (total_tokens / 1000) * cost_per_1k

logger = get_logger(__name__)
_http_client: Optional[httpx.AsyncClient] = None


def _get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled HTTP client for provider calls."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60.0)
    return _http_client


async def close_http_client():
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


class CircuitState(str, enum.Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """Circuit breaker per model protecting against cascading failures during outages."""

    def __init__(
        self,
        model: str,
        failure_threshold: Optional[int] = None,
        recovery_timeout: Optional[float] = None,
    ):
        self.model = model
        self.failure_threshold = failure_threshold or settings.circuit_breaker_failure_threshold
        self.recovery_timeout = recovery_timeout or settings.circuit_breaker_recovery_timeout
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self.total_trips = 0

    def can_attempt(self) -> bool:
        """Determines if a request may proceed or should fast-fail."""
        now = time.time()
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has elapsed (default 30s)
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = now
                logger.info(
                    "Circuit breaker entered HALF_OPEN (probing recovery)",
                    model=self.model,
                    elapsed_seconds=round(now - self.last_failure_time, 2),
                )
                return True
            return False
        return True

    def record_success(self):
        """Record a successful call: heals HALF_OPEN to CLOSED and resets failure counters."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker HEALED to CLOSED", model=self.model)
            self.state = CircuitState.CLOSED
            self.last_state_change = time.time()
        self.consecutive_failures = 0

    def record_failure(self, error: Optional[Exception] = None):
        """Record a failure: trips breaker to OPEN on threshold reached or failed canary probe."""
        now = time.time()
        self.consecutive_failures += 1
        self.last_failure_time = now

        if self.state == CircuitState.HALF_OPEN:
            # Canary probe failed: immediately re-trip to OPEN for another recovery period
            self.state = CircuitState.OPEN
            self.last_state_change = now
            self.total_trips += 1
            CIRCUIT_BREAKER_TRIPPED_COUNTER.labels(model=self.model).inc()
            logger.warning(
                "Circuit breaker probe failed in HALF_OPEN, re-tripping to OPEN",
                model=self.model,
                recovery_timeout=self.recovery_timeout,
                error=str(error) if error else None,
            )
        elif self.state == CircuitState.CLOSED and self.consecutive_failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self.last_state_change = now
            self.total_trips += 1
            CIRCUIT_BREAKER_TRIPPED_COUNTER.labels(model=self.model).inc()
            logger.warning(
                "Circuit breaker TRIPPED to OPEN due to consecutive failures",
                model=self.model,
                consecutive_failures=self.consecutive_failures,
                threshold=self.failure_threshold,
                recovery_timeout=self.recovery_timeout,
                error=str(error) if error else None,
            )

    def reset(self):
        """Reset circuit breaker to clean CLOSED state."""
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.last_state_change = time.time()

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        time_until_half_open = 0.0
        if self.state == CircuitState.OPEN and self.last_failure_time:
            time_until_half_open = max(0.0, self.recovery_timeout - (now - self.last_failure_time))
        return {
            "model": self.model,
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "total_trips": self.total_trips,
            "time_until_half_open_sec": round(time_until_half_open, 2),
        }


CIRCUIT_BREAKERS: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(model: str) -> CircuitBreaker:
    """Retrieve or initialize the circuit breaker for a given model."""
    if model not in CIRCUIT_BREAKERS:
        CIRCUIT_BREAKERS[model] = CircuitBreaker(model)
    return CIRCUIT_BREAKERS[model]


def get_all_circuit_statuses() -> Dict[str, Dict[str, Any]]:
    """Return status of all instantiated circuit breakers."""
    return {m: b.get_status() for m, b in CIRCUIT_BREAKERS.items()}


def reset_all_circuit_breakers():
    """Reset all circuit breakers to CLOSED."""
    for b in CIRCUIT_BREAKERS.values():
        b.reset()



async def _with_retry(coro_fn, *args, retries: int = 2, **kwargs):
    """Simple async retry helper with exponential backoff."""
    last_exc = None
    for attempt in range(retries):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
    raise last_exc


async def _call_litellm(model: str, prompt: str) -> Dict[str, Any]:
    """Call OpenAI / Gemini via LiteLLM."""
    import litellm
    api_key = settings.gemini_api_key if "gemini" in model.lower() else settings.openai_api_key
    if not api_key:
        raise RuntimeError(f"API key is not configured for {model}")

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        api_key=api_key,
    )
    text = response.choices[0].message.content
    usage = response.usage
    cost = estimate_cost(model, usage.prompt_tokens + usage.completion_tokens)
    return {"response": text, "cost": cost, "tokens": usage.total_tokens}


async def _call_openrouter(model: str, prompt: str) -> Dict[str, Any]:
    """Call OpenRouter via its OpenAI-compatible REST API."""
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    model_slug = model.removeprefix("openrouter/")

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    if settings.openrouter_site_url:
        headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_site_name:
        headers["X-Title"] = settings.openrouter_site_name

    payload = {
        "model": model_slug,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }

    resp = await _get_http_client().post(
            f"{settings.openrouter_base_url}/chat/completions",
            headers=headers,
            json=payload,
    )
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    cost = estimate_cost(model, total_tokens)
    return {"response": text, "cost": cost, "tokens": total_tokens}


async def _call_ollama(model: str, prompt: str) -> Dict[str, Any]:
    """Call local Ollama model via HTTP."""
    model_name = model.replace("ollama/", "")
    url = f"{settings.ollama_base_url}/api/generate"

    resp = await _get_http_client().post(url, json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
    })
    resp.raise_for_status()
    data = resp.json()
    return {
        "response": data.get("response", ""),
        "cost": 0.0,
        "tokens": data.get("eval_count", 0),
    }


async def _call_nvidia(model: str, prompt: str) -> Dict[str, Any]:
    """Call NVIDIA NIM API via OpenAI-compatible endpoint."""
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set")

    model_name = model.removeprefix("nvidia/")

    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.5,
    }

    resp = await _get_http_client().post(
            f"{settings.nvidia_base_url}/chat/completions",
            headers=headers,
            json=payload,
    )
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    total_tokens = usage.get("total_tokens", 0)
    cost = estimate_cost(model, total_tokens)
    return {"response": text, "cost": cost, "tokens": total_tokens}


async def call_model(model: str, prompt: str, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Dispatch to the correct provider with Circuit Breaker protection.
    Fast-fails and diverts to fallback if circuit breaker is OPEN.
    Traverses fallback chain to prevent cascading failure during outages.
    """
    if visited is None:
        visited = set()
    visited.add(model)

    meta = MODEL_REGISTRY.get(model, {})
    provider = meta.get("provider", "litellm")
    breaker = get_circuit_breaker(model)

    # 1. Check Circuit Breaker state
    if not breaker.can_attempt():
        fallback = meta.get("fallback")
        logger.warning(
            "Circuit breaker is OPEN: fast-failing and diverting to fallback",
            model=model,
            fallback=fallback,
            circuit_state=breaker.state.value,
        )
        if fallback and fallback not in visited:
            return await call_model(fallback, prompt, visited)
        resilient_fallback = "ollama/llama3"
        if resilient_fallback != model and resilient_fallback not in visited:
            return await call_model(resilient_fallback, prompt, visited)
        raise RuntimeError(f"Circuit breaker is OPEN for {model} and no healthy fallbacks available.")

    # 2. Attempt model call
    try:
        if provider == "openrouter":
            res = await _with_retry(_call_openrouter, model, prompt)
        elif provider == "nvidia":
            res = await _with_retry(_call_nvidia, model, prompt)
        elif provider == "ollama":
            res = await _call_ollama(model, prompt)
        else:
            res = await _with_retry(_call_litellm, model, prompt)

        breaker.record_success()
        return res
    except Exception as e:
        breaker.record_failure(e)
        fallback = meta.get("fallback")
        logger.warning(
            "Model call failed, trying fallback",
            model=model,
            fallback=fallback,
            circuit_state=breaker.state.value,
            error=str(e),
        )
        if fallback and fallback not in visited:
            return await call_model(fallback, prompt, visited)
        raise RuntimeError(f"All models failed for {model}: {e}") from e
