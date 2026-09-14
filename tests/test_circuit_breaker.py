"""Unit tests for Model Client Circuit Breaker Pattern."""
import pytest
import time
from unittest.mock import patch, AsyncMock

from app.models.client import (
    CircuitBreaker,
    CircuitState,
    get_circuit_breaker,
    get_all_circuit_statuses,
    reset_all_circuit_breakers,
    call_model,
)
from app.models.registry import MODEL_REGISTRY


@pytest.fixture(autouse=True)
def reset_breakers():
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


def test_circuit_breaker_initial_state():
    breaker = CircuitBreaker("test/model-1", failure_threshold=3, recovery_timeout=30.0)
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.can_attempt() is True
    assert breaker.total_trips == 0


def test_circuit_breaker_trips_to_open_after_consecutive_failures():
    breaker = CircuitBreaker("test/model-fail", failure_threshold=3, recovery_timeout=30.0)
    
    # 1st failure
    breaker.record_failure(RuntimeError("Connection refused 1"))
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 1
    assert breaker.can_attempt() is True

    # 2nd failure
    breaker.record_failure(RuntimeError("Connection refused 2"))
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 2
    assert breaker.can_attempt() is True

    # 3rd failure: trips to OPEN!
    breaker.record_failure(RuntimeError("Connection refused 3"))
    assert breaker.state == CircuitState.OPEN
    assert breaker.consecutive_failures == 3
    assert breaker.total_trips == 1
    # Immediately fast-fails subsequent attempts
    assert breaker.can_attempt() is False


def test_circuit_breaker_recovers_to_half_open_after_30s():
    breaker = CircuitBreaker("test/model-recovery", failure_threshold=2, recovery_timeout=30.0)
    breaker.record_failure(RuntimeError("Err 1"))
    breaker.record_failure(RuntimeError("Err 2"))
    assert breaker.state == CircuitState.OPEN
    assert breaker.can_attempt() is False

    # Simulate 15s elapsed: still OPEN
    breaker.last_failure_time = time.time() - 15.0
    assert breaker.can_attempt() is False
    assert breaker.state == CircuitState.OPEN

    # Simulate 31s elapsed: transitions to HALF_OPEN
    breaker.last_failure_time = time.time() - 31.0
    assert breaker.can_attempt() is True
    assert breaker.state == CircuitState.HALF_OPEN


def test_canary_probe_success_heals_breaker_to_closed():
    breaker = CircuitBreaker("test/model-canary", failure_threshold=2, recovery_timeout=30.0)
    breaker.record_failure(RuntimeError("Err 1"))
    breaker.record_failure(RuntimeError("Err 2"))
    assert breaker.state == CircuitState.OPEN

    # Transition to HALF_OPEN
    breaker.last_failure_time = time.time() - 31.0
    assert breaker.can_attempt() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Canary probe succeeds
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.consecutive_failures == 0
    assert breaker.can_attempt() is True


def test_canary_probe_failure_retrips_breaker_to_open():
    breaker = CircuitBreaker("test/model-re-trip", failure_threshold=2, recovery_timeout=30.0)
    breaker.record_failure(RuntimeError("Err 1"))
    breaker.record_failure(RuntimeError("Err 2"))
    assert breaker.state == CircuitState.OPEN
    assert breaker.total_trips == 1

    # Transition to HALF_OPEN
    breaker.last_failure_time = time.time() - 31.0
    assert breaker.can_attempt() is True
    assert breaker.state == CircuitState.HALF_OPEN

    # Canary probe fails: immediately re-trips to OPEN!
    breaker.record_failure(RuntimeError("Canary probe failed"))
    assert breaker.state == CircuitState.OPEN
    assert breaker.total_trips == 2
    assert breaker.can_attempt() is False


@pytest.mark.asyncio
async def test_call_model_fast_fails_open_breaker_to_fallback():
    model = "gpt-4o"
    fallback_model = MODEL_REGISTRY[model]["fallback"]
    breaker = get_circuit_breaker(model)
    
    # Trip breaker for primary model to OPEN
    breaker.state = CircuitState.OPEN
    breaker.last_failure_time = time.time()

    mock_fallback_resp = {
        "response": "Response from fallback model",
        "cost": 0.001,
        "tokens": 50,
        "model": fallback_model,
    }

    # Patch the primary provider caller to ensure it is NEVER called
    with patch("app.models.client._call_litellm", new_callable=AsyncMock) as mock_primary, \
         patch("app.models.client._call_openrouter", new_callable=AsyncMock) as mock_fallback:
        
        mock_fallback.return_value = mock_fallback_resp

        res = await call_model(model, "Hello world")
        
        # Primary call must NOT have been called (fast-failed by circuit breaker!)
        mock_primary.assert_not_called()
        # Fallback call was executed
        assert res["response"] == "Response from fallback model"


@pytest.mark.asyncio
async def test_call_model_trips_breaker_on_repeated_failures_and_falls_back():
    model = "gemini/gemini-1.5-flash"
    fallback_model = MODEL_REGISTRY[model]["fallback"]
    breaker = get_circuit_breaker(model)
    breaker.failure_threshold = 2

    mock_fallback_resp = {
        "response": "Fallback succeeded",
        "cost": 0.0001,
        "tokens": 20,
    }

    with patch("app.models.client._call_litellm", new_callable=AsyncMock) as mock_primary, \
         patch("app.models.client._call_openrouter", new_callable=AsyncMock) as mock_fallback:
        
        mock_primary.side_effect = RuntimeError("503 Service Unavailable")
        mock_fallback.return_value = mock_fallback_resp

        # 1st call fails primary, routes to fallback
        res1 = await call_model(model, "Prompt 1")
        assert res1["response"] == "Fallback succeeded"
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 1

        # 2nd call fails primary -> breaker TRIPS to OPEN!
        res2 = await call_model(model, "Prompt 2")
        assert res2["response"] == "Fallback succeeded"
        assert breaker.state == CircuitState.OPEN
        assert breaker.total_trips == 1

        # 3rd call: breaker is OPEN, primary is bypassed completely
        mock_primary.reset_mock()
        res3 = await call_model(model, "Prompt 3")
        assert res3["response"] == "Fallback succeeded"
        mock_primary.assert_not_called()


@pytest.mark.asyncio
async def test_cascading_fallback_traversal():
    # Test that if primary and secondary are both OPEN, it cascades to tertiary
    model_a = "test/cascade-a"
    model_b = "test/cascade-b"
    model_c = "ollama/llama3"

    MODEL_REGISTRY[model_a] = {"provider": "litellm", "fallback": model_b}
    MODEL_REGISTRY[model_b] = {"provider": "litellm", "fallback": model_c}

    breaker_a = get_circuit_breaker(model_a)
    breaker_b = get_circuit_breaker(model_b)
    breaker_a.state = CircuitState.OPEN
    breaker_a.last_failure_time = time.time()
    breaker_b.state = CircuitState.OPEN
    breaker_b.last_failure_time = time.time()

    mock_c_resp = {"response": "Tertiary fallback reached", "cost": 0.0, "tokens": 10}

    with patch("app.models.client._call_ollama", new_callable=AsyncMock) as mock_c:
        mock_c.return_value = mock_c_resp
        res = await call_model(model_a, "Cascade test prompt")
        assert res["response"] == "Tertiary fallback reached"


def test_circuit_breaker_status_and_reset():
    breaker = get_circuit_breaker("test/status-model")
    breaker.record_failure(RuntimeError("Test error"))
    
    statuses = get_all_circuit_statuses()
    assert "test/status-model" in statuses
    assert statuses["test/status-model"]["consecutive_failures"] == 1

    reset_all_circuit_breakers()
    statuses = get_all_circuit_statuses()
    assert statuses["test/status-model"]["consecutive_failures"] == 0
    assert statuses["test/status-model"]["state"] == CircuitState.CLOSED
