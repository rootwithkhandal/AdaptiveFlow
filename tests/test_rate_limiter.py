"""Unit and integration tests for per-tier sliding window rate limiting.

Tests cover:
- SlidingWindowRateLimiter core algorithm
- Free tier limit (10 req/min)
- Pro tier limit (60 req/min)
- Enterprise tier unlimited
- Sliding window timestamp expiration / recovery
- Per-user isolation
- HTTP 429 response, Retry-After header, and detail message
- Protection of Redis cache and FAISS semantic lookups
- Rate limiting on parallel and synthesis endpoints
- Inspection endpoint GET /ratelimit/{user_id}
- Rate limiter state reset
"""
import time
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.rate_limiter import SlidingWindowRateLimiter, rate_limiter
from app.dependencies import profile_manager, cache

client = TestClient(app)

MOCK_MODEL_RESPONSE = {
    "response": "Fast mocked LLM response.",
    "model": "ollama/llama3",
    "cost": 0.0,
    "latency": 0.05,
}


@pytest.fixture(autouse=True)
def reset_rate_limiter_state():
    """Ensure clean rate limiter state before every test."""
    rate_limiter.reset()
    yield
    rate_limiter.reset()


def test_free_tier_sliding_window_limit():
    """Free tier allows exactly 10 requests in a 60s window; 11th is rejected."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user = "free_user_unit_01"
    base_time = 1000.0

    for i in range(10):
        allowed, info = limiter.check(user, tier="free", now=base_time + i)
        assert allowed is True, f"Request {i+1} should have been allowed"
        assert info["remaining"] == 10 - (i + 1)
        assert info["limit"] == 10

    # 11th request must be rejected
    allowed, info = limiter.check(user, tier="free", now=base_time + 10.0)
    assert allowed is False
    assert info["remaining"] == 0
    assert info["retry_after"] >= 50  # 1000.0 + 60.0 - 1010.0 = 50.0s


def test_pro_tier_sliding_window_limit():
    """Pro tier allows 60 requests in a 60s window; 61st is rejected."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user = "pro_user_unit_01"
    base_time = 2000.0

    for i in range(60):
        allowed, info = limiter.check(user, tier="pro", now=base_time + (i * 0.5))
        assert allowed is True, f"Request {i+1} should have been allowed"
        assert info["remaining"] == 60 - (i + 1)

    # 61st request must be rejected
    allowed, info = limiter.check(user, tier="pro", now=base_time + 30.0)
    assert allowed is False
    assert info["remaining"] == 0
    assert info["limit"] == 60
    assert info["retry_after"] > 0


def test_enterprise_tier_unlimited():
    """Enterprise tier has no rate limit quota (unlimited requests)."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user = "enterprise_user_unit_01"
    base_time = 3000.0

    for i in range(120):
        allowed, info = limiter.check(user, tier="enterprise", now=base_time + (i * 0.1))
        assert allowed is True
        assert info["limit"] is None
        assert info["remaining"] is None


def test_sliding_window_expiration_and_recovery():
    """Old requests slide out of the 60s window, restoring quota."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user = "sliding_user_01"
    start_time = 5000.0

    # Exhaust free tier quota of 10 requests at t=5000..5009
    for i in range(10):
        limiter.check(user, tier="free", now=start_time + i)

    # At t=5015, still blocked
    allowed, info = limiter.check(user, tier="free", now=start_time + 15.0)
    assert allowed is False
    assert info["retry_after"] == 45  # (5000 + 60) - 5015 = 45s

    # At t=5060.5, exactly the first request (at 5000.0) has expired from window [5000.5..5060.5]
    allowed, info = limiter.check(user, tier="free", now=start_time + 60.5)
    assert allowed is True
    assert info["remaining"] == 0  # 9 remaining from previous window + 1 new = 10 active
    assert info["current_usage"] == 10


def test_per_user_rate_limit_isolation():
    """One user hitting rate limit does not affect other users."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user_a = "alice_free"
    user_b = "bob_free"
    t = 6000.0

    # Alice exhausts 10 requests
    for _ in range(10):
        limiter.check(user_a, tier="free", now=t)

    assert limiter.check(user_a, tier="free", now=t)[0] is False

    # Bob's quota is completely fresh and unaffected
    allowed_bob, info_bob = limiter.check(user_b, tier="free", now=t)
    assert allowed_bob is True
    assert info_bob["remaining"] == 9


@pytest.mark.asyncio
async def test_async_check_and_record():
    """Verify async check_and_record works in async execution contexts."""
    limiter = SlidingWindowRateLimiter(window_seconds=60.0)
    user = "async_user_01"
    t = 7000.0

    allowed, info = await limiter.check_and_record(user, tier="free", now=t)
    assert allowed is True
    assert info["current_usage"] == 1


def test_route_endpoint_rate_limiting_http_429():
    """POST /route returns HTTP 429 with Retry-After header when free quota is exceeded."""
    user_id = "api_user_free_429"
    profile_manager.set_tier(user_id, "free")

    with patch("app.rl_router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = dict(MOCK_MODEL_RESPONSE)

        # 10 successful requests
        for i in range(10):
            r = client.post("/route", json={"user_id": user_id, "prompt": f"Unique prompt {i}"})
            assert r.status_code == 200, f"Request {i} failed: {r.text}"

        # 11th request must receive HTTP 429
        r_blocked = client.post("/route", json={"user_id": user_id, "prompt": "Eleventh prompt"})
        assert r_blocked.status_code == 429
        assert "Retry-After" in r_blocked.headers
        assert int(r_blocked.headers["Retry-After"]) > 0
        data = r_blocked.json()
        assert "Rate limit exceeded" in data["detail"]
        assert "free" in data["detail"]


def test_rate_limit_protects_cache_and_faiss():
    """When a user is rate-limited, cache.get is never called, protecting Redis & FAISS."""
    user_id = "api_user_hammer_cache"
    profile_manager.set_tier(user_id, "free")

    with patch("app.rl_router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = dict(MOCK_MODEL_RESPONSE)

        # Exhaust quota of 10
        for i in range(10):
            client.post("/route", json={"user_id": user_id, "prompt": f"Initial query {i}"})

        # Spy on cache.get to ensure it is not called during rate-limited requests
        with patch.object(cache, "get", new_callable=AsyncMock) as mock_cache_get:
            r = client.post("/route", json={"user_id": user_id, "prompt": "Hammering query"})
            assert r.status_code == 429
            mock_cache_get.assert_not_called()


def test_parallel_route_rate_limiting():
    """POST /route/parallel enforces rate limits per tier."""
    user_id = "parallel_rate_limited_user"
    profile_manager.set_tier(user_id, "free")

    with patch("app.models.client.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = dict(MOCK_MODEL_RESPONSE)

        # Exhaust 10 requests
        for i in range(10):
            r = client.post("/route/parallel", json={
                "user_id": user_id,
                "prompt": f"Parallel prompt {i}",
                "strategy": "fastest",
            })
            assert r.status_code == 200

        # 11th request rejected with 429
        r_blocked = client.post("/route/parallel", json={
            "user_id": user_id,
            "prompt": "Parallel prompt 11",
            "strategy": "fastest",
        })
        assert r_blocked.status_code == 429
        assert "Retry-After" in r_blocked.headers


def test_synthesize_route_rate_limiting_for_pro_tier():
    """POST /route/synthesize enforces 60 req/min for pro tier."""
    user_id = "pro_synth_user"
    profile_manager.set_tier(user_id, "pro")

    limiter = rate_limiter
    # Pre-fill user window with 60 entries
    t_now = time.time()
    for i in range(60):
        limiter.check(user_id, tier="pro", now=t_now)

    # Attempt synthesis while rate-limited
    r = client.post("/route/synthesize", json={
        "user_id": user_id,
        "prompt": "Test synthesis query",
    })
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert "Rate limit exceeded" in r.json()["detail"]


def test_ratelimit_status_endpoint():
    """GET /ratelimit/{user_id} provides quota inspection."""
    user_id = "status_check_user"
    profile_manager.set_tier(user_id, "free")

    # Initially 0 usage
    r0 = client.get(f"/ratelimit/{user_id}")
    assert r0.status_code == 200
    data0 = r0.json()
    assert data0["tier"] == "free"
    assert data0["limit_per_minute"] == 10
    assert data0["current_window_usage"] == 0
    assert data0["remaining_in_window"] == 10

    # Record 3 requests
    rate_limiter.check(user_id, tier="free")
    rate_limiter.check(user_id, tier="free")
    rate_limiter.check(user_id, tier="free")

    r1 = client.get(f"/ratelimit/{user_id}")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["current_window_usage"] == 3
    assert data1["remaining_in_window"] == 7


def test_rate_limiter_reset():
    """Reset clears user timestamps and restores full quota immediately."""
    user_id = "reset_user_test"
    for _ in range(10):
        rate_limiter.check(user_id, tier="free")

    assert rate_limiter.check(user_id, tier="free")[0] is False

    # Specific reset
    rate_limiter.reset(user_id=user_id)
    assert rate_limiter.check(user_id, tier="free")[0] is True
