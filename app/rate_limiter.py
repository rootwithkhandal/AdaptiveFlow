"""Per-tier sliding window rate limiter.

Protects downstream inference engines, Redis, and FAISS vector indices
from excessive request volumes and burst hammering.

Tier Limits (sliding 60s window):
- free: 10 requests / minute
- pro: 60 requests / minute
- enterprise: unlimited (no rate limiting applied)
"""
import asyncio
import collections
import math
import threading
import time
import uuid
from typing import Dict, Any, Optional, Tuple, Deque

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIER_LIMITS: Dict[str, Optional[int]] = {
    "free": 10,
    "pro": 60,
    "enterprise": None,  # None means unlimited
}


class SlidingWindowRateLimiter:
    """Sliding window log rate limiter with per-tier quotas.
    
    Supports:
    - Precise sliding window timestamp pruning (no fixed-window boundary bursts).
    - Redis Sorted Set (ZSET) distributed backend with automatic in-memory fallback.
    - Tier-based configuration (free: 10/min, pro: 60/min, enterprise: unlimited).
    - Concurrency-safe in-memory deque state.
    """

    def __init__(
        self,
        tier_limits: Optional[Dict[str, Optional[int]]] = None,
        window_seconds: Optional[float] = None,
        redis_client=None,
    ):
        self.window_seconds = (
            window_seconds
            if window_seconds is not None
            else getattr(settings, "rate_limit_window_seconds", 60.0)
        )
        
        # Resolve tier limits
        if tier_limits is not None:
            self.tier_limits = dict(tier_limits)
        else:
            free_limit = getattr(settings, "rate_limit_free", 10)
            pro_limit = getattr(settings, "rate_limit_pro", 60)
            ent_cfg = getattr(settings, "rate_limit_enterprise", 0)
            ent_limit = None if ent_cfg == 0 else ent_cfg
            self.tier_limits = {
                "free": free_limit,
                "pro": pro_limit,
                "enterprise": ent_limit,
            }

        self._user_windows: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._redis = redis_client
        self._redis_attempted = False
        self._redis_cooldown_until = 0.0

    def _get_redis_client(self):
        if self._redis is not None:
            return self._redis
        if not self._redis_attempted:
            self._redis_attempted = True
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=0.2,
                    socket_timeout=0.2,
                )
                logger.info("Rate limiter connected to Redis", url=settings.redis_url)
            except Exception as e:
                logger.warning("Redis unavailable for rate limiting, using in-memory window", error=str(e))
                self._redis = None
                self._redis_cooldown_until = float("inf")
        return self._redis

    def get_limit_for_tier(self, tier: str) -> Optional[int]:
        """Return the requests-per-window quota for a given tier."""
        normalized_tier = (tier or "free").lower().strip()
        if normalized_tier in self.tier_limits:
            return self.tier_limits[normalized_tier]
        return self.tier_limits.get("free", 10)

    def check(
        self,
        user_id: str,
        tier: str = "free",
        now: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Synchronous rate limit check and record using in-memory sliding window.
        
        Returns:
            (allowed, info_dict)
        """
        current_time = now if now is not None else time.time()
        limit = self.get_limit_for_tier(tier)

        # Unlimited tier (e.g. enterprise)
        if limit is None:
            return True, {
                "allowed": True,
                "tier": tier,
                "limit": None,
                "remaining": None,
                "retry_after": 0,
                "reset_after": 0.0,
                "current_usage": len(self._user_windows.get(user_id, [])),
            }

        window_start = current_time - self.window_seconds

        with self._sync_lock:
            dq = self._user_windows[user_id]
            # Prune timestamps older than window_start
            while dq and dq[0] <= window_start:
                dq.popleft()

            current_usage = len(dq)

            if current_usage >= limit:
                oldest_ts = dq[0]
                time_remaining = (oldest_ts + self.window_seconds) - current_time
                retry_after = max(1, math.ceil(time_remaining))
                return False, {
                    "allowed": False,
                    "tier": tier,
                    "limit": limit,
                    "remaining": 0,
                    "retry_after": retry_after,
                    "reset_after": max(0.0, time_remaining),
                    "current_usage": current_usage,
                }

            # Record request timestamp
            dq.append(current_time)
            remaining = limit - (current_usage + 1)
            return True, {
                "allowed": True,
                "tier": tier,
                "limit": limit,
                "remaining": max(0, remaining),
                "retry_after": 0,
                "reset_after": self.window_seconds,
                "current_usage": current_usage + 1,
            }

    async def check_and_record(
        self,
        user_id: str,
        tier: str = "free",
        now: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Asynchronous rate limit check and record.
        
        Attempts distributed Redis ZSET if available; cleanly falls back
        to the in-memory sliding window.
        """
        current_time = now if now is not None else time.time()
        limit = self.get_limit_for_tier(tier)

        if limit is None:
            return True, {
                "allowed": True,
                "tier": tier,
                "limit": None,
                "remaining": None,
                "retry_after": 0,
                "reset_after": 0.0,
                "current_usage": self.get_usage(user_id, tier),
            }

        if time.time() < self._redis_cooldown_until:
            redis_client = None
        else:
            redis_client = self._get_redis_client()

        if redis_client is not None:
            try:
                key = f"ratelimit:{user_id}"
                window_start = current_time - self.window_seconds

                # Lua keeps pruning, counting, and recording in one Redis operation.
                member = f"{current_time}:{uuid.uuid4().hex}"
                script = """
                redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
                local count = redis.call('ZCARD', KEYS[1])
                if count >= tonumber(ARGV[2]) then
                    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
                    return {0, count, oldest[2] or 0}
                end
                redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
                redis.call('EXPIRE', KEYS[1], ARGV[5])
                return {1, count + 1, 0}
                """
                allowed_flag, current_usage, oldest_ts = await redis_client.eval(
                    script, 1, key, window_start, limit, current_time, member, int(self.window_seconds) + 10
                )
                if not allowed_flag:
                    time_remaining = (float(oldest_ts) + self.window_seconds) - current_time
                    retry_after = max(1, math.ceil(time_remaining))
                    return False, {
                        "allowed": False,
                        "tier": tier,
                        "limit": limit,
                        "remaining": 0,
                        "retry_after": retry_after,
                        "reset_after": max(0.0, time_remaining),
                        "current_usage": current_usage,
                    }

                remaining = limit - current_usage
                return True, {
                    "allowed": True,
                    "tier": tier,
                    "limit": limit,
                    "remaining": max(0, remaining),
                    "retry_after": 0,
                    "reset_after": self.window_seconds,
                    "current_usage": current_usage,
                }
            except Exception as e:
                self._redis_cooldown_until = time.time() + 30.0
                logger.warning("Redis rate limit operation failed, falling back to in-memory for 30s", error=str(e))

        # In-memory async fallback
        async with self._async_lock:
            return self.check(user_id=user_id, tier=tier, now=current_time)

    def get_usage(self, user_id: str, tier: str = "free", now: Optional[float] = None) -> int:
        """Return the count of active requests in the current sliding window."""
        current_time = now if now is not None else time.time()
        window_start = current_time - self.window_seconds

        with self._sync_lock:
            dq = self._user_windows.get(user_id)
            if not dq:
                return 0
            while dq and dq[0] <= window_start:
                dq.popleft()
            return len(dq)

    def reset(self, user_id: Optional[str] = None):
        """Reset rate limit history for a specific user or globally."""
        with self._sync_lock:
            if user_id is not None:
                self._user_windows.pop(user_id, None)
            else:
                self._user_windows.clear()
        logger.debug("Rate limiter state reset", user_id=user_id or "all")


# Global singleton instance
rate_limiter = SlidingWindowRateLimiter()

__all__ = [
    "SlidingWindowRateLimiter",
    "rate_limiter",
    "DEFAULT_TIER_LIMITS",
]
