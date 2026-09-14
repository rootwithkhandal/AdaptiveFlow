"""Redis-backed prompt cache with exact-match SHA256 and FAISS semantic caching.

Features:
- Exact-match via SHA256 hash in Redis with TTL.
- Semantic cache via FAISS IndexFlatIP (cosine similarity on normalized embeddings).
- Graceful degradation if Redis or sentence-transformers/faiss are unavailable.
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class RedisCache:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        enable_semantic_cache: bool = True,
        ttl: Optional[int] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.enable_semantic_cache = enable_semantic_cache
        self.ttl = ttl or settings.cache_ttl
        self._client = None
        self._embedder = None
        self._faiss_index = None
        self._cached_docs: List[Dict[str, Any]] = []  # metadata list matching FAISS IDs
        self._local_exact: Dict[str, Tuple[Dict[str, Any], float]] = {}  # In-memory fallback if Redis is offline
        self._redis_cooldown_until = 0.0

        self._connect_redis()
        if self.enable_semantic_cache:
            self._init_semantic_cache()

    def _connect_redis(self):
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=0.2,
                socket_timeout=0.2,
            )
            logger.info("Redis cache connected", url=settings.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable, exact cache will operate without Redis", error=str(e))
            self._client = None
            self._redis_cooldown_until = float("inf")

    def _init_semantic_cache(self):
        try:
            from sentence_transformers import SentenceTransformer
            import faiss

            self._embedder = SentenceTransformer(settings.embedding_model)
            dim = self._embedder.get_sentence_embedding_dimension()
            # IndexFlatIP calculates inner product; on normalized vectors this is cosine similarity
            self._faiss_index = faiss.IndexFlatIP(dim)
            logger.info("FAISS semantic cache initialized", dim=dim, threshold=self.similarity_threshold)
        except Exception as e:
            logger.warning("Semantic cache init failed, running exact-only", error=str(e))
            self._embedder = None
            self._faiss_index = None

    def _hash(self, prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        return vec / np.maximum(norm, 1e-12)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if not self._embedder:
            return None
        raw = self._embedder.encode(text, convert_to_numpy=True).astype("float32")
        return self._normalize(raw)

    async def get_exact(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Exact-match lookup via SHA256 key in Redis with in-memory fallback."""
        key = f"cache:{self._hash(prompt)}"
        if self._client and time.time() >= self._redis_cooldown_until:
            try:
                raw = await self._client.get(key)
                if raw:
                    data = json.loads(raw)
                    data["cache_type"] = "exact"
                    logger.debug("Exact cache hit (redis)", key=key)
                    return data
            except Exception as e:
                self._redis_cooldown_until = time.time() + 30.0
                logger.warning("Exact cache get error (falling back to memory for 30s)", error=str(e))

        # Check in-memory exact cache fallback
        if key in self._local_exact:
            val, expiry = self._local_exact[key]
            if time.time() < expiry:
                data = dict(val)
                data["cache_type"] = "exact"
                logger.debug("Exact cache hit (local memory)", key=key)
                return data
            else:
                del self._local_exact[key]

        return None

    def get_semantic(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Semantic similarity lookup via FAISS on normalized embeddings."""
        if not self._faiss_index or len(self._cached_docs) == 0:
            return None

        try:
            emb = self._embed(prompt)
            if emb is None:
                return None

            scores, indices = self._faiss_index.search(emb.reshape(1, -1), k=1)
            top_score = float(scores[0][0])
            top_idx = int(indices[0][0])

            if top_score >= self.similarity_threshold and top_idx >= 0 and top_idx < len(self._cached_docs):
                doc = dict(self._cached_docs[top_idx])
                doc["cache_type"] = "semantic"
                doc["similarity"] = round(top_score, 4)
                logger.debug("Semantic cache hit", similarity=top_score, threshold=self.similarity_threshold)
                return doc
        except Exception as e:
            logger.warning("Semantic cache lookup failed", error=str(e))

        return None

    async def get(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Unified cache get: tries exact SHA256 match first, then falls back to semantic FAISS."""
        # 1. Exact match (fastest path)
        exact_hit = await self.get_exact(prompt)
        if exact_hit:
            return exact_hit

        # 2. Semantic match (vector similarity)
        if self.enable_semantic_cache:
            semantic_hit = self.get_semantic(prompt)
            if semantic_hit:
                return semantic_hit

        return None

    async def set(self, prompt: str, value: Dict[str, Any]):
        """Store in both Redis (exact) and FAISS (semantic)."""
        key = f"cache:{self._hash(prompt)}"
        # Always store in in-memory fallback
        self._local_exact[key] = (dict(value), time.time() + self.ttl)

        # 1. Store in Redis
        if self._client and time.time() >= self._redis_cooldown_until:
            try:
                await self._client.setex(key, self.ttl, json.dumps(value))
                logger.debug("Redis cache set", key=key, ttl=self.ttl)
            except Exception as e:
                self._redis_cooldown_until = time.time() + 30.0
                logger.warning("Redis cache set error (falling back to memory for 30s)", error=str(e))

        # 2. Store in FAISS semantic index
        if self._faiss_index and self._embedder:
            try:
                emb = self._embed(prompt)
                if emb is not None:
                    self._faiss_index.add(emb.reshape(1, -1))
                    doc_meta = {
                        "prompt": prompt,
                        "response": value.get("response", ""),
                        "model_used": value.get("model_used", ""),
                        "task_type": value.get("task_type", "general"),
                        "timestamp": time.time(),
                    }
                    self._cached_docs.append(doc_meta)
                    logger.debug("FAISS semantic cache set", total_docs=len(self._cached_docs))
            except Exception as e:
                logger.warning("Semantic cache set failed", error=str(e))
