"""Unit tests for Redis & FAISS semantic cache."""
import pytest
import numpy as np
from unittest.mock import MagicMock
from app.redis_cache import RedisCache


def test_cache_hashing():
    cache = RedisCache(enable_semantic_cache=False)
    h1 = cache._hash("  Hello WORLD  ")
    h2 = cache._hash("hello world")
    assert h1 == h2


def test_semantic_cache_similarity():
    cache = RedisCache(similarity_threshold=0.85, enable_semantic_cache=True)

    # Mock embedder with synthetic normalized embeddings
    dim = 4
    cache._embedder = MagicMock()
    # Let prompt "A" have embedding [1, 0, 0, 0]
    # Let prompt "B" (similar) have embedding [0.95, 0.31, 0, 0]
    # Let prompt "C" (dissimilar) have embedding [0, 1, 0, 0]
    vec_a = np.array([1.0, 0.0, 0.0, 0.0], dtype="float32")
    vec_b = np.array([0.95, 0.3122, 0.0, 0.0], dtype="float32")
    vec_c = np.array([0.0, 1.0, 0.0, 0.0], dtype="float32")

    # Re-normalize
    vec_a = cache._normalize(vec_a)
    vec_b = cache._normalize(vec_b)
    vec_c = cache._normalize(vec_c)

    import faiss
    cache._faiss_index = faiss.IndexFlatIP(dim)
    cache._cached_docs = []

    # Store prompt A
    cache._faiss_index.add(vec_a.reshape(1, -1))
    cache._cached_docs.append({
        "prompt": "How do I sort a list in Python?",
        "response": "Use sorted(list)",
        "model_used": "ollama/llama3",
        "task_type": "code",
    })

    # Test lookup with similar prompt B (sim should be ~0.95 >= 0.85)
    cache._embed = MagicMock(return_value=vec_b)
    hit = cache.get_semantic("Sort Python list quickly")
    assert hit is not None
    assert hit["cache_type"] == "semantic"
    assert hit["response"] == "Use sorted(list)"
    assert hit["similarity"] >= 0.85

    # Test lookup with dissimilar prompt C (sim should be 0.0 < 0.85)
    cache._embed = MagicMock(return_value=vec_c)
    miss = cache.get_semantic("What is the weather in Tokyo?")
    assert miss is None
