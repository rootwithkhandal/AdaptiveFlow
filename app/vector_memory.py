"""Vector memory for context persistence.
Supports FAISS backend.
Stores and retrieves past user interactions as embeddings.
Feeds into prompt augmentation before model dispatch.
"""
import os
import json
import asyncio
from typing import List, Dict, Optional
from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)

MEMORY_DIR = "data/memory"
os.makedirs(MEMORY_DIR, exist_ok=True)


class VectorMemory:
    def __init__(self, backend: Optional[str] = None, top_k: Optional[int] = None):
        self.backend = backend or settings.memory_backend
        self.top_k = top_k or settings.top_k_context
        self._embedder = None
        self._faiss_index: Dict[str, object] = {}
        self._faiss_docs: Dict[str, List[Dict]] = {}  # user_id -> list of {text, embedding}
        self._init_backend()

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(settings.embedding_model)
        return self._embedder

    def _init_backend(self):
        logger.info("FAISS memory initialized")

    def _embed(self, text: str):
        return self._get_embedder().encode(text, convert_to_numpy=True)

    async def store(self, user_id: str, prompt: str, response: str):
        """Store a prompt+response pair as an embedding."""
        text = f"Q: {prompt}\nA: {response}"
        try:
            await asyncio.to_thread(self._faiss_store, user_id, text)
        except Exception as e:
            logger.warning("Memory store failed", error=str(e))

    async def retrieve(self, user_id: str, query: str) -> List[str]:
        """Retrieve top-k similar past interactions."""
        try:
            return await asyncio.to_thread(self._faiss_retrieve, user_id, query)
        except Exception as e:
            logger.warning("Memory retrieve failed", error=str(e))
            return []

    # --- FAISS backend ---
    def _faiss_store(self, user_id: str, text: str):
        import faiss
        import numpy as np
        emb = self._embed(text).astype("float32")
        if user_id not in self._faiss_docs:
            self._faiss_docs[user_id] = []
        docs = self._faiss_docs[user_id]
        docs.append({"text": text, "embedding": emb})
        max_entries = settings.max_vector_memory_entries_per_user
        if len(docs) > max_entries:
            del docs[:len(docs) - max_entries]
            self._faiss_index.pop(user_id, None)
        index = self._faiss_index.get(user_id)
        if index is None:
            index = faiss.IndexFlatL2(emb.shape[0])
            index.add(np.stack([d["embedding"] for d in docs]))
            self._faiss_index[user_id] = index
        else:
            index.add(emb.reshape(1, -1))

    def _faiss_retrieve(self, user_id: str, query: str) -> List[str]:
        import numpy as np
        import faiss

        docs = self._faiss_docs.get(user_id, [])
        if not docs:
            return []

        query_emb = self._embed(query).astype("float32")
        index = self._faiss_index.get(user_id)
        if index is None:
            index = faiss.IndexFlatL2(docs[0]["embedding"].shape[0])
            index.add(np.stack([d["embedding"] for d in docs]))
            self._faiss_index[user_id] = index
        k = min(self.top_k, len(docs))
        _, indices = index.search(query_emb.reshape(1, -1), k)
        return [docs[i]["text"] for i in indices[0] if i < len(docs)]
