"""Vector memory for context persistence.
Supports FAISS backend.
Stores and retrieves past user interactions as embeddings.
Feeds into prompt augmentation before model dispatch.
"""
import os
import json
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
        self._faiss_index = None
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

    def store(self, user_id: str, prompt: str, response: str):
        """Store a prompt+response pair as an embedding."""
        text = f"Q: {prompt}\nA: {response}"
        try:
            self._faiss_store(user_id, text)
        except Exception as e:
            logger.warning("Memory store failed", error=str(e))

    def retrieve(self, user_id: str, query: str) -> List[str]:
        """Retrieve top-k similar past interactions."""
        try:
            return self._faiss_retrieve(user_id, query)
        except Exception as e:
            logger.warning("Memory retrieve failed", error=str(e))
            return []

    # --- FAISS backend ---
    def _faiss_store(self, user_id: str, text: str):
        import numpy as np
        emb = self._embed(text).astype("float32")
        if user_id not in self._faiss_docs:
            self._faiss_docs[user_id] = []
        self._faiss_docs[user_id].append({"text": text, "embedding": emb})

    def _faiss_retrieve(self, user_id: str, query: str) -> List[str]:
        import numpy as np
        import faiss

        docs = self._faiss_docs.get(user_id, [])
        if not docs:
            return []

        query_emb = self._embed(query).astype("float32")
        embeddings = np.stack([d["embedding"] for d in docs])
        dim = embeddings.shape[1]

        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        k = min(self.top_k, len(docs))
        _, indices = index.search(query_emb.reshape(1, -1), k)
        return [docs[i]["text"] for i in indices[0] if i < len(docs)]
