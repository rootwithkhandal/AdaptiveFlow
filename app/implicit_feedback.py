"""Implicit Feedback Tracker: extracts passive reward signals from user interaction dynamics.

Signals tracked:
1. Follow-up prompt similarity (rephrased prompt within session = prior model failure)
2. Session dwell time after response (healthy read time 15s-180s = positive engagement)
3. Rapid bounce / skip (<5s = fast bounce penalty)
4. Abandonment / timeout (interrupted or canceled turn = harsh penalty)
"""
import time
from typing import Dict, Any, Optional
import numpy as np

from app.config import settings
from app.logger import get_logger

logger = get_logger(__name__)


class ImplicitFeedbackTracker:
    """Monitors multi-turn user sessions to infer reward signals without explicit rating."""

    def __init__(self, session_window: float = 300.0, rephrase_threshold: float = 0.80):
        self.session_window = session_window  # 5 minutes maximum for consecutive turn association
        self.rephrase_threshold = rephrase_threshold
        self._user_turns: Dict[str, Dict[str, Any]] = {}
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(settings.embedding_model)
            except Exception as e:
                logger.warning("Embedder unavailable for implicit feedback similarity", error=str(e))
                self._embedder = None
        return self._embedder

    def _cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Calculate cosine similarity between two prompts."""
        # Fast path: exact string match
        if text_a.strip().lower() == text_b.strip().lower():
            return 1.0

        embedder = self._get_embedder()
        if embedder:
            try:
                emb = embedder.encode([text_a, text_b], convert_to_numpy=True).astype("float32")
                norm = np.linalg.norm(emb, axis=-1, keepdims=True)
                emb_norm = emb / np.maximum(norm, 1e-12)
                sim = float(np.dot(emb_norm[0], emb_norm[1]))
                return max(0.0, min(1.0, sim))
            except Exception as e:
                logger.warning("Embedding cosine similarity calculation failed", error=str(e))

        # Fallback to Jaccard word token similarity if embedding model unavailable
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        return intersection / union if union > 0 else 0.0

    def evaluate_followup(self, user_id: str, current_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Evaluates current prompt against user's prior turn.
        If an implicit signal is inferred for the previous turn's model, returns signal details.
        """
        last_turn = self._user_turns.get(user_id)
        if not last_turn:
            return None

        now = time.time()
        dwell_time = now - last_turn["timestamp"]

        # If turn occurred outside the active session window, treat as fresh session
        if dwell_time > self.session_window:
            return None

        last_prompt = last_turn["prompt"]
        last_model = last_turn["model"]
        last_task = last_turn.get("task", "general")

        similarity = self._cosine_similarity(last_prompt, current_prompt)

        # 1. High similarity = User rephrased query because model failed
        if similarity >= self.rephrase_threshold:
            logger.info(
                "Implicit signal: Rephrase penalty detected",
                user_id=user_id,
                model=last_model,
                similarity=round(similarity, 3),
                dwell_time=round(dwell_time, 1),
            )
            return {
                "user_id": user_id,
                "model": last_model,
                "task": last_task,
                "signal_type": "rephrase_penalty",
                "reward_delta": -0.8,
                "similarity": round(similarity, 3),
                "dwell_time": round(dwell_time, 1),
                "rationale": f"User rephrased query (similarity {round(similarity, 2)}) within {round(dwell_time, 1)}s",
            }

        # 2. Short dwell (< 5s) with distinct query = fast bounce
        if dwell_time < 5.0:
            logger.debug(
                "Implicit signal: Fast bounce detected",
                user_id=user_id,
                model=last_model,
                dwell_time=round(dwell_time, 1),
            )
            return {
                "user_id": user_id,
                "model": last_model,
                "task": last_task,
                "signal_type": "fast_bounce",
                "reward_delta": -0.2,
                "similarity": round(similarity, 3),
                "dwell_time": round(dwell_time, 1),
                "rationale": f"Fast bounce: subsequent prompt sent in {round(dwell_time, 1)}s",
            }

        # 3. Healthy dwell time (15s to 180s) with progression = positive engagement
        if 15.0 <= dwell_time <= 180.0:
            logger.info(
                "Implicit signal: Positive engagement detected",
                user_id=user_id,
                model=last_model,
                dwell_time=round(dwell_time, 1),
            )
            return {
                "user_id": user_id,
                "model": last_model,
                "task": last_task,
                "signal_type": "engagement_reward",
                "reward_delta": +0.3,
                "similarity": round(similarity, 3),
                "dwell_time": round(dwell_time, 1),
                "rationale": f"Positive engagement: natural reading dwell time of {round(dwell_time, 1)}s",
            }

        return None

    def record_turn(self, user_id: str, prompt: str, model: str, task: str, latency: float):
        """Records the latest turn state for a user."""
        self._user_turns[user_id] = {
            "prompt": prompt,
            "model": model,
            "task": task,
            "latency": latency,
            "timestamp": time.time(),
        }

    def record_abandonment(self, user_id: str, reason: str = "client_abort") -> Optional[Dict[str, Any]]:
        """
        Triggered when a user aborts or disconnects prematurely.
        Applies a penalty to the model associated with the abandoned turn.
        """
        last_turn = self._user_turns.get(user_id)
        if not last_turn:
            return None

        # Expire turn so penalty isn't applied repeatedly
        del self._user_turns[user_id]

        logger.warning(
            "Implicit signal: Session abandonment detected",
            user_id=user_id,
            model=last_turn["model"],
            reason=reason,
        )

        return {
            "user_id": user_id,
            "model": last_turn["model"],
            "task": last_turn.get("task", "general"),
            "signal_type": "abandonment_penalty",
            "reward_delta": -1.5,
            "similarity": 0.0,
            "dwell_time": round(time.time() - last_turn["timestamp"], 1),
            "rationale": f"Session abandoned by client ({reason})",
        }
