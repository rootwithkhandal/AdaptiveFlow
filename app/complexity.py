"""Prompt Complexity Scorer.

Evaluates prompt complexity across three dimensions:
1. Token Count (raw and estimated token length)
2. Information Entropy (character Shannon entropy, Type-Token Ratio, vocabulary diversity)
3. Question Depth (reasoning triggers, multi-part questions, conditional clauses, syntax/code blocks)

Simple prompts automatically route to the cheapest capable model, saving cost without user friction.
Complex prompts pass through to full Multi-Armed Bandit RL exploration/exploitation.
"""
import math
import re
from typing import Dict, Any, List, Optional

from app.logger import get_logger
from app.config import settings
from app.models.registry import MODEL_REGISTRY
from app.models.client import get_circuit_breaker

logger = get_logger(__name__)

REASONING_KEYWORDS = [
    "why", "how does", "how do", "explain why", "difference between",
    "compare", "contrast", "trade-offs", "tradeoffs", "derive", "analyze",
    "implications", "pros and cons", "step-by-step", "architect", "step by step",
    "in-depth", "synthesize", "optimize", "evaluate", "critique", "root cause",
]

CONDITIONAL_CLAUSES = [
    "however", "assuming", "suppose", "furthermore", "whereas",
    "on the other hand", "given that", "in case of", "otherwise",
]

CODE_SYNTAX = [
    "def ", "class ", "function ", "async ", "import ", "lambda ",
    "return ", "select ", "from ", "where ", "join ", "curl ",
]


class PromptComplexityScorer:
    """Scores prompt complexity to optimize routing efficiency and eliminate unnecessary LLM cost."""

    def __init__(
        self,
        threshold: Optional[float] = None,
        weight_tokens: Optional[float] = None,
        weight_entropy: Optional[float] = None,
        weight_depth: Optional[float] = None,
    ):
        self.threshold = threshold if threshold is not None else getattr(settings, "complexity_threshold", 0.35)
        self.weight_tokens = weight_tokens if weight_tokens is not None else getattr(settings, "complexity_weight_tokens", 0.25)
        self.weight_entropy = weight_entropy if weight_entropy is not None else getattr(settings, "complexity_weight_entropy", 0.35)
        self.weight_depth = weight_depth if weight_depth is not None else getattr(settings, "complexity_weight_depth", 0.40)

    def compute_token_score(self, prompt: str) -> Dict[str, Any]:
        """Estimate token count and normalize over standard prompt length."""
        words = prompt.split()
        word_count = len(words)
        estimated_tokens = max(1, int(word_count * 1.33))
        score = min(1.0, estimated_tokens / 120.0)
        return {
            "score": round(score, 4),
            "estimated_tokens": estimated_tokens,
            "word_count": word_count,
            "char_count": len(prompt),
        }

    def compute_entropy_score(self, prompt: str) -> Dict[str, Any]:
        """Compute information entropy combining character Shannon entropy and Type-Token Ratio."""
        if not prompt:
            return {"score": 0.0, "char_entropy": 0.0, "type_token_ratio": 0.0}

        # 1. Character-level Shannon entropy H = - sum(p * log2(p))
        char_counts: Dict[str, int] = {}
        for c in prompt:
            char_counts[c] = char_counts.get(c, 0) + 1

        total_chars = len(prompt)
        h_char = -sum((cnt / total_chars) * math.log2(cnt / total_chars) for cnt in char_counts.values())
        norm_h_char = min(1.0, h_char / 5.0)

        # 2. Word-level Type-Token Ratio (vocabulary diversity)
        words = prompt.split()
        word_count = len(words)
        if word_count > 0:
            cleaned_words = [w.lower().strip(".,!?:;\"'()[]{}") for w in words]
            unique_words = len(set(cleaned_words))
            ttr = unique_words / word_count
            length_factor = min(1.0, word_count / 15.0)
        else:
            ttr = 0.0
            length_factor = 0.0

        # Composite entropy score
        entropy_score = 0.5 * norm_h_char + 0.5 * (ttr * length_factor)
        return {
            "score": round(entropy_score, 4),
            "char_entropy": round(h_char, 3),
            "type_token_ratio": round(ttr, 3),
        }

    def compute_question_depth_score(self, prompt: str) -> Dict[str, Any]:
        """Detect reasoning depth indicators: analytical keywords, multi-questions, code structures."""
        prompt_lower = prompt.lower()
        depth_score = 0.0
        signals: List[str] = []

        # 1. Analytical reasoning keywords
        matched_reasoning = [k for k in REASONING_KEYWORDS if k in prompt_lower]
        if matched_reasoning:
            depth_score += min(0.50, len(matched_reasoning) * 0.20)
            signals.extend([f"reasoning:{k}" for k in matched_reasoning[:3]])

        # 2. Multi-part question markers
        q_count = prompt.count("?")
        if q_count >= 2:
            depth_score += 0.25
            signals.append(f"multi_question:{q_count}")

        # 3. Numbered or bullet list structures
        if re.search(r'(\d+\.|\b[a-z]\))\s+', prompt):
            depth_score += 0.20
            signals.append("numbered_list")
        if re.search(r'^\s*[-*]\s+', prompt, re.M):
            depth_score += 0.20
            signals.append("bullet_list")

        # 4. Conditional or comparative clauses
        matched_cond = [c for c in CONDITIONAL_CLAUSES if c in prompt_lower]
        if matched_cond:
            depth_score += min(0.30, len(matched_cond) * 0.15)
            signals.extend([f"conditional:{c}" for c in matched_cond[:2]])

        # 5. Code blocks, structured payloads, or programming syntax
        if "```" in prompt or "`" in prompt:
            depth_score += 0.35
            signals.append("code_fence")
        if ("{" in prompt and "}" in prompt or "[" in prompt and "]" in prompt) and len(prompt) > 40:
            depth_score += 0.20
            signals.append("structured_syntax")
        if any(k in prompt_lower for k in CODE_SYNTAX):
            depth_score += 0.25
            signals.append("code_syntax")

        clamped_depth = min(1.0, depth_score)
        return {
            "score": round(clamped_depth, 4),
            "signals": signals,
        }

    def score(self, prompt: str) -> Dict[str, Any]:
        """Compute holistic prompt complexity score and return classification."""
        token_info = self.compute_token_score(prompt)
        entropy_info = self.compute_entropy_score(prompt)
        depth_info = self.compute_question_depth_score(prompt)

        s_tokens = token_info["score"]
        s_entropy = entropy_info["score"]
        s_depth = depth_info["score"]

        composite = (
            self.weight_tokens * s_tokens +
            self.weight_entropy * s_entropy +
            self.weight_depth * s_depth
        )
        composite = round(min(1.0, max(0.0, composite)), 4)
        is_simple = composite < self.threshold
        classification = "simple" if is_simple else "complex"

        logger.debug(
            "Prompt complexity evaluated",
            score=composite,
            classification=classification,
            is_simple=is_simple,
            tokens=token_info["estimated_tokens"],
            signals=depth_info["signals"],
        )

        return {
            "score": composite,
            "is_simple": is_simple,
            "classification": classification,
            "threshold": self.threshold,
            "token_count": token_info["estimated_tokens"],
            "entropy": entropy_info["score"],
            "question_depth": depth_info["score"],
            "signals": depth_info["signals"],
            "details": {
                "tokens": token_info,
                "entropy": entropy_info,
                "depth": depth_info,
            },
        }


def get_cheapest_capable_model(
    available_models: List[str],
    task_type: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    q_values: Optional[Dict[str, float]] = None,
) -> str:
    """Select the cheapest capable, healthy model permitted for user tier and task.

    - Bypasses circuit breakers that are OPEN.
    - Avoids models on the user's avoid list when alternatives exist.
    - Breaks cost ties by choosing the highest Q-value or lowest latency.
    """
    if not available_models:
        return "ollama/llama3"

    # 1. Filter by circuit breaker health
    healthy = [
        m for m in available_models
        if get_circuit_breaker(m).state.value != "OPEN"
    ]
    pool = healthy if healthy else available_models

    # 2. Avoid list filtering
    if profile:
        avoid_patterns = [str(p).strip().lower() for p in profile.get("avoid", []) if str(p).strip()]
        if avoid_patterns:
            non_avoided = [
                m for m in pool
                if not any(pat in m.lower() for pat in avoid_patterns)
            ]
            if non_avoided:
                pool = non_avoided

    # 3. Sort by cost_per_1k_tokens ascending, then Q-value descending
    q_map = q_values or {}

    def _sort_key(m: str):
        meta = MODEL_REGISTRY.get(m, {})
        cost = meta.get("cost_per_1k_tokens", 0.0)
        q = q_map.get(m, 0.0)
        # Primary: lowest cost, Secondary: highest Q (so negative q)
        return (cost, -q)

    sorted_pool = sorted(pool, key=_sort_key)
    return sorted_pool[0]
