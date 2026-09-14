"""Rule-based task classifier.
Classifies prompts into: code, security, creative, general.
Powers RL task-type routing.
"""
import re
from typing import Dict, List, Any

from app.logger import get_logger

logger = get_logger(__name__)

# Keyword patterns per task type
TASK_PATTERNS: Dict[str, List[str]] = {
    "code": [
        r"\b(code|function|class|debug|implement|algorithm|script|program|bug|error|syntax|compile|refactor|api|endpoint|sql|query|regex)\b",
    ],
    "security": [
        r"\b(password|auth|token|jwt|oauth|encrypt|decrypt|hash|vulnerability|exploit|injection|xss|csrf|pentest|firewall|ssl|tls|certificate)\b",
    ],
    "creative": [
        r"\b(write|story|poem|essay|creative|imagine|fiction|narrative|blog|article|describe|generate|draft)\b",
    ],
}


class TaskClassifier:
    """Classifies user prompts into task categories using regex keyword patterns."""

    def __init__(self, patterns: Dict[str, List[str]] = None):
        self.patterns = patterns or TASK_PATTERNS
        self._compiled = {
            task: [re.compile(p) for p in pat_list]
            for task, pat_list in self.patterns.items()
        }

    def classify(self, prompt: str) -> str:
        """Returns task type string: 'code', 'security', 'creative', or 'general'."""
        lower = prompt.lower()

        for task, compiled_patterns in self._compiled.items():
            for pattern in compiled_patterns:
                if pattern.search(lower):
                    logger.debug("Task classified", task=task, pattern=pattern.pattern)
                    return task

        return "general"

    def classify_with_details(self, prompt: str) -> Dict[str, Any]:
        """Returns detailed classification with matched keywords and confidence score."""
        lower = prompt.lower()
        matched_keywords = []
        best_task = "general"

        for task, compiled_patterns in self._compiled.items():
            task_matches = []
            for pattern in compiled_patterns:
                matches = pattern.findall(lower)
                if matches:
                    if isinstance(matches[0], tuple):
                        task_matches.extend([m for tup in matches for m in tup if m])
                    else:
                        task_matches.extend(matches)
            if task_matches and best_task == "general":
                best_task = task
                matched_keywords = list(dict.fromkeys(task_matches))

        confidence = 0.95 if best_task != "general" else 0.50
        return {
            "task_type": best_task,
            "matched_keywords": matched_keywords,
            "confidence": confidence,
        }
