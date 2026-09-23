"""Reinforcement Learning Router using Epsilon-Greedy or Thompson Sampling Multi-Armed Bandit.
Provides full Per-Task-Type RL State Split across code, security, creative, and general tasks,
giving each task class separate exploration parameters (epsilon arms), Q-tables, Beta distributions,
and state persistence.
"""
import os
import json
import random
import tempfile
from typing import Dict, Any, Optional, List

from app.task_classifier import TaskClassifier
from app.models.registry import MODEL_REGISTRY, get_models_for_profile
from app.models.client import call_model
from app.config import settings
from app.logger import get_logger
from app.complexity import PromptComplexityScorer, get_cheapest_capable_model

logger = get_logger(__name__)

SUPPORTED_TASKS = ["code", "security", "creative", "general"]
RL_STATE_FILE = "data/rl_state.json"
os.makedirs("data", exist_ok=True)


class RLRouter:
    def __init__(
        self,
        strategy: str = "epsilon_greedy",
        per_task_split: bool = True,
        state_file: Optional[str] = RL_STATE_FILE,
        complexity_scorer: Optional[PromptComplexityScorer] = None,
    ):
        """
        strategy: 'epsilon_greedy' or 'thompson_sampling'
        per_task_split: Default True. Each task class has its own independent Q-table & exploration rate.
        state_file: Path to persist learned weights across restarts, or None to disable.
        """
        self.strategy = strategy
        self.per_task_split = per_task_split
        self.state_file = state_file
        self.classifier = TaskClassifier()
        self.complexity_scorer = complexity_scorer or PromptComplexityScorer()

        # Global Q-values and aggregate telemetry
        self.epsilon = settings.epsilon
        self.q_values: Dict[str, float] = {m: 0.0 for m in MODEL_REGISTRY}
        self.counts: Dict[str, int] = {m: 0 for m in MODEL_REGISTRY}
        self.avg_latency: Dict[str, float] = {m: 0.0 for m in MODEL_REGISTRY}
        self.avg_cost: Dict[str, float] = {m: 0.0 for m in MODEL_REGISTRY}
        self.ts_alpha: Dict[str, float] = {m: 1.0 for m in MODEL_REGISTRY}
        self.ts_beta: Dict[str, float] = {m: 1.0 for m in MODEL_REGISTRY}

        # --- Independent Per-Task State Splits ---
        # Each task category maintains separate Q-tables, selection counts, and decay rates
        self.task_q_values: Dict[str, Dict[str, float]] = {
            task: {m: 0.0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }
        self.task_counts: Dict[str, Dict[str, int]] = {
            task: {m: 0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }
        self.task_epsilon: Dict[str, float] = {
            task: settings.epsilon for task in SUPPORTED_TASKS
        }
        self.task_ts_alpha: Dict[str, Dict[str, float]] = {
            task: {m: 1.0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }
        self.task_ts_beta: Dict[str, Dict[str, float]] = {
            task: {m: 1.0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }
        self.task_avg_latency: Dict[str, Dict[str, float]] = {
            task: {m: 0.0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }
        self.task_avg_cost: Dict[str, Dict[str, float]] = {
            task: {m: 0.0 for m in MODEL_REGISTRY} for task in SUPPORTED_TASKS
        }

        # Load persisted weights if available
        self._load_state()

    def _ensure_model(self, model: str):
        """Ensure model exists across all global and per-task data structures."""
        if model not in self.q_values:
            self.q_values[model] = 0.0
            self.counts[model] = 0
            self.avg_latency[model] = 0.0
            self.avg_cost[model] = 0.0
            self.ts_alpha[model] = 1.0
            self.ts_beta[model] = 1.0
            for task in SUPPORTED_TASKS:
                if task not in self.task_q_values:
                    self.task_q_values[task] = {}
                self.task_q_values[task][model] = 0.0
                if task not in self.task_counts:
                    self.task_counts[task] = {}
                self.task_counts[task][model] = 0
                if task not in self.task_ts_alpha:
                    self.task_ts_alpha[task] = {}
                self.task_ts_alpha[task][model] = 1.0
                if task not in self.task_ts_beta:
                    self.task_ts_beta[task] = {}
                self.task_ts_beta[task][model] = 1.0
                if task not in self.task_avg_latency:
                    self.task_avg_latency[task] = {}
                self.task_avg_latency[task][model] = 0.0
                if task not in self.task_avg_cost:
                    self.task_avg_cost[task] = {}
                self.task_avg_cost[task][model] = 0.0

    def _resolve_task(self, task_type: Optional[str]) -> str:
        """Ensures task_type resolves to a recognized task partition."""
        if task_type and task_type in SUPPORTED_TASKS:
            return task_type
        return "general"

    def _compute_preference_prior(
        self,
        model: str,
        profile: Optional[Dict[str, Any]] = None,
        task_type: Optional[str] = None,
    ) -> float:
        """Compute user preference prior offset for candidate model arm.
        
        - avoid list (e.g. ['gemini', 'ollama']): severe penalty (-5.0)
        - prefer persona/family (e.g. 'code_heavy', 'claude', 'fast', 'cost_saving'): positive boost (+0.75 to +1.0)
        """
        if not profile:
            return 0.0

        prior = 0.0
        model_lower = model.lower()

        # 1. Avoid list check (case-insensitive substring match)
        avoid_list = profile.get("avoid") or []
        for avoid_pattern in avoid_list:
            pat = str(avoid_pattern).strip().lower()
            if pat and pat in model_lower:
                prior -= 5.0
                break

        # 2. Prefer persona / specialization / model family check
        prefer = profile.get("prefer")
        if prefer:
            pref = str(prefer).strip().lower()
            meta = MODEL_REGISTRY.get(model, {})
            tasks = meta.get("tasks", [])
            cost_1k = meta.get("cost_per_1k_tokens", 0.0)

            if pref in ["code_heavy", "code"]:
                if "code" in tasks or any(k in model_lower for k in ["claude", "gpt-4o", "nemotron", "qwen"]):
                    prior += 0.75
            elif pref in ["creative", "creative_writing"]:
                if "creative" in tasks or any(k in model_lower for k in ["claude", "gemini"]):
                    prior += 0.75
            elif pref in ["security_first", "security"]:
                if "security" in tasks or any(k in model_lower for k in ["nemotron", "gpt-4o", "claude"]):
                    prior += 0.75
            elif pref in ["fast", "low_latency", "speed"]:
                if any(k in model_lower for k in ["flash", "mini", "8b", "llama3"]):
                    prior += 0.75
            elif pref in ["cost_saving", "cheap", "budget"]:
                if cost_1k <= 0.0002:
                    prior += 0.75
            elif pref in model_lower:
                prior += 1.0

        return prior

    def _select_model_epsilon_greedy(
        self,
        available_models: list,
        task_type: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Epsilon-greedy selection with independent per-task exploration arms and preference priors."""
        task = self._resolve_task(task_type)

        if self.per_task_split:
            eps = self.task_epsilon[task]
            q_dict = self.task_q_values[task]
        else:
            eps = self.epsilon
            q_dict = self.q_values

        # Apply user preference priors to effective Q-values for selection
        effective_q = {
            m: q_dict.get(m, 0.0) + self._compute_preference_prior(m, profile, task)
            for m in available_models
        }

        if random.random() < eps:
            # During exploration, steer clear of explicitly avoided models when possible
            non_avoided = [
                m for m in available_models
                if self._compute_preference_prior(m, profile, task) >= 0.0
            ]
            chosen = random.choice(non_avoided if non_avoided else available_models)
            logger.debug("Explore", model=chosen, epsilon=round(eps, 4), task=task)
        else:
            chosen = max(available_models, key=lambda m: effective_q[m])
            logger.debug("Exploit", model=chosen, task=task, q=round(q_dict.get(chosen, 0.0), 4), effective_q=round(effective_q[chosen], 4))

        # Decay exploration rate for this specific task arm
        if self.per_task_split:
            self.task_epsilon[task] = max(settings.min_epsilon, self.task_epsilon[task] * settings.epsilon_decay)
        else:
            self.epsilon = max(settings.min_epsilon, self.epsilon * settings.epsilon_decay)

        self._save_state()
        return chosen

    def _select_model_thompson(
        self,
        available_models: list,
        task_type: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Thompson Sampling: sample from Beta distribution per model biased by preference priors."""
        task = self._resolve_task(task_type)

        if self.per_task_split:
            alpha_dict = self.task_ts_alpha[task]
            beta_dict = self.task_ts_beta[task]
        else:
            alpha_dict = self.ts_alpha
            beta_dict = self.ts_beta

        samples = {}
        for m in available_models:
            a = alpha_dict.get(m, 1.0)
            b = beta_dict.get(m, 1.0)
            prior = self._compute_preference_prior(m, profile, task)
            if prior > 0:
                a += 2.0 * prior
            elif prior < 0:
                b += 2.0 * abs(prior)
            samples[m] = random.betavariate(a, b)

        chosen = max(samples, key=samples.get)
        logger.debug("Thompson sampling", model=chosen, task=task, samples=samples)
        return chosen

    def _select_model(
        self,
        available_models: list,
        task_type: Optional[str] = None,
        profile: Optional[Dict[str, Any]] = None,
    ) -> str:
        if self.strategy == "thompson_sampling":
            return self._select_model_thompson(available_models, task_type, profile)
        return self._select_model_epsilon_greedy(available_models, task_type, profile)

    async def route(
        self,
        user_id: str,
        prompt: str,
        profile: Dict,
        check_complexity: bool = True,
    ) -> Dict[str, Any]:
        """Classify task, evaluate prompt complexity, select model, call it, return result."""
        task_type = self.classifier.classify(prompt)
        available_models = get_models_for_profile(profile, task_type)

        if not available_models:
            available_models = ["ollama/llama3"]  # ultimate fallback

        complexity_info = self.complexity_scorer.score(prompt) if (check_complexity and self.complexity_scorer) else None

        if complexity_info and complexity_info["is_simple"]:
            # Route simple prompts to cheapest capable model automatically
            task_q = self.task_q_values.get(task_type, {})
            model = get_cheapest_capable_model(
                available_models,
                task_type=task_type,
                profile=profile,
                q_values=task_q,
            )
            routing_strategy = f"simple_bypass (cheapest capable: {model})"
            logger.info(
                "Routing simple prompt to cheapest capable model",
                user_id=user_id,
                model=model,
                complexity_score=complexity_info["score"],
            )
        else:
            model = self._select_model(available_models, task_type=task_type, profile=profile)
            routing_strategy = f"exploit (argmax Q in '{task_type}' arm)" if self.strategy != "thompson_sampling" else "thompson_sampling"
            logger.info("Routing complex prompt via full RL selection", user_id=user_id, model=model, task=task_type)

        result = await call_model(model, prompt)
        result["task_type"] = task_type
        result["model_used"] = model
        result["routing_strategy"] = routing_strategy
        if complexity_info:
            result["complexity"] = complexity_info
        return result

    def _compute_reward(self, cost: float, latency: float, accuracy: float = 1.0) -> float:
        """reward = (accuracy * 2) - (latency * 0.1) - (cost * 5)"""
        return (accuracy * 2) - (latency * 0.1) - (cost * 5)

    def update_reward(
        self,
        model: str,
        cost: float,
        latency: float,
        accuracy: float = 1.0,
        task_type: Optional[str] = None,
    ):
        """Update Q-value using incremental mean update for both global and per-task states."""
        self._ensure_model(model)
        reward = self._compute_reward(cost, latency, accuracy)
        task = self._resolve_task(task_type)

        # 1. Update global aggregate telemetry
        n = self.counts.get(model, 0) + 1
        self.counts[model] = n

        old_q = self.q_values.get(model, 0.0)
        self.q_values[model] = old_q + (reward - old_q) / n
        self.avg_latency[model] = self.avg_latency.get(model, 0.0) + (latency - self.avg_latency.get(model, 0.0)) / n
        self.avg_cost[model] = self.avg_cost.get(model, 0.0) + (cost - self.avg_cost.get(model, 0.0)) / n

        if reward > 0:
            self.ts_alpha[model] = self.ts_alpha.get(model, 1.0) + 1
        else:
            self.ts_beta[model] = self.ts_beta.get(model, 1.0) + 1

        # 2. Update task-specific Q-table and metrics (prevents muddy signal)
        task_n = self.task_counts[task].get(model, 0) + 1
        self.task_counts[task][model] = task_n

        t_old_q = self.task_q_values[task].get(model, 0.0)
        self.task_q_values[task][model] = t_old_q + (reward - t_old_q) / task_n

        self.task_avg_latency[task][model] = self.task_avg_latency[task].get(model, 0.0) + (
            latency - self.task_avg_latency[task].get(model, 0.0)
        ) / task_n
        self.task_avg_cost[task][model] = self.task_avg_cost[task].get(model, 0.0) + (
            cost - self.task_avg_cost[task].get(model, 0.0)
        ) / task_n

        if reward > 0:
            self.task_ts_alpha[task][model] = self.task_ts_alpha[task].get(model, 1.0) + 1
        else:
            self.task_ts_beta[task][model] = self.task_ts_beta[task].get(model, 1.0) + 1

        logger.debug(
            "Q-value updated",
            model=model,
            task=task,
            task_q=round(self.task_q_values[task][model], 4),
            reward=round(reward, 4),
        )

        self._save_state()

    def apply_feedback(self, model: str, rating: int, task_type: Optional[str] = None):
        """Incorporate explicit user feedback into global and task-specific reward scores."""
        self._ensure_model(model)
        feedback_reward = rating * 0.5
        task = self._resolve_task(task_type)

        # Global update
        old_q = self.q_values.get(model, 0.0)
        self.q_values[model] = old_q + feedback_reward * 0.1

        if rating > 0:
            self.ts_alpha[model] = self.ts_alpha.get(model, 1.0) + 1
        elif rating < 0:
            self.ts_beta[model] = self.ts_beta.get(model, 1.0) + 1

        # Task-specific update
        t_old_q = self.task_q_values[task].get(model, 0.0)
        self.task_q_values[task][model] = t_old_q + feedback_reward * 0.1
        if rating > 0:
            self.task_ts_alpha[task][model] = self.task_ts_alpha[task].get(model, 1.0) + 1
        elif rating < 0:
            self.task_ts_beta[task][model] = self.task_ts_beta[task].get(model, 1.0) + 1

        self._save_state()

    def apply_implicit_signal(
        self,
        model: str,
        signal_type: str,
        reward_delta: float,
        task_type: Optional[str] = None,
    ):
        """
        Incorporates passive, implicit signals (rephrases, engagement dwell, abandonment)
        directly into the task-specific Multi-Armed Bandit arms.
        """
        self._ensure_model(model)
        adjustment = reward_delta * 0.1
        task = self._resolve_task(task_type)

        # Global update
        old_q = self.q_values.get(model, 0.0)
        self.q_values[model] = old_q + adjustment

        if reward_delta > 0:
            self.ts_alpha[model] = self.ts_alpha.get(model, 1.0) + 1
        elif reward_delta < 0:
            self.ts_beta[model] = self.ts_beta.get(model, 1.0) + 1

        # Task-specific update
        t_old_q = self.task_q_values[task].get(model, 0.0)
        self.task_q_values[task][model] = t_old_q + adjustment
        if reward_delta > 0:
            self.task_ts_alpha[task][model] = self.task_ts_alpha[task].get(model, 1.0) + 1
        elif reward_delta < 0:
            self.task_ts_beta[task][model] = self.task_ts_beta[task].get(model, 1.0) + 1

        logger.info(
            "Implicit reward applied to task arm",
            model=model,
            signal=signal_type,
            task=task,
            delta=reward_delta,
            task_q=round(self.task_q_values[task][model], 4),
        )

        self._save_state()

    def seed_model_weights(self, model: str, warmup_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Seeds Q-table, counts, and Thompson Beta priors for a newly registered model
        using empirical results from the synthetic benchmark suite, eliminating blind exploration.
        """
        self._ensure_model(model)
        tasks_data = warmup_results.get("tasks", {})

        for task, stats in tasks_data.items():
            if task in SUPPORTED_TASKS:
                self.task_q_values[task][model] = stats.get("avg_reward", 0.0)
                self.task_counts[task][model] = stats.get("count", 0)
                self.task_avg_latency[task][model] = stats.get("avg_latency", 0.0)
                self.task_avg_cost[task][model] = stats.get("avg_cost", 0.0)
                self.task_ts_alpha[task][model] = 1.0 + float(stats.get("successes", 0))
                self.task_ts_beta[task][model] = 1.0 + float(stats.get("failures", 0))

        overall = warmup_results.get("overall", {})
        if overall:
            self.q_values[model] = overall.get("avg_reward", 0.0)
            self.counts[model] = overall.get("total_prompts", 0)
            self.avg_latency[model] = overall.get("avg_latency", 0.0)
            self.avg_cost[model] = overall.get("avg_cost", 0.0)
            total_successes = sum(s.get("successes", 0) for s in tasks_data.values())
            total_failures = sum(s.get("failures", 0) for s in tasks_data.values())
            self.ts_alpha[model] = 1.0 + float(total_successes)
            self.ts_beta[model] = 1.0 + float(total_failures)

        logger.info(
            "Seeded RL model weights from benchmark suite",
            model=model,
            q_value=self.q_values.get(model),
            tasks=list(tasks_data.keys()),
        )

        self._save_state()
        return {
            "model": model,
            "global_q": round(self.q_values[model], 4),
            "global_count": self.counts[model],
            "task_q_values": {t: round(self.task_q_values[t].get(model, 0.0), 4) for t in tasks_data},
        }

    def get_stats(self, task_type: Optional[str] = None) -> list:
        """Return model stats, optionally for a specific task arm."""
        if task_type and task_type in SUPPORTED_TASKS:
            target_q = self.task_q_values[task_type]
            target_counts = self.task_counts[task_type]
            target_latency = self.task_avg_latency[task_type]
            target_cost = self.task_avg_cost[task_type]
            target_eps = round(self.task_epsilon[task_type], 4)
            current_task = task_type
        else:
            target_q = self.q_values
            target_counts = self.counts
            target_latency = self.avg_latency
            target_cost = self.avg_cost
            target_eps = round(self.epsilon, 4)
            current_task = "all"

        all_models = sorted(list(set(list(MODEL_REGISTRY.keys()) + list(self.q_values.keys()))))

        return [
            {
                "model": m,
                "q_value": round(target_q.get(m, 0.0), 4),
                "selection_count": target_counts.get(m, 0),
                "avg_latency": round(target_latency.get(m, 0.0), 3),
                "avg_cost": round(target_cost.get(m, 0.0), 6),
                "task": current_task,
                "epsilon": target_eps,
            }
            for m in all_models
        ]

    def _save_state(self):
        """Persist learned per-task Q-tables and telemetry to disk."""
        if not self.state_file:
            return
        try:
            state = {
                "q_values": self.q_values,
                "counts": self.counts,
                "epsilon": self.epsilon,
                "task_q_values": self.task_q_values,
                "task_counts": self.task_counts,
                "task_epsilon": self.task_epsilon,
                "task_ts_alpha": self.task_ts_alpha,
                "task_ts_beta": self.task_ts_beta,
            }
            directory = os.path.dirname(self.state_file) or "."
            with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as tmp:
                json.dump(state, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, self.state_file)
        except Exception as e:
            logger.warning("Failed to save RL state", error=str(e))

    def _load_state(self):
        """Load persisted state if available."""
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            if "task_q_values" in state:
                self.task_q_values = state["task_q_values"]
            if "task_counts" in state:
                self.task_counts = state["task_counts"]
            if "task_epsilon" in state:
                self.task_epsilon = state["task_epsilon"]
            if "task_ts_alpha" in state:
                self.task_ts_alpha = state["task_ts_alpha"]
            if "task_ts_beta" in state:
                self.task_ts_beta = state["task_ts_beta"]
            if "q_values" in state:
                self.q_values = state["q_values"]
            if "counts" in state:
                self.counts = state["counts"]
            if "epsilon" in state:
                self.epsilon = state["epsilon"]
            logger.info("Loaded persisted RL state from disk", path=self.state_file)
        except Exception as e:
            logger.warning("Failed to load RL state from disk", error=str(e))

    def save_state(self):
        """Public alias to persist state."""
        self._save_state()

    def load_state(self):
        """Public alias to reload state."""
        self._load_state()

    def explain(self, user_id: str, prompt: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Explain routing decisions transparently: task class, candidates, Q-values, cost, reasoning."""
        from app.models.client import get_circuit_breaker

        # 1. Detailed classification
        classification = self.classifier.classify_with_details(prompt)
        task_type = classification["task_type"]
        matched_keywords = classification["matched_keywords"]
        confidence = classification["confidence"]

        # 2. Available models for user profile & task
        available_models = get_models_for_profile(profile, task_type)
        if not available_models:
            available_models = ["ollama/llama3"]

        # 3. Resolve task Q-values & exploration parameters
        task_q = self.task_q_values.get(task_type, self.q_values)
        task_eps = self.task_epsilon.get(task_type, self.epsilon) if self.per_task_split else self.epsilon

        # 4. Compute preference priors and rank available models by effective Q-value descending
        priors = {m: self._compute_preference_prior(m, profile, task_type) for m in available_models}
        effective_q = {m: task_q.get(m, 0.0) + priors[m] for m in available_models}
        sorted_models = sorted(available_models, key=lambda m: effective_q[m], reverse=True)
        top_models = sorted_models[:3]

        circuit_status = {m: get_circuit_breaker(m).state.value for m in available_models}
        top_candidates = []
        for m in top_models:
            cb_state = circuit_status[m]
            meta = MODEL_REGISTRY.get(m, {})
            cost_1k = meta.get("cost_per_1k_tokens", 0.0)
            avg_lat = self.task_avg_latency.get(task_type, {}).get(m, 0.0)
            top_candidates.append({
                "model": m,
                "q_value": round(task_q.get(m, 0.0), 4),
                "preference_prior": round(priors[m], 2),
                "avg_latency": round(avg_lat, 3),
                "cost_per_1k_tokens": cost_1k,
                "circuit_breaker_state": cb_state,
                "tier_allowed": True,
            })

        # 5. Determine complexity and selected model & mode
        complexity_info = self.complexity_scorer.score(prompt) if self.complexity_scorer else None
        is_simple = complexity_info["is_simple"] if complexity_info else False

        healthy_candidates = [
            m for m in sorted_models
            if get_circuit_breaker(m).state.value != "OPEN"
        ]

        if is_simple:
            selected_model = get_cheapest_capable_model(
                available_models,
                task_type=task_type,
                profile=profile,
                q_values=task_q,
            )
            selection_strategy = f"simple_bypass (cheapest capable: {selected_model})"
        else:
            selected_model = healthy_candidates[0] if healthy_candidates else sorted_models[0]
            selection_strategy = f"exploit (argmax Q in '{task_type}' arm)"

        # 6. Estimate token count and cost
        word_count = len(prompt.split())
        estimated_input_tokens = max(10, int(word_count * 1.33))
        estimated_total_tokens = estimated_input_tokens + 150
        estimated_cost = round((estimated_total_tokens / 1000) * MODEL_REGISTRY.get(selected_model, {}).get("cost_per_1k_tokens", 0.0), 6)

        # 7. Formulate structured selection reasoning
        user_tier = profile.get("tier", "free")
        winning_q = task_q.get(selected_model, 0.0)
        winning_prior = priors.get(selected_model, 0.0)

        reasoning_parts = [
            f"Prompt classified as '{task_type}' (confidence {int(confidence*100)}%) with matched signals: {matched_keywords or ['none']}."
        ]

        if is_simple and complexity_info:
            reasoning_parts.append(
                f"Prompt classified as simple (complexity={complexity_info['score']:.3f} < threshold {self.complexity_scorer.threshold}); automatically selected cheapest capable model '{selected_model}' to optimize cost."
            )
        elif complexity_info:
            reasoning_parts.append(
                f"Prompt classified as complex (complexity={complexity_info['score']:.3f} >= threshold {self.complexity_scorer.threshold}); routed through full RL multi-armed bandit selection."
            )

        if len(top_candidates) > 1:
            runner_up = top_candidates[1]["model"]
            runner_up_q = top_candidates[1]["q_value"]
            reasoning_parts.append(
                f"Model '{selected_model}' ranked #1 with Q-score {winning_q:.4f} (prior={winning_prior:+.2f}), leading '{runner_up}' (Q={runner_up_q:.4f}) in the '{task_type}' bandit arm."
            )
        else:
            reasoning_parts.append(
                f"Model '{selected_model}' selected as highest-performing arm with Q-score {winning_q:.4f} (prior={winning_prior:+.2f})."
            )

        prefer = profile.get("prefer")
        avoid = profile.get("avoid") or []
        if prefer:
            reasoning_parts.append(f"User preference prior prefer='{prefer}' applied.")
        if avoid:
            reasoning_parts.append(f"Avoid list {avoid} penalized.")

        if circuit_status.get(selected_model) == "OPEN":
            reasoning_parts.append(
                "Primary model circuit breaker is OPEN; request will automatically divert to fallback."
            )
        else:
            reasoning_parts.append(
                f"Circuit breaker status is healthy ({circuit_status.get(selected_model, 'CLOSED')})."
            )

        reasoning_parts.append(
            f"Authorized for user tier '{user_tier}' (exploration rate epsilon={task_eps:.4f})."
        )

        selection_reasoning = " ".join(reasoning_parts)

        return {
            "user_id": user_id,
            "user_tier": user_tier,
            "prompt": prompt,
            "predicted_task_class": task_type,
            "task_confidence": confidence,
            "matched_keywords": matched_keywords,
            "top_3_candidate_models": top_candidates,
            "selected_model": selected_model,
            "selection_strategy": selection_strategy,
            "estimated_cost": estimated_cost,
            "estimated_tokens": estimated_total_tokens,
            "selection_reasoning": selection_reasoning,
            "circuit_breaker_status": circuit_status,
            "complexity": complexity_info,
        }
