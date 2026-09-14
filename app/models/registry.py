"""Model registry: defines available models, their cost per token,
and which task types / user tiers they serve.
"""
from typing import Dict, Any, List, Optional

# Model definitions
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- OpenAI (via LiteLLM) ---
    "gpt-4o": {
        "provider": "litellm",
        "cost_per_1k_tokens": 0.005,
        "tiers": ["enterprise"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": "openrouter/openai/gpt-4o",
    },
    "gpt-4o-mini": {
        "provider": "litellm",
        "cost_per_1k_tokens": 0.00015,
        "tiers": ["pro", "enterprise"],
        "tasks": ["code", "creative", "general"],
        "fallback": "openrouter/openai/gpt-4o-mini",
    },
    # --- Gemini (via LiteLLM) ---
    "gemini/gemini-1.5-flash": {
        "provider": "litellm",
        "cost_per_1k_tokens": 0.000075,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["creative", "general"],
        "fallback": "openrouter/google/gemini-flash-1.5",
    },
    # --- OpenRouter models ---
    "openrouter/openai/gpt-4o": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.005,
        "tiers": ["enterprise"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": "openrouter/openai/gpt-4o-mini",
    },
    "openrouter/openai/gpt-4o-mini": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.00015,
        "tiers": ["pro", "enterprise"],
        "tasks": ["code", "creative", "general"],
        "fallback": "ollama/llama3",
    },
    "openrouter/anthropic/claude-3.5-sonnet": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.003,
        "tiers": ["enterprise"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": "openrouter/openai/gpt-4o-mini",
    },
    "openrouter/anthropic/claude-3-haiku": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.00025,
        "tiers": ["pro", "enterprise"],
        "tasks": ["code", "creative", "general"],
        "fallback": "ollama/llama3",
    },
    "openrouter/google/gemini-flash-1.5": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.000075,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["creative", "general"],
        "fallback": "ollama/llama3",
    },
    "openrouter/meta-llama/llama-3.1-8b-instruct:free": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["code", "creative", "general"],
        "fallback": "ollama/llama3",
    },
    "openrouter/mistralai/mistral-7b-instruct:free": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["code", "general"],
        "fallback": "ollama/mistral",
    },
    "openrouter/deepseek/deepseek-r1:free": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["code", "general"],
        "fallback": "ollama/llama3",
    },
    # --- NVIDIA models ---
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["code", "creative", "general"],
        "fallback": "openrouter/meta-llama/llama-3.1-8b-instruct:free",
    },
    "openrouter/nvidia/llama-3.1-nemotron-70b-instruct": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0007,
        "tiers": ["pro", "enterprise"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
    },
    "openrouter/nvidia/nemotron-mini-4b-instruct": {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0001,
        "tiers": ["free", "pro", "enterprise"],
        "tasks": ["code", "general"],
        "fallback": "ollama/llama3",
    },
    "nvidia/llama-3.1-nemotron-70b-instruct": {
        "provider": "nvidia",
        "cost_per_1k_tokens": 0.0007,
        "tiers": ["enterprise"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": "openrouter/nvidia/llama-3.1-nemotron-70b-instruct",
    },
    # --- Ollama (local) ---
    "ollama/llama3": {
        "provider": "ollama",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise", "high_risk"],
        "tasks": ["code", "security", "creative", "general"],
        "fallback": None,
    },
    "ollama/mistral": {
        "provider": "ollama",
        "cost_per_1k_tokens": 0.0,
        "tiers": ["free", "pro", "enterprise", "high_risk"],
        "tasks": ["code", "general"],
        "fallback": None,
    },
}

# Tier routing rules
TIER_RULES = {
    "free": {
        "allowed_models": [
            "gemini/gemini-1.5-flash",
            "openrouter/google/gemini-flash-1.5",
            "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            "openrouter/mistralai/mistral-7b-instruct:free",
            "openrouter/deepseek/deepseek-r1:free",
            "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
            "openrouter/nvidia/nemotron-mini-4b-instruct",
            "ollama/llama3",
            "ollama/mistral",
        ]
    },
    "pro": {
        "allowed_models": [
            "gpt-4o-mini",
            "gemini/gemini-1.5-flash",
            "openrouter/openai/gpt-4o-mini",
            "openrouter/anthropic/claude-3-haiku",
            "openrouter/google/gemini-flash-1.5",
            "openrouter/meta-llama/llama-3.1-8b-instruct:free",
            "openrouter/mistralai/mistral-7b-instruct:free",
            "openrouter/nvidia/llama-3.1-nemotron-70b-instruct",
            "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free",
            "openrouter/nvidia/nemotron-mini-4b-instruct",
            "ollama/llama3",
            "ollama/mistral",
        ]
    },
    "enterprise": {"allowed_models": list(MODEL_REGISTRY.keys())},
    "high_risk": {"allowed_models": ["ollama/llama3", "ollama/mistral"]},  # local only
}


def get_models_for_profile(profile: Dict, task_type: str) -> List[str]:
    """Return models available for a given user profile and task type."""
    tier = profile.get("tier", "free")

    # High-risk users always go local
    if profile.get("high_risk", False):
        tier = "high_risk"

    allowed = TIER_RULES.get(tier, TIER_RULES["free"])["allowed_models"]

    # Filter by task compatibility
    candidates = [
        m for m in allowed
        if task_type in MODEL_REGISTRY[m]["tasks"]
    ]

    return candidates if candidates else allowed


def register_model(
    model_name: str,
    config: Dict[str, Any],
    tiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Dynamically register a new model in the MODEL_REGISTRY and update TIER_RULES."""
    if not model_name:
        raise ValueError("Model name must not be empty.")

    required_keys = ["provider", "cost_per_1k_tokens", "tasks"]
    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing required model config key: '{k}'")

    if tiers is None:
        tiers = config.get("tiers", ["pro", "enterprise"])

    config["tiers"] = tiers
    if "fallback" not in config:
        config["fallback"] = None

    MODEL_REGISTRY[model_name] = config

    # Update tier rules
    for tier in tiers:
        if tier in TIER_RULES:
            if model_name not in TIER_RULES[tier]["allowed_models"]:
                TIER_RULES[tier]["allowed_models"].append(model_name)

    # Always ensure enterprise tier has access
    if "enterprise" in TIER_RULES and model_name not in TIER_RULES["enterprise"]["allowed_models"]:
        TIER_RULES["enterprise"]["allowed_models"].append(model_name)

    return {"model": model_name, "config": config}
