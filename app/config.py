"""Central configuration loaded from environment variables."""
import secrets
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_site_url: str = ""   # optional, shown on openrouter.ai rankings
    openrouter_site_name: str = "LLM Router"  # optional

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # App
    app_env: Literal["development", "production"] = "development"
    log_level: str = "INFO"
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    allowed_origins: list[str] = ["http://localhost:8000"]
    feedback_token_ttl_seconds: int = 900
    max_prompt_length: int = 20_000
    max_local_cache_entries: int = 10_000
    max_vector_memory_entries_per_user: int = 1_000
    max_implicit_feedback_users: int = 10_000

    # RL Router
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.01

    # NVIDIA NIM
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Vector Memory
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_context: int = 3
    memory_backend: Literal["faiss", "chroma"] = "faiss"

    # Circuit Breaker
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_recovery_timeout: float = 30.0

    # Rate Limiting
    rate_limit_free: int = 10          # 10 req/min
    rate_limit_pro: int = 60           # 60 req/min
    rate_limit_enterprise: int = 0     # 0 = unlimited
    rate_limit_window_seconds: float = 60.0

    # Prompt Complexity Scorer
    complexity_threshold: float = 0.35
    complexity_weight_tokens: float = 0.25
    complexity_weight_entropy: float = 0.35
    complexity_weight_depth: float = 0.40

    model_config = {"env_file": ".env", "extra": "ignore"}

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, value: str, info):
        env = info.data.get("app_env", "development")
        weak = {"", "change-me", "admin-secret-key", "change-me-in-production"}
        if env == "production" and (value in weak or len(value) < 32):
            raise ValueError("SECRET_KEY must be a unique value of at least 32 characters in production")
        return value


settings = Settings()
