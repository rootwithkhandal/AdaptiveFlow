"""Pydantic schemas for API request/response."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class RouteRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    prompt: str = Field(..., description="User prompt to route")


class RouteResponse(BaseModel):
    response: str
    model_used: str
    task_type: str
    cost: float
    latency: float
    cached: bool = False
    cache_type: Optional[str] = None
    similarity: Optional[float] = None
    context_injected: bool = False
    complexity: Optional[Dict[str, Any]] = None


class ParallelRouteRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    prompt: str = Field(..., description="User prompt to route")
    models: Optional[List[str]] = Field(None, description="Candidate models to dispatch concurrently")
    strategy: str = Field("fastest", description="Selection strategy: 'fastest', 'highest_q', or 'all'")


class ParallelRouteResponse(BaseModel):
    winner: str
    strategy: str
    response: str
    cost: float
    latency: float
    candidates: List[str]
    results: Optional[Dict[str, Any]] = None


class SynthesizeRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier (Must be Pro or Enterprise tier)")
    prompt: str = Field(..., description="User prompt to synthesize")
    candidate_models: Optional[List[str]] = Field(None, description="2-3 candidate models to evaluate concurrently")
    judge_model: Optional[str] = Field(None, description="Lightweight judge model (defaults to Claude 3 Haiku / Gemini Flash)")


class SynthesizeResponse(BaseModel):
    winner: str
    winner_response: str
    judge_model: str
    judge_rationale: str
    scores: Dict[str, Dict[str, Any]]
    candidates: List[str]
    results: Dict[str, Any]
    total_cost: float
    latency: float


class FeedbackRequest(BaseModel):
    user_id: str
    prompt: str
    model_used: str
    rating: int = Field(..., ge=-1, le=1, description="-1=bad, 0=neutral, 1=good")
    task_type: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class UserProfileResponse(BaseModel):
    user_id: str
    tier: str
    total_cost: float
    request_count: int
    budget_remaining: Optional[float]
    prefer: Optional[str] = None
    avoid: List[str] = []


class UserPreferencesRequest(BaseModel):
    prefer: Optional[str] = Field(default=None, description="Preferred specialization (e.g. 'code_heavy', 'fast') or model family ('claude', 'gpt')")
    avoid: Optional[List[str]] = Field(default=None, description="List of model patterns/substrings to avoid (e.g. ['gemini'])")


class UserPreferencesResponse(BaseModel):
    status: str
    user_id: str
    prefer: Optional[str] = None
    avoid: List[str] = []


class AbandonRequest(BaseModel):
    user_id: str = Field(..., description="User ID associated with the abandoned turn")
    reason: Optional[str] = Field("client_abort", description="Reason: client_abort, timeout, window_close")


class AbandonResponse(BaseModel):
    status: str
    signal_applied: bool
    details: Optional[Dict[str, Any]] = None


class ModelRegistrationRequest(BaseModel):
    model_name: str = Field(..., description="Unique model identifier, e.g. openrouter/qwen/qwen-2.5-72b-instruct")
    provider: str = Field(..., description="Provider type: openrouter, nvidia, litellm, ollama")
    cost_per_1k_tokens: float = Field(..., ge=0.0, description="Cost per 1k tokens in USD")
    tasks: List[str] = Field(default=["general"], description="Supported task types: code, security, creative, general")
    tiers: Optional[List[str]] = Field(default=None, description="Allowed user tiers: free, pro, enterprise")
    fallback: Optional[str] = Field(default=None, description="Fallback model identifier on failure")
    auto_warmup: bool = Field(default=True, description="Run curated synthetic benchmark suite and seed Q-table immediately")


class ModelRegistrationResponse(BaseModel):
    status: str
    model: str
    warmed_up: bool
    seeded_weights: Optional[Dict[str, Any]] = None
    warmup_scorecard: Optional[Dict[str, Any]] = None


class ModelWarmupRequest(BaseModel):
    model_name: str = Field(..., description="Registered model identifier to warm up")
    task_types: Optional[List[str]] = Field(default=None, description="Optional subset of task types to benchmark")


class ModelWarmupResponse(BaseModel):
    status: str
    model: str
    seeded_weights: Dict[str, Any]
    warmup_scorecard: Dict[str, Any]


class CandidateExplanation(BaseModel):
    model: str
    q_value: float
    preference_prior: float = 0.0
    avg_latency: float
    cost_per_1k_tokens: float
    circuit_breaker_state: str
    tier_allowed: bool


class ExplainResponse(BaseModel):
    user_id: str
    user_tier: str
    prompt: str
    predicted_task_class: str
    task_confidence: float
    matched_keywords: List[str]
    top_3_candidate_models: List[CandidateExplanation]
    selected_model: str
    selection_strategy: str
    estimated_cost: float
    estimated_tokens: int
    selection_reasoning: str
    cache_prediction: Dict[str, Any]
    circuit_breaker_status: Dict[str, str]
    complexity: Optional[Dict[str, Any]] = None


class ComplexityScoreRequest(BaseModel):
    prompt: str = Field(..., description="Prompt text to evaluate complexity for")
    user_id: Optional[str] = Field(default="anonymous", description="Optional user identifier")


class ComplexityScoreResponse(BaseModel):
    prompt: str
    score: float
    is_simple: bool
    classification: str
    threshold: float
    token_count: int
    entropy: float
    question_depth: float
    signals: List[str]
    cheapest_model: Optional[str] = None

