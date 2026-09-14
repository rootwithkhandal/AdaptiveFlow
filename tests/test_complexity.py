"""Unit and integration tests for Prompt Complexity Scorer.

Tests cover:
- Token count scoring across short vs long prompts
- Information entropy scoring (Shannon character entropy + Type-Token Ratio)
- Question depth scoring (reasoning keywords, multi-questions, code blocks, conditional clauses)
- Simple vs Complex prompt classification against threshold
- Cheapest capable model selection logic
- Circuit breaker awareness during cheapest model selection
- User avoid list awareness during cheapest model selection
- RL router simple prompt bypass (direct to cheapest model without RL exploration)
- RL router complex prompt handling (full RL multi-armed bandit selection)
- Routing explainability transparency (GET /route/explain returning complexity metadata)
- Dedicated complexity API endpoints (POST /prompt/complexity and GET /prompt/complexity)
- End-to-end routing with complexity optimization
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.complexity import PromptComplexityScorer, get_cheapest_capable_model
from app.dependencies import rl_router, profile_manager
from app.models.client import get_circuit_breaker, reset_all_circuit_breakers
from app.rl_router import RLRouter

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_circuit_breakers():
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


def test_token_count_scoring():
    """Short prompts score low in tokens, long prompts approach 1.0."""
    scorer = PromptComplexityScorer()

    short_res = scorer.compute_token_score("Hello there!")
    assert short_res["word_count"] == 2
    assert short_res["estimated_tokens"] <= 5
    assert short_res["score"] < 0.10

    long_prompt = "word " * 150
    long_res = scorer.compute_token_score(long_prompt)
    assert long_res["word_count"] == 150
    assert long_res["estimated_tokens"] > 120
    assert long_res["score"] == 1.0


def test_entropy_scoring():
    """Repetitive phrases have lower entropy than diverse technical content."""
    scorer = PromptComplexityScorer()

    repetitive = "echo echo echo echo echo echo echo echo"
    rep_res = scorer.compute_entropy_score(repetitive)

    diverse = "Quantum entanglement enables cryptographic key distribution across distributed fiber optic nodes"
    div_res = scorer.compute_entropy_score(diverse)

    assert div_res["score"] > rep_res["score"]
    assert div_res["type_token_ratio"] > rep_res["type_token_ratio"]


def test_question_depth_scoring_and_signals():
    """Detects analytical reasoning keywords, multi-part questions, and code blocks."""
    scorer = PromptComplexityScorer()

    # Simple prompt -> no depth signals
    simple_res = scorer.compute_question_depth_score("What is 2 + 2?")
    assert simple_res["score"] == 0.0
    assert len(simple_res["signals"]) == 0

    # Analytical prompt -> reasoning keywords
    analyt_res = scorer.compute_question_depth_score("Compare and contrast Paxos and Raft, explaining their trade-offs.")
    assert analyt_res["score"] >= 0.40
    assert any("reasoning:" in s for s in analyt_res["signals"])

    # Multi-question prompt
    multi_res = scorer.compute_question_depth_score("How does Redis persistence work? And what happens on replica failure?")
    assert any("multi_question" in s for s in multi_res["signals"])

    # Code block prompt
    code_res = scorer.compute_question_depth_score("Here is my script: ```def solve(): return True```. Fix the bug.")
    assert "code_fence" in code_res["signals"]
    assert "code_syntax" in code_res["signals"]
    assert code_res["score"] >= 0.50


def test_composite_complexity_simple_vs_complex():
    """Verifies that typical conversational/simple prompts are classified as simple, while technical tasks are complex."""
    scorer = PromptComplexityScorer(threshold=0.35)

    simple_prompts = [
        "What is the capital of France?",
        "Hello! Good morning.",
        "Translate 'thank you' to Spanish.",
        "What is 15 multiplied by 4?",
        "Tell me a short one-liner joke.",
    ]

    for p in simple_prompts:
        res = scorer.score(p)
        assert res["is_simple"] is True, f"Prompt '{p}' should be simple but scored {res['score']}"
        assert res["classification"] == "simple"
        assert res["score"] < 0.35

    complex_prompts = [
        "Compare and contrast optimistic vs pessimistic locking in high-concurrency databases, and explain their performance trade-offs.",
        "Write a Python function with asyncio and error handling to download files concurrently: ```def fetch(): pass```",
        "Why does my multithreaded application deadlock when lock A and lock B are acquired in reverse order? Explain step-by-step.",
        (
            "1. Explain the architectural difference between monolithic and microservice designs.\n"
            "2. When should a team migrate?\n"
            "3. What are the key latency implications?"
        ),
    ]

    for p in complex_prompts:
        res = scorer.score(p)
        assert res["is_simple"] is False, f"Prompt '{p}' should be complex but scored {res['score']}"
        assert res["classification"] == "complex"
        assert res["score"] >= 0.35


def test_get_cheapest_capable_model_selection():
    """Picks the model with lowest cost_per_1k_tokens among available candidates."""
    candidates = [
        "openrouter/anthropic/claude-3.5-sonnet",  # 0.003
        "openrouter/openai/gpt-4o-mini",          # 0.00015
        "gemini/gemini-1.5-flash",                # 0.00015
    ]

    cheapest = get_cheapest_capable_model(candidates)
    assert cheapest in ["openrouter/openai/gpt-4o-mini", "gemini/gemini-1.5-flash"]
    assert cheapest != "openrouter/anthropic/claude-3.5-sonnet"

    # When free local model is included
    candidates_with_ollama = ["ollama/llama3"] + candidates
    cheapest_free = get_cheapest_capable_model(candidates_with_ollama)
    assert cheapest_free == "ollama/llama3"


def test_cheapest_capable_model_circuit_breaker_resilience():
    """If cheapest model has OPEN circuit breaker, skips it and chooses next cheapest healthy model."""
    cb_mini = get_circuit_breaker("openrouter/openai/gpt-4o-mini")
    # Trip circuit breaker to OPEN
    cb_mini.record_failure()
    cb_mini.record_failure()
    cb_mini.record_failure()
    assert cb_mini.state.value == "OPEN"

    candidates = [
        "openrouter/openai/gpt-4o-mini",          # 0.00015 (OPEN)
        "gemini/gemini-1.5-flash",                # 0.00015 (CLOSED)
        "openrouter/anthropic/claude-3.5-sonnet",  # 0.003 (CLOSED)
    ]

    chosen = get_cheapest_capable_model(candidates)
    assert chosen == "gemini/gemini-1.5-flash"


def test_cheapest_capable_model_respects_user_avoid_list():
    """If user avoids gemini, get_cheapest_capable_model picks non-avoided cheap model."""
    candidates = [
        "gemini/gemini-1.5-flash",
        "openrouter/openai/gpt-4o-mini",
        "openrouter/anthropic/claude-3.5-sonnet",
    ]
    profile = {"user_id": "alice", "avoid": ["gemini"]}

    chosen = get_cheapest_capable_model(candidates, profile=profile)
    assert chosen == "openrouter/openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_rl_router_simple_prompt_bypasses_exploration():
    """RL router routes simple prompt to cheapest capable model and records bypass strategy."""
    router = RLRouter(state_file=None)
    profile = {"user_id": "u_simple", "tier": "pro"}
    simple_prompt = "What is 10 + 20?"

    with patch("app.rl_router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "response": "30",
            "tokens_used": 10,
            "cost": 0.0,
            "latency": 0.10,
        }
        res = await router.route("u_simple", simple_prompt, profile=profile)

        assert "simple_bypass" in res["routing_strategy"]
        assert "complexity" in res
        assert res["complexity"]["is_simple"] is True
        # Model should be cheapest capable model available for pro tier
        from app.models.registry import MODEL_REGISTRY
        cost_1k = MODEL_REGISTRY.get(res["model_used"], {}).get("cost_per_1k_tokens", 1.0)
        assert cost_1k <= 0.00015


@pytest.mark.asyncio
async def test_rl_router_complex_prompt_runs_full_rl():
    """RL router routes complex prompt through full bandit selection."""
    router = RLRouter(state_file=None)
    profile = {"user_id": "u_complex", "tier": "pro"}
    complex_prompt = (
        "Compare and contrast Paxos and Raft distributed consensus protocols. "
        "Detail the leader election mechanics and explain their trade-offs."
    )

    with patch("app.rl_router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "response": "Detailed consensus comparison...",
            "tokens_used": 150,
            "cost": 0.0005,
            "latency": 0.85,
        }
        res = await router.route("u_complex", complex_prompt, profile=profile)

        assert "simple_bypass" not in res["routing_strategy"]
        assert "complexity" in res
        assert res["complexity"]["is_simple"] is False


def test_explain_endpoint_displays_complexity_analysis():
    """GET /route/explain returns complexity breakdown and reflects classification in selection reasoning."""
    user_id = "explain_test_user"
    profile_manager.set_tier(user_id, "pro")

    # 1. Simple prompt explain
    r_simple = client.get("/route/explain", params={"user_id": user_id, "prompt": "What is 2 + 2?"})
    assert r_simple.status_code == 200
    data_simple = r_simple.json()

    assert "complexity" in data_simple
    assert data_simple["complexity"]["is_simple"] is True
    assert "simple" in data_simple["selection_strategy"].lower()
    assert "simple" in data_simple["selection_reasoning"].lower()

    # 2. Complex prompt explain
    comp_prompt = "Explain why my distributed database deadlocks during multi-region replication and propose an architectural fix."
    r_comp = client.get("/route/explain", params={"user_id": user_id, "prompt": comp_prompt})
    assert r_comp.status_code == 200
    data_comp = r_comp.json()

    assert "complexity" in data_comp
    assert data_comp["complexity"]["is_simple"] is False
    assert "complex" in data_comp["selection_reasoning"].lower()


def test_prompt_complexity_api_endpoints():
    """POST and GET /prompt/complexity endpoints return structured complexity evaluation."""
    # POST
    r_post = client.post(
        "/prompt/complexity",
        json={"prompt": "Write a python script using asyncio to scrape web pages", "user_id": "test_user"},
    )
    assert r_post.status_code == 200
    res_post = r_post.json()
    assert "score" in res_post
    assert "is_simple" in res_post
    assert "token_count" in res_post
    assert "entropy" in res_post
    assert "question_depth" in res_post
    assert "signals" in res_post
    assert res_post["cheapest_model"] is not None

    # GET
    r_get = client.get(
        "/prompt/complexity",
        params={"prompt": "Hello world!"},
    )
    assert r_get.status_code == 200
    res_get = r_get.json()
    assert res_get["is_simple"] is True
    assert res_get["classification"] == "simple"
