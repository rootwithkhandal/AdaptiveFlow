"""Unit tests for Synthetic Benchmarks and Cold-Start Warm-Up Engine."""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.benchmark import (
    BENCHMARK_SUITE,
    evaluate_benchmark_response,
    compute_benchmark_reward,
    run_model_warmup,
    SUPPORTED_TASKS,
)
from app.rl_router import RLRouter
from app.models.registry import register_model, MODEL_REGISTRY, TIER_RULES
from app.config import settings
from app.main import app

client = TestClient(app)


def test_benchmark_suite_size_and_structure():
    """Verify curated benchmark suite has 10-15 prompts per task type across all 4 tasks."""
    assert set(BENCHMARK_SUITE.keys()) == {"code", "security", "creative", "general"}
    for task in SUPPORTED_TASKS:
        prompts = BENCHMARK_SUITE[task]
        assert 10 <= len(prompts) <= 15, f"Task {task} has {len(prompts)} prompts, expected 10-15"
        for p in prompts:
            assert "id" in p
            assert "prompt" in p and len(p["prompt"]) > 10
            assert "keywords" in p and len(p["keywords"]) > 0


def test_evaluate_benchmark_response():
    prompt_meta = {"keywords": ["quicksort", "pivot", "partition"]}
    
    # 1. Empty response
    assert evaluate_benchmark_response("code", prompt_meta, "", 0.5) == 0.0
    assert evaluate_benchmark_response("code", prompt_meta, None, 0.5) == 0.0

    # 2. Too brief / refusal
    assert evaluate_benchmark_response("code", prompt_meta, "I cannot answer.", 0.5) == 0.2

    # 3. Coherent response matching keywords
    resp = "Here is an in-place quicksort algorithm using a pivot to partition the list."
    score = evaluate_benchmark_response("code", prompt_meta, resp, 0.5)
    assert score >= 0.90

    # 4. Latency penalty (> 20s)
    slow_score = evaluate_benchmark_response("code", prompt_meta, resp, 25.0)
    assert slow_score < score


def test_compute_benchmark_reward():
    # Reward = (acc * 2) - (lat * 0.1) - (cost * 5)
    # acc=1.0, lat=1.0, cost=0.01 -> 2.0 - 0.1 - 0.05 = 1.85
    rew = compute_benchmark_reward(accuracy=1.0, latency=1.0, cost=0.01)
    assert abs(rew - 1.85) < 1e-4


@pytest.mark.asyncio
async def test_run_model_warmup_execution():
    mock_resp = {
        "response": "Here is an implementation with def, pivot, and partition for quicksort.",
        "latency": 0.25,
        "cost": 0.0001,
        "model": "test/mock-model",
    }
    with patch("app.benchmark.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        warmup_card = await run_model_warmup("test/mock-model", task_types=["code", "security"])

        assert warmup_card["model"] == "test/mock-model"
        assert "code" in warmup_card["tasks"]
        assert "security" in warmup_card["tasks"]
        assert warmup_card["tasks"]["code"]["count"] == 12
        assert warmup_card["tasks"]["code"]["avg_reward"] > 1.5
        assert warmup_card["tasks"]["code"]["successes"] == 12
        assert warmup_card["overall"]["total_prompts"] == 24


def test_rl_router_seed_model_weights():
    router = RLRouter(per_task_split=True, state_file=None)
    new_model = "test/cold-start-model"

    # Prior to seeding, model is not known or at default
    assert new_model not in router.q_values

    mock_warmup_data = {
        "tasks": {
            "code": {
                "avg_reward": 1.92,
                "count": 12,
                "avg_latency": 0.35,
                "avg_cost": 0.0002,
                "successes": 12,
                "failures": 0,
            },
            "security": {
                "avg_reward": 1.88,
                "count": 12,
                "avg_latency": 0.40,
                "avg_cost": 0.0002,
                "successes": 12,
                "failures": 0,
            },
        },
        "overall": {
            "avg_reward": 1.90,
            "total_prompts": 24,
            "avg_latency": 0.375,
            "avg_cost": 0.0002,
        },
    }

    res = router.seed_model_weights(new_model, mock_warmup_data)
    assert res["model"] == new_model
    assert res["global_q"] == 1.9
    assert res["global_count"] == 24
    assert res["task_q_values"]["code"] == 1.92

    # Verify RL Router internal state
    assert router.task_q_values["code"][new_model] == 1.92
    assert router.task_counts["code"][new_model] == 12
    assert router.task_ts_alpha["code"][new_model] == 13.0  # 1.0 baseline + 12 successes
    assert router.task_ts_beta["code"][new_model] == 1.0

    # In exploit step with an unseeded model, the seeded model is chosen immediately!
    competitor = "unseeded/model"
    router._ensure_model(competitor)
    router.task_epsilon["code"] = 0.0  # Force 100% exploitation
    chosen = router._select_model_epsilon_greedy([new_model, competitor], task_type="code")
    assert chosen == new_model, "Seeded model should be chosen over unseeded model in exploit step"


def test_dynamic_model_registration():
    model_name = "test/dynamically-added-model"
    config = {
        "provider": "openrouter",
        "cost_per_1k_tokens": 0.0005,
        "tasks": ["code", "creative"],
    }
    res = register_model(model_name, config, tiers=["pro"])
    assert res["model"] == model_name
    assert model_name in MODEL_REGISTRY
    assert model_name in TIER_RULES["pro"]["allowed_models"]
    assert model_name in TIER_RULES["enterprise"]["allowed_models"]


def test_admin_register_and_warmup_endpoint():
    new_model = "openrouter/qwen/qwen-2.5-72b-instruct"
    mock_resp = {
        "response": "Here is a complete solution with def and keywords.",
        "latency": 0.3,
        "cost": 0.0004,
        "model": new_model,
    }

    with patch("app.benchmark.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp

        payload = {
            "model_name": new_model,
            "provider": "openrouter",
            "cost_per_1k_tokens": 0.0004,
            "tasks": ["code", "general"],
            "tiers": ["pro", "enterprise"],
            "auto_warmup": True,
        }

        # 1. Without admin key -> 403
        r_unauth = client.post("/admin/models/register", json=payload)
        assert r_unauth.status_code == 422 or r_unauth.status_code == 403

        # 2. With valid admin key -> 200
        headers = {"X-Admin-Key": settings.secret_key}
        r = client.post("/admin/models/register", json=payload, headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "registered"
        assert data["model"] == new_model
        assert data["warmed_up"] is True
        assert data["seeded_weights"] is not None
        assert data["warmup_scorecard"]["overall"]["total_prompts"] == 24

        # 3. Model is now in stats
        r_stats = client.get("/models/stats")
        assert r_stats.status_code == 200
        models_in_stats = [m["model"] for m in r_stats.json()]
        assert new_model in models_in_stats


def test_admin_warmup_existing_endpoint():
    headers = {"X-Admin-Key": settings.secret_key}
    mock_resp = {
        "response": "Valid response with required keywords.",
        "latency": 0.2,
        "cost": 0.0,
        "model": "ollama/llama3",
    }
    with patch("app.benchmark.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_resp
        r = client.post("/admin/models/warmup", json={
            "model_name": "ollama/llama3",
            "task_types": ["code"],
        }, headers=headers)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "warmed_up"
        assert data["model"] == "ollama/llama3"
        assert data["warmup_scorecard"]["tasks"]["code"]["count"] == 12
