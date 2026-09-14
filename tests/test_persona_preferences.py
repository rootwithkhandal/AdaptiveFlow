"""Unit and integration tests for Model Persona / Specialization Profiles.

Tests cover:
- User profile preference storage (prefer & avoid) in UserProfileManager
- Preference prior computation (_compute_preference_prior) for specializations and model families
- Avoid list severe penalty (-5.0) and case-insensitive substring matching
- RL arm selection with user prior as Bayesian guide (effective_q = Q + prior)
- Preservation of pure empirical Q-values (Q-table is never mutated by preference prior)
- Graceful fallback resilience when all models match avoid patterns
- Thompson Sampling integration adjusting Beta distribution
- Routing Explainability transparency (GET /route/explain returning preference_prior & reasoning)
- User preferences API endpoint (POST /profile/{user_id}/preferences)
- Admin user preferences endpoint (POST /admin/users/{user_id}/preferences with X-Admin-Key)
- End-to-end routing with persona prior
"""
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.main import app
from app.dependencies import profile_manager, rl_router
from app.rl_router import RLRouter
from app.profiles import UserProfileManager
from app.config import settings

client = TestClient(app)


def test_user_profile_preferences_defaults_and_mutation(tmp_path):
    """UserProfileManager provides clean defaults and allows setting routing hints."""
    profiles_file = str(tmp_path / "profiles_test.json")
    with patch("app.profiles.PROFILES_FILE", profiles_file):
        mgr = UserProfileManager()
        prof = mgr.get_or_create("user_fresh")
        assert prof["prefer"] is None
        assert prof["avoid"] == []

        # Update preferences
        mgr.set_preferences("user_fresh", prefer="code_heavy", avoid=["gemini", "ollama"])
        updated = mgr.get_or_create("user_fresh")
        assert updated["prefer"] == "code_heavy"
        assert updated["avoid"] == ["gemini", "ollama"]

        # Partial update (only prefer)
        mgr.set_preferences("user_fresh", prefer="fast")
        assert mgr.get_or_create("user_fresh")["prefer"] == "fast"
        assert mgr.get_or_create("user_fresh")["avoid"] == ["gemini", "ollama"]


def test_compute_preference_prior_avoid_penalty():
    """Avoid list causes -5.0 penalty for matching models (case-insensitive substring)."""
    router = RLRouter(state_file=None)
    profile = {
        "user_id": "test_avoid_user",
        "prefer": None,
        "avoid": ["gemini", "llama"],
    }

    # gemini models should get -5.0 penalty
    prior_gemini = router._compute_preference_prior("gemini/gemini-1.5-flash", profile, "general")
    assert prior_gemini == -5.0

    # llama models should get -5.0 penalty
    prior_llama = router._compute_preference_prior("ollama/llama3", profile, "code")
    assert prior_llama == -5.0

    # openai model should have 0.0 prior (unaffected)
    prior_openai = router._compute_preference_prior("openrouter/openai/gpt-4o-mini", profile, "general")
    assert prior_openai == 0.0


def test_compute_preference_prior_specialization_boosts():
    """Specialization keywords ('code_heavy', 'fast', 'creative', 'security_first') boost matching arms."""
    router = RLRouter(state_file=None)

    # Code heavy preference
    prof_code = {"user_id": "coder", "prefer": "code_heavy", "avoid": []}
    prior_qwen = router._compute_preference_prior("openrouter/qwen/qwen-2.5-72b-instruct", prof_code, "code")
    assert prior_qwen >= 0.75

    # Fast / low_latency preference
    prof_fast = {"user_id": "speeder", "prefer": "fast", "avoid": []}
    prior_flash = router._compute_preference_prior("gemini/gemini-1.5-flash", prof_fast, "general")
    assert prior_flash >= 0.75

    # Direct model family match (e.g. 'claude')
    prof_claude = {"user_id": "claude_fan", "prefer": "claude", "avoid": []}
    prior_claude = router._compute_preference_prior("openrouter/anthropic/claude-3.5-sonnet", prof_claude, "creative")
    assert prior_claude == 1.0


def test_selection_steered_by_prior_without_polluting_q_values():
    """RL selection uses effective Q (Q + prior), leaving the underlying Q-table pure."""
    router = RLRouter(state_file=None)
    router.epsilon = 0.0  # Force pure exploitation (deterministic argmax)
    router.task_epsilon["code"] = 0.0

    available = ["openrouter/openai/gpt-4o-mini", "gemini/gemini-1.5-flash"]

    # Seed model Q-values: gpt-4o-mini = 1.0, gemini-flash = 1.2
    router.task_q_values["code"]["openrouter/openai/gpt-4o-mini"] = 1.0
    router.task_q_values["code"]["gemini/gemini-1.5-flash"] = 1.2

    # Without preferences, gemini (1.2) wins over gpt-4o-mini (1.0)
    neutral_profile = {"user_id": "u1", "tier": "pro", "prefer": None, "avoid": []}
    choice_neutral = router._select_model_epsilon_greedy(available, task_type="code", profile=neutral_profile)
    assert choice_neutral == "gemini/gemini-1.5-flash"

    # With avoid=['gemini'], gemini effective_q = 1.2 - 5.0 = -3.8 -> gpt-4o-mini (1.0) wins
    avoid_profile = {"user_id": "u2", "tier": "pro", "prefer": None, "avoid": ["gemini"]}
    choice_avoid = router._select_model_epsilon_greedy(available, task_type="code", profile=avoid_profile)
    assert choice_avoid == "openrouter/openai/gpt-4o-mini"

    # Verify underlying Q-table was NEVER mutated or corrupted by the preference prior
    assert router.task_q_values["code"]["gemini/gemini-1.5-flash"] == 1.2
    assert router.task_q_values["code"]["openrouter/openai/gpt-4o-mini"] == 1.0


def test_graceful_fallback_when_all_models_avoided():
    """If user avoids all models, router does not crash and still selects a model."""
    router = RLRouter(state_file=None)
    router.epsilon = 0.0
    router.task_epsilon["general"] = 0.0

    available = ["openrouter/openai/gpt-4o-mini", "gemini/gemini-1.5-flash"]
    # User avoids both
    profile = {"user_id": "u_all_avoid", "prefer": None, "avoid": ["openai", "gemini"]}

    chosen = router._select_model_epsilon_greedy(available, task_type="general", profile=profile)
    assert chosen in available


def test_thompson_sampling_with_preference_prior():
    """Thompson sampling biases Beta distribution using preference prior."""
    router = RLRouter(state_file=None, strategy="thompson_sampling")
    available = ["openrouter/openai/gpt-4o-mini", "gemini/gemini-1.5-flash"]

    profile = {"user_id": "u_ts", "prefer": None, "avoid": ["gemini"]}
    # Run multiple selections - gpt-4o-mini should overwhelmingly win due to avoid penalty on gemini
    counts = {m: 0 for m in available}
    for _ in range(30):
        c = router._select_model_thompson(available, task_type="general", profile=profile)
        counts[c] += 1

    assert counts["openrouter/openai/gpt-4o-mini"] > counts["gemini/gemini-1.5-flash"]


def test_explain_endpoint_displays_preference_prior_and_reasoning():
    """GET /route/explain reflects user persona preferences and ranks by effective Q."""
    user_id = "persona_user_alice"
    profile_manager.set_tier(user_id, "pro")
    profile_manager.set_preferences(user_id, prefer="code_heavy", avoid=["gemini"])

    prompt = "Write a python function to implement binary search"
    r = client.get("/route/explain", params={"user_id": user_id, "prompt": prompt})
    assert r.status_code == 200
    data = r.json()

    # Top candidates must show preference_prior
    candidates = data["top_3_candidate_models"]
    assert len(candidates) > 0
    for c in candidates:
        assert "preference_prior" in c
        if "gemini" in c["model"].lower():
            assert c["preference_prior"] == -5.0

    # Reasoning must mention user preference hints
    reasoning = data["selection_reasoning"]
    assert "prefer='code_heavy'" in reasoning or "code_heavy" in reasoning
    assert "gemini" in reasoning


def test_user_preferences_api_endpoints():
    """POST /profile/{user_id}/preferences updates routing hints and GET /profile reflects them."""
    user_id = "test_pref_api_user"

    # 1. Update preferences via user route
    r = client.post(
        f"/profile/{user_id}/preferences",
        json={"prefer": "cost_saving", "avoid": ["claude"]},
    )
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "updated"
    assert res["user_id"] == user_id
    assert res["prefer"] == "cost_saving"
    assert res["avoid"] == ["claude"]

    # 2. Verify GET /profile/{user_id}
    r_get = client.get(f"/profile/{user_id}")
    assert r_get.status_code == 200
    prof = r_get.json()
    assert prof["prefer"] == "cost_saving"
    assert prof["avoid"] == ["claude"]


def test_admin_user_preferences_endpoint():
    """POST /admin/users/{user_id}/preferences requires X-Admin-Key and updates preferences."""
    user_id = "admin_managed_user"

    # Forbidden without key
    r_bad = client.post(
        f"/admin/users/{user_id}/preferences",
        json={"prefer": "security_first", "avoid": ["ollama"]},
    )
    assert r_bad.status_code == 422 or r_bad.status_code == 403

    # Forbidden with bad key
    r_wrong = client.post(
        f"/admin/users/{user_id}/preferences",
        headers={"x-admin-key": "wrong_key"},
        json={"prefer": "security_first", "avoid": ["ollama"]},
    )
    assert r_wrong.status_code == 403

    # Success with valid key
    r_ok = client.post(
        f"/admin/users/{user_id}/preferences",
        headers={"x-admin-key": settings.secret_key},
        json={"prefer": "security_first", "avoid": ["ollama"]},
    )
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["status"] == "updated"
    assert data["prefer"] == "security_first"
    assert data["avoid"] == ["ollama"]

    # Confirm persistence
    prof = profile_manager.get_or_create(user_id)
    assert prof["prefer"] == "security_first"
    assert prof["avoid"] == ["ollama"]


@pytest.mark.asyncio
async def test_end_to_end_route_respects_user_preferences():
    """POST /route uses preference prior when selecting model."""
    user_id = "e2e_pref_user"
    profile_manager.set_tier(user_id, "pro")
    profile_manager.set_preferences(user_id, prefer=None, avoid=["gemini"])

    prompt = "Write an essay about artificial intelligence"

    with patch("app.rl_router.call_model", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {
            "response": "Essay on AI...",
            "tokens_used": 120,
            "cost": 0.0001,
            "latency": 0.35,
        }
        r = client.post("/route", json={"user_id": user_id, "prompt": prompt})
        assert r.status_code == 200
        data = r.json()
        # Avoided gemini must not have been selected
        assert "gemini" not in data["model_used"].lower()
