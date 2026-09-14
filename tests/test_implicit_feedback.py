"""Unit tests for Implicit Feedback Signals and passive reward updates."""
import pytest
import time
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.implicit_feedback import ImplicitFeedbackTracker
from app.rl_router import RLRouter
from app.main import app


def test_rephrase_penalty_detection():
    tracker = ImplicitFeedbackTracker(rephrase_threshold=0.80)
    router = RLRouter()
    model = "openrouter/openai/gpt-4o-mini"
    initial_q = router.q_values[model]

    # Turn 1
    tracker.record_turn(
        user_id="user_test",
        prompt="Write a Python function to sort a list using quicksort",
        model=model,
        task="code",
        latency=1.2,
    )

    # Turn 2: rephrased query right after
    rephrased_prompt = "Write a quicksort function in Python"

    # Mock cosine similarity to be 0.88 (> 0.80)
    with patch.object(tracker, "_cosine_similarity", return_value=0.88):
        signal = tracker.evaluate_followup("user_test", rephrased_prompt)
        assert signal is not None
        assert signal["signal_type"] == "rephrase_penalty"
        assert signal["reward_delta"] == -0.8
        assert signal["model"] == model

        # Apply signal to RL router
        router.apply_implicit_signal(
            model=signal["model"],
            signal_type=signal["signal_type"],
            reward_delta=signal["reward_delta"],
            task_type=signal["task"],
        )
        assert router.q_values[model] < initial_q
        assert router.ts_beta[model] > 1.0


def test_engagement_reward_detection():
    tracker = ImplicitFeedbackTracker()
    router = RLRouter()
    model = "gemini/gemini-1.5-flash"
    initial_q = router.q_values[model]

    tracker.record_turn(
        user_id="user_test",
        prompt="Explain photosynthesis in simple terms",
        model=model,
        task="general",
        latency=0.8,
    )

    # Simulate 45 seconds natural reading dwell time
    tracker._user_turns["user_test"]["timestamp"] = time.time() - 45.0

    continuation_prompt = "How does cellular respiration differ from it?"

    with patch.object(tracker, "_cosine_similarity", return_value=0.35):
        signal = tracker.evaluate_followup("user_test", continuation_prompt)
        assert signal is not None
        assert signal["signal_type"] == "engagement_reward"
        assert signal["reward_delta"] == 0.3

        router.apply_implicit_signal(
            model=signal["model"],
            signal_type=signal["signal_type"],
            reward_delta=signal["reward_delta"],
            task_type=signal["task"],
        )
        assert router.q_values[model] > initial_q
        assert router.ts_alpha[model] > 1.0


def test_fast_bounce_detection():
    tracker = ImplicitFeedbackTracker()

    tracker.record_turn(
        user_id="user_test",
        prompt="Tell me a joke",
        model="ollama/llama3",
        task="creative",
        latency=0.5,
    )

    # Simulate 2 seconds dwell time (rapid click/skip)
    tracker._user_turns["user_test"]["timestamp"] = time.time() - 2.0

    different_prompt = "What is the capital of Japan?"
    with patch.object(tracker, "_cosine_similarity", return_value=0.10):
        signal = tracker.evaluate_followup("user_test", different_prompt)
        assert signal is not None
        assert signal["signal_type"] == "fast_bounce"
        assert signal["reward_delta"] == -0.2


def test_session_abandonment():
    tracker = ImplicitFeedbackTracker()
    router = RLRouter()
    model = "openrouter/nvidia/llama-3.1-nemotron-70b-instruct"
    initial_q = router.q_values[model]

    tracker.record_turn(
        user_id="user_abandon",
        prompt="Generate full stack application code",
        model=model,
        task="code",
        latency=5.0,
    )

    signal = tracker.record_abandonment("user_abandon", reason="client_abort")
    assert signal is not None
    assert signal["signal_type"] == "abandonment_penalty"
    assert signal["reward_delta"] == -1.5

    router.apply_implicit_signal(
        model=signal["model"],
        signal_type=signal["signal_type"],
        reward_delta=signal["reward_delta"],
    )
    assert router.q_values[model] < initial_q
    # Calling abandonment again returns None since turn expired
    assert tracker.record_abandonment("user_abandon") is None


def test_session_abandon_api_endpoint():
    client = TestClient(app)
    from app.dependencies import implicit_tracker

    # Record turn first
    implicit_tracker.record_turn(
        user_id="user_api_abandon",
        prompt="Test prompt",
        model="ollama/llama3",
        task="general",
        latency=1.0,
    )

    resp = client.post("/session/abandon", json={
        "user_id": "user_api_abandon",
        "reason": "tab_close",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["signal_applied"] is True
    assert data["details"]["signal_type"] == "abandonment_penalty"
