"""Unit tests for RL router, task classifier, and per-task state split."""
import pytest
import os
import json
from app.task_classifier import TaskClassifier
from app.rl_router import RLRouter
from app.config import settings


def test_classifier_code():
    c = TaskClassifier()
    assert c.classify("Write a Python function to sort a list") == "code"


def test_classifier_security():
    c = TaskClassifier()
    assert c.classify("How does JWT authentication work?") == "security"


def test_classifier_creative():
    c = TaskClassifier()
    assert c.classify("Write a short story about a dragon") == "creative"


def test_classifier_general():
    c = TaskClassifier()
    assert c.classify("What is the weather like today?") == "general"


def test_rl_router_default_mode():
    """Verify that per_task_split is enabled by default."""
    router = RLRouter(state_file=None)
    assert router.per_task_split is True
    assert set(router.task_q_values.keys()) == {"code", "security", "creative", "general"}
    assert set(router.task_epsilon.keys()) == {"code", "security", "creative", "general"}


def test_rl_router_q_update():
    router = RLRouter(state_file=None)
    model = "ollama/llama3"
    router.update_reward(model, cost=0.0, latency=1.0, accuracy=1.0)
    # reward = (1*2) - (1*0.1) - (0*5) = 1.9
    assert abs(router.q_values[model] - 1.9) < 0.01
    assert router.counts[model] == 1


def test_rl_router_per_task_split():
    router = RLRouter(per_task_split=True, state_file=None)
    model = "ollama/llama3"
    router.update_reward(model, cost=0.0, latency=1.0, accuracy=1.0, task_type="code")

    # Global updated
    assert abs(router.q_values[model] - 1.9) < 0.01
    # Code task Q updated
    assert abs(router.task_q_values["code"][model] - 1.9) < 0.01
    # Security task Q untouched
    assert router.task_q_values["security"][model] == 0.0


def test_rl_router_independent_task_epsilon_decay():
    """Verify that epsilon decays ONLY for the routed task category."""
    router = RLRouter(per_task_split=True, state_file=None)
    initial_code_eps = router.task_epsilon["code"]
    initial_sec_eps = router.task_epsilon["security"]
    initial_creative_eps = router.task_epsilon["creative"]
    initial_general_eps = router.task_epsilon["general"]

    # Select model for a code task
    router._select_model_epsilon_greedy(["ollama/llama3"], task_type="code")

    # Code epsilon decayed
    assert router.task_epsilon["code"] < initial_code_eps
    assert abs(router.task_epsilon["code"] - (initial_code_eps * settings.epsilon_decay)) < 1e-5

    # All other task epsilons remain completely untouched
    assert router.task_epsilon["security"] == initial_sec_eps
    assert router.task_epsilon["creative"] == initial_creative_eps
    assert router.task_epsilon["general"] == initial_general_eps


def test_rl_router_per_task_q_isolation():
    """
    Verify that GPT-4o dominating code does not muddy or inflate Q-values in general tasks,
    and Gemini Flash winning general tasks does not leak into code.
    """
    router = RLRouter(per_task_split=True, state_file=None)
    model_code = "gpt-4o"
    model_general = "gemini/gemini-1.5-flash"

    # Reward gpt-4o heavily on code
    for _ in range(3):
        router.update_reward(model_code, cost=0.01, latency=0.5, accuracy=1.0, task_type="code")

    # Reward gemini flash on general
    for _ in range(3):
        router.update_reward(model_general, cost=0.001, latency=0.2, accuracy=1.0, task_type="general")

    # Code arm: gpt-4o is high, gemini flash is 0.0
    assert router.task_q_values["code"][model_code] > 1.5
    assert router.task_q_values["code"][model_general] == 0.0
    assert router.task_counts["code"][model_code] == 3
    assert router.task_counts["code"][model_general] == 0

    # General arm: gemini flash is high, gpt-4o is 0.0
    assert router.task_q_values["general"][model_general] > 1.5
    assert router.task_q_values["general"][model_code] == 0.0
    assert router.task_counts["general"][model_general] == 3
    assert router.task_counts["general"][model_code] == 0

    # Security arm: completely pristine
    assert router.task_q_values["security"][model_code] == 0.0
    assert router.task_q_values["security"][model_general] == 0.0


def test_rl_router_feedback():
    router = RLRouter(state_file=None)
    model = "ollama/llama3"
    initial_q = router.q_values[model]
    router.apply_feedback(model, rating=1, task_type="code")
    assert router.q_values[model] > initial_q
    assert router.task_q_values["code"][model] > 0.0
    assert router.task_q_values["security"][model] == 0.0


def test_rl_router_epsilon_decay_global_mode():
    router = RLRouter(per_task_split=False, state_file=None)
    initial_eps = router.epsilon
    router._select_model_epsilon_greedy(["ollama/llama3"])
    assert router.epsilon <= initial_eps


def test_thompson_sampling_selection():
    router = RLRouter(strategy="thompson_sampling", state_file=None)
    models = ["ollama/llama3", "ollama/mistral"]
    chosen = router._select_model_thompson(models, task_type="code")
    assert chosen in models


def test_thompson_sampling_task_isolation():
    router = RLRouter(strategy="thompson_sampling", state_file=None)
    model = "ollama/llama3"

    # Reward on creative
    router.update_reward(model, cost=0.0, latency=0.5, accuracy=1.0, task_type="creative")

    # Creative Beta alpha incremented
    assert router.task_ts_alpha["creative"][model] > 1.0
    # Security Beta alpha remains baseline
    assert router.task_ts_beta["creative"][model] == 1.0
    assert router.task_ts_alpha["security"][model] == 1.0


def test_rl_router_state_persistence(tmp_path):
    """Verify state serialization to disk and reload upon initialization."""
    persisted_path = str(tmp_path / "rl_state.json")
    router1 = RLRouter(per_task_split=True, state_file=persisted_path)

    # Train router 1
    router1.update_reward("gpt-4o", cost=0.01, latency=0.4, accuracy=1.0, task_type="code")
    router1.update_reward("gemini/gemini-1.5-flash", cost=0.001, latency=0.2, accuracy=1.0, task_type="general")
    router1._select_model_epsilon_greedy(["gpt-4o"], task_type="code")

    saved_code_q = router1.task_q_values["code"]["gpt-4o"]
    saved_code_eps = router1.task_epsilon["code"]
    assert os.path.exists(persisted_path)

    # Instantiate router 2 from the same persistence path
    router2 = RLRouter(per_task_split=True, state_file=persisted_path)
    assert abs(router2.task_q_values["code"]["gpt-4o"] - saved_code_q) < 1e-6
    assert abs(router2.task_epsilon["code"] - saved_code_eps) < 1e-6
    assert router2.task_counts["code"]["gpt-4o"] == 1
    assert router2.task_counts["general"]["gemini/gemini-1.5-flash"] == 1
    assert router2.task_counts["security"]["gpt-4o"] == 0


def test_get_stats():
    router = RLRouter(state_file=None)
    stats = router.get_stats()
    assert isinstance(stats, list)
    assert all("model" in s and "q_value" in s for s in stats)

    # Per task stats
    code_stats = router.get_stats(task_type="code")
    assert any(s["task"] == "code" for s in code_stats)
    assert all("epsilon" in s for s in code_stats)
