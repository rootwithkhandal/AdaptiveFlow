"""Unit tests for NVIDIA model registry and client dispatch."""
import pytest
from unittest.mock import patch, AsyncMock
from app.models.registry import MODEL_REGISTRY, TIER_RULES, get_models_for_profile
from app.models.client import call_model, _call_nvidia
from app.config import settings


def test_nvidia_models_in_registry():
    nemotron_free = "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free"
    nemotron_paid = "openrouter/nvidia/llama-3.1-nemotron-70b-instruct"
    nemotron_mini = "openrouter/nvidia/nemotron-mini-4b-instruct"
    nvidia_nim = "nvidia/llama-3.1-nemotron-70b-instruct"

    for m in [nemotron_free, nemotron_paid, nemotron_mini, nvidia_nim]:
        assert m in MODEL_REGISTRY
        assert "tasks" in MODEL_REGISTRY[m]
        assert "provider" in MODEL_REGISTRY[m]

    # Verify providers
    assert MODEL_REGISTRY[nemotron_free]["provider"] == "openrouter"
    assert MODEL_REGISTRY[nvidia_nim]["provider"] == "nvidia"


def test_nvidia_tier_availability():
    free_models = TIER_RULES["free"]["allowed_models"]
    pro_models = TIER_RULES["pro"]["allowed_models"]
    ent_models = TIER_RULES["enterprise"]["allowed_models"]

    # Free tier gets free nemotron
    assert "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free" in free_models
    assert "openrouter/nvidia/nemotron-mini-4b-instruct" in free_models

    # Pro gets both paid and free
    assert "openrouter/nvidia/llama-3.1-nemotron-70b-instruct" in pro_models
    assert "openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free" in pro_models

    # Enterprise gets all
    assert "nvidia/llama-3.1-nemotron-70b-instruct" in ent_models


@pytest.mark.asyncio
async def test_nvidia_nim_dispatch():
    settings.nvidia_api_key = "nvapi-test-key"

    mock_resp = {
        "choices": [{"message": {"content": "Hello from NVIDIA Nemotron!"}}],
        "usage": {"total_tokens": 42},
    }

    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def post(self, url, headers=None, json=None):
            class Resp:
                def raise_for_status(self):
                    pass
                def json(self):
                    return mock_resp
            return Resp()

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        result = await _call_nvidia("nvidia/llama-3.1-nemotron-70b-instruct", "Hello")
        assert result["response"] == "Hello from NVIDIA Nemotron!"
        assert result["tokens"] == 42
