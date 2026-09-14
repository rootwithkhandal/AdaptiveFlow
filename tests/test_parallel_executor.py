"""Unit tests for ParallelExecutor."""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from app.parallel_executor import ParallelExecutor
from app.rl_router import RLRouter


@pytest.mark.asyncio
async def test_parallel_executor_fastest():
    executor = ParallelExecutor()

    async def mock_call(model, prompt):
        if model == "model_fast":
            await asyncio.sleep(0.01)
            return {"response": "Fast response", "cost": 0.001, "tokens": 10}
        else:
            await asyncio.sleep(0.2)
            return {"response": "Slow response", "cost": 0.005, "tokens": 50}

    with patch("app.models.client.call_model", side_effect=mock_call):
        res = await executor.dispatch(
            models=["model_slow", "model_fast"],
            prompt="Hello world",
            strategy="fastest",
        )
        assert res["winner"] == "model_fast"
        assert res["strategy"] == "fastest"
        assert res["response"] == "Fast response"


@pytest.mark.asyncio
async def test_parallel_executor_all():
    executor = ParallelExecutor()

    async def mock_call(model, prompt):
        return {"response": f"Response from {model}", "cost": 0.002, "tokens": 20}

    with patch("app.models.client.call_model", side_effect=mock_call):
        res = await executor.dispatch(
            models=["model_a", "model_b"],
            prompt="Compare models",
            strategy="all",
        )
        assert res["strategy"] == "all"
        assert "model_a" in res["results"]
        assert "model_b" in res["results"]
        assert res["results"]["model_a"]["success"] is True
        assert res["results"]["model_b"]["success"] is True


@pytest.mark.asyncio
async def test_parallel_executor_highest_q():
    router = RLRouter()
    router.q_values["model_high_q"] = 5.0
    router.q_values["model_low_q"] = 1.0

    executor = ParallelExecutor(rl_router=router)

    async def mock_call(model, prompt):
        return {"response": f"Response from {model}", "cost": 0.001, "tokens": 10}

    with patch("app.models.client.call_model", side_effect=mock_call):
        res = await executor.dispatch(
            models=["model_low_q", "model_high_q"],
            prompt="Best quality prompt",
            strategy="highest_q",
        )
        assert res["winner"] == "model_high_q"
        assert res["strategy"] == "highest_q"


@pytest.mark.asyncio
async def test_parallel_executor_one_fails():
    executor = ParallelExecutor()

    async def mock_call(model, prompt):
        if model == "broken_model":
            raise RuntimeError("API timeout")
        return {"response": "Healthy response", "cost": 0.001, "tokens": 10}

    with patch("app.models.client.call_model", side_effect=mock_call):
        res = await executor.dispatch(
            models=["broken_model", "working_model"],
            prompt="Fault tolerance test",
            strategy="fastest",
        )
        assert res["winner"] == "working_model"
        assert res["response"] == "Healthy response"
