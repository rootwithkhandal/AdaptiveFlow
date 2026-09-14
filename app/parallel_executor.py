"""Parallel Executor: multi-model concurrent dispatch.
Runs multiple models concurrently and selects or synthesizes the best result.
Foundation for multi-model synthesis and latency-hedging upgrades.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional

import app.models.client as model_client
from app.logger import get_logger

logger = get_logger(__name__)


class ParallelExecutor:
    """Dispatches prompts concurrently across multiple models."""

    def __init__(self, rl_router: Optional[Any] = None):
        self.rl_router = rl_router

    async def _execute_single(self, model: str, prompt: str) -> Dict[str, Any]:
        """Execute a single model and record execution metadata."""
        start = time.time()
        try:
            res = await model_client.call_model(model, prompt)
            latency = time.time() - start
            res["model_used"] = model
            res["latency"] = latency
            res["success"] = True
            return res
        except Exception as e:
            latency = time.time() - start
            logger.warning("Parallel model execution failed", model=model, error=str(e))
            return {
                "model_used": model,
                "response": "",
                "cost": 0.0,
                "tokens": 0,
                "latency": latency,
                "success": False,
                "error": str(e),
            }

    async def dispatch_fastest(self, models: List[str], prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Returns the first model that successfully completes."""
        start = time.time()
        tasks = [
            asyncio.create_task(self._execute_single(m, prompt), name=m)
            for m in models
        ]

        completed_results = {}
        last_error = None

        for future in asyncio.as_completed(tasks, timeout=timeout):
            try:
                res = await future
                completed_results[res["model_used"]] = res
                if res.get("success", False):
                    # Cancel remaining tasks to conserve resources
                    for t in tasks:
                        if not t.done():
                            t.cancel()
                    total_latency = time.time() - start
                    return {
                        "winner": res["model_used"],
                        "strategy": "fastest",
                        "response": res["response"],
                        "cost": res["cost"],
                        "tokens": res.get("tokens", 0),
                        "latency": total_latency,
                        "candidates": models,
                        "results": completed_results,
                    }
                else:
                    last_error = res.get("error")
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(f"All models failed in parallel dispatch. Last error: {last_error}")

    async def dispatch_all(self, models: List[str], prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Executes all models in parallel and returns all outputs."""
        start = time.time()
        tasks = [self._execute_single(m, prompt) for m in models]

        results_list = await asyncio.gather(*tasks, return_exceptions=False)
        total_latency = time.time() - start

        results_map = {r["model_used"]: r for r in results_list}
        successful = [r for r in results_list if r.get("success", False)]

        if not successful:
            raise RuntimeError(f"All models failed in parallel dispatch: {models}")

        # Default winner is the one with lowest latency among successful
        winner = min(successful, key=lambda r: r["latency"])

        return {
            "winner": winner["model_used"],
            "strategy": "all",
            "response": winner["response"],
            "total_cost": round(sum(r.get("cost", 0.0) for r in successful), 6),
            "latency": total_latency,
            "candidates": models,
            "results": results_map,
        }

    async def dispatch_highest_q(self, models: List[str], prompt: str, timeout: float = 60.0) -> Dict[str, Any]:
        """Executes candidates in parallel and selects the successful model with highest Q-value."""
        start = time.time()
        all_results = await self.dispatch_all(models, prompt, timeout=timeout)
        successful_results = [
            r for r in all_results["results"].values() if r.get("success", False)
        ]

        if not successful_results:
            raise RuntimeError("No successful models in parallel dispatch")

        # Select by highest Q-value
        q_vals = getattr(self.rl_router, "q_values", {}) if self.rl_router else {}
        winner = max(successful_results, key=lambda r: q_vals.get(r["model_used"], 0.0))

        return {
            "winner": winner["model_used"],
            "strategy": "highest_q",
            "response": winner["response"],
            "cost": winner["cost"],
            "tokens": winner.get("tokens", 0),
            "latency": time.time() - start,
            "candidates": models,
            "results": all_results["results"],
        }

    async def dispatch(
        self,
        models: List[str],
        prompt: str,
        strategy: str = "fastest",
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Unified entrypoint for parallel execution."""
        if not models:
            raise ValueError("No models provided for parallel execution")

        logger.info("Parallel dispatch started", models=models, strategy=strategy)

        if strategy == "fastest":
            return await self.dispatch_fastest(models, prompt, timeout=timeout)
        elif strategy == "highest_q":
            return await self.dispatch_highest_q(models, prompt, timeout=timeout)
        elif strategy == "all":
            return await self.dispatch_all(models, prompt, timeout=timeout)
        else:
            return await self.dispatch_fastest(models, prompt, timeout=timeout)
