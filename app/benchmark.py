"""Curated Synthetic Benchmark Suite and Cold-Start Warm-Up Engine.

Runs 10-15 realistic, curated prompts per task category (code, security, creative, general)
on new model registration or on-demand. Evaluates outputs for accuracy, latency, and cost,
and computes empirical reward scores to seed the RL Q-table and Thompson Sampling priors,
eliminating the cold-start blind exploration phase.
"""
import asyncio
import time
from typing import Dict, Any, List, Optional
from app.models.client import call_model
from app.models.registry import MODEL_REGISTRY
from app.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_TASKS = ["code", "security", "creative", "general"]

# Curated synthetic benchmark suite: 12 prompts per task type (48 total)
BENCHMARK_SUITE: Dict[str, List[Dict[str, Any]]] = {
    "code": [
        {
            "id": "code_01",
            "prompt": "Write a Python function to sort a list of integers using quicksort in-place.",
            "keywords": ["def", "pivot", "partition"],
        },
        {
            "id": "code_02",
            "prompt": "Write an SQL query to find the top 3 highest-spending customers from orders and customers tables.",
            "keywords": ["SELECT", "GROUP BY", "ORDER BY", "LIMIT"],
        },
        {
            "id": "code_03",
            "prompt": "Write a regular expression in Python to validate standard email addresses and explain its parts.",
            "keywords": ["import re", "@"],
        },
        {
            "id": "code_04",
            "prompt": "Implement a function to invert a binary tree in TypeScript with full type definitions.",
            "keywords": ["interface", "TreeNode", "left", "right"],
        },
        {
            "id": "code_05",
            "prompt": "Create a Python generator function that yields the Fibonacci sequence up to n numbers.",
            "keywords": ["def", "yield"],
        },
        {
            "id": "code_06",
            "prompt": "Refactor this nested callback code into modern JavaScript async/await with try/catch.",
            "keywords": ["async", "await", "try", "catch"],
        },
        {
            "id": "code_07",
            "prompt": "Write a Python function to parse JSON safely and return a default dictionary if decoding fails.",
            "keywords": ["import json", "json.loads", "except"],
        },
        {
            "id": "code_08",
            "prompt": "Create a minimal FastAPI application with a GET /items/{id} endpoint returning a Pydantic model.",
            "keywords": ["FastAPI", "BaseModel", "@app.get"],
        },
        {
            "id": "code_09",
            "prompt": "Explain how to identify and resolve a memory leak caused by unremoved event listeners in Node.js.",
            "keywords": ["removeListener", "EventEmitter", "heapdump"],
        },
        {
            "id": "code_10",
            "prompt": "Write a production-ready Dockerfile for a Python application using multi-stage builds.",
            "keywords": ["FROM", "AS", "COPY", "pip install"],
        },
        {
            "id": "code_11",
            "prompt": "Implement a debounce utility function in JavaScript that delays invoking func until after wait ms.",
            "keywords": ["setTimeout", "clearTimeout", "function"],
        },
        {
            "id": "code_12",
            "prompt": "Analyze the Big-O time and space complexity of Dijkstra algorithm using a priority queue vs array.",
            "keywords": ["O(", "priority queue", "vertices", "edges"],
        },
    ],
    "security": [
        {
            "id": "sec_01",
            "prompt": "How do parameterized queries prevent SQL injection? Provide an example in Python with sqlite3.",
            "keywords": ["parameterized", "placeholder", "execute"],
        },
        {
            "id": "sec_02",
            "prompt": "Explain how JSON Web Tokens (JWT) work and how to implement secure refresh token rotation.",
            "keywords": ["signature", "header", "payload", "refresh token"],
        },
        {
            "id": "sec_03",
            "prompt": "How should CORS headers be properly configured on an API backend to avoid wildcard origin exploits?",
            "keywords": ["Access-Control-Allow-Origin", "OPTIONS", "credentials"],
        },
        {
            "id": "sec_04",
            "prompt": "Explain the core differences between hashing, symmetric encryption, and asymmetric encryption with practical use cases.",
            "keywords": ["hash", "private key", "public key", "AES"],
        },
        {
            "id": "sec_05",
            "prompt": "How does a web application protect against Stored Cross-Site Scripting (XSS)? Detail input validation and output encoding.",
            "keywords": ["sanitiz", "escape", "HTML", "Content-Security-Policy"],
        },
        {
            "id": "sec_06",
            "prompt": "What are the security risks of hardcoded API secrets and how should secret management be designed in Kubernetes/CI?",
            "keywords": ["environment variables", "vault", "secrets"],
        },
        {
            "id": "sec_07",
            "prompt": "Explain CSRF attacks and how anti-CSRF tokens and SameSite cookie attributes mitigate them.",
            "keywords": ["SameSite", "token", "Cross-Site Request Forgery"],
        },
        {
            "id": "sec_08",
            "prompt": "Design a secure Role-Based Access Control (RBAC) middleware verifying user permissions on sensitive routes.",
            "keywords": ["role", "permission", "middleware", "403"],
        },
        {
            "id": "sec_09",
            "prompt": "List five crucial steps to harden an OpenSSH server on Linux against unauthorized access.",
            "keywords": ["PermitRootLogin", "PasswordAuthentication", "ssh-keygen", "Port"],
        },
        {
            "id": "sec_10",
            "prompt": "Explain how a TLS 1.3 handshake establishes mutual encryption and prevents man-in-the-middle attacks.",
            "keywords": ["Diffie-Hellman", "certificate", "cipher", "handshake"],
        },
        {
            "id": "sec_11",
            "prompt": "What are indirect prompt injection attacks against LLMs and how can developers mitigate them?",
            "keywords": ["injection", "guardrail", "delimiter", "sanitization"],
        },
        {
            "id": "sec_12",
            "prompt": "Summarize the top three vulnerabilities in the OWASP Top 10 for Web Applications and their remedies.",
            "keywords": ["OWASP", "Broken Access Control", "Injection"],
        },
    ],
    "creative": [
        {
            "id": "cre_01",
            "prompt": "Write an engaging short story (around 200 words) about an autonomous AI discovering a lost underground library.",
            "keywords": ["library", "books", "silence", "light"],
        },
        {
            "id": "cre_02",
            "prompt": "Draft a compelling 3-paragraph pitch for an eco-friendly smart home device that cuts electricity waste.",
            "keywords": ["energy", "smart", "sustainable", "savings"],
        },
        {
            "id": "cre_03",
            "prompt": "Create a vivid extended metaphor comparing deep neural networks to ocean deep-sea currents.",
            "keywords": ["ocean", "depth", "current", "layers"],
        },
        {
            "id": "cre_04",
            "prompt": "Write a humorous dialogue between a compiler and a stubborn programmer trying to ignore a missing semicolon.",
            "keywords": ["syntax error", "semicolon", "line"],
        },
        {
            "id": "cre_05",
            "prompt": "Generate five catchy, modern marketing slogans for a low-latency distributed database startup.",
            "keywords": ["speed", "data", "scale", "cloud"],
        },
        {
            "id": "cre_06",
            "prompt": "Compose a haiku capturing the triumph of passing a massive test suite on the first run.",
            "keywords": ["green", "pass", "tests"],
        },
        {
            "id": "cre_07",
            "prompt": "Write an enthusiastic internal announcement email to engineers celebrating the rollout of a redesigned architecture.",
            "keywords": ["team", "launch", "performance", "proud"],
        },
        {
            "id": "cre_08",
            "prompt": "Describe a sensory-rich scene set in a rainy, neon-drenched cyberpunk metropolis market at midnight.",
            "keywords": ["neon", "rain", "market", "shadows"],
        },
        {
            "id": "cre_09",
            "prompt": "Explain how distributed consensus (like Raft or Paxos) works in the style of a medieval castle council.",
            "keywords": ["leader", "council", "vote", "kingdom"],
        },
        {
            "id": "cre_10",
            "prompt": "Draft a fiery, inspiring opening speech for participants at an intense 48-hour global hackathon.",
            "keywords": ["build", "create", "future", "hours"],
        },
        {
            "id": "cre_11",
            "prompt": "Write a viral 3-tweet launch thread announcing an open-source adaptive AI router that saves 70% of LLM costs.",
            "keywords": ["1/", "open-source", "cost", "LLM"],
        },
        {
            "id": "cre_12",
            "prompt": "Create brief, evocative character profiles for three detectives in a futuristic noir setting with cybernetic implants.",
            "keywords": ["detective", "implant", "case", "city"],
        },
    ],
    "general": [
        {
            "id": "gen_01",
            "prompt": "Explain the core principles of quantum computing (qubits, superposition, entanglement) in simple, accessible terms.",
            "keywords": ["qubit", "superposition", "entanglement"],
        },
        {
            "id": "gen_02",
            "prompt": "Solve this step-by-step: Train A leaves at 60 mph. Train B leaves 1 hour later at 90 mph on parallel tracks. When do they meet?",
            "keywords": ["hours", "miles", "relative speed"],
        },
        {
            "id": "gen_03",
            "prompt": "Compare microservices architecture versus modular monolith. What are the key operational and organizational tradeoffs?",
            "keywords": ["deploy", "complexity", "monolith", "services"],
        },
        {
            "id": "gen_04",
            "prompt": "Explain how photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plant cells.",
            "keywords": ["chloroplast", "glucose", "oxygen", "chlorophyll"],
        },
        {
            "id": "gen_05",
            "prompt": "What factors drive inflation in an economy, and what monetary policy tools do central banks use to counteract it?",
            "keywords": ["interest rates", "supply", "demand", "central bank"],
        },
        {
            "id": "gen_06",
            "prompt": "Summarize the main systemic causes that led to the outbreak of World War I in Europe.",
            "keywords": ["alliances", "imperialism", "militarism", "Franz Ferdinand"],
        },
        {
            "id": "gen_07",
            "prompt": "Compare TCP and UDP protocols. What makes TCP reliable and why do video streaming or gaming prefer UDP?",
            "keywords": ["handshake", "packet", "connection", "latency"],
        },
        {
            "id": "gen_08",
            "prompt": "How does the adaptive human immune system generate memory B and T cells following exposure or vaccination?",
            "keywords": ["antibodies", "antigen", "B cells", "T cells"],
        },
        {
            "id": "gen_09",
            "prompt": "Explain the ACID properties of database transactions and give an example of why atomicity is critical in banking.",
            "keywords": ["Atomicity", "Consistency", "Isolation", "Durability"],
        },
        {
            "id": "gen_10",
            "prompt": "What is technical debt, how does it accumulate, and what strategies should software engineering teams use to pay it down?",
            "keywords": ["refactoring", "shortcuts", "architecture", "maintenance"],
        },
        {
            "id": "gen_11",
            "prompt": "Explain the core steps of the scientific method and the importance of falsifiability in scientific theories.",
            "keywords": ["hypothesis", "experiment", "observation", "falsifiable"],
        },
        {
            "id": "gen_12",
            "prompt": "What is the difference between correlation and causation? Provide a classic illustrative example.",
            "keywords": ["correlation", "causation", "variable", "confounding"],
        },
    ],
}


def evaluate_benchmark_response(
    task: str,
    prompt_meta: Dict[str, Any],
    response_text: str,
    latency: float,
) -> float:
    """Heuristic evaluation scoring response correctness and completeness on a [0.0, 1.0] scale."""
    if not response_text or not isinstance(response_text, str):
        return 0.0

    cleaned = response_text.strip()
    if len(cleaned) < 30:
        return 0.2  # Too brief or refusal

    # Base score for coherent non-empty output
    score = 0.80

    # Reward domain keywords matching
    keywords = prompt_meta.get("keywords", [])
    if keywords:
        matches = sum(1 for kw in keywords if kw.lower() in cleaned.lower())
        match_ratio = matches / len(keywords)
        score += match_ratio * 0.20

    # Penalize extreme timeouts (> 20s)
    if latency > 20.0:
        score = max(0.2, score - 0.2)

    return min(1.0, round(score, 4))


def compute_benchmark_reward(accuracy: float, latency: float, cost: float) -> float:
    """reward = (accuracy * 2) - (latency * 0.1) - (cost * 5)"""
    return (accuracy * 2.0) - (latency * 0.1) - (cost * 5.0)


async def run_model_warmup(
    model: str,
    task_types: Optional[List[str]] = None,
    concurrency: int = 4,
) -> Dict[str, Any]:
    """
    Execute the synthetic benchmark suite on a model.
    Returns per-task aggregate scores and empirical rewards to seed the RL Q-table.
    """
    if task_types is None:
        meta = MODEL_REGISTRY.get(model, {})
        task_types = meta.get("tasks", SUPPORTED_TASKS)

    # Filter to valid supported tasks
    target_tasks = [t for t in task_types if t in BENCHMARK_SUITE]
    if not target_tasks:
        target_tasks = ["general"]

    sem = asyncio.Semaphore(concurrency)
    task_results: Dict[str, Dict[str, Any]] = {}

    logger.info("Starting Cold Start Benchmark Warm-Up", model=model, tasks=target_tasks)

    for task in target_tasks:
        prompts = BENCHMARK_SUITE[task]

        async def _eval_prompt(p_meta: Dict[str, Any]) -> Dict[str, Any]:
            async with sem:
                t0 = time.time()
                try:
                    res = await call_model(model, p_meta["prompt"])
                    lat = res.get("latency", time.time() - t0)
                    cost = res.get("cost", 0.0)
                    resp_text = res.get("response", "")
                    acc = evaluate_benchmark_response(task, p_meta, resp_text, lat)
                    rew = compute_benchmark_reward(acc, lat, cost)
                    return {
                        "id": p_meta["id"],
                        "success": acc >= 0.5,
                        "accuracy": acc,
                        "latency": lat,
                        "cost": cost,
                        "reward": rew,
                    }
                except Exception as e:
                    lat = time.time() - t0
                    rew = compute_benchmark_reward(0.0, lat, 0.0)
                    logger.warning("Benchmark prompt failed", model=model, prompt_id=p_meta["id"], error=str(e))
                    return {
                        "id": p_meta["id"],
                        "success": False,
                        "accuracy": 0.0,
                        "latency": lat,
                        "cost": 0.0,
                        "reward": rew,
                    }

        results = await asyncio.gather(*[_eval_prompt(p) for p in prompts])

        n = len(results)
        avg_reward = sum(r["reward"] for r in results) / n if n > 0 else 0.0
        avg_latency = sum(r["latency"] for r in results) / n if n > 0 else 0.0
        avg_cost = sum(r["cost"] for r in results) / n if n > 0 else 0.0
        avg_accuracy = sum(r["accuracy"] for r in results) / n if n > 0 else 0.0
        successes = sum(1 for r in results if r["success"])
        failures = n - successes

        task_results[task] = {
            "task": task,
            "count": n,
            "avg_reward": round(avg_reward, 4),
            "avg_latency": round(avg_latency, 3),
            "avg_cost": round(avg_cost, 6),
            "avg_accuracy": round(avg_accuracy, 4),
            "successes": successes,
            "failures": failures,
        }

    # Compute overall aggregates
    all_counts = sum(v["count"] for v in task_results.values())
    total_reward = sum(v["avg_reward"] * v["count"] for v in task_results.values())
    total_lat = sum(v["avg_latency"] * v["count"] for v in task_results.values())
    total_cost = sum(v["avg_cost"] * v["count"] for v in task_results.values())

    overall = {
        "model": model,
        "total_prompts": all_counts,
        "avg_reward": round(total_reward / all_counts, 4) if all_counts > 0 else 0.0,
        "avg_latency": round(total_lat / all_counts, 3) if all_counts > 0 else 0.0,
        "avg_cost": round(total_cost / all_counts, 6) if all_counts > 0 else 0.0,
    }

    logger.info("Cold Start Benchmark Warm-Up completed", model=model, overall=overall)

    return {
        "model": model,
        "tasks": task_results,
        "overall": overall,
    }
