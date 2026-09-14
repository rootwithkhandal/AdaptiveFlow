# AdaptiveFlow

> **adapts to every prompt**

A production-grade, enterprise LLM load balancer that dynamically routes prompts to the optimal AI model using reinforcement learning, prompt complexity scoring, persona priors, vector memory, semantic prompt caching, and resilient circuit breakers.

---

## Key Capabilities

Every incoming request passes through an intelligent, multi-stage routing pipeline:

1. **Security Screening & Compliance Audit Logging**
   - Regex-based guardrails protecting against prompt injections, jailbreaks, and sensitive data leakage.
   - Dual-persistence compliance audit logging in both structured SQLite (`data/security_audit.db`) and append-only JSONL (`data/security_audit.jsonl`).
2. **Per-Tier Sliding Window Rate Limiter**
   - Strict sliding-window rate limits (`free`: 10 req/min, `pro`: 60 req/min, `enterprise`: unlimited).
   - Operates *before* Redis and FAISS vector computations to prevent burst hammering and denial-of-service.
3. **Dual-Tier Prompt Cache**
   - Instant returns ($O(1)$) on exact SHA256 matches in Redis.
   - Cosine semantic similarity cache using SentenceTransformers embeddings into a FAISS index.
4. **Prompt Complexity Scorer (Cost Optimization)**
   - Pre-evaluates prompt complexity across **token length**, **information entropy** (character Shannon entropy + Type-Token Ratio), and **question depth** (reasoning triggers, multi-part questions, syntax/code blocks).
   - **Simple prompts** ($\\text{Score} < 0.35$) automatically bypass RL exploration and route to the **cheapest capable model** (e.g. local Ollama at $0.0 or Mini/Flash at $0.00015/1k tokens).
   - **Complex prompts** ($\\text{Score} \\ge 0.35$) route through full RL bandit selection.
5. **Per-Task-Type RL State Split (Multi-Armed Bandit)**
   - Partitions Q-tables, selection counts, and decay rates across `code`, `security`, `creative`, and `general` task categories.
   - Independent $\\epsilon$-greedy exploration arms and Thompson Sampling Beta distributions prevent "muddy signal" averaging across domains.
6. **Model Persona & Specialization Profiles (Bayesian Priors)**
   - Allows users and admins to define routing hints (e.g., `{"prefer": "code_heavy", "avoid": ["gemini"]}`).
   - Acts as a Bayesian prior at selection time without corrupting the underlying empirical environment Q-values.
7. **Circuit Breakers with Cascading Fallbacks**
   - State machine (`CLOSED` $\\to$ `OPEN` on 3 consecutive failures $\\to$ `HALF_OPEN` after 30s recovery timeout).
   - Outbound network calls are skipped ($O(1)$ fast-fail) when failing, traversing provider fallback chains down to local Ollama.
8. **Cold Start Warm-Up via Synthetic Benchmarks**
   - Onboard new models without blind exploration using a curated 48-prompt synthetic benchmark suite (12 per task class) that seeds Q-values and Beta parameters.
9. **Routing Explainability & Transparency (`GET /route/explain`)**
   - Exposes predicted task classification, complexity breakdown, top 3 candidate models with Q-values and priors, estimated token costs, cache previews, and clear human-readable selection reasoning.
10. **Multi-Model Synthesis & Parallel Dispatch**
    - Concurrent dispatch strategies (`fastest`, `highest_q`, `all`).
    - Multi-model synthesis (Fugu-Ultra) dispatching 2–3 models concurrently with an LLM judge evaluating correctness and conciseness (Pro+ tier gated).
11. **Passive Implicit Feedback Loop**
    - Automatically rewards or penalizes models via user behavior: rephrase detection (-0.8), engagement dwell time (+0.3), fast bounces (-0.2), and premature client abandonment (-1.5).

---

## Project Structure

```
.
├── app/
│   ├── main.py                # FastAPI app factory, CORS, metrics, and lifecycle
│   ├── config.py              # Central BaseSettings (keys, circuit breaker, rate limit, complexity)
│   ├── dependencies.py        # Centralized singleton instances (single source of truth)
│   ├── complexity.py          # Prompt Complexity Scorer (tokens + entropy + depth)
│   ├── rl_router.py           # Multi-Armed Bandit RL router (task split, priors, explainability)
│   ├── benchmark.py           # Curated synthetic benchmark suite & cold-start Q-table seeding
│   ├── rate_limiter.py        # Sliding window per-tier rate limiter (protects Redis/FAISS)
│   ├── prompt_filter.py       # Security filter & compliance audit log (SQLite + JSONL)
│   ├── redis_cache.py         # Redis SHA256 + FAISS vector similarity semantic cache
│   ├── vector_memory.py       # FAISS / ChromaDB conversational context memory
│   ├── task_classifier.py     # Rule-based NLP task classifier (code, security, creative, general)
│   ├── parallel_executor.py   # Concurrent multi-model dispatcher (fastest, highest_q, all)
│   ├── synthesis.py           # Multi-Model Synthesis (Fugu-Ultra) with LLM judge scoring
│   ├── implicit_feedback.py   # Passive feedback signals (rephrase, dwell time, abandonment)
│   ├── profiles.py            # User tiers (free, pro, enterprise), budgets, and persona hints
│   ├── metrics.py             # Prometheus metrics (requests, latency, costs, rate limits, CB)
│   ├── logger.py              # Structured JSON logging via structlog
│   ├── cost.py                # Token cost estimator
│   │
│   ├── api/                   # REST API layer
│   │   ├── routes.py          # Core endpoints (/route, /route/explain, /prompt/complexity, etc.)
│   │   ├── admin.py           # Admin endpoints (costs, tiers, security audit, model warmup)
│   │   └── schemas.py         # Pydantic validation schemas
│   │
│   └── models/                # LLM execution & circuit breaker layer
│       ├── registry.py        # Model catalog, pricing, fallback pointers, and tier access
│       └── client.py          # Model dispatchers & per-model CircuitBreaker instances
│
├── infra/                     # Prometheus & Grafana provisioning
├── ui/                        # Web dashboard frontend
├── data/                      # Local storage (profiles.json, rl_state.json, security_audit.db)
├── tests/                     # Comprehensive test suite (14 test suites, 105 tests)
├── pyproject.toml             # Standard PEP 517/518/621 & dependency group configuration
├── uv.lock                    # Fully pinned reproducible lockfile generated by uv
├── .gitignore                 # Optimized ignore rules for bytecode, venv, caches, and local data
└── README.md                  # System architecture, API documentation, and quick-start guide
```

---

## API Surface Map

### Routing & Optimization

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/route` | Primary routing endpoint (security $\\to$ rate limit $\\to$ cache $\\to$ complexity $\\to$ RL) |
| `GET` | `/route/explain` | Full decision transparency: task class, complexity score, candidate models, Q-values, cost, reasoning |
| `POST` | `/prompt/complexity` | Score prompt complexity across token count, entropy, and depth, showing cheapest model |
| `GET` | `/prompt/complexity` | Query parameter variant of prompt complexity evaluation |
| `POST` | `/route/parallel` | Concurrently dispatch to candidate models (`fastest`, `highest_q`, `all`) |
| `POST` | `/route/synthesize` | Multi-Model Synthesis (Fugu-Ultra) with LLM judge layer (**Pro+ only**) |
| `POST` | `/session/abandon` | Client abort beacon applying abandonment penalties to active turn |
| `POST` | `/feedback` | Submit explicit feedback ratings (+1 / -1) to update task bandit rewards |

### Identity, Quotas & Telemetry

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/profile/{user_id}` | Retrieve user tier, accrued costs, request count, and persona hints (`prefer`, `avoid`) |
| `POST` | `/profile/{user_id}/preferences` | Update user routing hints (e.g. `{"prefer": "code_heavy", "avoid": ["gemini"]}`) |
| `GET` | `/ratelimit/{user_id}` | Inspect sliding window rate limit usage and remaining quota |
| `GET` | `/models/stats` | View RL Q-values, selection counts, latency, and costs (supports `?task_type=`) |
| `GET` | `/health` | Liveness health check |
| `GET` | `/metrics` | Prometheus metrics endpoint |
| `GET` | `/ui` | Web GUI dashboard |

### Admin Management (`X-Admin-Key` Required)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/admin/costs` | View total accrued costs per user |
| `POST` | `/admin/users/{user_id}/tier` | Set user tier (`free`, `pro`, `enterprise`) and budget limits |
| `POST` | `/admin/users/{user_id}/preferences` | Inspect or override user routing hints/persona |
| `POST` | `/admin/users/{user_id}/flag` | Flag user as high-risk (sandboxes routing to local Ollama) |
| `GET` | `/admin/rl/stats` | View internal RL weights across all models and tasks |
| `POST` | `/admin/models/register` | Dynamically register model + auto-run synthetic benchmark warm-up |
| `POST` | `/admin/models/warmup` | Re-run synthetic benchmark suite on existing model to re-seed Q-values |
| `GET` | `/admin/security/audit` | Query structured compliance security audit logs (filtered by user or threat) |
| `GET` | `/admin/security/stats` | Summary statistics of all security events and top flagged actors |

---

## Quick Start

### 1. Environment Configuration

Copy the sample environment file and add your API keys:

```bash
cp .env.example .env
```

Key environment variables:
```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key
NVIDIA_API_KEY=your_nvidia_key
REDIS_URL=redis://localhost:6379
SECRET_KEY=admin-secret-key
```

### 2. Fast Setup with uv (Recommended)

The project is fully configured for [uv](https://github.com/astral-sh/uv) with `pyproject.toml` and `uv.lock`:

```bash
# Sync virtual environment and install all dependencies & dev tools
uv sync

# Run the API server with auto-reload
uv run uvicorn app.main:app --reload --port 8000

# Execute the complete automated test suite
uv run pytest -v
```

### 3. Alternative: Docker Compose

```bash
docker-compose up --build
```

Access:
- **API**: `http://localhost:8000`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **Web UI**: `http://localhost:8000/ui`
- **Grafana Dashboards**: `http://localhost:3000`

---

## Running the Automated Test Suite

The test suite contains **105 automated unit and integration tests across 14 test modules**:

```powershell
# Run natively with uv
uv run pytest -v

# Or run a specific test suite
uv run pytest tests/test_complexity.py -v
```

### Verified Test Suites (105/105 Passing)

- `tests/test_complexity.py`: Token count, Shannon entropy, question depth scoring, and cheapest capable model bypass.
- `tests/test_persona_preferences.py`: User and admin routing hints (`prefer`, `avoid`) acting as Bayesian priors.
- `tests/test_explain.py`: Algorithmic transparency in `GET /route/explain`.
- `tests/test_security.py`: Guardrail regex blocking, SQLite & JSONL audit logs, and compliance APIs.
- `tests/test_rate_limiter.py`: Sliding window quotas per tier, Redis/FAISS protection, and HTTP 429 retries.
- `tests/test_circuit_breaker.py`: Per-model failure thresholding, 30s recovery canary probes, and fallback cascades.
- `tests/test_benchmarks.py`: 48-prompt synthetic benchmark cold-start warm-up and Q-table seeding.
- `tests/test_rl_router.py`: Independent per-task RL arms, isolated epsilon decay, and Q-table persistence.
- `tests/test_implicit_feedback.py`: Rephrase penalty, dwell engagement, and session abandonment signals.
- `tests/test_nvidia_models.py`: NVIDIA NIM and Nemotron model integrations.
- `tests/test_parallel_executor.py`: Concurrent multi-model execution (`fastest`, `highest_q`, `all`).
- `tests/test_synthesis.py`: Fugu-Ultra multi-model synthesis and LLM judge scoring.
- `tests/test_semantic_cache.py`: Dual SHA256 and FAISS cosine similarity semantic caching.
- `tests/test_router.py`: End-to-end API integration tests.

---

## License

MIT
