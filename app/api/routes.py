"""FastAPI route definitions."""
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from app.api.schemas import (
    RouteRequest, RouteResponse,
    ParallelRouteRequest, ParallelRouteResponse,
    SynthesizeRequest, SynthesizeResponse,
    FeedbackRequest, FeedbackResponse,
    UserProfileResponse, UserPreferencesRequest, UserPreferencesResponse,
    AbandonRequest, AbandonResponse,
    ExplainResponse,
    ComplexityScoreRequest, ComplexityScoreResponse,
)
from app.dependencies import (
    rl_router, cache, memory, profile_manager, prompt_filter,
    parallel_executor, synthesizer, implicit_tracker, rate_limiter,
    complexity_scorer,
)
from app.models.registry import get_models_for_profile
from app.complexity import get_cheapest_capable_model
from app.metrics import (
    REQUEST_COUNTER, LATENCY_HISTOGRAM, COST_COUNTER, CACHE_HIT_COUNTER,
    IMPLICIT_FEEDBACK_COUNTER, RATE_LIMIT_EXCEEDED_COUNTER
)
from app.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/route", response_model=RouteResponse)
async def route_prompt(req: RouteRequest):
    """Main routing endpoint — classifies, caches, routes, and responds."""
    start = time.time()

    # 1. Security check with audit logging
    is_safe, threat_type = prompt_filter.check(req.prompt, user_id=req.user_id)
    if not is_safe:
        logger.warning("Blocked prompt", user_id=req.user_id, threat=threat_type)
        raise HTTPException(status_code=400, detail=f"Prompt blocked: {threat_type}")

    # 2. Per-user sliding window rate limit check (prevents hammering Redis cache & FAISS)
    profile = profile_manager.get_or_create(req.user_id)
    user_tier = profile.get("tier", "free")
    allowed, limit_info = await rate_limiter.check_and_record(req.user_id, tier=user_tier)
    if not allowed:
        RATE_LIMIT_EXCEEDED_COUNTER.labels(tier=user_tier).inc()
        logger.warning(
            "Rate limit exceeded",
            user_id=req.user_id,
            tier=user_tier,
            limit=limit_info["limit"],
            retry_after=limit_info["retry_after"],
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for tier '{user_tier}'. Limit: {limit_info['limit']} req/min. Please retry in {limit_info['retry_after']}s.",
            headers={"Retry-After": str(limit_info["retry_after"])},
        )

    # 3. Evaluate implicit feedback signal from previous turn (e.g. rephrase, dwell)
    implicit_signal = implicit_tracker.evaluate_followup(req.user_id, req.prompt)
    if implicit_signal:
        rl_router.apply_implicit_signal(
            model=implicit_signal["model"],
            signal_type=implicit_signal["signal_type"],
            reward_delta=implicit_signal["reward_delta"],
            task_type=implicit_signal.get("task"),
        )
        IMPLICIT_FEEDBACK_COUNTER.labels(
            model=implicit_signal["model"],
            signal_type=implicit_signal["signal_type"],
        ).inc()

    # 3. Check cache (exact SHA256 or semantic FAISS)
    cached = await cache.get(req.prompt)
    if cached:
        latency = time.time() - start
        cache_type = cached.get("cache_type", "exact")
        similarity = cached.get("similarity")

        CACHE_HIT_COUNTER.labels(type=cache_type).inc()
        REQUEST_COUNTER.labels(model="cache", task="cached").inc()

        return RouteResponse(
            response=cached["response"],
            model_used=cached["model_used"],
            task_type=cached["task_type"],
            cost=0.0,
            latency=latency,
            cached=True,
            cache_type=cache_type,
            similarity=similarity,
        )

    # 3. Load user profile
    profile = profile_manager.get_or_create(req.user_id)

    # 4. Retrieve context from vector memory
    context_docs = memory.retrieve(req.user_id, req.prompt)
    enriched_prompt = req.prompt
    context_injected = False
    if context_docs:
        context_str = "\n".join(f"- {d}" for d in context_docs)
        enriched_prompt = f"Relevant context:\n{context_str}\n\nUser query: {req.prompt}"
        context_injected = True

    # 5. Route via RL router (respects user profile constraints)
    result = await rl_router.route(
        user_id=req.user_id,
        prompt=enriched_prompt,
        profile=profile,
    )

    latency = time.time() - start

    # 6. Update RL reward (updates both global and per-task state)
    rl_router.update_reward(
        result["model_used"],
        result["cost"],
        latency,
        task_type=result.get("task_type"),
    )

    # 7. Store interaction in vector memory
    memory.store(req.user_id, req.prompt, result["response"])

    # 8. Cache the response in both Redis and semantic FAISS index
    await cache.set(req.prompt, {
        "response": result["response"],
        "model_used": result["model_used"],
        "task_type": result["task_type"],
    })

    # 9. Update user profile usage
    profile_manager.record_usage(req.user_id, result["cost"], result["model_used"])

    # 10. Emit metrics
    REQUEST_COUNTER.labels(model=result["model_used"], task=result["task_type"]).inc()
    LATENCY_HISTOGRAM.labels(model=result["model_used"]).observe(latency)
    COST_COUNTER.labels(model=result["model_used"]).inc(result["cost"])

    # 11. Record turn in implicit feedback tracker
    implicit_tracker.record_turn(
        user_id=req.user_id,
        prompt=req.prompt,
        model=result["model_used"],
        task=result["task_type"],
        latency=latency,
    )

    return RouteResponse(
        response=result["response"],
        model_used=result["model_used"],
        task_type=result["task_type"],
        cost=result["cost"],
        latency=latency,
        cached=False,
        context_injected=context_injected,
        complexity=result.get("complexity"),
    )


@router.post("/route/parallel", response_model=ParallelRouteResponse)
async def route_prompt_parallel(req: ParallelRouteRequest):
    """Concurrent multi-model dispatch endpoint."""
    # 1. Security screening
    is_safe, threat_type = prompt_filter.check(req.prompt, user_id=req.user_id)
    if not is_safe:
        logger.warning("Blocked prompt in parallel route", user_id=req.user_id, threat=threat_type)
        raise HTTPException(status_code=400, detail=f"Prompt blocked: {threat_type}")

    # 2. Rate limit check (sliding window per tier)
    profile = profile_manager.get_or_create(req.user_id)
    user_tier = profile.get("tier", "free")
    allowed, limit_info = await rate_limiter.check_and_record(req.user_id, tier=user_tier)
    if not allowed:
        RATE_LIMIT_EXCEEDED_COUNTER.labels(tier=user_tier).inc()
        logger.warning(
            "Rate limit exceeded in parallel route",
            user_id=req.user_id,
            tier=user_tier,
            limit=limit_info["limit"],
            retry_after=limit_info["retry_after"],
        )
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for tier '{user_tier}'. Limit: {limit_info['limit']} req/min. Please retry in {limit_info['retry_after']}s.",
            headers={"Retry-After": str(limit_info["retry_after"])},
        )

    # 3. Candidate model resolution
    candidates = req.models
    if not candidates:
        task_type = rl_router.classifier.classify(req.prompt)
        candidates = get_models_for_profile(profile, task_type)

    if not candidates:
        candidates = ["ollama/llama3"]

    # 3. Parallel execution
    result = await parallel_executor.dispatch(
        models=candidates,
        prompt=req.prompt,
        strategy=req.strategy,
    )

    # 4. Usage tracking for winning model
    cost = result.get("cost", 0.0)
    profile_manager.record_usage(req.user_id, cost, result["winner"])

    # 5. Metrics
    REQUEST_COUNTER.labels(model=result["winner"], task="parallel").inc()
    LATENCY_HISTOGRAM.labels(model=result["winner"]).observe(result["latency"])
    COST_COUNTER.labels(model=result["winner"]).inc(cost)

    return ParallelRouteResponse(
        winner=result["winner"],
        strategy=result["strategy"],
        response=result["response"],
        cost=cost,
        latency=result["latency"],
        candidates=candidates,
        results=result.get("results"),
    )


@router.post("/route/synthesize", response_model=SynthesizeResponse)
async def route_prompt_synthesize(req: SynthesizeRequest):
    """Multi-Model Synthesis endpoint (Fugu-Ultra equivalent).
    Sends prompt to 2-3 models in parallel, runs a lightweight judge layer
    scoring on correctness and conciseness, and returns the winning response.
    Gated to Pro+ (pro and enterprise) tiers only.
    """
    # 1. Security screening
    is_safe, threat_type = prompt_filter.check(req.prompt, user_id=req.user_id)
    if not is_safe:
        logger.warning("Blocked prompt in synthesize route", user_id=req.user_id, threat=threat_type)
        raise HTTPException(status_code=400, detail=f"Prompt blocked: {threat_type}")

    # 2. Gate to Pro+ tiers only
    profile = profile_manager.get_or_create(req.user_id)
    tier = profile.get("tier", "free")
    if profile.get("high_risk", False) or tier not in ["pro", "enterprise"]:
        logger.warning("Synthesis access denied for non-pro user", user_id=req.user_id, tier=tier)
        raise HTTPException(
            status_code=403,
            detail="Multi-Model Synthesis is restricted to Pro and Enterprise tiers only. Please upgrade your tier.",
        )

    # 3. Rate limit check (sliding window per tier)
    allowed, limit_info = await rate_limiter.check_and_record(req.user_id, tier=tier)
    if not allowed:
        RATE_LIMIT_EXCEEDED_COUNTER.labels(tier=tier).inc()
        logger.warning("Rate limit exceeded in synthesize route", user_id=req.user_id, tier=tier)
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded for tier '{tier}'. Limit: {limit_info['limit']} req/min. Please retry in {limit_info['retry_after']}s.",
            headers={"Retry-After": str(limit_info["retry_after"])},
        )

    # 4. Determine candidate models (2-3 models)
    candidates = req.candidate_models
    if not candidates:
        task_type = rl_router.classifier.classify(req.prompt)
        available = get_models_for_profile(profile, task_type)
        q_vals = getattr(rl_router, "q_values", {})
        sorted_available = sorted(available, key=lambda m: q_vals.get(m, 0.0), reverse=True)
        candidates = sorted_available[:3]
        if len(candidates) < 2:
            candidates = ["openrouter/openai/gpt-4o-mini", "gemini/gemini-1.5-flash"]

    # 4. Run synthesis & judge evaluation
    result = await synthesizer.synthesize(
        prompt=req.prompt,
        candidate_models=candidates,
        judge_model=req.judge_model,
    )

    # 5. Usage tracking and metrics
    total_cost = result.get("total_cost", 0.0)
    profile_manager.record_usage(req.user_id, total_cost, result["winner"])

    REQUEST_COUNTER.labels(model=result["winner"], task="synthesis").inc()
    LATENCY_HISTOGRAM.labels(model=result["winner"]).observe(result["latency"])
    COST_COUNTER.labels(model=result["winner"]).inc(total_cost)

    return SynthesizeResponse(
        winner=result["winner"],
        winner_response=result["winner_response"],
        judge_model=result["judge_model"],
        judge_rationale=result["judge_rationale"],
        scores=result["scores"],
        candidates=result["candidates"],
        results=result["results"],
        total_cost=total_cost,
        latency=result["latency"],
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Accept user feedback and update RL reward scores."""
    rl_router.apply_feedback(req.model_used, req.rating, task_type=req.task_type)
    return FeedbackResponse(status="ok")


@router.post("/session/abandon", response_model=AbandonResponse)
async def report_session_abandon(req: AbandonRequest):
    """Reports premature client abandonment / tab close / request cancellation.
    Triggers an abandonment penalty on the model of the active turn.
    """
    signal = implicit_tracker.record_abandonment(req.user_id, reason=req.reason)
    if signal:
        rl_router.apply_implicit_signal(
            model=signal["model"],
            signal_type=signal["signal_type"],
            reward_delta=signal["reward_delta"],
            task_type=signal.get("task"),
        )
        IMPLICIT_FEEDBACK_COUNTER.labels(
            model=signal["model"],
            signal_type=signal["signal_type"],
        ).inc()
        return AbandonResponse(status="ok", signal_applied=True, details=signal)
    return AbandonResponse(status="ok", signal_applied=False, details=None)


@router.get("/profile/{user_id}", response_model=UserProfileResponse)
async def get_profile(user_id: str):
    profile = profile_manager.get_or_create(user_id)
    return UserProfileResponse(
        user_id=user_id,
        tier=profile["tier"],
        total_cost=profile["total_cost"],
        request_count=profile["request_count"],
        budget_remaining=profile.get("budget_limit", None),
        prefer=profile.get("prefer"),
        avoid=profile.get("avoid", []),
    )


@router.post("/profile/{user_id}/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(user_id: str, req: UserPreferencesRequest):
    """Set or update routing preferences (e.g. prefer='code_heavy', avoid=['gemini'])."""
    updated = profile_manager.set_preferences(
        user_id=user_id,
        prefer=req.prefer,
        avoid=req.avoid,
    )
    return UserPreferencesResponse(
        status="updated",
        user_id=user_id,
        prefer=updated.get("prefer"),
        avoid=updated.get("avoid", []),
    )


@router.get("/ratelimit/{user_id}")
async def get_ratelimit_status(user_id: str):
    """Inspect current sliding window rate limit usage and remaining quota."""
    profile = profile_manager.get_or_create(user_id)
    tier = profile.get("tier", "free")
    limit = rate_limiter.get_limit_for_tier(tier)
    usage = rate_limiter.get_usage(user_id, tier=tier)
    remaining = None if limit is None else max(0, limit - usage)
    return {
        "user_id": user_id,
        "tier": tier,
        "limit_per_minute": limit,
        "current_window_usage": usage,
        "remaining_in_window": remaining,
        "window_seconds": rate_limiter.window_seconds,
    }


@router.get("/route/explain", response_model=ExplainResponse)
async def explain_route(
    user_id: str = Query(..., description="Unique user identifier"),
    prompt: str = Query(..., description="Prompt to analyze and explain"),
):
    """Routing Explainability Endpoint.
    Returns: predicted task class, top 3 candidate models with Q-values,
    estimated cost, and human-readable selection reasoning.
    """
    # 1. Security screening
    is_safe, threat_type = prompt_filter.check(prompt, user_id=user_id)
    if not is_safe:
        logger.warning("Blocked prompt in explain route", user_id=user_id, threat=threat_type)
        raise HTTPException(status_code=400, detail=f"Prompt blocked: {threat_type}")

    # 2. Retrieve user profile
    profile = profile_manager.get_or_create(user_id)

    # 3. Dry-run cache preview
    cached = await cache.get(prompt)
    cache_prediction = {
        "will_cache_hit": cached is not None,
        "cache_type": cached.get("cache_type", "miss") if cached else "miss",
        "cached_model": cached.get("model_used") if cached else None,
        "similarity": cached.get("similarity") if cached else None,
    }

    # 4. Generate decision explanation
    explanation = rl_router.explain(user_id=user_id, prompt=prompt, profile=profile)
    explanation["cache_prediction"] = cache_prediction

    return ExplainResponse(**explanation)


@router.get("/models/stats")
async def get_model_stats(task_type: Optional[str] = Query(None, description="Optional task filter (code, security, creative, general)")):
    """Return current RL Q-values and stats for all models (global or per-task)."""
    return rl_router.get_stats(task_type=task_type)


@router.post("/prompt/complexity", response_model=ComplexityScoreResponse)
async def score_prompt_complexity(req: ComplexityScoreRequest):
    """Score prompt complexity across token count, information entropy, and question depth.
    Indicates whether the prompt qualifies for simple-path cheapest model bypass.
    """
    res = complexity_scorer.score(req.prompt)
    profile = profile_manager.get_or_create(req.user_id) if req.user_id else None
    task_type = rl_router.classifier.classify(req.prompt)
    available_models = get_models_for_profile(profile, task_type) if profile else None
    cheapest = None
    if available_models:
        cheapest = get_cheapest_capable_model(
            available_models,
            task_type=task_type,
            profile=profile,
            q_values=rl_router.task_q_values.get(task_type, {}),
        )
    return ComplexityScoreResponse(
        prompt=req.prompt,
        score=res["score"],
        is_simple=res["is_simple"],
        classification=res["classification"],
        threshold=res["threshold"],
        token_count=res["token_count"],
        entropy=res["entropy"],
        question_depth=res["question_depth"],
        signals=res["signals"],
        cheapest_model=cheapest,
    )


@router.get("/prompt/complexity", response_model=ComplexityScoreResponse)
async def score_prompt_complexity_get(
    prompt: str = Query(..., description="Prompt text to evaluate complexity for"),
    user_id: str = Query("anonymous", description="User ID"),
):
    """GET endpoint for prompt complexity scoring."""
    return await score_prompt_complexity(ComplexityScoreRequest(prompt=prompt, user_id=user_id))


@router.get("/health")
async def health():
    return {"status": "ok"}
