"""Admin endpoints for cost monitoring, user management, and model registration/warmup."""
from typing import Optional
from fastapi import APIRouter, Header, HTTPException, Query, Depends
from app.dependencies import rl_router, profile_manager, prompt_filter
from app.config import settings
from app.models.registry import register_model, MODEL_REGISTRY
from app.benchmark import run_model_warmup
from app.api.schemas import (
    ModelRegistrationRequest,
    ModelRegistrationResponse,
    ModelWarmupRequest,
    ModelWarmupResponse,
    UserPreferencesRequest,
    UserPreferencesResponse,
)

def _verify_admin(x_admin_key: str = Header(...)):
    if x_admin_key != settings.secret_key:
        raise HTTPException(status_code=403, detail="Invalid admin key")

admin_router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(_verify_admin)])


@admin_router.get("/costs")
async def all_user_costs():
    return profile_manager.get_all_costs()


@admin_router.post("/users/{user_id}/tier")
async def set_user_tier(user_id: str, tier: str, budget: float = None):
    profile_manager.set_tier(user_id, tier, budget)
    return {"status": "updated", "user_id": user_id, "tier": tier}


@admin_router.post("/users/{user_id}/preferences", response_model=UserPreferencesResponse)
async def admin_set_user_preferences(
    user_id: str,
    req: UserPreferencesRequest,
):
    """Admin endpoint to set or override user routing hints/persona preferences."""
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


@admin_router.post("/users/{user_id}/flag")
async def flag_user(user_id: str):
    profile_manager.flag_high_risk(user_id)
    return {"status": "flagged", "user_id": user_id}


@admin_router.get("/rl/stats")
async def rl_stats():
    return rl_router.get_stats()


@admin_router.post("/models/register", response_model=ModelRegistrationResponse)
async def register_new_model(req: ModelRegistrationRequest):
    """Dynamically registers a new model and auto-runs synthetic benchmark warm-up."""
    config = {
        "provider": req.provider,
        "cost_per_1k_tokens": req.cost_per_1k_tokens,
        "tasks": req.tasks,
        "fallback": req.fallback,
    }
    register_model(req.model_name, config, req.tiers)

    seeded_weights = None
    warmup_card = None
    if req.auto_warmup:
        warmup_card = await run_model_warmup(req.model_name, task_types=req.tasks)
        seeded_weights = rl_router.seed_model_weights(req.model_name, warmup_card)

    return ModelRegistrationResponse(
        status="registered",
        model=req.model_name,
        warmed_up=req.auto_warmup,
        seeded_weights=seeded_weights,
        warmup_scorecard=warmup_card,
    )


@admin_router.post("/models/warmup", response_model=ModelWarmupResponse)
async def warmup_existing_model(req: ModelWarmupRequest):
    """Runs synthetic benchmarks on an existing model and re-seeds its RL weights."""
    if req.model_name not in MODEL_REGISTRY and req.model_name not in rl_router.q_values:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' not found in registry.")

    warmup_card = await run_model_warmup(req.model_name, task_types=req.task_types)
    seeded_weights = rl_router.seed_model_weights(req.model_name, warmup_card)

    return ModelWarmupResponse(
        status="warmed_up",
        model=req.model_name,
        seeded_weights=seeded_weights,
        warmup_scorecard=warmup_card,
    )


@admin_router.get("/security/audit")
async def get_security_audit_logs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    threat_type: Optional[str] = Query(None, description="Filter by threat type"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Retrieve structured security audit logs for compliance inspections."""
    return prompt_filter.get_audit_logs(
        limit=limit,
        offset=offset,
        user_id=user_id,
        threat_type=threat_type,
    )


@admin_router.get("/security/stats")
async def get_security_audit_stats():
    """Summary of all security events: counts by threat type and top flagged users."""
    return prompt_filter.get_audit_stats()

