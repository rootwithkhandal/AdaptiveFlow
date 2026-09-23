"""User profile management.
Stores tier, budget, usage stats, and risk flags.
"""
import json
import os
import tempfile
import threading
from typing import Dict, Any, Optional, List
from app.logger import get_logger

logger = get_logger(__name__)

PROFILES_FILE = "data/profiles.json"
os.makedirs("data", exist_ok=True)

DEFAULT_PROFILES = {
    "free": {"tier": "free", "budget_limit": 1.0, "high_risk": False},
    "pro": {"tier": "pro", "budget_limit": 50.0, "high_risk": False},
    "enterprise": {"tier": "enterprise", "budget_limit": None, "high_risk": False},
    "high_risk": {"tier": "free", "budget_limit": 0.0, "high_risk": True},
}


class UserProfileManager:
    def __init__(self):
        self._profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        if os.path.exists(PROFILES_FILE):
            try:
                with open(PROFILES_FILE) as f:
                    self._profiles = json.load(f)
            except Exception:
                self._profiles = {}

    def _save(self):
        try:
            directory = os.path.dirname(PROFILES_FILE) or "."
            with tempfile.NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as tmp:
                json.dump(self._profiles, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_name = tmp.name
            os.replace(tmp_name, PROFILES_FILE)
        except Exception as e:
            logger.warning("Profile save failed", error=str(e))

    def get_or_create(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self._profiles:
            self._profiles[user_id] = {
                "user_id": user_id,
                "tier": "free",
                "budget_limit": 1.0,
                "high_risk": False,
                "total_cost": 0.0,
                "request_count": 0,
                "model_usage": {},
                "prefer": None,
                "avoid": [],
            }
            self._save()
        else:
            # Ensure preference keys exist on legacy records
            if "prefer" not in self._profiles[user_id]:
                self._profiles[user_id]["prefer"] = None
            if "avoid" not in self._profiles[user_id]:
                self._profiles[user_id]["avoid"] = []
        return self._profiles[user_id]

    def set_preferences(
        self,
        user_id: str,
        prefer: Optional[str] = None,
        avoid: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Set user-defined routing hints (e.g. prefer='code_heavy', avoid=['gemini'])."""
        profile = self.get_or_create(user_id)
        if prefer is not None:
            profile["prefer"] = prefer
        if avoid is not None:
            profile["avoid"] = avoid
        self._save()
        logger.info("Updated user routing preferences", user_id=user_id, prefer=profile.get("prefer"), avoid=profile.get("avoid"))
        return profile

    def set_tier(self, user_id: str, tier: str, budget_limit: Optional[float] = None):
        profile = self.get_or_create(user_id)
        defaults = DEFAULT_PROFILES.get(tier, DEFAULT_PROFILES["free"])
        profile["tier"] = tier
        profile["high_risk"] = defaults["high_risk"]
        if budget_limit is not None:
            profile["budget_limit"] = budget_limit
        elif defaults["budget_limit"] is not None:
            profile["budget_limit"] = defaults["budget_limit"]
        self._save()

    def record_usage(self, user_id: str, cost: float, model: str):
        profile = self.get_or_create(user_id)
        profile["total_cost"] = round(profile.get("total_cost", 0.0) + cost, 6)
        profile["request_count"] = profile.get("request_count", 0) + 1
        usage = profile.setdefault("model_usage", {})
        usage[model] = usage.get(model, 0) + 1
        self._save()
        logger.debug("Usage recorded", user_id=user_id, cost=cost, model=model)

    def get_all_costs(self) -> Dict[str, float]:
        """Return total cost per user — for admin dashboard."""
        return {uid: p.get("total_cost", 0.0) for uid, p in self._profiles.items()}

    def flag_high_risk(self, user_id: str):
        profile = self.get_or_create(user_id)
        profile["high_risk"] = True
        self._save()
