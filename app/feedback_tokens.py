"""Short-lived, single-use feedback capability tokens."""
import base64
import hashlib
import hmac
import json
import secrets
import time
from threading import Lock
from typing import Any, Dict

from app.config import settings


class FeedbackTokenManager:
    def __init__(self):
        self._redeemed: Dict[str, float] = {}
        self._lock = Lock()

    def issue(self, user_id: str, model: str, task_type: str) -> str:
        payload = {"u": user_id, "m": model, "t": task_type, "exp": int(time.time()) + settings.feedback_token_ttl_seconds, "jti": secrets.token_urlsafe(16)}
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        sig = hmac.new(settings.secret_key.encode(), raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=") + "." + base64.urlsafe_b64encode(sig).decode().rstrip("=")

    def validate_and_redeem(self, token: str, user_id: str, model: str, task_type: str | None) -> bool:
        try:
            payload_part, signature_part = token.split(".", 1)
            raw = base64.urlsafe_b64decode(payload_part + "=" * (-len(payload_part) % 4))
            supplied = base64.urlsafe_b64decode(signature_part + "=" * (-len(signature_part) % 4))
            expected = hmac.new(settings.secret_key.encode(), raw, hashlib.sha256).digest()
            payload: Dict[str, Any] = json.loads(raw)
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if not hmac.compare_digest(supplied, expected):
            return False
        if payload.get("exp", 0) < time.time() or payload.get("u") != user_id or payload.get("m") != model:
            return False
        if task_type is not None and payload.get("t") != task_type:
            return False
        with self._lock:
            now = time.time()
            self._redeemed = {jti: exp for jti, exp in self._redeemed.items() if exp >= now}
            jti = payload.get("jti")
            if not jti or jti in self._redeemed:
                return False
            self._redeemed[jti] = payload["exp"]
        return True


feedback_tokens = FeedbackTokenManager()
