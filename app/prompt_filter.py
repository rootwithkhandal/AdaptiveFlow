"""Security layer: detects adversarial / malicious prompts via regex.
Includes dual-backend structured audit logging (SQLite + append-only JSONL)
capturing what was flagged, why, by whom, and exact timestamps for compliance.
"""
import re
import os
import json
import time
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict, Any

from app.logger import get_logger

logger = get_logger(__name__)

AUDIT_LOG_FILE = "data/security_audit.jsonl"
SQLITE_DB_PATH = "data/security_audit.db"
os.makedirs("data", exist_ok=True)

# Adversarial patterns to block
ADVERSARIAL_PATTERNS = [
    (r"\bignore\s+(?:(?:all|previous|prior)\s+)*(?:instructions?|prompts?|rules?)\b", "prompt_injection"),
    (r"\bjailbreak\b", "jailbreak"),
    (r"\bdan mode\b", "jailbreak"),
    (r"\bact as (an? )?(unrestricted|evil|malicious|hacker)\b", "jailbreak"),
    (r"\b(rm -rf|format c:|del /f|drop table|exec\(|eval\(|__import__)\b", "code_injection"),
    (r"\b(ssn|social security number|credit card|cvv|bank account)\b", "pii_extraction"),
    (r"(https?://[^\s]+\.(exe|bat|sh|ps1))", "malicious_url"),
]


class PromptFilter:
    """Security filter with dual-backend structured compliance audit logging."""

    def __init__(
        self,
        audit_log_path: Optional[str] = None,
        sqlite_db_path: Optional[str] = None,
        enable_sqlite: bool = True,
        patterns: Optional[List[Tuple[str, str]]] = None,
    ):
        self.audit_log_path = audit_log_path or AUDIT_LOG_FILE
        self.sqlite_db_path = sqlite_db_path or SQLITE_DB_PATH
        self.enable_sqlite = enable_sqlite
        self.patterns = patterns or ADVERSARIAL_PATTERNS
        self._lock = threading.Lock()

        if self.enable_sqlite:
            self._init_sqlite()

    def _init_sqlite(self):
        """Initialize SQLite audit log table and indexes."""
        try:
            with self._lock:
                if self.sqlite_db_path:
                    db_dir = os.path.dirname(self.sqlite_db_path)
                    if db_dir:
                        os.makedirs(db_dir, exist_ok=True)
                    with sqlite3.connect(self.sqlite_db_path) as conn:
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS security_audit (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT NOT NULL,
                                epoch REAL NOT NULL,
                                user_id TEXT NOT NULL,
                                threat_type TEXT NOT NULL,
                                matched_pattern TEXT NOT NULL,
                                prompt_snippet TEXT NOT NULL,
                                full_prompt TEXT NOT NULL,
                                prompt_length INTEGER NOT NULL,
                                action_taken TEXT NOT NULL
                            )
                        """)
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_user ON security_audit(user_id)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_threat ON security_audit(threat_type)")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON security_audit(timestamp)")
                        conn.commit()
        except Exception as e:
            logger.error("Failed to initialize SQLite security audit db", error=str(e))

    def _log_audit_event(self, threat: str, pattern: str, prompt: str, user_id: Optional[str] = None):
        """Append an audit record for security forensics and compliance.
        
        Captures:
        - Timestamp (ISO 8601 UTC & epoch)
        - By whom (user_id)
        - Why (threat_type and matched_pattern)
        - What was flagged (prompt_snippet, full_prompt, prompt_length)
        - Action taken (blocked)
        """
        now = datetime.now(timezone.utc)
        record = {
            "timestamp": now.isoformat(),
            "epoch": time.time(),
            "user_id": user_id or "anonymous",
            "threat_type": threat,
            "matched_pattern": pattern,
            "prompt_snippet": prompt[:120],
            "full_prompt": prompt,
            "prompt_length": len(prompt),
            "action_taken": "blocked",
        }

        # 1. Append to append-only JSONL file
        if self.audit_log_path:
            try:
                log_dir = os.path.dirname(self.audit_log_path)
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error("Failed to write security audit JSONL file", error=str(e))

        # 2. Insert into SQLite table
        if self.enable_sqlite and self.sqlite_db_path:
            try:
                with self._lock:
                    with sqlite3.connect(self.sqlite_db_path) as conn:
                        conn.execute("""
                            INSERT INTO security_audit (
                                timestamp, epoch, user_id, threat_type,
                                matched_pattern, prompt_snippet, full_prompt,
                                prompt_length, action_taken
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            record["timestamp"],
                            record["epoch"],
                            record["user_id"],
                            record["threat_type"],
                            record["matched_pattern"],
                            record["prompt_snippet"],
                            record["full_prompt"],
                            record["prompt_length"],
                            record["action_taken"],
                        ))
                        conn.commit()
            except Exception as e:
                logger.error("Failed to insert security audit event into SQLite", error=str(e))

    def check(self, prompt: str, user_id: Optional[str] = None) -> Tuple[bool, str]:
        """Returns (is_safe, threat_type). is_safe=False means block request."""
        lower = prompt.lower()

        for pattern, threat in self.patterns:
            if re.search(pattern, lower):
                logger.warning("Adversarial prompt detected", threat=threat, user_id=user_id)
                self._log_audit_event(threat=threat, pattern=pattern, prompt=prompt, user_id=user_id)
                return False, threat

        return True, "none"

    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        user_id: Optional[str] = None,
        threat_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve structured audit records with filtering for compliance reporting."""
        # Try SQLite first if enabled
        if self.enable_sqlite and self.sqlite_db_path and os.path.exists(self.sqlite_db_path):
            try:
                with self._lock:
                    with sqlite3.connect(self.sqlite_db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        query = "SELECT * FROM security_audit WHERE 1=1"
                        params = []
                        if user_id:
                            query += " AND user_id = ?"
                            params.append(user_id)
                        if threat_type:
                            query += " AND threat_type = ?"
                            params.append(threat_type)
                        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
                        params.extend([limit, offset])

                        cursor = conn.execute(query, params)
                        return [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.warning("SQLite audit read failed, falling back to JSONL", error=str(e))

        # Fallback to JSONL file
        results = []
        if self.audit_log_path and os.path.exists(self.audit_log_path):
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for line in reversed(lines):
                        rec = json.loads(line)
                        if user_id and rec.get("user_id") != user_id:
                            continue
                        if threat_type and rec.get("threat_type") != threat_type:
                            continue
                        results.append(rec)
                        if len(results) >= offset + limit:
                            break
                return results[offset:offset + limit]
            except Exception as e:
                logger.error("JSONL audit read failed", error=str(e))

        return results

    def get_audit_stats(self) -> Dict[str, Any]:
        """Aggregate compliance summary: total flagged, by threat type, top violators."""
        logs = self.get_audit_logs(limit=1000)
        by_threat: Dict[str, int] = {}
        by_user: Dict[str, int] = {}

        for l in logs:
            tt = l.get("threat_type", "unknown")
            uid = l.get("user_id", "anonymous")
            by_threat[tt] = by_threat.get(tt, 0) + 1
            by_user[uid] = by_user.get(uid, 0) + 1

        return {
            "total_flagged_events": len(logs),
            "threat_breakdown": by_threat,
            "top_flagged_users": dict(sorted(by_user.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
