"""Audit logger — track all state-changing actions for security compliance.

Logs: who (user/IP), what (action), when (timestamp), where (resource),
      result (success/failure), and correlation ID.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    """Auditable actions in the system."""

    # Auth
    LOGIN = "auth.login"
    LOGIN_FAILED = "auth.login_failed"
    REGISTER = "auth.register"
    TOKEN_REFRESH = "auth.token_refresh"

    # Model
    MODEL_DEPLOY = "model.deploy"
    MODEL_ROLLBACK = "model.rollback"
    MODEL_STAGE_CHANGE = "model.stage_change"

    # Data
    DATA_INGEST = "data.ingest"
    DATA_VALIDATE = "data.validate"

    # Config
    CONFIG_CHANGE = "config.change"

    # Prediction
    PREDICTION = "prediction.request"
    BATCH_PREDICTION = "prediction.batch"


@dataclass
class AuditEntry:
    """Single audit log entry."""

    action: str
    user: str  # username or "anonymous"
    ip_address: str
    resource: str  # what was acted upon
    timestamp: str = ""
    success: bool = True
    details: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(UTC).isoformat()


class AuditLogger:
    """Structured audit logging for compliance and security.

    Writes to:
    1. Python logger (goes through structured JSON logging)
    2. Local audit file (append-only, for offline analysis)
    """

    def __init__(
        self,
        audit_file: str = "logs/audit.jsonl",
    ) -> None:
        self._audit_file = Path(audit_file)
        self._audit_logger = logging.getLogger("phoenix.audit")

    def log(
        self,
        action: AuditAction,
        user: str = "anonymous",
        ip_address: str = "unknown",
        resource: str = "",
        success: bool = True,
        details: dict[str, Any] | None = None,
        correlation_id: str = "",
    ) -> AuditEntry:
        """Record an auditable action."""
        entry = AuditEntry(
            action=action.value,
            user=user,
            ip_address=ip_address,
            resource=resource,
            success=success,
            details=details or {},
            correlation_id=correlation_id,
        )

        # Log to structured logger
        self._audit_logger.info(
            "AUDIT: %s by %s on %s [%s]",
            entry.action,
            entry.user,
            entry.resource,
            "OK" if entry.success else "FAIL",
            extra={"audit": asdict(entry)},
        )

        # Append to audit file
        self._write_to_file(entry)

        return entry

    def _write_to_file(self, entry: AuditEntry) -> None:
        """Append audit entry to JSONL file."""
        try:
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_file.open("a") as f:
                f.write(json.dumps(asdict(entry), default=str) + "\n")
        except Exception as e:
            logger.warning("Failed to write audit log: %s", e)

    def query(
        self,
        action: str | None = None,
        user: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query recent audit entries from file."""
        if not self._audit_file.exists():
            return []
        results: list[dict[str, Any]] = []
        for line in self._audit_file.read_text().strip().split("\n"):
            if not line:
                continue
            try:
                entry = json.loads(line)
                if action and entry.get("action") != action:
                    continue
                if user and entry.get("user") != user:
                    continue
                results.append(entry)
            except json.JSONDecodeError:
                continue
        return results[-limit:]


# Global singleton
audit_logger = AuditLogger()
