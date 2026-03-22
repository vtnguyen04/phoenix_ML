"""Tests for audit logger."""

import json
from pathlib import Path

import pytest

from src.infrastructure.logging.audit_logger import (
    AuditAction,
    AuditLogger,
)


@pytest.fixture
def audit(tmp_path: Path) -> AuditLogger:
    return AuditLogger(audit_file=str(tmp_path / "audit.jsonl"))


class TestAuditLogger:
    def test_log_creates_entry(self, audit: AuditLogger) -> None:
        entry = audit.log(
            AuditAction.LOGIN,
            user="admin",
            ip_address="127.0.0.1",
            resource="/api/v1/auth/login",
        )
        assert entry.action == "auth.login"
        assert entry.user == "admin"
        assert entry.success is True
        assert entry.timestamp != ""

    def test_log_writes_to_file(self, audit: AuditLogger) -> None:
        audit.log(AuditAction.PREDICTION, user="user1", resource="model-a")
        audit.log(AuditAction.DATA_INGEST, user="user2", resource="data.csv")

        content = Path(audit._audit_file).read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        entry0 = json.loads(lines[0])
        assert entry0["action"] == "prediction.request"

    def test_query_by_action(self, audit: AuditLogger) -> None:
        audit.log(AuditAction.LOGIN, user="u1", resource="r1")
        audit.log(AuditAction.PREDICTION, user="u2", resource="r2")
        audit.log(AuditAction.LOGIN, user="u3", resource="r3")

        results = audit.query(action="auth.login")
        assert len(results) == 2

    def test_query_by_user(self, audit: AuditLogger) -> None:
        audit.log(AuditAction.LOGIN, user="alice", resource="r1")
        audit.log(AuditAction.PREDICTION, user="bob", resource="r2")

        results = audit.query(user="alice")
        assert len(results) == 1
        assert results[0]["user"] == "alice"

    def test_query_limit(self, audit: AuditLogger) -> None:
        for i in range(10):
            audit.log(AuditAction.PREDICTION, user=f"u{i}", resource="r")

        results = audit.query(limit=3)
        assert len(results) == 3

    def test_failed_action(self, audit: AuditLogger) -> None:
        entry = audit.log(
            AuditAction.LOGIN_FAILED,
            user="hacker",
            ip_address="10.0.0.1",
            resource="/auth/login",
            success=False,
            details={"reason": "invalid password"},
        )
        assert entry.success is False
        assert entry.details["reason"] == "invalid password"
