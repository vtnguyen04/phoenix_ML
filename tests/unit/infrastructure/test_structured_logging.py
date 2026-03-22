"""Tests for structured logging and correlation middleware."""

import json
import logging

import pytest

from src.infrastructure.http.middleware.correlation_middleware import correlation_id_var
from src.infrastructure.logging.logging_config import JSONFormatter


@pytest.fixture
def json_formatter() -> JSONFormatter:
    return JSONFormatter()


class TestJSONFormatter:
    def test_produces_valid_json(self, json_formatter: JSONFormatter) -> None:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert "timestamp" in parsed

    def test_includes_correlation_id(self, json_formatter: JSONFormatter) -> None:
        token = correlation_id_var.set("req-12345")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="with cid",
                args=None,
                exc_info=None,
            )
            output = json_formatter.format(record)
            parsed = json.loads(output)
            assert parsed["correlation_id"] == "req-12345"
        finally:
            correlation_id_var.reset(token)

    def test_no_correlation_id_when_empty(self, json_formatter: JSONFormatter) -> None:
        token = correlation_id_var.set("")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.WARNING,
                pathname="",
                lineno=0,
                msg="no cid",
                args=None,
                exc_info=None,
            )
            output = json_formatter.format(record)
            parsed = json.loads(output)
            assert "correlation_id" not in parsed
        finally:
            correlation_id_var.reset(token)

    def test_includes_exception_info(self, json_formatter: JSONFormatter) -> None:
        try:
            raise ValueError("test error")
        except ValueError:
            import sys  # noqa: PLC0415

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="error occurred",
            args=None,
            exc_info=exc_info,
        )
        output = json_formatter.format(record)
        parsed = json.loads(output)
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "test error"


class TestCorrelationIdVar:
    def test_default_empty(self) -> None:
        token = correlation_id_var.set("")
        try:
            assert correlation_id_var.get() == ""
        finally:
            correlation_id_var.reset(token)

    def test_set_and_get(self) -> None:
        token = correlation_id_var.set("abc-123")
        try:
            assert correlation_id_var.get() == "abc-123"
        finally:
            correlation_id_var.reset(token)
