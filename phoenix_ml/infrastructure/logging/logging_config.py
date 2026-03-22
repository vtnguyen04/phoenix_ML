"""Structured JSON logging configuration with correlation ID support."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from phoenix_ml.infrastructure.http.middleware.correlation_middleware import correlation_id_var


class JSONFormatter(logging.Formatter):
    """JSON log formatter with correlation ID injection."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject correlation ID from context
        cid = correlation_id_var.get("")
        if cid:
            log_entry["correlation_id"] = cid

        # Add extra fields if present
        if hasattr(record, "model_id"):
            log_entry["model_id"] = record.model_id
        if hasattr(record, "latency_ms"):
            log_entry["latency_ms"] = record.latency_ms

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
            }

        return json.dumps(log_entry, default=str)


def configure_logging(
    level: str = "INFO",
    json_format: bool = True,
) -> None:
    """Configure root logger with JSON formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, use JSON format. If False, use standard format.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("uvicorn.access", "aiokafka", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
