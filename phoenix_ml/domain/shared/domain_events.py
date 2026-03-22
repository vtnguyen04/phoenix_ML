"""
Domain Events — Typed event dataclasses for the Observer Pattern.

Each event represents something that HAPPENED in the domain.
Handlers emit events; subscribers react independently.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class PredictionCompleted:
    """Emitted after a successful prediction."""

    model_id: str
    version: str
    latency: float
    confidence: float
    status: str  # "success" | "error"
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class DriftDetected:
    """Emitted when statistical drift is detected."""

    model_id: str
    feature_name: str
    score: float
    method: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class DriftScorePublished:
    """Emitted after every drift check (detected or not)."""

    model_id: str
    feature_name: str
    method: str
    score: float


@dataclass(frozen=True)
class ModelRetrained:
    """Emitted after a model retraining completes."""

    model_id: str
    version: str
    metrics: dict[str, float]
    promoted: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
