from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class PredictionMade:
    """
    Domain Event triggered when a prediction is successfully generated.
    """
    prediction_id: str
    model_id: str
    model_version: str
    entity_id: str | None
    features: dict[str, Any] | None
    result: Any
    confidence: float
    latency_ms: float
    occurred_at: datetime = datetime.now(UTC)
