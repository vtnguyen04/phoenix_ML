from datetime import datetime, timezone
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
from typing import Any

from src.domain.inference.value_objects.confidence_score import ConfidenceScore

class Prediction(BaseModel):
    """
    Entity representing a single prediction result.
    """
    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    model_id: str
    model_version: str
    result: Any
    confidence: ConfidenceScore
    latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_confident(self, threshold: float) -> bool:
        return self.confidence.value >= threshold