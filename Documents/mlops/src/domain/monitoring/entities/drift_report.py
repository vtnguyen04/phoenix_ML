from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class DriftReport(BaseModel):
    """
    Value Object representing the result of a drift detection check.
    """

    model_config = ConfigDict(frozen=True)

    feature_name: str
    drift_detected: bool
    p_value: float
    statistic: float
    threshold: float
    analyzed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
