"""Domain event emitted when a training job completes successfully."""

from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass(frozen=True)
class TrainingCompleted:
    """
    Domain event raised when a training job completes successfully.

    Consumers can use this to trigger model registration, A/B test setup,
    or notification workflows.
    """

    job_id: str
    model_id: str
    model_artifact_path: str
    accuracy: float
    f1_score: float
    duration_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
