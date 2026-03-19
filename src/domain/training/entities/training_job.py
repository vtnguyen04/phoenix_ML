"""
Training Job Entity — Aggregate Root for the Training Bounded Context.

Represents a single training job with its lifecycle states.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.training.entities.training_config import TrainingConfig


class TrainingStatus(Enum):
    """Lifecycle stages for a training job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingMetrics:
    """Value object for training evaluation metrics."""

    accuracy: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    loss: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """
    Aggregate Root for training operations.

    Encapsulates the complete lifecycle of a model training run,
    from creation through execution to completion or failure.
    """

    model_id: str
    config: "TrainingConfig"
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TrainingStatus = TrainingStatus.PENDING
    metrics: TrainingMetrics | None = None
    model_artifact_path: str | None = None
    error_message: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def start(self) -> None:
        """Transition to RUNNING state."""
        if self.status != TrainingStatus.PENDING:
            raise ValueError(
                f"Cannot start job in {self.status.value} state; must be PENDING"
            )
        self.status = TrainingStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, metrics: TrainingMetrics, artifact_path: str) -> None:
        """Transition to COMPLETED with results."""
        if self.status != TrainingStatus.RUNNING:
            raise ValueError(
                f"Cannot complete job in {self.status.value} state; must be RUNNING"
            )
        self.status = TrainingStatus.COMPLETED
        self.metrics = metrics
        self.model_artifact_path = artifact_path
        self.completed_at = datetime.now(UTC)

    def fail(self, error: str) -> None:
        """Transition to FAILED with error message."""
        if self.status not in (TrainingStatus.PENDING, TrainingStatus.RUNNING):
            raise ValueError(
                f"Cannot fail job in {self.status.value} state"
            )
        self.status = TrainingStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now(UTC)

    def cancel(self) -> None:
        """Cancel a pending or running job."""
        if self.status not in (TrainingStatus.PENDING, TrainingStatus.RUNNING):
            raise ValueError(
                f"Cannot cancel job in {self.status.value} state"
            )
        self.status = TrainingStatus.CANCELLED
        self.completed_at = datetime.now(UTC)

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed time for the job, or None if not started."""
        if self.started_at is None:
            return None
        end = self.completed_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()

    @property
    def is_terminal(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self.status in (
            TrainingStatus.COMPLETED,
            TrainingStatus.FAILED,
            TrainingStatus.CANCELLED,
        )


