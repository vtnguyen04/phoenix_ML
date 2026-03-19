"""
Training Repository — Abstract Interface.

Defines the contract for persisting and retrieving training jobs.
Concrete implementations live in the infrastructure layer.
"""

from abc import ABC, abstractmethod

from src.domain.training.entities.training_job import TrainingJob


class TrainingRepository(ABC):
    """
    Interface for training job persistence.

    Following DDD principles, this interface is defined in the domain layer
    and implemented by infrastructure adapters (PostgreSQL, in-memory, etc.).
    """

    @abstractmethod
    async def save(self, job: TrainingJob) -> None:
        """Persist a training job."""
        ...

    @abstractmethod
    async def get_by_id(self, job_id: str) -> TrainingJob | None:
        """Retrieve a training job by ID."""
        ...

    @abstractmethod
    async def get_by_model_id(
        self, model_id: str, limit: int = 10
    ) -> list[TrainingJob]:
        """Retrieve training jobs for a given model, most recent first."""
        ...

    @abstractmethod
    async def get_latest_completed(self, model_id: str) -> TrainingJob | None:
        """Retrieve the most recent completed training job for a model."""
        ...
