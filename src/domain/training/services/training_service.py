"""
Training Service — Domain Service.

Orchestrates the training workflow: job creation, execution coordination,
and result handling. Pure domain logic; no infrastructure dependencies.
"""

import logging

from src.domain.training.entities.training_config import TrainingConfig
from src.domain.training.entities.training_job import TrainingJob, TrainingMetrics
from src.domain.training.events.training_completed import TrainingCompleted
from src.domain.training.repositories.training_repository import TrainingRepository

logger = logging.getLogger(__name__)


class TrainingService:
    """
    Domain Service for orchestrating model training.

    Manages job lifecycle, validates configurations, and emits domain
    events on completion. Actual model training is delegated to an
    infrastructure-level trainer passed as a callback.
    """

    def __init__(self, repository: TrainingRepository) -> None:
        self._repository = repository
        self._event_log: list[TrainingCompleted] = []

    async def create_job(self, model_id: str, config: TrainingConfig) -> TrainingJob:
        """
        Create and persist a new training job.

        Returns the job in PENDING state.
        """
        if not model_id:
            raise ValueError("model_id cannot be empty")
        if not config.dataset_path:
            raise ValueError("dataset_path cannot be empty")

        job = TrainingJob(model_id=model_id, config=config)
        await self._repository.save(job)
        logger.info("Created training job %s for model %s", job.job_id, model_id)
        return job

    async def start_job(self, job_id: str) -> TrainingJob:
        """Transition a job to RUNNING state."""
        job = await self._repository.get_by_id(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        job.start()
        await self._repository.save(job)
        logger.info("Started training job %s", job_id)
        return job

    async def complete_job(
        self,
        job_id: str,
        metrics: TrainingMetrics,
        artifact_path: str,
    ) -> TrainingJob:
        """Mark a job as completed with results and emit a domain event."""
        job = await self._repository.get_by_id(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        job.complete(metrics=metrics, artifact_path=artifact_path)
        await self._repository.save(job)

        event = TrainingCompleted(
            job_id=job.job_id,
            model_id=job.model_id,
            model_artifact_path=artifact_path,
            accuracy=metrics.accuracy,
            f1_score=metrics.f1_score,
            duration_seconds=job.duration_seconds or 0.0,
        )
        self._event_log.append(event)
        logger.info(
            "Training job %s completed — accuracy=%.4f, f1=%.4f",
            job_id,
            metrics.accuracy,
            metrics.f1_score,
        )
        return job

    async def fail_job(self, job_id: str, error: str) -> TrainingJob:
        """Mark a job as failed."""
        job = await self._repository.get_by_id(job_id)
        if job is None:
            raise ValueError(f"Training job {job_id} not found")

        job.fail(error)
        await self._repository.save(job)
        logger.warning("Training job %s failed: %s", job_id, error)
        return job

    async def get_history(self, model_id: str, limit: int = 10) -> list[TrainingJob]:
        """Retrieve training history for a model."""
        return await self._repository.get_by_model_id(model_id, limit)

    @property
    def events(self) -> list[TrainingCompleted]:
        """Return emitted domain events (for testing / event bus integration)."""
        return list(self._event_log)
