import uuid
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.confidence_score import ConfidenceScore
from phoenix_ml.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from phoenix_ml.infrastructure.persistence.models import PredictionLogORM


class PostgresPredictionLogRepository(PredictionLogRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log(self, command: PredictCommand, prediction: Prediction) -> None:
        orm = PredictionLogORM(
            id=str(uuid.uuid4()),
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            features=command.features if command.features else [],
            result=prediction.result,
            confidence=prediction.confidence.value,
            latency_ms=prediction.latency_ms,
            created_at=datetime.now(UTC),
        )
        self._session.add(orm)
        await self._session.commit()

    async def update_ground_truth(
        self, prediction_id: str, ground_truth: int | float | str
    ) -> None:
        stmt = (
            update(PredictionLogORM)
            .where(PredictionLogORM.id == prediction_id)
            .values(ground_truth=ground_truth)
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get_recent_logs(
        self, model_id: str, limit: int = 1000
    ) -> list[tuple[PredictCommand, Prediction]]:
        query = (
            select(PredictionLogORM)
            .where(PredictionLogORM.model_id == model_id)
            .order_by(PredictionLogORM.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(query)
        orms = result.scalars().all()

        return [
            (
                PredictCommand(
                    model_id=o.model_id,
                    model_version=o.model_version,
                    features=o.features,
                ),
                Prediction(
                    model_id=o.model_id,
                    model_version=o.model_version,
                    result=o.result,
                    confidence=ConfidenceScore(value=float(o.confidence)),
                    latency_ms=o.latency_ms,
                ),
            )
            for o in orms
        ]
