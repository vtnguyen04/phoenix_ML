import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.commands.predict_command import PredictCommand
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.infrastructure.persistence.models import PredictionLogORM


class PostgresPredictionLogRepository(PredictionLogRepository):
    """
    Persistent implementation of PredictionLogRepository using PostgreSQL.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def log(self, command: PredictCommand, prediction: Prediction) -> None:
        orm_log = PredictionLogORM(
            id=str(uuid.uuid4()),
            model_id=prediction.model_id,
            model_version=prediction.model_version,
            features=command.features or [],
            result=prediction.result, # type: ignore
            confidence=prediction.confidence.value,
            latency_ms=prediction.latency_ms,
            created_at=datetime.now(UTC)
        )
        self._session.add(orm_log)
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
        rows = result.scalars().all()
        
        # Convert ORM back to Domain/Command
        # (Simplified conversion for monitoring service)
        return [
            (
                PredictCommand(
                    model_id=row.model_id, 
                    model_version=row.model_version, 
                    features=row.features
                ),
                Prediction(
                    model_id=row.model_id,
                    model_version=row.model_version,
                    result=row.result,
                    confidence=Any, # Not used in drift calc
                    latency_ms=row.latency_ms
                )
            )
            for row in rows
        ]
