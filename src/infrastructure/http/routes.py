import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.application.handlers.retrain_handler import RetrainHandler
from src.application.services.monitoring_service import MonitoringService
from src.config import get_settings
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.entities.drift_report import DriftReport
from src.infrastructure.http.container import (
    drift_calculator,
    find_project_root,
    kafka_producer,
    model_evaluator,
)
from src.infrastructure.http.dependencies import get_predict_handler
from src.infrastructure.persistence.database import get_db
from src.infrastructure.persistence.postgres_drift_repo import (
    PostgresDriftReportRepository,
)
from src.infrastructure.persistence.postgres_log_repo import (
    PostgresPredictionLogRepository,
)
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    prediction_id: str
    ground_truth: int


async def _log_prediction_background(
    command: PredictCommand,
    prediction: Prediction,
    prediction_id: str,
    db: AsyncSession,
) -> None:
    try:
        repo = PostgresPredictionLogRepository(db)
        await repo.log(command, prediction)
    except Exception as e:
        logger.error("⚠️ DB Log failed: %s", e)

    try:
        event = {
            "event_id": prediction_id,
            "model_id": prediction.model_id,
            "version": prediction.model_version,
            "features": command.features,
            "result": prediction.result,
            "confidence": prediction.confidence.value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        await kafka_producer.publish("inference-events", event)
    except Exception as e:
        logger.error("⚠️ Kafka Log failed: %s", e)


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": get_settings().APP_VERSION}


@router.post("/predict")
async def predict(
    command: PredictCommand,
    background_tasks: BackgroundTasks,
    handler: PredictHandler = Depends(get_predict_handler),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    try:
        prediction = await handler.execute(command)
        prediction_id = str(uuid.uuid4())
        background_tasks.add_task(
            _log_prediction_background, command, prediction, prediction_id, db
        )
        return {
            "prediction_id": prediction_id,
            "model_id": prediction.model_id,
            "version": prediction.model_version,
            "result": prediction.result,
            "confidence": {"value": prediction.confidence.value},
            "latency_ms": round(prediction.latency_ms, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/feedback")
async def feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, str]:
    try:
        repo = PostgresPredictionLogRepository(db)
        await repo.update_ground_truth(request.prediction_id, request.ground_truth)
        return {"status": "feedback_received"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/monitoring/drift/{model_id}")
async def check_drift(
    model_id: str,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DriftReport:
    try:
        log_repo = PostgresPredictionLogRepository(db)
        drift_repo = PostgresDriftReportRepository(db)
        model_repo = PostgresModelRegistry(db)
        root = find_project_root()
        rh = RetrainHandler(root, model_repo, model_evaluator)
        ms = MonitoringService(log_repo, drift_calculator, drift_repo, rh)

        mock_reference_data = [float(x) for x in range(100)]
        return await ms.check_drift(
            model_id=model_id, reference_data=mock_reference_data, feature_index=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
