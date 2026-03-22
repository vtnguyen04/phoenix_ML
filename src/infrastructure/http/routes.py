import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.application.handlers.query_handlers import (
    GetDriftReportQueryHandler,
    GetModelPerformanceQueryHandler,
    GetModelQueryHandler,
)
from src.application.queries import (
    GetDriftReportQuery,
    GetModelPerformanceQuery,
    GetModelQuery,
)
from src.application.services.monitoring_service import MonitoringService
from src.config import get_settings
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.entities.drift_report import DriftReport
from src.infrastructure.bootstrap.container import (
    drift_calculator,
    event_bus,
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
    ground_truth: int | float | str


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


@router.get("/health/deep")
async def deep_health_check() -> dict[str, Any]:
    """Deep health check — verify all infrastructure dependencies."""
    from src.infrastructure.monitoring.health_check import HealthChecker  # noqa: PLC0415

    checker = HealthChecker()
    report = await checker.check_all()
    return report.to_dict()



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


class BatchPredictRequest(BaseModel):
    model_id: str
    batch: list[list[float]]
    model_version: str | None = None
    entity_ids: list[str] | None = None


@router.post("/predict/batch")
async def batch_predict(
    request: BatchPredictRequest,
    handler: PredictHandler = Depends(get_predict_handler),  # noqa: B008
) -> dict[str, Any]:
    """Batch prediction — process multiple inputs in one request."""
    from src.application.commands.batch_predict_command import (  # noqa: PLC0415
        BatchPredictCommand,
    )
    from src.application.handlers.batch_predict_handler import (  # noqa: PLC0415
        BatchPredictHandler,
    )

    try:
        batch_handler = BatchPredictHandler(handler)
        command = BatchPredictCommand(
            model_id=request.model_id,
            batch=request.batch,
            model_version=request.model_version,
            entity_ids=request.entity_ids,
        )
        return await batch_handler.handle(command)
    except Exception as e:
        logger.exception("Batch inference failed")
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

        from src.infrastructure.bootstrap.lifespan import (  # noqa: PLC0415
            alert_manager,
        )

        ms = MonitoringService(
            log_repo,
            drift_calculator,
            drift_repo,
            alert_manager=alert_manager,
            event_bus=event_bus,
        )

        root = find_project_root()
        reference_data = _load_reference_distributions(root)
        return await ms.check_drift(
            model_id=model_id, reference_data=reference_data, feature_index=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/models")
async def list_models(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[dict[str, Any]]:
    """List all registered models."""
    model_repo = PostgresModelRegistry(db)
    models = await model_repo.list_all()
    return [
        {
            "model_id": m.id,
            "version": m.version,
            "status": m.stage.value,
            "metadata": m.metadata,
        }
        for m in models
    ]


@router.get("/models/{model_id}")
async def get_model(
    model_id: str,
    version: str | None = None,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Retrieve model information from the registry."""
    model_repo = PostgresModelRegistry(db)
    handler = GetModelQueryHandler(model_repo)
    model = await handler.execute(GetModelQuery(model_id=model_id, version=version))
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return {
        "model_id": model.id,
        "version": model.version,
        "status": model.stage.value,
        "metadata": model.metadata,
    }


@router.get("/monitoring/reports/{model_id}")
async def get_drift_reports(
    model_id: str,
    limit: int = 10,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> list[DriftReport]:
    """Return historical drift reports for a model."""
    drift_repo = PostgresDriftReportRepository(db)
    handler = GetDriftReportQueryHandler(drift_repo)
    return await handler.execute(GetDriftReportQuery(model_id=model_id, limit=limit))


@router.get("/monitoring/performance/{model_id}")
async def get_model_performance(
    model_id: str,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Return aggregated performance metrics for a model."""
    log_repo = PostgresPredictionLogRepository(db)
    handler = GetModelPerformanceQueryHandler(log_repo, model_evaluator)
    return await handler.execute(GetModelPerformanceQuery(model_id=model_id))


class RollbackRequest(BaseModel):
    model_id: str


class RegisterModelRequest(BaseModel):
    model_id: str
    version: str
    uri: str
    framework: str = "onnx"
    stage: str = "challenger"
    metadata: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)


@router.post("/models/rollback")
async def rollback_challengers(
    request: RollbackRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Archive all active challenger models, keep champion serving."""
    model_repo = PostgresModelRegistry(db)
    active = await model_repo.get_active_versions(request.model_id)

    champion = None
    archived: list[str] = []
    for model in active:
        role = model.metadata.get("role", "")
        if role == "champion":
            champion = model.version
        elif role == "challenger":
            await model_repo.update_stage(request.model_id, model.version, "archived")
            archived.append(model.version)

    return {
        "model_id": request.model_id,
        "champion": champion,
        "archived_challengers": archived,
    }


@router.post("/models/register")
async def register_model(
    request: RegisterModelRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Register a new model version in the model registry."""
    from src.domain.inference.entities.model import Model  # noqa: PLC0415

    metadata = {**request.metadata, "role": request.stage, "metrics": request.metrics}
    model = Model(
        id=request.model_id,
        version=request.version,
        uri=request.uri,
        framework=request.framework,
        metadata=metadata,
        is_active=True,
    )

    model_repo = PostgresModelRegistry(db)
    await model_repo.save(model)

    return {
        "model_id": model.id,
        "version": model.version,
        "stage": request.stage,
        "status": "registered",
    }


@router.post("/models/{model_id}/retrain")
async def trigger_retrain(model_id: str) -> dict[str, Any]:
    """Manually trigger model retraining via Airflow.

    Triggers the self_healing_pipeline DAG with the given model_id.
    Works for any model regardless of its configured retrain trigger.
    """
    import httpx as _httpx  # noqa: PLC0415

    airflow_url = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")
    airflow_user = os.environ.get("AIRFLOW_USER", "admin")
    airflow_password = os.environ.get("AIRFLOW_PASSWORD", "admin")

    try:
        resp = _httpx.post(
            f"{airflow_url}/api/v1/dags/self_healing_pipeline/dagRuns",
            json={
                "conf": {
                    "model_id": model_id,
                    "reason": "Manual retrain triggered via API",
                    "drift_score": 0.0,
                },
            },
            auth=(airflow_user, airflow_password),
            timeout=10.0,
        )
        resp.raise_for_status()
        result = resp.json()
        return {
            "model_id": model_id,
            "status": "triggered",
            "dag_run_id": result.get("dag_run_id", "unknown"),
            "message": f"Retrain pipeline triggered for {model_id}",
        }
    except _httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to trigger Airflow: {e.response.status_code}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Airflow connection failed: {e}",
        ) from e


def _load_reference_distributions(project_root: Path) -> list[float]:
    """Load reference feature distributions from training data.

    Falls back to synthetic data if the reference file doesn't exist.
    """
    ref_path = project_root / "data" / "reference_data.json"
    if ref_path.exists():
        with open(ref_path) as f:
            data = json.load(f)
            if isinstance(data, list) and len(data) > 0:
                return [
                    float(record[0]) if isinstance(record, list) else float(record)
                    for record in data
                ]
    logger.warning("Reference data not found at %s, using synthetic fallback", ref_path)
    return [float(x) for x in range(100)]
