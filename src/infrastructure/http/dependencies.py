import os
from typing import Any

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.services.inference_service import InferenceService
from src.domain.inference.services.routing_strategy import ABTestStrategy
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.infrastructure.http.container import (
    artifact_storage,
    batch_manager,
    event_bus,
    feature_store,
    inference_engine,
)
from src.infrastructure.persistence.database import get_db
from src.infrastructure.persistence.mlflow_model_registry import (
    MlflowModelRegistry,
)
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)

# ── Model Registry Factory (OCP: add new backends via dict entry) ──
_REGISTRY_FACTORIES: dict[str, Any] = {
    "mlflow": lambda db: MlflowModelRegistry(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    ),
    "postgres": lambda db: PostgresModelRegistry(db),
}


async def get_predict_handler(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PredictHandler:
    registry_backend = os.getenv("MODEL_REGISTRY_BACKEND", "postgres").strip().lower()
    factory = _REGISTRY_FACTORIES.get(registry_backend, _REGISTRY_FACTORIES["postgres"])
    model_repo: ModelRepository = factory(db)

    inference_service = InferenceService(
        model_repo=model_repo,
        inference_engine=inference_engine,
        batch_manager=batch_manager,
        feature_store=feature_store,
        artifact_storage=artifact_storage,
        routing_strategy=ABTestStrategy(0.5),
    )
    return PredictHandler(inference_service, event_bus)
