from collections.abc import Callable

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix_ml.application.handlers.predict_handler import PredictHandler
from phoenix_ml.config import get_settings
from phoenix_ml.domain.inference.services.inference_service import InferenceService
from phoenix_ml.domain.inference.services.routing_strategy import ABTestStrategy
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository
from phoenix_ml.infrastructure.bootstrap.container import (
    artifact_storage,
    batch_manager,
    event_bus,
    feature_store,
    inference_engine,
)
from phoenix_ml.infrastructure.persistence.database import get_db
from phoenix_ml.infrastructure.persistence.mlflow_model_registry import (
    MlflowModelRegistry,
)
from phoenix_ml.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)

settings = get_settings()

# ── Model Registry Factory (OCP: add new backends via dict entry) ──
_REGISTRY_FACTORIES: dict[str, Callable[..., ModelRepository]] = {
    "mlflow": lambda db: MlflowModelRegistry(
        tracking_uri=settings.MLFLOW_TRACKING_URI,
    ),
    "postgres": lambda db: PostgresModelRegistry(db),
}

# Read backend from config (not os.getenv)
_registry_backend = settings.MODEL_REGISTRY_BACKEND


async def get_predict_handler(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PredictHandler:
    factory = _REGISTRY_FACTORIES.get(_registry_backend, _REGISTRY_FACTORIES["postgres"])
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
