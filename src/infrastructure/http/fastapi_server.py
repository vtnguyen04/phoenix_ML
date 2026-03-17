import asyncio
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.application.handlers.retrain_handler import RetrainHandler
from src.application.services.monitoring_service import MonitoringService
from src.config import get_settings
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.services.batch_manager import BatchConfig, BatchManager
from src.domain.inference.services.inference_service import (
    InferenceService,
)
from src.domain.inference.services.routing_strategy import ABTestStrategy
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.domain.monitoring.services.model_evaluator import ModelEvaluator
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.feature_store.redis_feature_store import RedisFeatureStore
from src.infrastructure.messaging.kafka_producer import KafkaProducer
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.infrastructure.persistence.database import Base, engine, get_db
from src.infrastructure.persistence.models import ModelORM
from src.infrastructure.persistence.postgres_drift_repo import (
    PostgresDriftReportRepository,
)
from src.infrastructure.persistence.postgres_log_repo import (
    PostgresPredictionLogRepository,
)
from src.infrastructure.persistence.postgres_model_registry import (
    PostgresModelRegistry,
)
from src.shared.utils.model_generator import generate_simple_onnx

settings = get_settings()
logger = logging.getLogger(__name__)

# --- Global Components ---
artifact_storage = LocalArtifactStorage(base_dir=Path("/tmp/phoenix/remote_storage"))
inference_engine = ONNXInferenceEngine(cache_dir=Path("/tmp/phoenix/model_cache"))
batch_config = BatchConfig(max_batch_size=16, max_wait_time_ms=10)
batch_manager = BatchManager(inference_engine, config=batch_config)
kafka_producer = KafkaProducer(bootstrap_servers=settings.KAFKA_URL)
drift_calculator = DriftCalculator()
model_evaluator = ModelEvaluator()

# Conditional Feature Store Initialization
feature_store: FeatureStore
if settings.USE_REDIS:
    feature_store = RedisFeatureStore(redis_url=settings.REDIS_URL)
else:
    feature_store = InMemoryFeatureStore()

_shutdown_event = asyncio.Event()


def find_project_root() -> Path:
    """Find root by searching for pyproject.toml upwards from this file."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_model_exists() -> Path:
    """Ensures model exists, generating a VALID ONNX one if in test context."""
    root = find_project_root()
    model_path = root / "models" / "credit_risk" / "v1" / "model.onnx"

    if model_path.exists():
        return model_path.absolute()

    is_ci = os.getenv("GITHUB_ACTIONS")
    is_test = "test" in str(Path.cwd())

    if is_ci or is_test:
        logger.warning(
            "🧪 CI/Test context. Generating valid ONNX model at %s", model_path
        )
        generate_simple_onnx(model_path)
        return model_path.absolute()

    msg = f"Model not found at {model_path} and not in CI environment."
    raise FileNotFoundError(msg)


async def run_monitoring_loop() -> None:
    """Background task to check drift periodically."""
    logger.info("🚀 Starting Drift Monitoring Loop...")
    reference_data = np.random.normal(0, 1, 100).tolist()

    while not _shutdown_event.is_set():
        try:
            for _ in range(50):
                if _shutdown_event.is_set():
                    return
                await asyncio.sleep(0.1)

            async for db in get_db():
                log_repo = PostgresPredictionLogRepository(db)
                drift_repo = PostgresDriftReportRepository(db)
                model_repo = PostgresModelRegistry(db)

                # Loop needs its own handler connected to current session
                rh = RetrainHandler(find_project_root(), model_repo, model_evaluator)
                ms = MonitoringService(log_repo, drift_calculator, drift_repo, rh)

                await ms.check_drift(
                    model_id="credit-risk",
                    reference_data=reference_data,
                    feature_index=0,
                )
                break
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("⚠️ Monitoring Loop Error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Initialize Database Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Seed Initial Data
    async for db in get_db():
        model_repo = PostgresModelRegistry(db)

        result = await db.execute(
            select(ModelORM).where(
                ModelORM.id == "credit-risk", ModelORM.version == "v1"
            )
        )
        if not result.scalar_one_or_none():
            try:
                real_model_path = ensure_model_exists()
                logger.info("✅ Seeding model from %s", real_model_path)
                credit_model_v1 = Model(
                    id="credit-risk",
                    version="v1",
                    uri=f"local://{real_model_path}",
                    framework="onnx",
                    metadata={
                        "features": ["income", "debt", "age", "credit_history"],
                        "role": "champion",
                        "metrics": {"accuracy": 0.85, "f1_score": 0.84},
                    },
                )
                await model_repo.save(credit_model_v1)
                await model_repo.update_stage("credit-risk", "v1", "champion")
                await db.commit()
                logger.info("✅ Successfully registered Credit Risk model v1")
            except Exception as e:
                logger.error("❌ Failed to seed model: %s", e)

        await feature_store.add_features(
            "customer-good", {"f1": 2.0, "f2": -1.5, "f3": 1.0, "f4": 1.5}
        )
        break

    # 3. Start Background Components
    _shutdown_event.clear()
    await kafka_producer.start()
    monitor_task = asyncio.create_task(run_monitoring_loop())

    yield

    # Cleanup
    logger.info("🧹 Lifespan shutdown started...")
    _shutdown_event.set()
    monitor_task.cancel()
    await batch_manager.stop()
    await kafka_producer.stop()
    try:
        await asyncio.wait_for(monitor_task, timeout=2.0)
    except (TimeoutError, asyncio.CancelledError):
        pass
    await engine.dispose()
    logger.info("✅ Cleanup complete.")


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


async def get_predict_handler(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PredictHandler:
    model_repo: ModelRepository = PostgresModelRegistry(db)
    inference_service = InferenceService(
        model_repo=model_repo,
        inference_engine=inference_engine,
        batch_manager=batch_manager,
        feature_store=feature_store,
        artifact_storage=artifact_storage,
        routing_strategy=ABTestStrategy(0.5),
    )
    return PredictHandler(inference_service)


async def log_prediction_background(
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


class FeedbackRequest(BaseModel):
    prediction_id: str
    ground_truth: int


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.post("/predict")
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
            log_prediction_background, command, prediction, prediction_id, db
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


@app.post("/feedback")
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


@app.get("/monitoring/drift/{model_id}")
async def check_drift(
    model_id: str,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> DriftReport:
    try:
        log_repo = PostgresPredictionLogRepository(db)
        drift_repo = PostgresDriftReportRepository(db)
        model_repo = PostgresModelRegistry(db)

        # Handler needs to be created with current DB session context
        root = find_project_root()
        rh = RetrainHandler(root, model_repo, model_evaluator)
        ms = MonitoringService(log_repo, drift_calculator, drift_repo, rh)

        mock_reference_data = [float(x) for x in range(100)]
        return await ms.check_drift(
            model_id=model_id, reference_data=mock_reference_data, feature_index=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
