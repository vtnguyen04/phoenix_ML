import shutil
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.application.services.monitoring_service import MonitoringService
from src.config import get_settings
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.monitoring.entities.drift_report import DriftReport
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.feature_store.redis_feature_store import RedisFeatureStore
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.infrastructure.monitoring.in_memory_log_repo import (
    InMemoryPredictionLogRepository,
)
from src.infrastructure.persistence.database import Base, engine, get_db
from src.infrastructure.persistence.postgres_model_registry import PostgresModelRegistry

settings = get_settings()

# --- Global Components ---
artifact_storage = LocalArtifactStorage(base_dir=Path("/tmp/phoenix/remote_storage"))
inference_engine = ONNXInferenceEngine(cache_dir=Path("/tmp/phoenix/model_cache"))
log_repo = InMemoryPredictionLogRepository()
drift_calculator = DriftCalculator()
monitoring_service = MonitoringService(log_repo, drift_calculator)

# Conditional Feature Store Initialization
feature_store: FeatureStore
if settings.USE_REDIS:
    feature_store = RedisFeatureStore(redis_url=settings.REDIS_URL)
else:
    feature_store = InMemoryFeatureStore()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Initialize Database Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Seed Initial Data
    async for db in get_db():
        model_repo = PostgresModelRegistry(db)

        # --- SETUP REAL MODEL ---
        # Copy the real trained model to the artifact storage location
        real_model_path = Path("models/credit_risk/v1/model.onnx")
        storage_path = Path("/tmp/phoenix/remote_storage/credit-risk/v1/model.onnx")
        
        if real_model_path.exists():
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(real_model_path, storage_path)
            print(f"✅ Loaded real model from {real_model_path}")
        else:
            print("⚠️ Real model not found, skipping model registration")

        # Register the Credit Risk Model
        credit_model = Model(
            id="credit-risk",
            version="v1",
            uri=f"local://{storage_path}",
            framework="onnx",
            metadata={"features": ["income", "debt", "age", "credit_history"]}
        )
        await model_repo.save(credit_model)

        # --- SEED REAL FEATURE DATA ---
        # Feature order: [Income, Debt, Age, CreditHistory]
        # Data is scaled/normalized as per training script assumption
        
        # Profile 1: Good Customer (High Income, Low Debt) 
        # -> Expect Low Risk (Class 0)
        # Features: [2.0, -1.5, 1.0, 1.5]
        await feature_store.add_features(
            "customer-good", 
            {"f1": 2.0, "f2": -1.5, "f3": 1.0, "f4": 1.5}
        )

        # Profile 2: Risky Customer (Low Income, High Debt) 
        # -> Expect High Risk (Class 1)
        # Features: [-1.5, 2.0, -0.5, -1.0]
        await feature_store.add_features(
            "customer-bad", 
            {"f1": -1.5, "f2": 2.0, "f3": -0.5, "f4": -1.0}
        )
        
        print("✅ Seeded customer data into Feature Store")
        break  # Only need one session

    yield
    # Cleanup logic if any
    await engine.dispose()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)


# --- Dependencies ---
async def get_predict_handler(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PredictHandler:
    model_repo = PostgresModelRegistry(db)
    return PredictHandler(model_repo, inference_engine, feature_store, artifact_storage)


async def log_prediction_background(
    command: PredictCommand, prediction: Prediction
) -> None:
    """Background task to log prediction without blocking response"""
    await log_repo.log(command, prediction)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.post("/predict")
async def predict(
    command: PredictCommand,
    background_tasks: BackgroundTasks,
    handler: PredictHandler = Depends(get_predict_handler),  # noqa: B008
) -> dict[str, Any]:
    try:
        prediction = await handler.execute(command)

        # Async Logging
        background_tasks.add_task(log_prediction_background, command, prediction)

        return {
            "model_id": prediction.model_id,
            "version": prediction.model_version,
            "result": prediction.result,
            "confidence": prediction.confidence.value,
            "latency_ms": round(prediction.latency_ms, 2),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e


@app.get("/monitoring/drift/{model_id}")
async def check_drift(model_id: str) -> DriftReport:
    try:
        # Mock Reference Data
        mock_reference_data = [float(x) for x in range(100)]

        report = await monitoring_service.check_drift(
            model_id=model_id, reference_data=mock_reference_data, feature_index=0
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
