import asyncio
import shutil
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
from src.infrastructure.messaging.kafka_producer import KafkaProducer
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.infrastructure.monitoring.in_memory_log_repo import (
    InMemoryPredictionLogRepository,
)
from src.infrastructure.persistence.database import Base, engine, get_db
from src.infrastructure.persistence.postgres_log_repo import (
    PostgresPredictionLogRepository,
)
from src.infrastructure.persistence.postgres_model_registry import PostgresModelRegistry

settings = get_settings()

# --- Global Components ---
artifact_storage = LocalArtifactStorage(base_dir=Path("/tmp/phoenix/remote_storage"))
inference_engine = ONNXInferenceEngine(cache_dir=Path("/tmp/phoenix/model_cache"))
kafka_producer = KafkaProducer(bootstrap_servers=settings.KAFKA_URL)
drift_calculator = DriftCalculator()

# log_repo will be initialized per-request for DB session or globally for Memory
global_log_repo: Any = InMemoryPredictionLogRepository() 
monitoring_service = MonitoringService(global_log_repo, drift_calculator)

# Conditional Feature Store Initialization
feature_store: FeatureStore
if settings.USE_REDIS:
    feature_store = RedisFeatureStore(redis_url=settings.REDIS_URL)
else:
    feature_store = InMemoryFeatureStore()

async def run_monitoring_loop() -> None:
    """
    Background task to check drift periodically (every 10s).
    Simulates a continuous monitoring system.
    """
    print("🚀 Starting Drift Monitoring Loop...")
    # Mock Reference Data (Standard Normal Distribution) from training
    # Simulating Feature 0 (Income) from training set
    reference_data = np.random.normal(0, 1, 100).tolist()
    
    while True:
        try:
            await asyncio.sleep(5) # Check every 5 seconds for demo
            
            # Check drift for 'credit-risk' model, feature 0 (income)
            await monitoring_service.check_drift(
                model_id="credit-risk",
                reference_data=reference_data,
                feature_index=0
            )
        except ValueError:
             # Not enough data yet, silent fail
             pass
        except Exception as e:
            print(f"⚠️ Monitoring Loop Error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # 1. Initialize Database Tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 2. Seed Initial Data
    async for db in get_db():
        model_repo = PostgresModelRegistry(db)

        # --- SETUP REAL MODEL (V1 - CHAMPION) ---
        real_model_path_v1 = Path("models/credit_risk/v1/model.onnx")
        storage_path_v1 = Path("/tmp/phoenix/remote_storage/credit-risk/v1/model.onnx")
        
        if real_model_path_v1.exists():
            storage_path_v1.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(real_model_path_v1, storage_path_v1)
            print(f"✅ Loaded Champion model (v1) from {real_model_path_v1}")

        credit_model_v1 = Model(
            id="credit-risk",
            version="v1",
            uri=f"local://{storage_path_v1}",
            framework="onnx",
            metadata={
                "features": ["income", "debt", "age", "credit_history"], 
                "role": "champion"
            }
        )
        await model_repo.save(credit_model_v1)

        # --- SETUP REAL MODEL (V2 - CHALLENGER) ---
        real_model_path_v2 = Path("models/credit_risk/v2/model.onnx")
        storage_path_v2 = Path("/tmp/phoenix/remote_storage/credit-risk/v2/model.onnx")
        
        if real_model_path_v2.exists():
            storage_path_v2.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(real_model_path_v2, storage_path_v2)
            print(f"✅ Loaded Challenger model (v2) from {real_model_path_v2}")

        credit_model_v2 = Model(
            id="credit-risk",
            version="v2",
            uri=f"local://{storage_path_v2}",
            framework="onnx",
            metadata={
                "features": ["income", "debt", "age", "credit_history"], 
                "role": "challenger"
            }
        )
        await model_repo.save(credit_model_v2)

        # --- SEED REAL FEATURE DATA ---
        await feature_store.add_features(
            "customer-good", 
            {"f1": 2.0, "f2": -1.5, "f3": 1.0, "f4": 1.5}
        )
        await feature_store.add_features(
            "customer-bad", 
            {"f1": -1.5, "f2": 2.0, "f3": -0.5, "f4": -1.0}
        )
        
        print("✅ Seeded customer data into Feature Store")
        break  # Only need one session

    # 3. Start Background Monitoring
    await kafka_producer.start()
    monitor_task = asyncio.create_task(run_monitoring_loop())

    yield
    
    # Cleanup
    await kafka_producer.stop()
    monitor_task.cancel()
    await engine.dispose()


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus Metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# --- Dependencies ---
async def get_predict_handler(
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> PredictHandler:
    model_repo = PostgresModelRegistry(db)
    return PredictHandler(model_repo, inference_engine, feature_store, artifact_storage)


async def log_prediction_background(
    command: PredictCommand, prediction: Prediction, db: AsyncSession
) -> None:
    """
    Background task to log prediction to DB and Kafka.
    """
    # 1. Persistent Storage (Postgres)
    repo = PostgresPredictionLogRepository(db)
    await repo.log(command, prediction)
    
    # 2. Real-time Streaming (Kafka)
    event = {
        "event_id": str(uuid.uuid4()),
        "model_id": prediction.model_id,
        "version": prediction.model_version,
        "features": command.features,
        "result": prediction.result,
        "confidence": prediction.confidence.value,
        "timestamp": datetime.now(UTC).isoformat()
    }
    await kafka_producer.publish("inference-events", event)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": settings.APP_VERSION}


@app.post("/predict")
async def predict(
    command: PredictCommand,
    background_tasks: BackgroundTasks,
    handler: PredictHandler = Depends(get_predict_handler),  # noqa: B008
    db: AsyncSession = Depends(get_db), # noqa: B008
) -> dict[str, Any]:
    try:
        prediction = await handler.execute(command)

        # Async Logging (DB + Kafka)
        background_tasks.add_task(log_prediction_background, command, prediction, db)

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