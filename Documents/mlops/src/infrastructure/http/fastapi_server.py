from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from src.application.dto.prediction_request import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.config import get_settings
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.feature_store.redis_feature_store import RedisFeatureStore
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# --- Dependency Injection ---
model_repo = InMemoryModelRepository()
inference_engine = MockInferenceEngine()

# Conditional Feature Store Initialization
feature_store: FeatureStore
if settings.USE_REDIS:
    feature_store = RedisFeatureStore(redis_url=settings.REDIS_URL)
    print(f"✅ Initialized RedisFeatureStore at {settings.REDIS_URL}")
else:
    feature_store = InMemoryFeatureStore()
    print("⚠️  Initialized InMemoryFeatureStore (Use REDIS_URL to switch)")


def get_predict_handler() -> PredictHandler:
    return PredictHandler(model_repo, inference_engine, feature_store)


@app.on_event("startup")
async def startup_event() -> None:
    # Register a default model
    demo_model = Model(
        id="demo-model",
        version="v1",
        uri=settings.DEFAULT_MODEL_PATH,
        framework="onnx",
    )
    await model_repo.save(demo_model)

    # Seed data if using InMemory (for demo purposes)
    if isinstance(feature_store, InMemoryFeatureStore):
        feature_store.add_features("user-123", {"f1": 0.5, "f2": 1.5, "f3": 2.5})


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/predict")
async def predict(
    command: PredictCommand,
    handler: PredictHandler = Depends(get_predict_handler),  # noqa: B008
) -> dict[str, Any]:
    try:
        prediction = await handler.execute(command)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)