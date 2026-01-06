from typing import Any

from fastapi import Depends, FastAPI, HTTPException

from src.application.dto.prediction_request import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.ml_engines.mock_engine import MockInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository

app = FastAPI(title="Phoenix ML Platform", version="0.1.0")

# --- Dependency Injection ---
# In a real app, use a proper DI container like 'dependency-injector' or 'punq'
# For now, we'll use singleton instances for simplicity
model_repo = InMemoryModelRepository()
inference_engine = MockInferenceEngine()
feature_store = InMemoryFeatureStore()


def get_predict_handler() -> PredictHandler:
    return PredictHandler(model_repo, inference_engine, feature_store)


@app.on_event("startup")
async def startup_event() -> None:
    # Register a default model for demo purposes
    demo_model = Model(
        id="demo-model",
        version="v1",
        uri="local://models/demo.onnx",
        framework="onnx",
    )
    await model_repo.save(demo_model)
    
    # Seed feature store for demo
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