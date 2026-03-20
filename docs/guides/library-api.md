# Library API Guide

Phoenix ML Platform can be used as a Python library for building ML inference services.

## Installation

```bash
# Install with all dependencies
pip install -e "."

# With development tools
pip install -e ".[dev]"

# Or using uv
uv sync
```

## Quick Start

### 1. Basic Inference

```python
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine

# Initialize engine
engine = ONNXInferenceEngine()

# Load model
engine.load("my-model", "v1", "/path/to/model.onnx")

# Predict
result = engine.predict("my-model", "v1", [1.0, 2.0, 3.0])
print(result)  # {"output": [0.85], "probabilities": [0.15, 0.85]}
```

### 2. Full Inference Service

```python
import asyncio
from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.services.inference_service import InferenceService
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine

async def main():
    engine = ONNXInferenceEngine()
    # InferenceService accepts model_repo, engine, routing, feature_store, etc.
    # See container.py for the full wiring setup
    service = InferenceService(...)
    handler = PredictHandler(service)

    cmd = PredictCommand(
        model_id="credit-risk",
        model_version="v1",
        features=[0.5, 0.3, 0.8, 1.2, -0.4, ...]  # 30 features
    )
    prediction = await handler.execute(cmd)
    print(f"Result: {prediction.result}, Confidence: {prediction.confidence.value}")

asyncio.run(main())
```

### 3. Batch Prediction

```python
from src.application.commands.batch_predict_command import BatchPredictCommand
from src.application.handlers.batch_predict_handler import BatchPredictHandler

batch_cmd = BatchPredictCommand(
    model_id="credit-risk",
    batch=[[1.0]*30, [2.0]*30, [0.5]*30],
    model_version="v1",
)
batch_handler = BatchPredictHandler(predict_handler)
result = await batch_handler.handle(batch_cmd)
# {"predictions": [...], "total": 3, "successful": 3, "errors": [], "batch_latency_ms": 2.5}
```

### 4. Drift Detection

```python
from src.domain.monitoring.services.drift_calculator import DriftCalculator

calculator = DriftCalculator()
report = calculator.calculate(
    reference_data=[0.1, 0.2, 0.3, 0.15, 0.25],
    current_data=[0.5, 0.6, 0.7, 0.55, 0.65],
    feature_name="income",
    model_id="credit-risk",
)
print(f"Drift: {report.drift_detected}, p-value: {report.p_value}")
```

### 5. Custom Inference Engine

Implement the `InferenceEngine` interface to add support for a new ML runtime:

```python
from src.domain.inference.services.inference_engine import InferenceEngine
from typing import Any

class MyCustomEngine(InferenceEngine):
    def load(self, model_id: str, version: str, artifact_uri: str) -> None:
        # Load your model from the artifact URI
        pass

    def predict(self, model_id: str, version: str, features: list[float]) -> dict[str, Any]:
        # Run inference and return results
        return {"output": [0.5]}

    def is_loaded(self, model_id: str, version: str) -> bool:
        return True
```

### 6. Training a New Model

```python
# Create examples/my_problem/train.py
def train_and_export(output_path, metrics_path, reference_path):
    """Train model, export ONNX, save metrics and reference data."""
    # 1. Load and prepare data
    # 2. Train model (any sklearn/xgboost/etc.)
    # 3. Export to ONNX
    # 4. Save metrics JSON
    # 5. Save reference distributions for drift detection

# Create model_configs/my-model.yaml
# model_id: my-model
# version: v1
# framework: onnx
# task_type: classification
# model_path: models/my_model/v1/model.onnx
# train_script: examples/my_problem/train.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions (concurrent) |
| POST | `/feedback` | Submit ground truth |
| GET | `/models/{id}` | Get model info |
| POST | `/models/register` | Register new model version |
| POST | `/models/rollback` | Rollback challengers |
| GET | `/monitoring/drift/{id}` | Trigger drift check |
| GET | `/monitoring/reports/{id}` | Drift reports history |
| GET | `/monitoring/performance/{id}` | Performance metrics |
| GET | `/metrics` | Prometheus metrics |

## CLI

```bash
# Start server (after pip install -e .)
phoenix-serve

# Or with uvicorn directly
uv run uvicorn src.infrastructure.http.fastapi_server:app --reload --port 8000

# Key environment variables
export DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
export REDIS_URL=redis://localhost:6379
export KAFKA_URL=localhost:9092
export DEFAULT_MODEL_ID=credit-risk
export MLFLOW_TRACKING_URI=http://localhost:5000
export API_URL=http://localhost:8000
```
