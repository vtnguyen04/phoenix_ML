# Customization Guide

Hướng dẫn tùy chỉnh Phoenix ML Platform: thêm model mới, custom engines, data loaders, plugins, alert notifiers.

## Thêm Model Mới

### Bước 1: Tạo Model Config (YAML)

Tạo file `model_configs/<model-id>.yaml`:

```yaml
# model_configs/sentiment-analysis.yaml
model_id: sentiment-analysis
version: v1
feature_names:
  - text_length
  - avg_word_length
  - sentiment_score
  - exclamation_count
  - question_count
data_loader: tabular              # hoặc "image" cho image models
train_script: examples/sentiment/train.py
monitoring:
  drift_test: psi                 # ks, psi, hoặc chi2
  drift_threshold: 0.25           # Custom threshold (default: 0.3)
```

### Bước 2: Tạo Training Script

```python
# examples/sentiment/train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json
from pathlib import Path

def train():
    # Load data
    df = pd.read_csv("data/sentiment/dataset.csv")
    X = df.drop("label", axis=1).values.astype(np.float32)
    y = df["label"].values
    
    # Train
    model = GradientBoostingClassifier(n_estimators=100)
    model.fit(X, y)
    
    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, X.shape[1]]))],
    )
    
    # Save
    output_dir = Path("models/sentiment_analysis/v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Save metrics
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1_score": float(f1_score(y_test, preds, average="weighted")),
        "num_features": X.shape[1],
        "feature_names": list(df.drop("label", axis=1).columns),
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train()
```

### Bước 3: Add to DVC Pipeline

```yaml
# Thêm vào dvc.yaml:
  train_sentiment:
    cmd: uv run python examples/sentiment/train.py
    deps:
      - data/sentiment/dataset.csv
      - examples/sentiment/train.py
    outs:
      - models/sentiment_analysis/v1/
```

### Bước 4: Train & Run

```bash
uv run python examples/sentiment/train.py
# Model sẽ tự động load khi API start (lifespan.py scan model_configs/)
```

## Custom Inference Engine

Implement `InferenceEngine` ABC:

```python
# src/infrastructure/ml_engines/custom_engine.py
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.domain.inference.value_objects.confidence_score import ConfidenceScore

class CustomEngine(InferenceEngine):
    async def load(self, model: Model) -> None:
        """Load model into memory."""
        self._model = load_custom_model(model.uri)
    
    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        """Run inference."""
        result = self._model.predict(features.values)
        return Prediction(
            model_id=model.id,
            model_version=model.version,
            result=float(result),
            confidence=ConfidenceScore(value=0.95),
            latency_ms=1.5,
        )
    
    async def batch_predict(self, model: Model, features_list: list[FeatureVector]) -> list[Prediction]:
        """Batch inference."""
        return [await self.predict(model, f) for f in features_list]
    
    async def optimize(self, model: Model) -> None:
        """Optimize model (optional)."""
        pass
```

Đăng ký trong `container.py`:

```python
# Thêm vào _ENGINE_FACTORIES:
_ENGINE_FACTORIES = {
    "onnx": lambda: ONNXInferenceEngine(...),
    "tensorrt": lambda: TensorRTExecutor(...),
    "triton": lambda: TritonInferenceClient(...),
    "custom": lambda: CustomEngine(),  # ← NEW
}
```

```bash
# Sử dụng:
INFERENCE_ENGINE=custom uv run uvicorn ...
```

## Custom Data Loader

Implement `IDataLoader` ABC:

```python
# src/infrastructure/data_loaders/text_loader.py
from src.domain.training.services.data_loader_plugin import IDataLoader, DatasetInfo
import numpy as np

class TextDataLoader(IDataLoader):
    async def load(self, data_path: str) -> tuple[np.ndarray, DatasetInfo]:
        """Load text dataset."""
        # Load and tokenize text data
        texts, labels = load_texts(data_path)
        features = vectorize(texts)  # TF-IDF, embeddings, etc.
        
        info = DatasetInfo(
            num_samples=len(texts),
            num_features=features.shape[1],
            feature_names=[f"feature_{i}" for i in range(features.shape[1])],
            data_format="text",
        )
        return features, info
    
    async def split(self, data, test_size: float = 0.2):
        """Train/test split."""
        from sklearn.model_selection import train_test_split
        return train_test_split(data, test_size=test_size)
```

Đăng ký trong `registry.py`:

```python
# src/infrastructure/data_loaders/registry.py
_LOADER_REGISTRY = {
    "tabular": TabularDataLoader,
    "image": ImageDataLoader,
    "text": TextDataLoader,  # ← NEW
}
```

## Custom Pre/Post Processor Plugins

```python
# Custom preprocessor cho model cụ thể
from src.domain.inference.services.processor_plugin import IPreprocessor, IPostprocessor
import numpy as np

class NormalizationPreprocessor(IPreprocessor):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std = std
    
    def transform(self, raw_input: dict) -> np.ndarray:
        features = np.array(raw_input["features"], dtype=np.float32)
        return (features - self.mean) / self.std

class MultiLabelPostprocessor(IPostprocessor):
    def __init__(self, labels: list[str], threshold: float = 0.5):
        self.labels = labels
        self.threshold = threshold
    
    def transform(self, model_output: np.ndarray, labels: list[str]) -> dict:
        active = [l for l, p in zip(self.labels, model_output) if p > self.threshold]
        return {"labels": active, "probabilities": model_output.tolist()}
```

Đăng ký trong `container.py` (hoặc `lifespan.py`):

```python
plugin_registry.register_model(
    model_id="sentiment-analysis",
    preprocessor=NormalizationPreprocessor(mean=train_mean, std=train_std),
    postprocessor=MultiLabelPostprocessor(labels=["positive", "negative", "neutral"]),
    data_loader=TextDataLoader(),
)
```

## Custom Alert Notifier

```python
# src/infrastructure/monitoring/teams_notifier.py
from src.domain.monitoring.services.alert_notifier import IAlertNotifier
import httpx

class TeamsNotifier(IAlertNotifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def notify(self, alert) -> bool:
        payload = {
            "@type": "MessageCard",
            "summary": f"Phoenix Alert: {alert.name}",
            "sections": [{
                "facts": [
                    {"name": "Alert", "value": alert.name},
                    {"name": "Severity", "value": alert.severity},
                    {"name": "Value", "value": str(alert.actual_value)},
                ]
            }]
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.webhook_url, json=payload)
            return resp.status_code == 200
```

## Cấu hình Monitoring Thresholds

Trong `model_configs/<model>.yaml`:

```yaml
monitoring:
  drift_test: ks              # Algorithm: ks, psi, chi2
  drift_threshold: 0.25       # Khi nào coi là drifted
```

Trong `.env`:

```bash
MONITORING_INTERVAL_SECONDS=30   # Check drift mỗi 30s
DRIFT_THRESHOLD=0.3              # Default threshold (override bằng YAML)
```

## Thêm Routing Strategy

```python
from src.domain.inference.services.routing_strategy import RoutingStrategy
from src.domain.inference.entities.model import Model

class WeightedStrategy(RoutingStrategy):
    """Route based on model performance weights."""
    
    def __init__(self, weights: dict[str, float]):
        self.weights = weights  # {version: weight}
    
    def select(self, champion: Model, challengers: list[Model]) -> Model:
        import random
        all_models = [champion] + challengers
        versions = [m.version for m in all_models]
        weights = [self.weights.get(v, 0.1) for v in versions]
        return random.choices(all_models, weights=weights, k=1)[0]
```

---
*Document Status: v4.0 — Updated March 2026*
