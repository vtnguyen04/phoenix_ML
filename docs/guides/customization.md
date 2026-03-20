# Customization Guide

This guide explains how to add your own models, configure the platform, and customize every aspect of Phoenix ML.

---

## 1. Adding a New Model

### Step 1: Create Training Script

Create a file at `examples/<problem_name>/train.py` with a `train_and_export` function:

```python
# examples/sentiment_analysis/train.py
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def train_and_export(
    output_path: str,
    metrics_path: str = "",
    reference_path: str = "",
) -> None:
    """Train model, export to ONNX, save metrics + reference data."""
    # 1. Load data (your dataset)
    X_train, y_train, X_test, y_test = load_my_data()

    # 2. Train
    model = SGDClassifier()
    model.fit(X_train, y_train)

    # 3. Evaluate
    accuracy = model.score(X_test, y_test)

    # 4. Export to ONNX
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    onnx_model = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, X_train.shape[1]]))],
    )
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 5. Save metrics JSON
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump({
                "accuracy": accuracy,
                "n_features": X_train.shape[1],
                "n_samples": len(X_train),
                "model_type": "SGDClassifier",
                "task_type": "classification",
            }, f, indent=2)

    # 6. Save reference features (for drift detection)
    if reference_path:
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        with open(reference_path, "w") as f:
            json.dump(X_test[:100].tolist(), f)
```

**Key requirements**:

- Function must be named `train_and_export` (primary) or `train`, `train_model`, `main` (fallbacks)
- First argument: `output_path` — where to save the ONNX file
- Optional: `metrics_path`, `reference_path` for metrics and drift baseline
- The function is called automatically by the Airflow self-healing pipeline

### Step 2: Create Model Config

Create `model_configs/<model-id>.yaml`:

```yaml
# model_configs/sentiment-analysis.yaml
model_id: sentiment-analysis
version: v1
framework: onnx
task_type: classification          # classification | regression
model_path: models/sentiment_analysis/v1/model.onnx
train_script: examples/sentiment_analysis/train.py

# Feature names (used by frontend inference panel)
feature_names:
  - text_length
  - word_count
  - sentiment_score
  - capital_ratio

# Reference data for drift detection
reference_path: models/sentiment_analysis/v1/reference_features.json

# Dataset source (used by DVC)
dataset_name: sentiment-reviews

metadata:
  role: champion
  description: "Sentiment analysis on product reviews"
```

**Fields explained**:

| Field | Required | Description |
|-------|----------|-------------|
| `model_id` | Yes | Unique ID, used in API calls (`/predict`) |
| `version` | Yes | Semantic version for initial deployment |
| `framework` | Yes | Always `onnx` (models exported via ONNX) |
| `task_type` | Yes | `classification` or `regression` (affects metrics display) |
| `model_path` | Yes | Path to the ONNX file |
| `train_script` | Yes | Path to training script (used by Airflow DAG) |
| `feature_names` | No | Feature names for frontend display |
| `reference_path` | No | Baseline data for drift detection |
| `metadata.role` | No | `champion` (default) or `challenger` |

### Step 3: Add DVC Stage (Optional)

In `dvc.yaml`, add a stage for reproducible training:

```yaml
stages:
  train_sentiment:
    cmd: uv run python examples/sentiment_analysis/train.py
         models/sentiment_analysis/v1/model.onnx
         --metrics-path models/sentiment_analysis/v1/metrics.json
         --reference-path models/sentiment_analysis/v1/reference_features.json
    deps:
      - examples/sentiment_analysis/train.py
      - data/sentiment/
    outs:
      - models/sentiment_analysis/v1/model.onnx
    metrics:
      - models/sentiment_analysis/v1/metrics.json:
          cache: false
```

### Step 4: Train and Deploy

```bash
# Train
uv run python examples/sentiment_analysis/train.py \
  models/sentiment_analysis/v1/model.onnx \
  --metrics-path models/sentiment_analysis/v1/metrics.json

# Restart API to auto-discover new model
docker compose restart api

# Verify
curl http://localhost:8001/models/sentiment-analysis
```

---

## 2. Configuring Monitoring Thresholds

Edit `src/config/monitoring.py` or set environment variables:

```python
# src/config/monitoring.py
MONITORING_INTERVAL_SECONDS = 30       # How often drift checks run
DRIFT_THRESHOLD = 0.05                 # p-value threshold for KS test
ANOMALY_Z_SCORE_THRESHOLD = 3.0        # Z-score for confidence anomalies
ANOMALY_LATENCY_MULTIPLIER = 3.0       # Multiplier for latency spike detection
ANOMALY_ERROR_RATE_THRESHOLD = 0.05    # Error rate above 5% = anomaly

# Rollback thresholds
ROLLBACK_ERROR_RATE_THRESHOLD = 0.10   # Archive challenger if >10% error rate
ROLLBACK_LATENCY_THRESHOLD_MS = 500.0  # Archive if avg latency >500ms
ROLLBACK_MIN_REQUESTS = 50             # Minimum requests before evaluating

# Alert cooldown
ALERT_COOLDOWN_SECONDS = 300           # Don't re-fire same alert for 5 min
```

**Environment variable overrides** (in `.env` or `docker-compose.yaml`):

```bash
MONITORING_INTERVAL=30
DRIFT_THRESHOLD=0.05
ROLLBACK_ERROR_RATE=0.10
ROLLBACK_LATENCY_MS=500
ALERT_COOLDOWN=300
```

---

## 3. Configuring the Frontend

### API Target

Edit `frontend/vite.config.ts` to change the backend URL:

```ts
// For local development (default)
target: process.env.VITE_API_TARGET ?? 'http://localhost:8001'

// For Docker
target: 'http://api:8000'
```

### Adding Dashboard Panels

All dashboard components are in `frontend/src/components/dashboard/`. To add a new panel:

```tsx
// frontend/src/components/dashboard/MyPanel.tsx
import { useEffect, useState } from 'react';
import { Card } from './Card';
import { mlService } from '../../api/mlService';

export function MyPanel({ modelId }: { modelId: string }) {
  const [data, setData] = useState(null);

  useEffect(() => {
    mlService.get(`/my-endpoint/${modelId}`).then(setData);
  }, [modelId]);

  return (
    <Card title="My Custom Panel">
      {/* your content */}
    </Card>
  );
}
```

Then add it to `frontend/src/App.tsx`.

### Adding Services to Health Check

Edit `frontend/src/config.ts`:

```ts
export const SERVICES = [
  { name: 'API',        url: '/health',          port: 8001 },
  { name: 'My Service', url: 'http://my-svc:80', port: 80   },
];
```

---

## 4. Custom Inference Engine

To use a runtime other than ONNX Runtime (e.g., TensorRT, Triton):

```python
# src/infrastructure/ml_engines/my_engine.py
from src.domain.inference.services.inference_engine import InferenceEngine

class MyCustomEngine(InferenceEngine):
    def load(self, model_id: str, version: str, artifact_uri: str) -> None:
        # Load model from artifact_uri
        ...

    def predict(
        self, model_id: str, version: str, features: list[float]
    ) -> dict[str, Any]:
        # Run inference
        return {"output": [...], "probabilities": [...]}

    def is_loaded(self, model_id: str, version: str) -> bool:
        return True
```

Register in `src/infrastructure/http/container.py`:

```python
engine = MyCustomEngine()
# Replace or add alongside ONNXInferenceEngine
```

---

## 5. Custom Alert Notifications

Create a new notifier by implementing `AlertNotifier`:

```python
# src/infrastructure/monitoring/slack_notifier.py
from src.domain.monitoring.services.alert_notifier import AlertNotifier

class SlackNotifier(AlertNotifier):
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, alert) -> None:
        import httpx
        payload = {
            "text": f"*{alert.severity}* — {alert.message}",
            "channel": "#ml-alerts",
        }
        async with httpx.AsyncClient() as client:
            await client.post(self.webhook_url, json=payload)
```

Configure via environment variable:

```bash
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
```

---

## 6. Custom Routing Strategy

Create a new routing strategy for A/B testing, canary, or shadow traffic:

```python
from src.domain.inference.services.routing_strategy import RoutingStrategy

class MyRoutingStrategy(RoutingStrategy):
    def select_version(
        self, model_id: str, versions: list[str]
    ) -> str:
        # Your logic (e.g., geo-based, user segment)
        return versions[0]
```

---

## 7. Database Configuration

### PostgreSQL

```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/phoenix
```

### Alembic Migrations

```bash
# Create a new migration
uv run alembic revision --autogenerate -m "add my_table"

# Apply
uv run alembic upgrade head

# Rollback
uv run alembic downgrade -1
```

### Redis (Feature Store)

```bash
REDIS_URL=redis://localhost:6379
USE_REDIS=true    # Set to false to use in-memory store
```

---

## 8. Grafana Dashboard Customization

Dashboard JSON is in `grafana/provisioning/dashboards/`:

```bash
# Edit existing dashboard
grafana/provisioning/dashboards/phoenix-ml-dashboard.json

# Add new dashboard — drop a JSON file in the same directory
# Grafana auto-provisions on restart
docker compose restart grafana
```

**Datasource**: Prometheus at `http://prometheus:9090` (pre-configured).

---

## Summary: Minimal Steps to Add a New Model

```bash
# 1. Create training script
mkdir -p examples/my_problem
# Implement train_and_export() in train.py

# 2. Create config
cat > model_configs/my-model.yaml << 'EOF'
model_id: my-model
version: v1
framework: onnx
task_type: classification
model_path: models/my_model/v1/model.onnx
train_script: examples/my_problem/train.py
feature_names: [f1, f2, f3]
EOF

# 3. Train
uv run python examples/my_problem/train.py models/my_model/v1/model.onnx

# 4. Deploy
docker compose restart api

# 5. Test
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my-model", "features": [1.0, 2.0, 3.0]}'
```
