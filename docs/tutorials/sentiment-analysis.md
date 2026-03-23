# Tutorial: Sentiment Analysis — Step-by-Step Guide

A **practical, detailed** guide to using Phoenix ML framework for Sentiment Analysis.

> **Summary**: User only needs to (1) define config, (2) write training script, (3) update `.env` for infra. The framework handles the rest.

---

## 📁 User Directory Structure

```
my-sentiment-project/
├── .env                           # ← User configures infra URLs here
├── model_configs/
│   └── sentiment.yaml             # ← Define use case
├── my_training/
│   ├── train.py                   # ← Training script
│   ├── data_loader.py             # ← (Optional) Custom data loader
│   └── preprocessor.py            # ← (Optional) Custom preprocessor
├── data/
│   └── sentiment/
│       └── reviews.csv            # ← Dataset
└── models/                        # ← Framework auto-creates during training
```

---

## 1️⃣ Step 1 — Define your use case (YAML Config)

```yaml
# model_configs/sentiment.yaml
model_id: sentiment
version: v1
framework: onnx
task_type: classification
model_path: models/sentiment/v1/model.onnx
train_script: my_training/train.py
data_path: data/sentiment/reviews.csv
dataset_name: product-reviews

# Features the model receives (after preprocessing)
feature_names:
  - text_length
  - avg_word_length
  - positive_ratio
  - negative_ratio
  - exclamation_count
  - question_count
  - capital_ratio
  - tfidf_0
  - tfidf_1
  - tfidf_2
  # ... (depends on how you extract features)

# Custom data loader (optional)
# If empty → framework uses TabularDataLoader by default
data_loader: my_training.data_loader.SentimentDataLoader

# Data source
data_source:
  type: file                      # file | database | dvc
  # type: database → add:
  #   query: "SELECT text, label FROM reviews WHERE ..."
  #   connection: main_postgres

# Retrain strategy
retrain:
  trigger: scheduled              # drift | manual | data_change | scheduled
  schedule: "0 0 * * 0"           # Cron: every Sunday at 00:00
  drift_detection: true

# Monitoring
monitoring:
  drift_test: psi                 # ks | psi | wasserstein | chi2
  drift_threshold: 0.25
  primary_metric: f1_score
```

### Field-by-field Explanation

| Field | Required | Description |
|-------|---------|-------|
| `model_id` | ✅ | Unique ID, used in API calls |
| `task_type` | ✅ | `classification` → framework selects appropriate defaults |
| `train_script` | ✅ | Path to the `train_and_export()` function |
| `feature_names` | ✅ | List of features, used for drift monitoring |
| `data_loader` | ❌ | Custom class path, if empty → `TabularDataLoader` |
| `data_source.type` | ❌ | Default: `file`. `database` for SQL, `dvc` for large datasets |
| `retrain.trigger` | ❌ | Default: `drift`. See triggers table below |
| `monitoring.drift_test` | ❌ | Default: `ks` for classification |

---

## 2️⃣ Step 2 — Write Training Script

### Basics (enough to run)

```python
# my_training/train.py
"""Training script for sentiment analysis.

Framework requires function: train_and_export(output_path, **kwargs)
Optional parameters: metrics_path, data_path, reference_path
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def extract_text_features(texts: pd.Series) -> np.ndarray:
    """Extract features from text.

    THIS IS THE USER-DEFINED PART — custom preprocessing logic for your use case.
    """
    features = pd.DataFrame()
    features["text_length"] = texts.str.len()
    features["avg_word_length"] = texts.str.split().apply(
        lambda words: np.mean([len(w) for w in words]) if words else 0
    )
    features["exclamation_count"] = texts.str.count("!")
    features["question_count"] = texts.str.count(r"\?")
    features["capital_ratio"] = texts.apply(
        lambda t: sum(1 for c in t if c.isupper()) / max(len(t), 1)
    )

    # TF-IDF features (top N)
    tfidf = TfidfVectorizer(max_features=50)
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{i}" for i in range(50)]
    )

    return pd.concat([features, tfidf_df], axis=1).values


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    data_path: str | None = None,
    reference_path: str | None = None,
) -> None:
    """Entry point — Framework calls this function during retrain."""

    # 1. Load data
    df = pd.read_csv(data_path or "data/sentiment/reviews.csv")
    X = extract_text_features(df["text"])
    y = df["label"].values  # 0=negative, 1=positive

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X.shape[1],
    }

    # 5. Export ONNX
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 6. Save metrics (framework sends to MLflow)
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # 7. Save reference data (framework uses for drift detection)
    if reference_path:
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        ref_data = X_train[:100].tolist()  # 100 samples as baseline
        with open(reference_path, "w") as f:
            json.dump(ref_data, f)

    print(f"✅ Model saved: {output_path}")
    print(f"📊 Metrics: {metrics}")
```

---

## 3️⃣ Optional: Hyperparameter Tuning (Optuna)

Framework integrates Optuna — user **just needs to call it** in the training script:

```python
# my_training/train.py — advanced version with Optuna
from phoenix_ml.domain.training.services.optuna_optimizer import OptunaOptimizer


def train_and_export(output_path, metrics_path=None, data_path=None, **kwargs):
    df = pd.read_csv(data_path or "data/sentiment/reviews.csv")
    X = extract_text_features(df["text"])
    y = df["label"].values

    # ── Automatically find best hyperparameters ──
    optimizer = OptunaOptimizer(
        n_trials=50,                 # Number of trials
        task_type="classification",  # or "regression"
        metric="f1",                 # accuracy | f1 | rmse | r2
    )
    result = optimizer.optimize(X, y)
    print(f"Best params: {result.best_params}")
    print(f"Best F1: {result.best_score:.4f}")

    # ── Train final model with best params ──
    from xgboost import XGBClassifier
    model = XGBClassifier(**result.best_params)
    model.fit(X, y)

    # Export ONNX...
```

### Custom search space

```python
# User defines search space
custom_space = {
    "n_estimators": ("int", 50, 500),
    "max_depth": ("int", 3, 10),
    "learning_rate": ("float", 0.01, 0.3),
    "subsample": ("float", 0.6, 1.0),
}

optimizer = OptunaOptimizer(
    n_trials=100,
    task_type="classification",
    search_space=custom_space,  # Override defaults
)
```

---

## 4️⃣ Optional: Custom Data Loader

If your data format is special, implement `IDataLoader`:

```python
# my_training/data_loader.py
from phoenix_ml.domain.training.services.data_loader_plugin import IDataLoader, DatasetInfo
import pandas as pd


class SentimentDataLoader(IDataLoader):
    """Custom data loader for sentiment analysis.

    Processes CSV with 'text' and 'label' columns (positive/negative).
    """

    async def load(self, data_path: str, **kwargs) -> tuple:
        df = pd.read_csv(data_path)

        # Custom: read from multiple sources
        if data_path.endswith(".jsonl"):
            df = pd.read_json(data_path, lines=True)

        # Custom logic: filter data, clean text, etc.
        df["text"] = df["text"].str.lower().str.strip()
        df = df.dropna(subset=["text", "label"])

        info = DatasetInfo(
            num_samples=len(df),
            num_features=0,  # text, features not yet extracted
            class_labels=["negative", "positive"],
            data_format="text_classification",
            metadata={"source": data_path, "avg_text_length": df["text"].str.len().mean()},
        )
        return df, info

    async def split(self, data, test_size=0.2, random_seed=42):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=test_size, random_state=random_seed)
        return train, test
```

**Register in YAML:**
```yaml
# model_configs/sentiment.yaml
data_loader: my_training.data_loader.SentimentDataLoader
```

**Or register via code:**
```python
from phoenix_ml.infrastructure.data_loaders.registry import DataLoaderRegistry
DataLoaderRegistry.register("sentiment", SentimentDataLoader)
```

### What if I don't customize?

No need — framework uses `TabularDataLoader` by default (reads CSV, splits train/test).

---

## 5️⃣ Optional: Custom Pre/Post Processor

If you want the framework to process data **before/after inference** (instead of client-side):

```python
# my_training/preprocessor.py
from phoenix_ml.domain.inference.services.processor_plugin import IPreprocessor, IPostprocessor
import numpy as np


class SentimentPreprocessor(IPreprocessor):
    """Text → feature vector before entering model."""

    async def preprocess(self, raw_input: dict, model_config: dict) -> list[float]:
        # Receives text from API, returns feature vector
        text = raw_input.get("text", "")
        features = []
        features.append(float(len(text)))                          # text_length
        words = text.split()
        features.append(np.mean([len(w) for w in words]) if words else 0)
        features.append(float(text.count("!")))
        features.append(float(text.count("?")))
        # ... add TF-IDF features
        return features


class SentimentPostprocessor(IPostprocessor):
    """Model output → response with label text."""

    async def postprocess(self, model_output: list[float], model_config: dict) -> dict:
        labels = model_config.get("class_labels", ["negative", "positive"])
        if len(model_output) == 1:
            idx = int(model_output[0] >= 0.5)
        else:
            idx = int(np.argmax(model_output))
        return {
            "sentiment": labels[idx],
            "confidence": float(max(model_output)),
            "label_id": idx,
        }
```

**Register via code (in startup script):**
```python
from phoenix_ml.infrastructure.bootstrap.container import plugin_registry

plugin_registry.register_preprocessor("sentiment", SentimentPreprocessor())
plugin_registry.register_postprocessor("sentiment", SentimentPostprocessor())
```

**Result**: User calls API with **raw text**, framework handles processing:
```bash
POST /predict
{"model_id": "sentiment", "text": "Great product!"}
# → {"sentiment": "positive", "confidence": 0.94, "label_id": 1}
```

### What if I don't customize?

Not needed! User sends `features: [float, ...]` directly. Framework uses `PassthroughPreprocessor` (pass-through) + `ClassificationPostprocessor` (argmax + confidence).

---

## 6️⃣ Infrastructure — Just Update `.env`

Framework reads **all connection URLs from environment variables**. User only needs to **update 1 file `.env`**:

```bash
# .env — Copy from .env.example and update URLs

# ── Cloud PostgreSQL ──────────────────────
# Local: sqlite+aiosqlite:///./phoenix.db (default, no setup needed)
# Cloud: update URL → framework auto-connects
DATABASE_URL=postgresql+asyncpg://user:pass@my-cloud-db.aws.com:5432/phoenix

# ── Cloud Redis ───────────────────────────
# false = framework uses in-memory (no Redis server needed)
# true + URL = framework auto-connects to Redis
USE_REDIS=true
REDIS_URL=redis://my-redis.aws.com:6379

# ── Kafka ─────────────────────────────────
# Framework auto-connects. No Kafka? Framework still runs (graceful degradation)
KAFKA_URL=my-kafka.aws.com:9092

# ── MLflow ────────────────────────────────
MLFLOW_TRACKING_URI=http://my-mlflow.aws.com:5000

# ── Jaeger ────────────────────────────────
JAEGER_ENDPOINT=http://my-jaeger.aws.com:4317

# ── Airflow ───────────────────────────────
AIRFLOW_API_URL=http://my-airflow.aws.com:8080/api/v1
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=my-secret
```

### Table: What does the user need to set up?

| Service | Required? | If absent? |
|---------|----------|--------------|
| **PostgreSQL** | ❌ No | Framework uses SQLite (auto-creates file) |
| **Redis** | ❌ No | `USE_REDIS=false` → uses in-memory cache |
| **Kafka** | ❌ No | Graceful degradation, logs events locally |
| **MLflow** | ❌ No | Metrics saved locally, not sent to MLflow |
| **Airflow** | ❌ No | Manual retrain via API instead of pipeline |
| **Grafana** | ❌ No | Prometheus metrics still exposed, no dashboards |

> **Conclusion**: Framework runs **WITHOUT any external service** — only Python + model file needed. Services are optional, enabled by updating URLs in `.env`.

---

## 7️⃣ Deploy & Run

### Local (no Docker)

```bash
# Only Python needed
uv run python -m phoenix_ml.infrastructure.http.fastapi_server
# → http://localhost:8000/docs
```

### Docker Compose (full stack)

```bash
# Clone repo, update .env, run
docker compose up -d
# → API: localhost:8001, Grafana: localhost:3001, MLflow: localhost:5001
```

### Cloud (user deploys)

```bash
# 1. Push Docker image
docker build -t my-registry.com/phoenix-ml .
docker push my-registry.com/phoenix-ml

# 2. Deploy with Helm chart (reference)
helm install phoenix-ml deploy/helm/phoenix-ml/ \
  --set database.url="postgresql://..." \
  --set redis.url="redis://..." \
  --set mlflow.url="http://..."
```

---

## 8️⃣ Call the API

```bash
# ── Predict (send extracted features) ──────────────
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentiment", "features": [120, 5.2, 0.3, 0.05, 2, 0, 0.1, ...]}'

# ── Or send raw text (if custom preprocessor exists) ──
curl -X POST http://localhost:8000/predict \
  -d '{"model_id": "sentiment", "text": "This product is amazing!"}'

# ── Batch predict ──────────────────────────────────
curl -X POST http://localhost:8000/predict/batch \
  -d '{"model_id": "sentiment", "batch": [
    {"features": [120, 5.2, ...]},
    {"features": [80, 4.1, ...]}
  ]}'

# ── Manual retrain ─────────────────────────────────
curl -X POST http://localhost:8000/models/sentiment/retrain
# → Triggers Airflow DAG

# ── Check drift ────────────────────────────────────
curl http://localhost:8000/monitoring/drift/sentiment

# ── Drift history ──────────────────────────────────
curl http://localhost:8000/monitoring/reports/sentiment

# ── Model details ──────────────────────────────────
curl http://localhost:8000/models/sentiment

# ── Export fresh data ──────────────────────────────
curl -X POST http://localhost:8000/data/export-training \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentiment", "min_samples": 10, "include_baseline": true}'
```

---

## Summary: What You Do vs What the Framework Does

| Area | User does | Framework provides |
|------|-------------|-------------------|
| **Define use case** | ✍️ Write YAML config | ✅ Parse, validate, smart defaults |
| **Training** | ✍️ Write `train_and_export()` | ✅ Calls function, manages versions |
| **Hyperparameters** | ✍️ Call `OptunaOptimizer` in code | ✅ Optuna TPE integrated, logs results |
| **Data loading** | ❌ Not needed (or ✍️ custom) | ✅ TabularLoader/ImageLoader by default |
| **Preprocessing** | ❌ Not needed (or ✍️ custom) | ✅ PassthroughPreprocessor by default |
| **Inference** | ❌ Not needed | ✅ ONNX Runtime, sub-50ms |
| **Drift monitoring** | ❌ Not needed | ✅ KS/PSI/Chi²/Wasserstein auto |
| **Retrain** | ❌ Not needed | ✅ Auto-trigger (drift/schedule/data_change) |
| **MLflow logging** | ❌ Not needed | ✅ Auto-log metrics, params, artifacts |
| **A/B testing** | ❌ Not needed | ✅ Champion/Challenger routing |
| **Infrastructure** | ✍️ Update URLs in `.env` | ✅ Auto-connect, graceful degradation |
| **Deploy** | ✍️ `docker compose up` or Helm | ✅ Docker/Compose/Helm reference provided |
