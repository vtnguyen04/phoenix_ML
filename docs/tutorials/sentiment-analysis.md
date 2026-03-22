# Tutorial: Sentiment Analysis — Chi tiết từ A đến Z

Hướng dẫn **thực tế, chi tiết** cách sử dụng Phoenix ML framework cho bài toán Sentiment Analysis.

> **Tóm tắt**: User chỉ cần (1) định nghĩa config, (2) viết training script, (3) đổi `.env` cho infra. Framework lo hết phần còn lại.

---

## 📁 Cấu trúc thư mục của User

```
my-sentiment-project/
├── .env                           # ← User config infra URLs tại đây
├── model_configs/
│   └── sentiment.yaml             # ← Định nghĩa bài toán
├── my_training/
│   ├── train.py                   # ← Training script
│   ├── data_loader.py             # ← (Tùy chọn) Custom data loader
│   └── preprocessor.py            # ← (Tùy chọn) Custom preprocessor
├── data/
│   └── sentiment/
│       └── reviews.csv            # ← Dataset
└── models/                        # ← Framework tự tạo khi train
```

---

## 1️⃣ Bước 1 — Định nghĩa bài toán (YAML Config)

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

# Features mà model nhận (sau khi preprocessing)
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
  # ... (tùy vào cách bạn extract features)

# Custom data loader (tùy chọn)
# Nếu để trống → framework dùng TabularDataLoader mặc định
data_loader: my_training.data_loader.SentimentDataLoader

# Nguồn dữ liệu
data_source:
  type: file                      # file | database | dvc
  # type: database → thêm:
  #   query: "SELECT text, label FROM reviews WHERE ..."
  #   connection: main_postgres

# Chiến lược retrain
retrain:
  trigger: scheduled              # drift | manual | data_change | scheduled
  schedule: "0 0 * * 0"           # Cron: mỗi chủ nhật lúc 00:00
  drift_detection: true

# Monitoring
monitoring:
  drift_test: psi                 # ks | psi | wasserstein | chi2
  drift_threshold: 0.25
  primary_metric: f1_score
```

### Giải thích từng field

| Field | Bắt buộc | Mô tả |
|-------|---------|-------|
| `model_id` | ✅ | ID duy nhất, dùng trong API call |
| `task_type` | ✅ | `classification` → framework tự chọn defaults phù hợp |
| `train_script` | ✅ | Path tới function `train_and_export()` |
| `feature_names` | ✅ | Danh sách features, dùng cho drift monitoring |
| `data_loader` | ❌ | Custom class path, nếu trống → `TabularDataLoader` |
| `data_source.type` | ❌ | Default: `file`. `database` cho SQL, `dvc` cho large datasets |
| `retrain.trigger` | ❌ | Default: `drift`. Xem bảng triggers bên dưới |
| `monitoring.drift_test` | ❌ | Default: `ks` cho classification |

---

## 2️⃣ Bước 2 — Viết Training Script

### Cơ bản (đủ để chạy)

```python
# my_training/train.py
"""Training script cho sentiment analysis.

Framework yêu cầu function: train_and_export(output_path, **kwargs)
Tham số tùy chọn: metrics_path, data_path, reference_path
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
    """Trích xuất features từ text.

    ĐÂY LÀ PHẦN USER TỰ DEFINE — logic preprocessing riêng cho bài toán.
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
    """Entry point — Framework gọi function này khi retrain."""

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

    # 6. Save metrics (framework gửi lên MLflow)
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    # 7. Save reference data (framework dùng cho drift detection)
    if reference_path:
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        ref_data = X_train[:100].tolist()  # 100 samples làm baseline
        with open(reference_path, "w") as f:
            json.dump(ref_data, f)

    print(f"✅ Model saved: {output_path}")
    print(f"📊 Metrics: {metrics}")
```

---

## 3️⃣ Tùy chọn: Tinh chỉnh Hyperparameters (Optuna)

Framework tích hợp Optuna — user **chỉ cần gọi** trong training script:

```python
# my_training/train.py — phiên bản nâng cao với Optuna
from src.domain.training.services.optuna_optimizer import OptunaOptimizer


def train_and_export(output_path, metrics_path=None, data_path=None, **kwargs):
    df = pd.read_csv(data_path or "data/sentiment/reviews.csv")
    X = extract_text_features(df["text"])
    y = df["label"].values

    # ── Tự động tìm hyperparameters tốt nhất ──
    optimizer = OptunaOptimizer(
        n_trials=50,                 # Số lần thử
        task_type="classification",  # hoặc "regression"
        metric="f1",                 # accuracy | f1 | rmse | r2
    )
    result = optimizer.optimize(X, y)
    print(f"Best params: {result.best_params}")
    print(f"Best F1: {result.best_score:.4f}")

    # ── Train final model với best params ──
    from xgboost import XGBClassifier
    model = XGBClassifier(**result.best_params)
    model.fit(X, y)

    # Export ONNX...
```

### Custom search space

```python
# User tự define search space
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

## 4️⃣ Tùy chọn: Custom Data Loader

Nếu data format đặc biệt, user implement `IDataLoader`:

```python
# my_training/data_loader.py
from src.domain.training.services.data_loader_plugin import IDataLoader, DatasetInfo
import pandas as pd


class SentimentDataLoader(IDataLoader):
    """Custom data loader cho sentiment analysis.

    Xử lý CSV có cột 'text' và 'label' (positive/negative).
    """

    async def load(self, data_path: str, **kwargs) -> tuple:
        df = pd.read_csv(data_path)

        # Custom: đọc nhiều sources
        if data_path.endswith(".jsonl"):
            df = pd.read_json(data_path, lines=True)

        # Custom logic: lọc data, clean text, etc.
        df["text"] = df["text"].str.lower().str.strip()
        df = df.dropna(subset=["text", "label"])

        info = DatasetInfo(
            num_samples=len(df),
            num_features=0,  # text, chưa extract features
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

**Đăng ký trong YAML:**
```yaml
# model_configs/sentiment.yaml
data_loader: my_training.data_loader.SentimentDataLoader
```

**Hoặc đăng ký bằng code:**
```python
from src.infrastructure.data_loaders.registry import DataLoaderRegistry
DataLoaderRegistry.register("sentiment", SentimentDataLoader)
```

### Nếu KHÔNG custom?

Không cần làm gì — framework tự dùng `TabularDataLoader` (đọc CSV, split train/test).

---

## 5️⃣ Tùy chọn: Custom Pre/Post Processor

Nếu muốn framework xử lý data **trước/sau inference** (thay vì client tự làm):

```python
# my_training/preprocessor.py
from src.domain.inference.services.processor_plugin import IPreprocessor, IPostprocessor
import numpy as np


class SentimentPreprocessor(IPreprocessor):
    """Text → feature vector trước khi vào model."""

    async def preprocess(self, raw_input: dict, model_config: dict) -> list[float]:
        # Nhận text từ API, trả về feature vector
        text = raw_input.get("text", "")
        features = []
        features.append(float(len(text)))                          # text_length
        words = text.split()
        features.append(np.mean([len(w) for w in words]) if words else 0)
        features.append(float(text.count("!")))
        features.append(float(text.count("?")))
        # ... thêm TF-IDF features
        return features


class SentimentPostprocessor(IPostprocessor):
    """Model output → response có label text."""

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

**Đăng ký bằng code (trong startup script):**
```python
from src.infrastructure.bootstrap.container import plugin_registry

plugin_registry.register_preprocessor("sentiment", SentimentPreprocessor())
plugin_registry.register_postprocessor("sentiment", SentimentPostprocessor())
```

**Kết quả**: User gọi API với **raw text**, framework tự xử lý:
```bash
POST /predict
{"model_id": "sentiment", "text": "Sản phẩm rất tốt!"}
# → {"sentiment": "positive", "confidence": 0.94, "label_id": 1}
```

### Nếu KHÔNG custom?

Không cần! User gửi `features: [float, ...]` trực tiếp. Framework dùng `PassthroughPreprocessor` (pass-through) + `ClassificationPostprocessor` (argmax + confidence).

---

## 6️⃣ Infrastructure — User chỉ cần đổi `.env`

Framework đọc **tất cả connection URLs từ environment variables**. User chỉ cần **đổi 1 file `.env`**:

```bash
# .env — Copy từ .env.example và đổi URLs

# ── Cloud PostgreSQL ──────────────────────
# Local: sqlite+aiosqlite:///./phoenix.db (mặc định, KHÔNG cần setup gì)
# Cloud: đổi URL → framework TỰ CONNECT
DATABASE_URL=postgresql+asyncpg://user:pass@my-cloud-db.aws.com:5432/phoenix

# ── Cloud Redis ───────────────────────────
# false = framework dùng in-memory (không cần Redis server)
# true + URL = framework tự connect Redis
USE_REDIS=true
REDIS_URL=redis://my-redis.aws.com:6379

# ── Kafka ─────────────────────────────────
# Framework auto-connect. Không có Kafka? Framework vẫn chạy (graceful degradation)
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

### Bảng: User cần setup gì?

| Service | Bắt buộc? | Nếu không có? |
|---------|----------|--------------|
| **PostgreSQL** | ❌ Không | Framework dùng SQLite (tự tạo file) |
| **Redis** | ❌ Không | `USE_REDIS=false` → dùng in-memory cache |
| **Kafka** | ❌ Không | Graceful degradation, log events locally |
| **MLflow** | ❌ Không | Metrics vẫn save local, không gửi MLflow |
| **Airflow** | ❌ Không | Manual retrain qua API thay vì pipeline |
| **Grafana** | ❌ Không | Prometheus metrics vẫn expose, không có dashboards |

> **Kết luận**: Framework chạy được **KHÔNG CẦN bất kỳ service nào** — chỉ cần Python + model file. Services là optional, bật lên bằng cách đổi URLs trong `.env`.

---

## 7️⃣ Deploy & Chạy

### Local (không Docker)

```bash
# Chỉ cần Python
uv run python -m src.infrastructure.http.fastapi_server
# → http://localhost:8000/docs
```

### Docker Compose (full stack)

```bash
# Clone repo, đổi .env, run
docker compose up -d
# → API: localhost:8000, Grafana: localhost:3000, MLflow: localhost:5000
```

### Cloud (user tự deploy)

```bash
# 1. Push Docker image
docker build -t my-registry.com/phoenix-ml .
docker push my-registry.com/phoenix-ml

# 2. Deploy với Helm chart (reference)
helm install phoenix-ml deploy/helm/phoenix-ml/ \
  --set database.url="postgresql://..." \
  --set redis.url="redis://..." \
  --set mlflow.url="http://..."
```

---

## 8️⃣ Gọi API

```bash
# ── Predict (gửi features đã extract) ──────────────
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "sentiment", "features": [120, 5.2, 0.3, 0.05, 2, 0, 0.1, ...]}'

# ── Hoặc gửi raw text (nếu có custom preprocessor) ──
curl -X POST http://localhost:8000/predict \
  -d '{"model_id": "sentiment", "text": "Sản phẩm rất tuyệt vời!"}'

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
curl http://localhost:8000/monitoring/drift-report?model_id=sentiment

# ── Model details ──────────────────────────────────
curl http://localhost:8000/models/sentiment
```

---

## Tóm tắt: User làm gì vs Framework làm gì

| Phần | User tự làm | Framework hỗ trợ |
|------|-------------|-------------------|
| **Định nghĩa bài toán** | ✍️ Viết YAML config | ✅ Parse, validate, smart defaults |
| **Training** | ✍️ Viết `train_and_export()` | ✅ Gọi function, manage versions |
| **Hyperparameters** | ✍️ Gọi `OptunaOptimizer` trong code | ✅ Optuna TPE integrated, log results |
| **Data loading** | ❌ Không cần (hoặc ✍️ custom) | ✅ TabularLoader/ImageLoader mặc định |
| **Preprocessing** | ❌ Không cần (hoặc ✍️ custom) | ✅ PassthroughPreprocessor mặc định |
| **Inference** | ❌ Không cần | ✅ ONNX Runtime, sub-50ms |
| **Drift monitoring** | ❌ Không cần | ✅ KS/PSI/Chi²/Wasserstein auto |
| **Retrain** | ❌ Không cần | ✅ Auto-trigger (drift/schedule/data_change) |
| **MLflow logging** | ❌ Không cần | ✅ Auto-log metrics, params, artifacts |
| **A/B testing** | ❌ Không cần | ✅ Champion/Challenger routing |
| **Infrastructure** | ✍️ Đổi URLs trong `.env` | ✅ Auto-connect, graceful degradation |
| **Deploy** | ✍️ `docker compose up` hoặc Helm | ✅ Docker/Compose/Helm reference cung cấp |
