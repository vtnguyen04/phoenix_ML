# Getting Started — Build Your ML System with Phoenix ML

Hướng dẫn từ A đến Z: cách sử dụng Phoenix ML framework để xây dựng hệ thống ML production cho **bài toán của bạn**.

!!! info "Bạn chỉ cần 2 thứ"
    1. **YAML config** — Định nghĩa bài toán
    2. **Training script** — Train + export ONNX

    Framework lo hết phần còn lại: inference, monitoring, drift detection, auto-retrain, A/B testing, logging.

---

## Tổng quan: User Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. Cài đặt framework                                  │
│  2. Tạo model_configs/my-model.yaml   ← Bài toán       │
│  3. Viết my_project/train.py          ← Train + ONNX   │
│  4. Chạy framework (docker/local)                       │
│  5. POST /predict → inference                           │
│     ↓ Framework tự lo:                                  │
│     ├── ONNX inference (sub-50ms)                       │
│     ├── Drift monitoring                                │
│     ├── Auto retrain (drift/schedule/manual/data_change)│
│     ├── MLflow experiment logging                       │
│     ├── Champion/Challenger A/B testing                 │
│     └── Prometheus metrics + alerting                   │
└─────────────────────────────────────────────────────────┘
```

---

## Bước 1: Cài đặt

```bash
# Clone repo + install
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML
uv sync   # hoặc pip install -e .
```

---

## Bước 2: Định nghĩa bài toán (YAML Config)

Tạo file `model_configs/<model-id>.yaml`:

### Cấu trúc config

```yaml
# ── BẮT BUỘC ──────────────────────────────
model_id: my-model              # ID duy nhất
version: v1
framework: onnx                 # onnx | tensorrt | triton
task_type: classification       # classification | regression | object_detection | nlp | custom
model_path: models/my_model/v1/model.onnx
train_script: my_project/train.py

# ── FEATURES (bỏ trống nếu raw input) ────
feature_names:
  - feature_1
  - feature_2

# ── DATA SOURCE ──────────────────────────
data_source:
  type: file                    # file | database | dvc
  # type: database → thêm:
  #   query: "SELECT * FROM features WHERE ..."
  #   connection: "my_postgres"
  # type: dvc → thêm:
  #   path: data/images/

# ── RETRAIN STRATEGY ─────────────────────
retrain:
  trigger: drift                # drift | manual | data_change | scheduled
  # trigger: scheduled → thêm:
  #   schedule: "0 0 * * 0"    # Cron expression
  drift_detection: true         # false cho object_detection, NLP

# ── MONITORING ───────────────────────────
monitoring:
  drift_test: ks                # ks | psi | wasserstein | chi2
  primary_metric: accuracy      # accuracy | rmse | f1_score | map
```

---

## Bước 3: Viết Training Script

Framework yêu cầu 1 function: `train_and_export(output_path, **kwargs)`

```python
# my_project/train.py
def train_and_export(output_path, metrics_path=None, data_path=None):
    """Train model và export ra ONNX.

    Args:
        output_path: Đường dẫn save model.onnx
        metrics_path: (Optional) Đường dẫn save metrics.json
        data_path: (Optional) Đường dẫn đến training data
    """
    import json
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from pathlib import Path

    # 1. Load data
    df = pd.read_csv(data_path or "data/my_model/dataset.csv")
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # 2. Train
    model = LogisticRegression()
    model.fit(X, y)

    # 3. Export ONNX
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 4. Save metrics (optional, dùng cho MLflow logging)
    if metrics_path:
        metrics = {"accuracy": float(model.score(X, y)), "n_samples": len(y)}
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
```

---

## Bước 4: Chạy Framework

### Option A: Docker Compose (khuyến nghị)

```bash
# Khởi động toàn bộ stack: API, PostgreSQL, Redis, Kafka, MLflow, Airflow, Grafana
docker compose up -d

# Xem logs
docker compose logs -f api
```

### Option B: Local (dev)

```bash
# Chạy server trực tiếp
uv run python -m src.infrastructure.http.fastapi_server
# → http://localhost:8000
```

---

## Bước 5: Sử dụng API

### Predict

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "my-model", "features": [1.5, 2.3, 0.7]}'

# Response:
# {"result": 1, "confidence": 0.92, "model_version": "v1", "latency_ms": 12.5}
```

### Batch Predict

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "my-model",
    "batch": [
      {"features": [1.5, 2.3, 0.7]},
      {"features": [0.8, 1.1, 2.4]}
    ]
  }'
```

### Manual Retrain

```bash
curl -X POST http://localhost:8000/models/my-model/retrain
# → Triggers Airflow DAG to retrain your model
```

### Check Drift

```bash
curl http://localhost:8000/monitoring/drift-report?model_id=my-model
```

---

## Ví dụ thực tế

### Ví dụ 1: Sentiment Analysis

```yaml
# model_configs/sentiment.yaml
model_id: sentiment
version: v1
framework: onnx
task_type: classification
model_path: models/sentiment/v1/model.onnx
train_script: examples/sentiment/train.py
data_path: data/sentiment/reviews.csv
dataset_name: product-reviews

feature_names:
  - text_length
  - avg_word_length
  - positive_word_count
  - negative_word_count
  - exclamation_count
  - question_count
  - capital_ratio
  - tfidf_0
  - tfidf_1
  - tfidf_2
  # ... (TF-IDF hoặc embedding features)

data_source:
  type: file

retrain:
  trigger: scheduled
  schedule: "0 0 * * 0"       # Mỗi chủ nhật
  drift_detection: true

monitoring:
  drift_test: psi
  primary_metric: f1_score
```

**User app preprocessing:**

```python
# Client-side: text → features → API call
import httpx

def predict_sentiment(text: str) -> dict:
    features = extract_features(text)  # TF-IDF, word counts, etc.
    resp = httpx.post("http://localhost:8000/predict", json={
        "model_id": "sentiment",
        "features": features
    })
    return resp.json()
    # → {"result": 1, "confidence": 0.94}  ← 1 = positive

predict_sentiment("Sản phẩm rất tốt, giao hàng nhanh!")
```

---

### Ví dụ 2: Product Recommendation

```yaml
# model_configs/product-recommend.yaml
model_id: product-recommend
version: v1
framework: onnx
task_type: regression
model_path: models/product_recommend/v1/model.onnx
train_script: examples/recommendation/train.py
data_path: data/recommend/interactions.csv

feature_names:
  - user_age
  - user_purchase_count
  - user_avg_spend
  - item_category
  - item_price
  - item_rating
  - user_item_similarity
  - days_since_last_purchase

data_source:
  type: database
  query: "SELECT * FROM user_item_features WHERE created_at > '{{last_train_date}}'"
  connection: main_postgres

retrain:
  trigger: drift
  drift_detection: true

monitoring:
  drift_test: wasserstein
  primary_metric: rmse
```

**User app:**

```python
# Get recommendation score for user + item
resp = httpx.post("http://localhost:8000/predict", json={
    "model_id": "product-recommend",
    "features": [25, 12, 150.0, 3, 299.99, 4.5, 0.85, 7]
})
score = resp.json()["result"]  # 0.92 → highly relevant
```

---

## Retrain Triggers — Chi tiết

| Trigger | Khi nào dùng | Cách hoạt động |
|---------|-------------|----------------|
| `drift` | Tabular data, production traffic | Framework tự detect drift → trigger Airflow DAG |
| `manual` | Khi user muốn kiểm soát hoàn toàn | `POST /models/{id}/retrain` hoặc Airflow UI |
| `data_change` | DVC datasets (images, text corpora) | DAG check `dvc status` mỗi 6h → retrain nếu data thay đổi |
| `scheduled` | Retrain định kỳ | Cron expression: `"0 0 * * 0"` = mỗi chủ nhật |

---

## Setup DVC (cho datasets lớn)

Nếu bài toán cần version dữ liệu lớn (ảnh, text corpora):

```bash
# 1. Init DVC
dvc init

# 2. Config remote storage (MinIO, S3, GCS)
dvc remote add -d storage s3://my-bucket/dvc-data
dvc remote modify storage endpointurl http://minio:9000

# 3. Track dataset
dvc add data/my_dataset/
git add data/my_dataset.dvc .gitignore

# 4. Push data
dvc push

# 5. Set model config:
#    data_source.type: dvc
#    retrain.trigger: data_change
```

---

## Monitoring Dashboard

Sau khi deploy, truy cập:

| Service | URL | Chức năng |
|---------|-----|-----------|
| **API** | `http://localhost:8000/docs` | Swagger UI |
| **Grafana** | `http://localhost:3000` | Metrics dashboards |
| **MLflow** | `http://localhost:5000` | Experiment tracking |
| **Airflow** | `http://localhost:8080` | Pipeline management |
| **Frontend** | `http://localhost:5173` | React dashboard |

---

## FAQ

**Q: Framework hỗ trợ framework ML nào?**
Bất kỳ framework nào export được ONNX: scikit-learn, XGBoost, LightGBM, PyTorch, TensorFlow, Keras.

**Q: Input format là gì?**
Array of floats: `[f1, f2, f3, ...]`. User tự preprocessing (text → TF-IDF, image → embedding, etc.) trước khi gọi API.

**Q: Có thể chạy nhiều model cùng lúc không?**
Có. Mỗi YAML config = 1 model. Framework load tất cả model configs khi khởi động.

**Q: Model tự retrain thế nào?**
Framework detect drift → trigger Airflow DAG → chạy `train_script` → export ONNX mới → log MLflow → register challenger → so sánh với champion.
