# Tutorial: Fraud Detection — Step-by-Step Guide

A **practical, detailed** guide to using Phoenix ML framework for Fraud Detection (Highly Imbalanced Classification).

> **Summary**: Detect fraudulent transactions in real-time.

---

## 📁 User Directory Structure

```
my-fraud-project/
├── .env                           # ← User configures infra URLs here
├── model_configs/
│   └── fraud-detection.yaml       # ← Define use case
├── my_training/
│   └── train.py                   # ← Training script
├── data/
│   └── fraud/
│       └── transactions.csv       # ← Dataset
└── models/                        # ← Framework auto-creates during training
```

---

## 1️⃣ Step 1 — Define your use case (YAML Config)

```yaml
# model_configs/fraud-detection.yaml
model_id: fraud-detection
version: v1
framework: onnx
task_type: classification
model_path: models/fraud_detection/v1/model.onnx
train_script: my_training/train.py

# Features
feature_names:
  - amount
  - time_since_last_txn
  - merchant_risk_score
  - location_distance
  # ... (and many others)

# Data source
data_source:
  type: file
  path: data/fraud/transactions.csv

# Retrain strategy
retrain:
  trigger: scheduled
  schedule: "0 2 * * *"           # Retrain every night at 2 AM
  drift_detection: true

# Monitoring
monitoring:
  drift_test: psi                 # PSI is often preferred for categorical/binned risk data
  drift_threshold: 0.2
  primary_metric: f1_score        # Accuracy is bad for imbalanced data!
```

---

## 2️⃣ Step 2 — Write Training Script

```python
# my_training/train.py
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_and_export(output_path: str, metrics_path: str = None, data_path: str = None, reference_path: str = None) -> None:
    # 1. Load data
    df = pd.read_csv(data_path or "data/fraud/transactions.csv")
    X = df.drop(columns=["is_fraud"]).values
    y = df["is_fraud"].values

    # 2. Train (Anomaly detection approach for imbalanced classes)
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X)

    # 3. Evaluate
    # IsolationForest returns -1 (anomaly) and 1 (normal). Convert to 1 (fraud) and 0 (normal)
    y_pred_raw = model.predict(X)
    y_pred = [1 if p == -1 else 0 for p in y_pred_raw]
    
    metrics = {
        "f1_score": float(f1_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred))
    }

    # 4. Export ONNX
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 5. Save metrics
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
```

---

## 3️⃣ Call the API

Detect anomalies in real velocity:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "fraud-detection", 
    "features": [1500.00, 2.5, 0.8, 5000.0]
  }'

# Response:
# {"result": 1.0, "confidence": 0.98, "latency_ms": 1.5}
# (1.0 = Fraud detected!)
```
