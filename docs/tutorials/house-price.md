# Tutorial: House Price Prediction — Step-by-Step Guide

A **practical, detailed** guide to using Phoenix ML framework for House Price Prediction (Regression).

> **Summary**: Build a regression model to estimate the market value of properties.

---

## 📁 User Directory Structure

```
my-housing-project/
├── .env                           # ← User configures infra URLs here
├── model_configs/
│   └── house-price.yaml           # ← Define use case
├── my_training/
│   └── train.py                   # ← Training script
├── data/
│   └── housing/
│       └── properties.csv         # ← Dataset
└── models/                        # ← Framework auto-creates during training
```

---

## 1️⃣ Step 1 — Define your use case (YAML Config)

```yaml
# model_configs/house-price.yaml
model_id: house-price
version: v1
framework: onnx
task_type: regression
model_path: models/house_price/v1/model.onnx
train_script: my_training/train.py

# Features
feature_names:
  - medinc
  - houseage
  - averooms
  - avebedrms
  - population
  - aveoccup
  - latitude
  - longitude

# Data source
data_source:
  type: file
  path: data/housing/properties.csv

# Retrain strategy
retrain:
  trigger: drift
  drift_detection: true

# Monitoring
monitoring:
  drift_test: wasserstein         # Wasserstein distance is best for continuous regression targets
  drift_threshold: 0.1
  primary_metric: rmse            # Use Root Mean Squared Error for regression
```

---

## 2️⃣ Step 2 — Write Training Script

```python
# my_training/train.py
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def train_and_export(output_path: str, metrics_path: str = None, data_path: str = None, reference_path: str = None) -> None:
    # 1. Load data
    df = pd.read_csv(data_path or "data/housing/properties.csv")
    X = df.drop(columns=["price"]).values
    y = df["price"].values

    # 2. Train Regressor
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X, y)

    # 3. Evaluate
    y_pred = model.predict(X)
    metrics = {
        "rmse": float(mean_squared_error(y, y_pred, squared=False)),
        "r2_score": float(r2_score(y, y_pred))
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

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "house-price", 
    "features": [8.3, 41.0, 6.9, 1.0, 322.0, 2.5, 37.8, -122.2]
  }'

# Response:
# {"result": 452000.0, "confidence": 1.0, "latency_ms": 12.0}
# (result = predicted house price)
```
