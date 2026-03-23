# Tutorial: Credit Risk Scoring — Step-by-Step Guide

A **practical, detailed** guide to using Phoenix ML framework for Credit Risk Scoring (Tabular Classification).

> **Summary**: Build a model that predicts loan default probability based on customer profiles.

---

## 📁 User Directory Structure

```
my-credit-project/
├── .env                           # ← User configures infra URLs here
├── model_configs/
│   └── credit-risk.yaml           # ← Define use case
├── my_training/
│   └── train.py                   # ← Training script
├── data/
│   └── credit/
│       └── customers.csv          # ← Dataset
└── models/                        # ← Framework auto-creates during training
```

---

## 1️⃣ Step 1 — Define your use case (YAML Config)

```yaml
# model_configs/credit-risk.yaml
model_id: credit-risk
version: v1
framework: onnx
task_type: classification
model_path: models/credit_risk/v1/model.onnx
train_script: my_training/train.py

# Features the model receives
feature_names:
  - age
  - income
  - loan_amount
  - credit_score
  - months_employed
  - debt_to_income
  - previous_defaults

# Data source
data_source:
  type: database                  # Pulling customer data from SQL
  query: "SELECT * FROM loan_applications WHERE status = 'approved'"
  connection: main_postgres

# Retrain strategy
retrain:
  trigger: drift                  # Retrain when customer profiles shift
  drift_detection: true

# Monitoring
monitoring:
  drift_test: ks                  # Good for continuous financial data
  drift_threshold: 0.3
  primary_metric: accuracy
```

---

## 2️⃣ Step 2 — Write Training Script

```python
# my_training/train.py
import json
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost

def train_and_export(output_path: str, metrics_path: str = None, data_path: str = None, reference_path: str = None) -> None:
    # 1. Load data
    df = pd.read_csv(data_path or "data/credit/customers.csv")
    X = df.drop(columns=["default"]).values
    y = df["default"].values

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Train XGBoost
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

    # 5. Export ONNX
    update_registered_converter(XGBClassifier, 'XGBoostXGBClassifier', calculate_linear_classifier_output_shapes, convert_xgboost)
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # 6. Save metrics & reference data
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
            
    if reference_path:
        Path(reference_path).parent.mkdir(parents=True, exist_ok=True)
        with open(reference_path, "w") as f:
            json.dump(X_train[:100].tolist(), f)
```

---

## 3️⃣ Call the API

Users can predict loan default probability in real-time:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "credit-risk", 
    "features": [35, 85000, 250000, 720, 48, 0.35, 0]
  }'

# Response:
# {"result": 0.0, "confidence": 0.92, "latency_ms": 3.2}
# (0.0 = no default, 92% confidence)
```
