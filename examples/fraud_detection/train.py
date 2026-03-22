"""
[EXAMPLE] Train Fraud Detection Model — XGBoost Classification.

Demonstrates XGBoost integration with the Phoenix ML framework.
Uses DataLoader to load data from disk (data/fraud_detection/dataset.csv).

Usage:
    python examples/fraud_detection/train.py
    python examples/fraud_detection/train.py --output models/fraud_detection/v1/model.onnx
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from phoenix_ml.infrastructure.data_loaders.registry import resolve_data_loader

logger = logging.getLogger(__name__)

MODEL_ID = "fraud-detection"
DEFAULT_DATA_PATH = "data/fraud_detection/dataset.csv"

FEATURE_NAMES = [
    "transaction_amount",
    "merchant_category",
    "card_type",
    "transaction_hour",
    "distance_from_home",
    "distance_from_last_txn",
    "ratio_to_median_amount",
    "is_weekend",
    "is_night",
    "num_txn_last_24h",
    "num_txn_last_7d",
    "avg_amount_last_30d",
]
N_FEATURES = len(FEATURE_NAMES)


async def _load_data(data_path: str) -> tuple[Any, Any, Any, Any]:
    """Load and split data using the DataLoader framework."""
    loader = resolve_data_loader(MODEL_ID)
    data, info = await loader.load(data_path, target_column="target")
    (x_train, y_train), (x_test, y_test) = await loader.split(data, test_size=0.2)

    print(f"📐 Loaded: {info.num_samples} samples × {info.num_features} features")
    print(f"   Train: {len(x_train)}, Test: {len(x_test)}")
    print(f"   Class distribution: legit={int(sum(y_train == 0))}, fraud={int(sum(y_train == 1))}")
    return x_train, x_test, y_train, y_test


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
    data_path: str | None = None,
) -> None:
    """Train XGBoost fraud detection model.

    Framework-standard entry point — called by Airflow self-healing DAG.
    """
    resolved_data = data_path or DEFAULT_DATA_PATH
    x_train, x_test, y_train, y_test = asyncio.run(_load_data(resolved_data))

    # XGBoost with scale_pos_weight to handle imbalance
    n_legit = int(sum(y_train == 0))
    n_fraud = int(sum(y_train == 1))
    scale_weight = n_legit / max(n_fraud, 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    print("\n🏋️ Training XGBClassifier...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": N_FEATURES,
        "n_fraud_train": n_fraud,
        "n_legit_train": n_legit,
        "dataset": "synthetic-fraud",
        "model_type": "XGBClassifier",
        "all_features": FEATURE_NAMES,
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        if k not in ("all_features",):
            print(f"   {k}: {v}")

    # Save metrics
    met = Path(metrics_path) if metrics_path else Path(output_path).parent / "metrics.json"
    met.parent.mkdir(parents=True, exist_ok=True)
    with open(met, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics → {met}")

    # Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_xgboost(model, initial_types=initial_type)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ ONNX → {out}")

    # Save reference distributions
    ref_path = reference_path or str(out.parent / "reference_features.json")
    reference: dict[str, Any] = {
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "reference_distributions": {},
    }
    for i, fname in enumerate(FEATURE_NAMES):
        reference["reference_distributions"][fname] = x_train[:, i].tolist()
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"✅ Reference data → {ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--output", default="models/fraud_detection/v1/model.onnx")
    parser.add_argument("--metrics", default="models/fraud_detection/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to dataset CSV")
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference, args.data)
