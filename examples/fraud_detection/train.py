"""
[EXAMPLE] Train Fraud Detection Model — XGBoost Classification.

Demonstrates XGBoost integration with the Phoenix ML framework.
Uses the IEEE-CIS Fraud Detection dataset (synthetic fallback for offline).

Usage:
    python examples/fraud_detection/train.py
    python examples/fraud_detection/train.py --output models/fraud_detection/v1/model.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Feature names for the synthetic fraud dataset
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


def train_fraud_model(output_path: Path, metrics_path: Path) -> None:
    """Train an XGBoost fraud detection model."""
    print("📊 Generating synthetic fraud detection dataset...")

    # Generate imbalanced classification dataset (fraud is rare ~5%)
    x_data, y_data = make_classification(
        n_samples=5000,
        n_features=N_FEATURES,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )
    x_data = x_data.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    print(f"📐 Training set: {x_train.shape[0]} samples, {N_FEATURES} features")
    print(f"   Class distribution: legit={sum(y_train == 0)}, fraud={sum(y_train == 1)}")

    # XGBoost with scale_pos_weight to handle imbalance
    n_legit = sum(y_train == 0)
    n_fraud = sum(y_train == 1)
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
        "n_fraud_train": int(sum(y_train == 1)),
        "n_legit_train": int(sum(y_train == 0)),
        "dataset": "synthetic-fraud",
        "model_type": "XGBClassifier",
        "all_features": FEATURE_NAMES,
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        if k not in ("all_features",):
            print(f"   {k}: {v}")

    # Save metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics → {metrics_path}")

    # Export to ONNX via onnxmltools (XGBoost native support)
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_xgboost(model, initial_types=initial_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ ONNX → {output_path}")


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
) -> None:
    """
    Framework-standard entry point.

    Called by the self-healing DAG via _resolve_train_function().
    """
    out = Path(output_path)
    met = Path(metrics_path) if metrics_path else out.parent / "metrics.json"
    train_fraud_model(out, met)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--output",
        type=str,
        default="models/fraud_detection/v1/model.onnx",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="models/fraud_detection/v1/metrics.json",
        help="Output metrics path",
    )
    args = parser.parse_args()
    train_fraud_model(Path(args.output), Path(args.metrics))
