"""
[EXAMPLE] Train Credit-Risk Classification Model — GradientBoosting.

Reference implementation for the German Credit dataset.
Uses DataLoader to load data from disk (data/credit_risk/dataset.csv).

Usage:
    python examples/credit_risk/train.py
    python examples/credit_risk/train.py --output models/my_model/v1/model.onnx
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.infrastructure.data_loaders.registry import resolve_data_loader

logger = logging.getLogger(__name__)

MODEL_ID = "credit-risk"
DEFAULT_DATA_PATH = "data/credit_risk/dataset.csv"

FEATURE_NAMES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
    "credit_per_month",
    "age_credit_ratio",
    "installment_credit_ratio",
    "age_employment_score",
    "credit_risk_density",
    "duration_installment",
    "checking_savings_interact",
    "age_checking_interact",
    "credit_existing_interact",
    "log_credit_amount",
]
N_FEATURES = len(FEATURE_NAMES)  # 30


async def _load_data(data_path: str) -> tuple[Any, Any, Any, Any]:
    """Load and split data using the DataLoader framework."""
    loader = resolve_data_loader(MODEL_ID)
    data, info = await loader.load(data_path, target_column="target")
    (x_train, y_train), (x_test, y_test) = await loader.split(data, test_size=0.2)

    print(f"📐 Loaded: {info.num_samples} samples × {info.num_features} features")
    print(f"   Train: {len(x_train)}, Test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
    data_path: str | None = None,
) -> None:
    """Train GradientBoosting with feature engineering, export to ONNX.

    Framework-standard entry point — called by Airflow self-healing DAG.
    """
    resolved_data = data_path or DEFAULT_DATA_PATH
    x_train, x_test, y_train, y_test = asyncio.run(_load_data(resolved_data))

    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=5,
        subsample=0.8,
        max_features="sqrt",
        random_state=42,
    )

    print("\n🏋️ Training GradientBoostingClassifier (30 features)...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
    }

    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_acc = cross_val_score(model, x_all, y_all, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, x_all, y_all, cv=cv, scoring="f1")

    metrics = {
        **test_metrics,
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": N_FEATURES,
        "dataset": "german-credit-openml",
        "model_type": "GradientBoostingClassifier",
        "all_features": FEATURE_NAMES,
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        if k != "all_features":
            print(f"   {k}: {v}")

    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics → {metrics_path}")

    # Export to ONNX
    pipeline = Pipeline([("classifier", model)])
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ ONNX → {path}")

    # Save reference distributions for drift detection
    ref_path = reference_path or str(
        Path(output_path).parent.parent.parent / "data" / "reference_data.json"
    )
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
    parser = argparse.ArgumentParser(description="Train credit risk champion model")
    parser.add_argument("--output", default="models/credit_risk/v1/model.onnx")
    parser.add_argument("--metrics", default="models/credit_risk/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to dataset CSV")
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference, args.data)
