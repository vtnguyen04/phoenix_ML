"""
[EXAMPLE] Train Credit-Risk Classification Model — GradientBoosting.

This is a REFERENCE IMPLEMENTATION for the German Credit dataset.
Adapt this script for your own ML problem type and dataset.

Uses all 20 base features + 10 engineered features from German Credit
dataset. Achieves >78% accuracy, >85% F1 with robust cross-validation.

Usage:
    python examples/credit_risk/train.py
    python examples/credit_risk/train.py --output models/my_model/v1/model.onnx
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NUMERIC_FEATURES = [
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
]
CATEGORICAL_FEATURES = [
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
]
ALL_BASE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENGINEERED_NAMES = [
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
FINAL_FEATURE_NAMES = ALL_BASE_FEATURES + ENGINEERED_NAMES
N_FEATURES = len(FINAL_FEATURE_NAMES)  # 30
MIN_RETRAIN_SAMPLES = 50


def _engineer_features(X: np.ndarray) -> np.ndarray:
    """Create 10 interaction & ratio features from encoded data."""
    eps = 1e-8
    dur = X[:, 0:1]
    credit = X[:, 1:2]
    inst = X[:, 2:3]
    age = X[:, 4:5]
    existing = X[:, 5:6]
    n_num = len(NUMERIC_FEATURES)
    emp_i = n_num + CATEGORICAL_FEATURES.index("employment")
    chk_i = n_num + CATEGORICAL_FEATURES.index("checking_status")
    sav_i = n_num + CATEGORICAL_FEATURES.index("savings_status")
    emp = X[:, emp_i : emp_i + 1]
    chk = X[:, chk_i : chk_i + 1]
    sav = X[:, sav_i : sav_i + 1]

    engineered = np.hstack(
        [
            credit / (dur + eps),  # credit_per_month
            age / (credit + eps),  # age_credit_ratio
            inst / (credit + eps),  # installment_credit_ratio
            age * (emp + 1),  # age_employment_score
            credit * inst / (dur + eps),  # credit_risk_density
            dur * inst,  # duration_installment
            chk * sav,  # checking_savings_interact
            age * chk,  # age_checking_interact
            credit * existing,  # credit_existing_interact
            np.log1p(np.abs(credit)),  # log_credit_amount
        ]
    )
    return np.hstack([X, engineered])


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load German Credit dataset — local cache first, OpenML fallback."""
    # Try local reference data first (works in Docker without internet)
    local_paths = [
        Path(__file__).parent.parent / "data" / "reference_features.json",
        Path("/app/data/reference_features.json"),
    ]
    for local_path in local_paths:
        if local_path.exists():
            print(f"📥 Loading German Credit dataset from local cache: {local_path}")
            with open(local_path) as f:
                records = json.load(f)
            # Extract feature matrix and labels from seeded records
            feature_list = []
            labels = []
            for rec in records:
                features = rec.get("features", {})
                if isinstance(features, dict):
                    row = [features.get(k, 0.0) for k in FINAL_FEATURE_NAMES]
                elif isinstance(features, list):
                    row = features[:N_FEATURES]
                    row += [0.0] * max(0, N_FEATURES - len(row))
                else:
                    continue
                feature_list.append(row)
                labels.append(rec.get("label", rec.get("result", 1)))
            if len(feature_list) >= MIN_RETRAIN_SAMPLES:
                X = np.array(feature_list, dtype=np.float32)
                y = np.array(labels, dtype=int)
                print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
                print(f"   Class: good={y.sum()}, bad={len(y) - y.sum()}")
                return X, y

    # Fallback: fetch from OpenML (requires internet + write access)
    print("📥 Fetching German Credit dataset from OpenML...")
    os.makedirs(os.path.expanduser("~/scikit_learn_data"), exist_ok=True)
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_base = preprocessor.fit_transform(df[ALL_BASE_FEATURES]).astype(np.float32)
    X = _engineer_features(X_base).astype(np.float32)
    y = (df["class"] == "good").astype(int).values

    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"   Base: {len(ALL_BASE_FEATURES)}, Engineered: {len(ENGINEERED_NAMES)}")
    print(f"   Class: good={y.sum()}, bad={len(y) - y.sum()}")
    return X, y


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
) -> None:
    """Train GradientBoosting with feature engineering, export to ONNX."""
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )

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
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    metrics = {
        **test_metrics,
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": N_FEATURES,
        "dataset": "german-credit-openml",
        "model_type": "GradientBoostingClassifier",
        "all_features": FINAL_FEATURE_NAMES,
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        if k != "all_features":
            print(f"   {k}: {v}")

    importances = model.feature_importances_
    top = sorted(
        zip(FINAL_FEATURE_NAMES, importances, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )[:7]
    print("\n🔍 Top 7 features:")
    for fname, imp in top:
        print(f"   {fname}: {imp:.4f}")

    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics → {metrics_path}")

    # Export to ONNX via skl2onnx (native GradientBoosting support)
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
        "feature_names": FINAL_FEATURE_NAMES,
        "n_features": N_FEATURES,
        "reference_distributions": {},
    }
    for i, fname in enumerate(FINAL_FEATURE_NAMES):
        reference["reference_distributions"][fname] = X_train[:, i].tolist()

    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"✅ Reference data → {ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train credit risk champion model")
    parser.add_argument("--output", default="models/credit_risk/v1/model.onnx")
    parser.add_argument("--metrics", default="models/credit_risk/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference)
