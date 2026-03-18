"""
Train Challenger Model — LogisticRegression on 30 Features.

Uses same feature engineering as champion but LogisticRegression
for A/B comparison.

Usage:
    python scripts/train_challenger.py
"""

import json
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
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
ALL_BASE = NUMERIC_FEATURES + CATEGORICAL_FEATURES
ENGINEERED = [
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
ALL_FEATURES = ALL_BASE + ENGINEERED
N_FEATURES = len(ALL_FEATURES)


def _engineer_features(X: np.ndarray) -> np.ndarray:
    eps = 1e-8
    dur, credit, inst, age = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 4:5]
    existing = X[:, 5:6]
    n = len(NUMERIC_FEATURES)
    emp = X[
        :,
        n + CATEGORICAL_FEATURES.index("employment") : n
        + CATEGORICAL_FEATURES.index("employment")
        + 1,
    ]
    chk = X[
        :,
        n + CATEGORICAL_FEATURES.index("checking_status") : n
        + CATEGORICAL_FEATURES.index("checking_status")
        + 1,
    ]
    sav = X[
        :,
        n + CATEGORICAL_FEATURES.index("savings_status") : n
        + CATEGORICAL_FEATURES.index("savings_status")
        + 1,
    ]
    return np.hstack(
        [
            X,
            credit / (dur + eps),
            age / (credit + eps),
            inst / (credit + eps),
            age * (emp + 1),
            credit * inst / (dur + eps),
            dur * inst,
            chk * sav,
            age * chk,
            credit * existing,
            np.log1p(np.abs(credit)),
        ]
    )


def train_challenger(
    output_path: str = "models/credit_risk/v2/model.onnx",
    metrics_path: str = "models/credit_risk/v2/metrics.json",
) -> None:
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    X_base = preprocessor.fit_transform(df[ALL_BASE]).astype(np.float32)
    X = _engineer_features(X_base).astype(np.float32)
    y = (df["class"] == "good").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=99,
        stratify=y,
    )

    model = LogisticRegression(C=0.5, max_iter=1000, random_state=99)
    pipeline = Pipeline([("classifier", model)])
    print("🏋️ Training LogisticRegression (Challenger, 30 features)...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "n_features": N_FEATURES,
        "model_type": "LogisticRegression",
    }
    print(f"📊 {metrics}")

    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ {output_path}")


if __name__ == "__main__":
    train_challenger()
