"""
[EXAMPLE] Seed Real Features — German Credit (30 Features with Engineering).

This is a REFERENCE IMPLEMENTATION for the German Credit dataset.
Adapt this script for your own ML problem type and dataset.

Usage:
    python scripts/seed_features.py
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
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


def _engineer_features(X: np.ndarray) -> np.ndarray:
    """Same feature engineering as train_model.py."""
    eps = 1e-8
    dur, credit, inst, age = X[:, 0:1], X[:, 1:2], X[:, 2:3], X[:, 4:5]
    existing = X[:, 5:6]
    n_num = len(NUMERIC_FEATURES)
    emp = X[
        :,
        n_num + CATEGORICAL_FEATURES.index("employment") : n_num
        + CATEGORICAL_FEATURES.index("employment")
        + 1,
    ]
    chk = X[
        :,
        n_num + CATEGORICAL_FEATURES.index("checking_status") : n_num
        + CATEGORICAL_FEATURES.index("checking_status")
        + 1,
    ]
    sav = X[
        :,
        n_num + CATEGORICAL_FEATURES.index("savings_status") : n_num
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


def seed_features(
    output_path: str = "data/reference_features.json",
    num_records: int = 100,
) -> None:
    """Generate real feature records with 30 features (20 base + 10 eng)."""
    print("📥 Fetching German Credit dataset...")
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

    records = []
    for i in range(min(num_records, len(X))):
        features = {name: round(float(X[i, j]), 6) for j, name in enumerate(FINAL_FEATURE_NAMES)}
        records.append(
            {
                "entity_id": f"customer-{i:04d}",
                "features": features,
                "label": int(y[i]),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"✅ {len(records)} records ({len(FINAL_FEATURE_NAMES)} features)")

    ref_path = path.parent / "reference_data.json"
    reference: dict[str, Any] = {
        "feature_names": FINAL_FEATURE_NAMES,
        "n_features": len(FINAL_FEATURE_NAMES),
        "reference_distributions": {},
    }
    for i, fname in enumerate(FINAL_FEATURE_NAMES):
        reference["reference_distributions"][fname] = X[:, i].tolist()
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"✅ Reference distributions → {ref_path}")


if __name__ == "__main__":
    seed_features()
