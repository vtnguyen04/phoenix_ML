"""
[EXAMPLE] Train House Price Regression Model — California Housing.

Reference implementation for a REGRESSION task.
Uses DataLoader to load data from disk (data/house_price/dataset.csv).

Usage:
    python examples/house_price/train.py
    python examples/house_price/train.py --output models/house_price/v1/model.onnx
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from phoenix_ml.infrastructure.data_loaders.registry import resolve_data_loader

logger = logging.getLogger(__name__)

MODEL_ID = "house-price"
DEFAULT_DATA_PATH = "data/house_price/dataset.csv"


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
    """Train GradientBoosting regressor, export to ONNX.

    Framework-standard entry point — called by Airflow self-healing DAG.
    """
    resolved_data = data_path or DEFAULT_DATA_PATH
    x_train, x_test, y_train, y_test = asyncio.run(_load_data(resolved_data))

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    print("\n🏋️ Training GradientBoostingRegressor...")
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE:  {mae:.4f}")
    print(f"✅ R²:   {r2:.4f}")

    feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    n_features = len(feature_names)

    # Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"📦 ONNX → {out}")

    # Save metrics
    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "n_features": n_features,
        "all_features": feature_names,
        "dataset": "california-housing",
        "model_type": "GradientBoostingRegressor",
        "task_type": "regression",
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
    }
    met = Path(metrics_path) if metrics_path else out.parent / "metrics.json"
    met.parent.mkdir(parents=True, exist_ok=True)
    with open(met, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics → {met}")

    # Save reference distributions
    ref_path = reference_path or str(out.parent / "reference_features.json")
    reference: dict[str, Any] = {
        "feature_names": feature_names,
        "n_features": n_features,
        "reference_distributions": {},
    }
    for i, fname in enumerate(feature_names):
        reference["reference_distributions"][fname] = x_train[:, i].tolist()
    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"✅ Reference data → {ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price model")
    parser.add_argument("--output", default="models/house_price/v1/model.onnx")
    parser.add_argument("--metrics", default="models/house_price/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to dataset CSV")
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference, args.data)
