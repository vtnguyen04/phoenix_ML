"""
[EXAMPLE] Train House Price Regression Model — California Housing.

This is a REFERENCE IMPLEMENTATION for a REGRESSION task,
demonstrating the framework's multi-task-type support.

Uses scikit-learn's California Housing dataset with GradientBoosting.
Exports to ONNX format for inference.

Usage:
    python examples/house_price/train.py
    python examples/house_price/train.py --output models/house_price/v1/model.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def train_house_price(output_path: Path, metrics_path: Path) -> None:
    """Train a regression model on California Housing dataset."""
    print("📊 Loading California Housing dataset...")
    data = fetch_california_housing()
    x_data = data.data
    y_data = data.target
    feature_names = list(data.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    print(f"📐 Training set: {x_train.shape[0]} samples, {len(feature_names)} features")
    print(f"📐 Test set: {x_test.shape[0]} samples")

    # Train GradientBoosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(x_train, y_train)

    # Evaluate
    y_pred = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ MAE:  {mae:.4f}")
    print(f"✅ R²:   {r2:.4f}")

    # Export to ONNX
    n_features = x_train.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"📦 ONNX model saved to {output_path}")

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
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"📊 Metrics saved to {metrics_path}")


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
) -> None:
    """
    Framework-standard entry point.

    Called by the self-healing DAG via _resolve_train_function().
    Matches the same signature as examples/credit_risk/train.train_and_export().
    """
    out = Path(output_path)
    met = Path(metrics_path) if metrics_path else out.parent / "metrics.json"
    train_house_price(out, met)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train house price model")
    parser.add_argument(
        "--output",
        type=str,
        default="models/house_price/v1/model.onnx",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="models/house_price/v1/metrics.json",
        help="Output metrics path",
    )
    args = parser.parse_args()
    train_house_price(Path(args.output), Path(args.metrics))

