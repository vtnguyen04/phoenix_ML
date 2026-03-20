"""
[EXAMPLE] Image Classification — Fashion-MNIST with Neural Network.

Demonstrates image classification within the Phoenix ML framework.
Uses DataLoader to load data from disk (data/image_class/dataset.npz).

Input: 784 floats (28×28 pixel values, normalized to [0, 1])
Output: 10 classes (T-shirt, Trouser, Pullover, Dress, Coat,
                     Sandal, Shirt, Sneaker, Bag, Ankle boot)

Usage:
    python examples/image_classification/train.py
    python examples/image_classification/train.py --output models/image_class/v1/model.onnx
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.infrastructure.data_loaders.registry import resolve_data_loader

logger = logging.getLogger(__name__)

MODEL_ID = "image-class"
DEFAULT_DATA_PATH = "data/image_class/dataset.npz"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
N_FEATURES = 784  # 28 × 28 pixels
N_CLASSES = 10


async def _load_data(data_path: str) -> tuple[Any, Any, Any, Any]:
    """Load and split data using the DataLoader framework."""
    loader = resolve_data_loader(MODEL_ID)
    data, info = await loader.load(
        data_path,
        normalize=True,
        class_names=CLASS_NAMES,
        max_samples=25000,
    )
    (x_train, y_train), (x_test, y_test) = await loader.split(data, test_size=0.2)

    print(f"📐 Loaded: {info.num_samples} samples × {info.num_features} features")
    print(f"   Classes: {N_CLASSES} ({', '.join(CLASS_NAMES[:5])}...)")
    print(f"   Train: {len(x_train)}, Test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
    data_path: str | None = None,
) -> None:
    """Train MLP image classifier, export to ONNX.

    Framework-standard entry point — called by Airflow self-healing DAG.
    """
    resolved_data = data_path or DEFAULT_DATA_PATH
    x_train, x_test, y_train, y_test = asyncio.run(_load_data(resolved_data))

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.001,
                    max_iter=50,
                    batch_size=256,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                    verbose=True,
                ),
            ),
        ]
    )

    print("\n🏋️ Training MLPClassifier (256→128 hidden layers)...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "image_size": "28x28",
        "dataset": "fashion-mnist",
        "model_type": "MLPClassifier",
        "class_names": CLASS_NAMES,
        "per_class_f1": {name: round(report[name]["f1-score"], 4) for name in CLASS_NAMES},
    }

    print(f"\n📊 Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}")

    # Save metrics
    met = Path(metrics_path) if metrics_path else Path(output_path).parent / "metrics.json"
    met.parent.mkdir(parents=True, exist_ok=True)
    with open(met, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics → {met}")

    # Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ ONNX → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument("--output", default="models/image_class/v1/model.onnx")
    parser.add_argument("--metrics", default="models/image_class/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to dataset NPZ")
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference, args.data)
