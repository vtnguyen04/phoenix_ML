"""
[EXAMPLE] Image Classification — Fashion-MNIST with Neural Network.

Demonstrates image classification within the Phoenix ML framework.
Uses sklearn's MLPClassifier (multi-layer perceptron) to classify
28×28 grayscale fashion images into 10 categories.

Input: 784 floats (28×28 pixel values, normalized to [0, 1])
Output: 10 classes (T-shirt, Trouser, Pullover, Dress, Coat,
                     Sandal, Shirt, Sneaker, Bag, Ankle boot)

Usage:
    python examples/image_classification/train.py
    python examples/image_classification/train.py --output models/image_class/v1/model.onnx
"""

import argparse
import json
from pathlib import Path

import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 10 Fashion-MNIST class labels
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


def train_image_classifier(output_path: Path, metrics_path: Path) -> None:
    """Train a neural network for Fashion-MNIST image classification."""
    print("📊 Loading Fashion-MNIST dataset...")

    data = fetch_openml("Fashion-MNIST", version=1, as_frame=False, parser="auto")
    x_all = data.data.astype(np.float32)
    y_all = data.target.astype(int)

    print(f"   Total samples: {x_all.shape[0]}, Features: {x_all.shape[1]} (28×28 pixels)")
    print(f"   Classes: {N_CLASSES} ({', '.join(CLASS_NAMES[:5])}...)")

    # Normalize pixel values to [0, 1]
    x_all = x_all / 255.0

    # Use a subset for faster training (full 70k is slow for MLP)
    n_train_total = 20000
    n_test_total = 5000
    indices = np.random.RandomState(42).permutation(len(x_all))
    x_subset = x_all[indices[: n_train_total + n_test_total]]
    y_subset = y_all[indices[: n_train_total + n_test_total]]

    x_train, x_test, y_train, y_test = train_test_split(
        x_subset, y_subset, test_size=n_test_total, random_state=42, stratify=y_subset
    )

    print(f"   Train: {len(x_train)}, Test: {len(x_test)}")

    # MLP Neural Network (2 hidden layers)
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

    # Per-class report
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "image_size": "28x28",
        "dataset": "fashion-mnist",
        "model_type": "MLPClassifier",
        "class_names": CLASS_NAMES,
        "per_class_f1": {
            name: round(report[name]["f1-score"], 4) for name in CLASS_NAMES
        },
    }

    print("\n📊 Model Metrics:")
    print(f"   Accuracy:    {accuracy:.4f}")
    print(f"   F1 (macro):  {f1_macro:.4f}")
    print(f"   F1 (weight): {f1_weighted:.4f}")
    print("\n   Per-class F1:")
    for name in CLASS_NAMES:
        f1_val = report[name]["f1-score"]
        print(f"     {name:15s}: {f1_val:.4f}")

    # Save metrics
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n✅ Metrics → {metrics_path}")

    # Export to ONNX
    initial_type = [("float_input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

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
    train_image_classifier(out, met)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument(
        "--output",
        type=str,
        default="models/image_class/v1/model.onnx",
        help="Output model path",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="models/image_class/v1/metrics.json",
        help="Output metrics path",
    )
    args = parser.parse_args()
    train_image_classifier(Path(args.output), Path(args.metrics))
