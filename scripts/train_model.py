import argparse
import json
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def train_and_export_model(output_path: str, metrics_path: str | None = None) -> None:
    """
    Trains a simple credit risk model, evaluates it, and exports to ONNX.
    """
    # 1. Generate synthetic data (4 features)
    X, y = make_classification(
        n_samples=2000,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Train Logistic Regression model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
    }

    print(f"📊 Model Metrics: {metrics}")

    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"✅ Metrics saved to {metrics_path}")

    # 4. Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = convert_sklearn(clf, initial_types=initial_type)

    # 5. Save file
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(onx.SerializeToString())

    print(f"✅ Model successfully saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="models/credit_risk/v1/model.onnx",
        help="Path to save the ONNX model",
    )
    parser.add_argument(
        "--metrics", type=str, default=None, help="Path to save the metrics JSON"
    )
    args = parser.parse_args()
    train_and_export_model(args.output, args.metrics)
