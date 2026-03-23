"""
[EXAMPLE] Train Sentiment Classification Model — Real IMDB Reviews.

Production-grade implementation using TF-IDF features on real movie reviews.
Uses the framework's DataLoader plugin (TextDataLoader) for data loading
and splitting — same pattern as credit_risk/train.py.

Usage:
    python examples/sentiment/train.py
    python examples/sentiment/train.py --data data/sentiment/dataset.csv
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from phoenix_ml.infrastructure.data_loaders.registry import resolve_data_loader

logger = logging.getLogger(__name__)

MODEL_ID = "sentiment"
DEFAULT_DATA_PATH = "data/sentiment/dataset.csv"

# TF-IDF outputs a fixed number of features for ONNX compatibility
MAX_FEATURES = 5000


async def _load_data(
    data_path: str,
) -> tuple[Any, Any, Any, Any]:
    """Load and split data using the framework's DataLoader plugin.

    Resolves TextDataLoader via registry (model_configs/sentiment.yaml
    has task_type: text_classification → TextDataLoader).
    """
    loader = resolve_data_loader(MODEL_ID)
    data, info = await loader.load(
        data_path,
        text_column="review",
        target_column="sentiment",
    )
    (x_train, y_train), (x_test, y_test) = await loader.split(data, test_size=0.2)

    print(f"📐 Loaded: {info.num_samples} samples (via {type(loader).__name__})")
    print(f"   Format: {info.data_format}")
    print(f"   Classes: {info.class_labels}")
    print(f"   Train: {len(x_train)}, Test: {len(x_test)}")
    return x_train, x_test, y_train, y_test


def train_and_export(
    output_path: str,
    metrics_path: str | None = None,
    reference_path: str | None = None,
    data_path: str | None = None,
) -> None:
    """Train TF-IDF + LogisticRegression on real reviews, export to ONNX.

    Framework-standard entry point — called by Airflow self-healing DAG.

    Args:
        output_path: Path to write the ONNX model file.
        metrics_path: Optional path to write metrics.json.
        reference_path: Optional path to write reference feature distributions.
        data_path: Optional path to training dataset CSV.
    """
    resolved_data = data_path or DEFAULT_DATA_PATH
    x_train, x_test, y_train, y_test = asyncio.run(_load_data(resolved_data))

    # TF-IDF vectorization
    print(f"\n📝 Fitting TF-IDF vectorizer (max_features={MAX_FEATURES})...")
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    x_train_tfidf = tfidf.fit_transform(x_train).astype(np.float32)
    x_test_tfidf = tfidf.transform(x_test).astype(np.float32)
    actual_features = x_train_tfidf.shape[1]

    print(f"   Vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"   Feature matrix: {x_train_tfidf.shape}")

    # Train classifier
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )

    print("\n🏋️ Training LogisticRegression...")
    model.fit(x_train_tfidf, y_train)

    y_pred = model.predict(x_test_tfidf)
    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "n_features": actual_features,
        "max_tfidf_features": MAX_FEATURES,
        "dataset": "imdb-movie-reviews",
        "model_type": "TF-IDF + LogisticRegression",
    }

    print("\n📊 Model Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

    # Save metrics
    if metrics_path:
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Metrics → {metrics_path}")

    # Export classifier to ONNX (TF-IDF applied at inference time)
    pipeline = Pipeline([("classifier", model)])
    initial_type = [("float_input", FloatTensorType([None, actual_features]))]
    onx = convert_sklearn(pipeline, initial_types=initial_type)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"✅ ONNX → {path} ({path.stat().st_size / 1024:.1f} KB)")

    # Save TF-IDF vocabulary for inference preprocessing
    vocab_path = str(Path(output_path).parent / "tfidf_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump({
            "vocabulary": {k: int(v) for k, v in tfidf.vocabulary_.items()},
            "idf": tfidf.idf_.tolist(),
            "max_features": MAX_FEATURES,
        }, f)
    print(f"✅ TF-IDF vocabulary → {vocab_path}")

    # Save reference distributions for drift detection
    ref_path = reference_path or str(
        Path(output_path).parent / "reference_data.json"
    )

    feature_names = tfidf.get_feature_names_out()[:50].tolist()
    reference: dict[str, Any] = {
        "feature_names": feature_names,
        "n_features": actual_features,
        "reference_distributions": {},
    }
    train_dense = x_train_tfidf[:, :50].toarray()
    for i, fname in enumerate(feature_names):
        reference["reference_distributions"][fname] = train_dense[:, i].tolist()

    Path(ref_path).parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(reference, f)
    print(f"✅ Reference data → {ref_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment model on real reviews")
    parser.add_argument("--output", default="models/sentiment/v1/model.onnx")
    parser.add_argument("--metrics", default="models/sentiment/v1/metrics.json")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to reviews CSV")
    args = parser.parse_args()
    train_and_export(args.output, args.metrics, args.reference, args.data)
