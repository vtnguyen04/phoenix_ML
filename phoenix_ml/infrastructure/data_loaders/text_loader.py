# ruff: noqa: PLC0415
"""
TextDataLoader — CSV/JSON loader for NLP tasks (sentiment, classification, etc.).

Handles text data where the input is a string column (reviews, tweets, etc.)
and the target is a label column. Returns raw text arrays for downstream
vectorization (TF-IDF, embeddings, etc.) in the training script.

Expected data format:
    - CSV with header row
    - One text column (specified via ``text_column`` kwarg or auto-detected)
    - One target/label column

Usage:
    loader = TextDataLoader()
    data, info = await loader.load("data/sentiment/dataset.csv",
                                    text_column="review",
                                    target_column="sentiment")
    train, test = await loader.split(data, test_size=0.2)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phoenix_ml.domain.training.services.data_loader_plugin import DatasetInfo, IDataLoader

logger = logging.getLogger(__name__)


class TextDataLoader(IDataLoader):
    """Load text datasets from CSV files.

    Returns (texts, labels) arrays where texts are raw strings
    for downstream feature extraction (TF-IDF, embeddings, etc.).
    """

    async def load(self, data_path: str, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load text dataset from CSV.

        Args:
            data_path: Path to CSV file.
            **kwargs:
                text_column: Name of the text column (auto-detected if absent).
                target_column: Name of target column (default: last column).
                max_samples: Limit number of rows loaded.

        Returns:
            Tuple of ((texts, labels) arrays, DatasetInfo metadata).
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        df = pd.read_csv(path)

        max_samples = kwargs.get("max_samples")
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # Resolve target column
        target_col = kwargs.get("target_column", df.columns[-1])

        # Resolve text column — explicit kwarg, or auto-detect first object/string column
        text_col = kwargs.get("text_column")
        if text_col is None:
            object_cols = df.select_dtypes(include=["object"]).columns
            text_col = object_cols[0] if len(object_cols) > 0 else df.columns[0]

        texts = df[text_col].fillna("").astype(str).values
        labels = df[target_col].values

        # Convert string labels to int if needed (e.g., "positive" → 1)
        if labels.dtype == object:
            unique_labels = sorted(set(labels))
            label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
            labels = np.array([label_map[lbl] for lbl in labels])

        info = DatasetInfo(
            num_samples=len(df),
            num_features=0,  # text → features are extracted at training time
            feature_names=[],
            class_labels=[str(v) for v in sorted(np.unique(labels))],
            data_format="text_csv",
            metadata={
                "text_column": str(text_col),
                "target_column": str(target_col),
                "source_path": str(path),
                "avg_text_length": float(np.mean([len(t) for t in texts])),
            },
        )

        logger.info(
            "Loaded %d text samples from %s (text=%s, target=%s)",
            info.num_samples,
            path.name,
            text_col,
            target_col,
        )
        return (texts, labels), info

    async def split(
        self,
        data: Any,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split (texts, labels) into train and test sets with stratification.

        Returns:
            Tuple of ((X_train, y_train), (X_test, y_test)).
        """
        from sklearn.model_selection import train_test_split

        texts, labels = data

        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=random_seed,
            stratify=labels,
        )

        logger.info("Split: train=%d, test=%d (stratified)", len(x_train), len(x_test))
        return (x_train, y_train), (x_test, y_test)
