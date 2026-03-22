# ruff: noqa: PLC0415
"""
TabularDataLoader — Generic CSV/Parquet loader for any tabular ML task.

Works for classification, regression, anomaly detection, or any task
where data is stored as rows × columns in CSV or Parquet format.

Expected data format:
    - CSV/Parquet with a header row
    - Last column = target (label/value) by default
    - Or specify target_column in kwargs

Usage:
    loader = TabularDataLoader()
    data, info = await loader.load("data/my_model/dataset.csv")
    train, test = await loader.split(data, test_size=0.2)
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phoenix_ml.domain.training.services.data_loader_plugin import DatasetInfo, IDataLoader

logger = logging.getLogger(__name__)

_MAX_CLASSIFICATION_CLASSES = 20


class TabularDataLoader(IDataLoader):
    """Load tabular data from CSV or Parquet files.

    Supports any tabular ML problem (classification, regression, etc.).
    Returns (X, y) numpy arrays with metadata in DatasetInfo.
    """

    async def load(self, data_path: str, **kwargs: Any) -> tuple[Any, DatasetInfo]:
        """Load tabular dataset from CSV or Parquet.

        Args:
            data_path: Path to CSV or Parquet file.
            **kwargs:
                target_column: Name of target column (default: last column).
                feature_columns: Explicit list of feature columns.
                max_samples: Limit number of rows loaded.

        Returns:
            Tuple of ((X, y) arrays, DatasetInfo metadata).
        """
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        # Read file
        if path.suffix in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        max_samples = kwargs.get("max_samples")
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)

        # Split features and target
        target_col = kwargs.get("target_column", df.columns[-1])
        feature_cols = kwargs.get("feature_columns")
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        x_data = df[feature_cols].values.astype(np.float32)
        y_data = df[target_col].values

        info = DatasetInfo(
            num_samples=len(df),
            num_features=len(feature_cols),
            feature_names=list(feature_cols),
            data_format="tabular_csv" if path.suffix == ".csv" else "tabular_parquet",
            metadata={
                "target_column": str(target_col),
                "source_path": str(path),
                "dtypes": {c: str(df[c].dtype) for c in feature_cols[:5]},
            },
        )

        # Detect classification labels
        unique_values = np.unique(y_data)
        if len(unique_values) <= _MAX_CLASSIFICATION_CLASSES:
            info.class_labels = [str(v) for v in sorted(unique_values)]

        logger.info(
            "Loaded %d samples × %d features from %s (target=%s)",
            info.num_samples,
            info.num_features,
            path.name,
            target_col,
        )
        return (x_data, y_data), info

    async def split(
        self,
        data: Any,
        test_size: float = 0.2,
        random_seed: int = 42,
    ) -> tuple[Any, Any]:
        """Split (X, y) into train and test sets.

        Automatically uses stratified split for classification tasks.

        Returns:
            Tuple of ((X_train, y_train), (X_test, y_test)).
        """
        from sklearn.model_selection import train_test_split

        x_data, y_data = data

        # Use stratified split for classification (discrete targets)
        unique = np.unique(y_data)
        stratify = y_data if len(unique) <= _MAX_CLASSIFICATION_CLASSES else None

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )

        logger.info(
            "Split: train=%d, test=%d (stratified=%s)",
            len(x_train),
            len(x_test),
            stratify is not None,
        )
        return (x_train, y_train), (x_test, y_test)
