"""Feature transforms & pipeline — chained feature engineering steps.

Transform types:
- StandardScaler: (x - mean) / std
- MinMaxScaler: (x - min) / (max - min)
- LogTransform: log1p(x)
- OneHotEncoder: categorical → binary columns
- Imputer: fill missing values (mean, median, constant)
- FeaturePipeline: chain transforms in order
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureTransform(ABC):
    """Base class for feature transforms."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        """Fit transform on training data."""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transform to data."""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


class StandardScaler(FeatureTransform):
    """Standardize features to zero mean and unit variance."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        cols = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            self._means[col] = float(df[col].mean())
            self._stds[col] = float(df[col].std()) or 1.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mean in self._means.items():
            if col in df.columns:
                df[col] = (df[col] - mean) / self._stds[col]
        return df


class MinMaxScaler(FeatureTransform):
    """Scale features to [0, 1] range."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns
        self._mins: dict[str, float] = {}
        self._maxs: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        cols = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            self._mins[col] = float(df[col].min())
            self._maxs[col] = float(df[col].max())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, mn in self._mins.items():
            if col in df.columns:
                rng = self._maxs[col] - mn
                df[col] = (df[col] - mn) / rng if rng != 0 else 0
        return df


class LogTransform(FeatureTransform):
    """Apply log1p transform to reduce skewness."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(lower=0))
        return df


class Imputer(FeatureTransform):
    """Fill missing values using strategy (mean, median, constant)."""

    def __init__(
        self, strategy: str = "median", fill_value: float = 0.0, columns: list[str] | None = None
    ) -> None:
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._fill_values: dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        cols = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        for col in cols:
            if self.strategy == "mean":
                self._fill_values[col] = float(df[col].mean())
            elif self.strategy == "median":
                self._fill_values[col] = float(df[col].median())
            else:
                self._fill_values[col] = self.fill_value
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, val in self._fill_values.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(val)
        return df


class OneHotEncoder(FeatureTransform):
    """One-hot encode categorical columns."""

    def __init__(self, columns: list[str] | None = None, drop_first: bool = True) -> None:
        self.columns = columns
        self.drop_first = drop_first
        self._categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> FeatureTransform:
        cols = self.columns or list(df.select_dtypes(include=["object", "category"]).columns)
        for col in cols:
            self._categories[col] = sorted(df[col].dropna().unique().tolist())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, cats in self._categories.items():
            if col not in df.columns:
                continue
            for i, cat in enumerate(cats):
                if self.drop_first and i == 0:
                    continue
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df = df.drop(columns=[col])
        return df


@dataclass
class FeaturePipeline:
    """Chain multiple feature transforms in sequence.

    Usage:
        pipeline = FeaturePipeline(steps=[
            Imputer(strategy="median"),
            StandardScaler(columns=["income", "age"]),
            OneHotEncoder(columns=["category"]),
        ])
        df_train = pipeline.fit_transform(train_df)
        df_test = pipeline.transform(test_df)
    """

    steps: list[FeatureTransform] = field(default_factory=list)

    def add(self, transform: FeatureTransform) -> FeaturePipeline:
        self.steps.append(transform)
        return self

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        current = df
        for step in self.steps:
            step.fit(current)
            current = step.transform(current)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df
        for step in self.steps:
            current = step.transform(current)
        return current

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def describe(self) -> list[dict[str, Any]]:
        """Return pipeline step descriptions."""
        return [
            {"step": i, "type": type(step).__name__}
            for i, step in enumerate(self.steps)
        ]
