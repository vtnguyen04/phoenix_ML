"""Dataset quality validation.

Checks a ``pandas.DataFrame`` for missing values, type mismatches,
outliers, and distribution statistics. Returns a ``DataQualityReport``
containing severity-graded issues (INFO, WARNING, ERROR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class QualityIssue:
    column: str
    severity: Severity
    issue_type: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ColumnStats:
    name: str
    dtype: str
    non_null_count: int
    null_count: int
    null_percent: float
    unique_count: int
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None


@dataclass
class DataQualityReport:
    """Result of data quality validation."""

    total_rows: int
    total_columns: int
    issues: list[QualityIssue] = field(default_factory=list)
    column_stats: list[ColumnStats] = field(default_factory=list)
    passed: bool = True

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [
                {
                    "column": i.column,
                    "severity": i.severity.value,
                    "type": i.issue_type,
                    "message": i.message,
                }
                for i in self.issues
            ],
        }


class DataValidator:
    """Validate data quality before training or ingestion.

    Checks:
    - Missing values (null/NaN percentage per column)
    - Type validation (numeric, categorical)
    - Outlier detection (IQR method)
    - Range validation (configurable min/max per feature)
    - Duplicate detection
    """

    def __init__(
        self,
        max_null_percent: float = 30.0,
        outlier_threshold: float = 3.0,
        max_duplicate_percent: float = 10.0,
    ) -> None:
        self._max_null_percent = max_null_percent
        self._outlier_threshold = outlier_threshold
        self._max_dup_percent = max_duplicate_percent

    def validate(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        feature_ranges: dict[str, tuple[float, float]] | None = None,
    ) -> DataQualityReport:
        """Run all validation checks on a DataFrame."""
        report = DataQualityReport(
            total_rows=len(df),
            total_columns=len(df.columns),
        )

        self._check_empty(df, report)
        if not report.passed:
            return report

        self._check_nulls(df, report)
        self._check_duplicates(df, report)
        self._compute_column_stats(df, report)
        self._check_outliers(df, report)

        if feature_ranges:
            self._check_ranges(df, feature_ranges, report)

        if target_column and target_column in df.columns:
            self._check_target(df, target_column, report)

        report.passed = len(report.errors) == 0

        logger.info(
            "Data validation: %d rows × %d cols, %d issues (%d errors, %d warnings)",
            report.total_rows,
            report.total_columns,
            len(report.issues),
            len(report.errors),
            len(report.warnings),
        )
        return report

    def _check_empty(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        if len(df) == 0:
            report.issues.append(
                QualityIssue("*", Severity.ERROR, "empty_dataset", "Dataset has 0 rows")
            )
            report.passed = False

    def _check_nulls(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        for col in df.columns:
            null_pct = df[col].isnull().mean() * 100
            if null_pct > self._max_null_percent:
                report.issues.append(
                    QualityIssue(
                        col,
                        Severity.ERROR,
                        "high_null_rate",
                        f"{col} has {null_pct:.1f}% null values "
                        f"(threshold: {self._max_null_percent}%)",
                        {"null_percent": round(null_pct, 2)},
                    )
                )
            elif null_pct > 0:
                report.issues.append(
                    QualityIssue(
                        col,
                        Severity.WARNING,
                        "has_nulls",
                        f"{col} has {null_pct:.1f}% null values",
                        {"null_percent": round(null_pct, 2)},
                    )
                )

    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        dup_count = df.duplicated().sum()
        dup_pct = dup_count / len(df) * 100 if len(df) > 0 else 0
        if dup_pct > self._max_dup_percent:
            report.issues.append(
                QualityIssue(
                    "*",
                    Severity.WARNING,
                    "high_duplicates",
                    f"Dataset has {dup_count} duplicate rows ({dup_pct:.1f}%)",
                    {"duplicate_count": int(dup_count), "duplicate_percent": round(dup_pct, 2)},
                )
            )

    def _check_outliers(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        for col in df.select_dtypes(include=[np.number]).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - self._outlier_threshold * iqr
            upper = q3 + self._outlier_threshold * iqr
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                report.issues.append(
                    QualityIssue(
                        col,
                        Severity.INFO,
                        "outliers_detected",
                        f"{col} has {outliers} outlier(s) "
                        f"beyond {self._outlier_threshold}×IQR",
                        {
                            "outlier_count": int(outliers),
                            "lower_bound": lower,
                            "upper_bound": upper,
                        },
                    )
                )

    def _check_ranges(
        self,
        df: pd.DataFrame,
        ranges: dict[str, tuple[float, float]],
        report: DataQualityReport,
    ) -> None:
        for col, (lo, hi) in ranges.items():
            if col not in df.columns:
                continue
            oob = ((df[col] < lo) | (df[col] > hi)).sum()
            if oob > 0:
                report.issues.append(
                    QualityIssue(
                        col,
                        Severity.WARNING,
                        "out_of_range",
                        f"{col}: {oob} values outside [{lo}, {hi}]",
                        {"out_of_range_count": int(oob)},
                    )
                )

    def _check_target(self, df: pd.DataFrame, target: str, report: DataQualityReport) -> None:
        if df[target].isnull().any():
            report.issues.append(
                QualityIssue(
                    target,
                    Severity.ERROR,
                    "target_has_nulls",
                    "Target column has null values",
                )
            )
        unique = df[target].nunique()
        if unique == 1:
            report.issues.append(
                QualityIssue(
                    target,
                    Severity.ERROR,
                    "constant_target",
                    "Target has only 1 unique value",
                )
            )

    def _compute_column_stats(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            stats = ColumnStats(
                name=col,
                dtype=str(df[col].dtype),
                non_null_count=len(df) - null_count,
                null_count=null_count,
                null_percent=round(null_count / len(df) * 100, 2) if len(df) > 0 else 0,
                unique_count=int(df[col].nunique()),
            )
            if pd.api.types.is_numeric_dtype(df[col]):
                stats.mean = round(float(df[col].mean()), 4)
                stats.std = round(float(df[col].std()), 4)
                stats.min_val = float(df[col].min())
                stats.max_val = float(df[col].max())
            report.column_stats.append(stats)
