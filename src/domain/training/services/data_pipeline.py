"""Data Pipeline — Orchestrate: collect → validate → transform → store.

Provides end-to-end real data ingestion with quality gates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.domain.training.services.data_validator import DataQualityReport, DataValidator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Outcome of a pipeline run."""

    success: bool
    rows_processed: int = 0
    rows_stored: int = 0
    quality_report: DataQualityReport | None = None
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rows_processed": self.rows_processed,
            "rows_stored": self.rows_stored,
            "quality_report": self.quality_report.to_dict() if self.quality_report else None,
            "errors": self.errors,
        }


class DataPipeline:
    """End-to-end data pipeline: ingest → validate → transform → store.

    Steps:
    1. Load raw data from source (CSV, Parquet, DB, API)
    2. Validate data quality (nulls, types, outliers)
    3. Apply feature transforms (impute, scale, encode)
    4. Store processed features into Feature Store
    """

    def __init__(
        self,
        validator: DataValidator | None = None,
    ) -> None:
        self._validator = validator or DataValidator()

    async def run_from_file(
        self,
        source_path: str,
        target_column: str | None = None,
        feature_ranges: dict[str, tuple[float, float]] | None = None,
        output_path: str | None = None,
    ) -> PipelineResult:
        """Run pipeline from a file source.

        Args:
            source_path: Path to CSV or Parquet file.
            target_column: Name of target column.
            feature_ranges: Optional {column: (min, max)} for validation.
            output_path: Where to save processed data (optional).
        """
        result = PipelineResult(success=False)

        # Step 1: Load
        try:
            df = self._load(source_path)
            result.rows_processed = len(df)
        except Exception as e:
            result.errors.append(f"Load failed: {e}")
            return result

        # Step 2: Validate
        report = self._validator.validate(df, target_column, feature_ranges)
        result.quality_report = report
        if not report.passed:
            result.errors.append(
                f"Validation failed: {len(report.errors)} error(s)"
            )
            logger.warning("Pipeline validation failed: %s", result.errors)
            return result

        # Step 3: Clean & Transform
        df = self._basic_clean(df)

        # Step 4: Store
        if output_path:
            self._save(df, output_path)
            result.rows_stored = len(df)
        else:
            result.rows_stored = len(df)

        result.success = True
        logger.info(
            "Pipeline completed: %d rows processed, %d stored",
            result.rows_processed,
            result.rows_stored,
        )
        return result

    async def run_from_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        output_path: str | None = None,
    ) -> PipelineResult:
        """Run pipeline from an existing DataFrame."""
        result = PipelineResult(success=False, rows_processed=len(df))

        report = self._validator.validate(df, target_column)
        result.quality_report = report
        if not report.passed:
            result.errors.append(f"Validation failed: {len(report.errors)} error(s)")
            return result

        df = self._basic_clean(df)

        if output_path:
            self._save(df, output_path)

        result.rows_stored = len(df)
        result.success = True
        return result

    def _load(self, path: str) -> pd.DataFrame:
        """Load data from CSV or Parquet."""
        p = Path(path)
        if not p.exists():
            msg = f"File not found: {path}"
            raise FileNotFoundError(msg)
        if p.suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        return pd.read_csv(p)

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: drop full-NaN rows, fill numeric NaN with median."""
        df = df.dropna(how="all")
        for col in df.select_dtypes(include=["float64", "float32", "int64"]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        return df

    def _save(self, df: pd.DataFrame, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in (".parquet", ".pq"):
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False)
        logger.info("Saved processed data: %s (%d rows)", p, len(df))
