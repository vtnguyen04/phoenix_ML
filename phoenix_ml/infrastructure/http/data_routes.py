"""Data pipeline API routes — trigger ingestion, validate data, check pipeline status."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from phoenix_ml.domain.training.services.data_pipeline import DataPipeline
from phoenix_ml.domain.training.services.data_validator import DataValidator
from phoenix_ml.infrastructure.bootstrap.model_config_loader import load_model_config
from phoenix_ml.infrastructure.persistence.database import get_db
from phoenix_ml.infrastructure.persistence.postgres_log_repo import (
    PostgresPredictionLogRepository,
)

logger = logging.getLogger(__name__)
data_router = APIRouter(prefix="/data", tags=["Data Pipeline"])

_pipeline = DataPipeline()
_validator = DataValidator()


# ── Request Models ──────────────────────────────────────────────


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to CSV/Parquet file")
    target_column: str | None = None
    output_path: str | None = None
    feature_ranges: dict[str, list[float]] | None = None


class ValidateRequest(BaseModel):
    source_path: str = Field(..., description="Path to CSV/Parquet file")
    target_column: str | None = None


class ExportTrainingRequest(BaseModel):
    model_id: str = Field(..., description="Model to export data for")
    min_samples: int = Field(500, description="Minimum labeled samples required")
    include_baseline: bool = Field(
        True, description="Merge with baseline training data"
    )
    max_fresh_samples: int = Field(10000, description="Max fresh samples to export")


# ── Ingest / Validate Endpoints ─────────────────────────────────


@data_router.post("/ingest")
async def ingest_data(body: IngestRequest) -> dict[str, Any]:
    """Run the data pipeline: load → validate → clean → store."""
    if not Path(body.source_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.source_path}")

    _MIN_MAX_PAIR_LEN = 2
    ranges = None
    if body.feature_ranges:
        ranges = {
            k: (v[0], v[1])
            for k, v in body.feature_ranges.items()
            if len(v) == _MIN_MAX_PAIR_LEN
        }

    result = await _pipeline.run_from_file(
        source_path=body.source_path,
        target_column=body.target_column,
        feature_ranges=ranges,
        output_path=body.output_path,
    )
    return result.to_dict()


@data_router.post("/validate")
async def validate_data(body: ValidateRequest) -> dict[str, Any]:
    """Validate data quality without ingesting."""
    path = Path(body.source_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.source_path}")

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    report = _validator.validate(df, body.target_column)
    return report.to_dict()


# ── Export Training Data (for self-healing retrain) ─────────────
#
# Refactored into small, single-responsibility helper functions:
#   _fetch_labeled_logs    → queries DB for prediction logs with ground truth
#   _load_baseline_data    → loads baseline CSV from model config
#   _build_fresh_dataframe → converts log dicts → DataFrame, aligns columns
#   _merge_datasets        → combines baseline + fresh DataFrames
#   _write_export_csv      → writes timestamped CSV to disk


async def _fetch_labeled_logs(
    db: AsyncSession,
    model_id: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    """Query prediction logs WHERE ground_truth IS NOT NULL (SRP: DB access only)."""
    repo = PostgresPredictionLogRepository(db)
    return await repo.export_labeled_logs(model_id, limit=max_samples)


def _load_baseline_data(model_id: str) -> pd.DataFrame:
    """Load baseline training CSV from model config (SRP: file I/O only)."""
    config_path = Path("model_configs") / f"{model_id}.yaml"
    if not config_path.exists():
        return pd.DataFrame()

    config = load_model_config(config_path)
    if not config.data_path:
        return pd.DataFrame()

    baseline_path = Path(config.data_path)
    if not baseline_path.exists():
        return pd.DataFrame()

    return pd.read_csv(baseline_path)


def _build_fresh_dataframe(
    rows: list[dict[str, Any]],
    baseline_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Convert labeled log dicts → DataFrame with aligned columns (SRP: transform only)."""
    if not rows:
        return pd.DataFrame()

    feature_lists = [r["features"] for r in rows]
    targets = [r["target"] for r in rows]
    df = pd.DataFrame(feature_lists)

    # Align column names with baseline if available
    if baseline_columns:
        feature_cols = [c for c in baseline_columns if c != "target"]
        if len(feature_cols) == df.shape[1]:
            df.columns = feature_cols
        else:
            df.columns = [f"feature_{i}" for i in range(df.shape[1])]

    df["target"] = targets
    return df


def _merge_datasets(baseline: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Merge baseline + fresh DataFrames (SRP: concat only)."""
    has_baseline = not baseline.empty
    has_fresh = not fresh.empty

    if has_baseline and has_fresh:
        return pd.concat([baseline, fresh], ignore_index=True)
    if has_baseline:
        return baseline
    if has_fresh:
        return fresh
    return pd.DataFrame()


def _write_export_csv(df: pd.DataFrame, model_id: str) -> Path:
    """Write DataFrame to timestamped CSV (SRP: file write only)."""
    fs_model_id = model_id.replace("-", "_")
    output_dir = Path("data") / fs_model_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"retrain_{int(time.time())}.csv"
    df.to_csv(output_path, index=False)
    return output_path


@data_router.post("/export-training")
async def export_training_data(
    body: ExportTrainingRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> dict[str, Any]:
    """Export fresh training data from prediction logs for retrain.

    Queries prediction_logs where ground_truth IS NOT NULL,
    optionally merges with baseline training data from model config,
    and writes a timestamped CSV for the retrain pipeline.
    """
    # 1. Fetch labeled logs from prediction DB
    fresh_rows = await _fetch_labeled_logs(db, body.model_id, body.max_fresh_samples)
    fresh_count = len(fresh_rows)
    logger.info("Export: %d labeled logs for %s", fresh_count, body.model_id)

    # 2. Load baseline data (also provides column names for alignment)
    baseline_df = _load_baseline_data(body.model_id) if body.include_baseline else pd.DataFrame()
    baseline_count = len(baseline_df)

    # 3. Build fresh DataFrame with aligned columns
    baseline_cols = list(baseline_df.columns) if not baseline_df.empty else None
    fresh_df = _build_fresh_dataframe(fresh_rows, baseline_cols)

    # 4. Merge
    combined_df = _merge_datasets(baseline_df, fresh_df)
    total_samples = len(combined_df)

    if total_samples < body.min_samples:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Insufficient data: {total_samples} samples (need {body.min_samples}). "
                f"Fresh: {fresh_count}, baseline: {baseline_count}."
            ),
        )

    # 5. Write to disk
    output_path = _write_export_csv(combined_df, body.model_id)
    logger.info("Exported %d samples → %s", total_samples, output_path)

    return {
        "export_path": str(output_path),
        "total_samples": total_samples,
        "fresh_samples": fresh_count,
        "baseline_samples": baseline_count,
        "model_id": body.model_id,
    }
