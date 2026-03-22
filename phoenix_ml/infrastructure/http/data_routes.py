"""Data pipeline API routes — trigger ingestion, validate data, check pipeline status."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from phoenix_ml.domain.training.services.data_pipeline import DataPipeline
from phoenix_ml.domain.training.services.data_validator import DataValidator

logger = logging.getLogger(__name__)
data_router = APIRouter(prefix="/data", tags=["Data Pipeline"])

_pipeline = DataPipeline()
_validator = DataValidator()


class IngestRequest(BaseModel):
    source_path: str = Field(..., description="Path to CSV/Parquet file")
    target_column: str | None = None
    output_path: str | None = None
    feature_ranges: dict[str, list[float]] | None = None


class ValidateRequest(BaseModel):
    source_path: str = Field(..., description="Path to CSV/Parquet file")
    target_column: str | None = None


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
    import pandas as pd  # noqa: PLC0415

    path = Path(body.source_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {body.source_path}")

    df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    report = _validator.validate(df, body.target_column)
    return report.to_dict()
