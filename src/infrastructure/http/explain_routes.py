"""Explain routes — model explainability via perturbation-based importance."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.monitoring.services.explainability_service import ExplainabilityService
from src.infrastructure.bootstrap.container import inference_engine
from src.infrastructure.persistence.database import get_db
from src.infrastructure.persistence.postgres_model_registry import PostgresModelRegistry

logger = logging.getLogger(__name__)
explain_router = APIRouter(tags=["Explainability"])

_explainability = ExplainabilityService()


class ExplainRequest(BaseModel):
    model_id: str
    model_version: str | None = None
    features: list[float] = Field(..., min_length=1)
    feature_names: list[str] | None = None


class ExplainResponse(BaseModel):
    model_id: str
    prediction: float
    confidence: float
    importances: dict[str, float]
    top_features: list[str]
    method: str


@explain_router.post("/explain", response_model=ExplainResponse)
async def explain_prediction(
    body: ExplainRequest,
    db: AsyncSession = Depends(get_db),  # noqa: B008
) -> ExplainResponse:
    """Run prediction with feature importance analysis."""
    registry = PostgresModelRegistry(db)
    model = await registry.get_champion(body.model_id)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{body.model_id}' not found")

    features = np.array(body.features, dtype=np.float32)

    try:
        result = await _explainability.explain(
            engine=inference_engine,
            model=model,
            features=features,
            feature_names=body.feature_names,
        )
    except Exception as e:
        logger.exception("Explainability failed for model %s", body.model_id)
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")  # noqa: B904

    return ExplainResponse(
        model_id=body.model_id,
        prediction=result["prediction"],
        confidence=result["confidence"],
        importances=result["importances"],
        top_features=result["top_features"],
        method=result["method"],
    )
