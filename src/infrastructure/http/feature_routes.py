import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.domain.feature_store.entities.feature_registry import FeatureMetadata
from src.infrastructure.http.container import feature_store

logger = logging.getLogger(__name__)

feature_router = APIRouter(prefix="/features", tags=["Feature Store"])


class IngestRequest(BaseModel):
    entity_id: str
    features: dict[str, float]


@feature_router.get("/{entity_id}")
async def get_features(entity_id: str, keys: str = "") -> dict[str, Any]:
    """Retrieve online features for a specific entity."""
    feature_names = keys.split(",") if keys else []
    if not feature_names:
        raise HTTPException(status_code=400, detail="Must provide ?keys=f1,f2")

    features = await feature_store.get_online_features(entity_id, feature_names)
    if not features:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    return {"entity_id": entity_id, "features": dict(zip(feature_names, features, strict=False))}


@feature_router.post("/ingest")
async def ingest_features(request: IngestRequest) -> dict[str, str]:
    """Ingest new features into the online store."""
    await feature_store.add_features(request.entity_id, request.features)
    logger.info("Ingested %d features for %s", len(request.features), request.entity_id)
    return {"status": "success", "message": f"Ingested {len(request.features)} features"}


@feature_router.get("/metadata/{feature_name}")
async def get_feature_metadata(feature_name: str) -> FeatureMetadata:
    """Retrieve metadata about a specific feature from the registry."""
    return FeatureMetadata(
        name=feature_name,
        dtype="float",
        description=f"Auto-generated feature metadata for {feature_name}",
        owner="ml-platform-team",
        data_source=f"{feature_name}-event-stream",
    )
