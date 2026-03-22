from pydantic import BaseModel, Field


class PredictionRequestDTO(BaseModel):
    """Data Transfer Object for incoming prediction requests."""

    model_id: str = Field(..., description="Unique model identifier")
    model_version: str | None = Field(None, description="Specific version or None for latest")
    entity_id: str | None = Field(None, description="Entity ID for feature store lookup")
    features: list[float] | None = Field(None, description="Pre-computed feature values")
