from pydantic import BaseModel, Field


class PredictionResponseDTO(BaseModel):
    """Data Transfer Object for prediction responses."""

    prediction_id: str = Field(..., description="Unique prediction identifier")
    model_id: str = Field(..., description="Model that produced the prediction")
    version: str = Field(..., description="Model version used")
    result: list[float] = Field(..., description="Prediction output values")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    latency_ms: float = Field(..., ge=0.0, description="Inference latency in ms")
