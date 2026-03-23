from pydantic import BaseModel


class PredictCommand(BaseModel):
    """Input DTO for a single prediction request."""

    model_id: str
    model_version: str | None = None
    features: list[float] | None = None
    entity_id: str | None = None
