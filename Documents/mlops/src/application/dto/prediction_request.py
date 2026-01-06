
from pydantic import BaseModel


class PredictCommand(BaseModel):
    """
    Data Transfer Object (DTO) for prediction requests.
    """
    model_id: str
    model_version: str
    features: list[float]
