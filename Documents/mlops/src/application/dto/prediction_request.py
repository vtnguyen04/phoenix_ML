from pydantic import BaseModel
from typing import List, Any

class PredictCommand(BaseModel):
    """
    Data Transfer Object (DTO) for prediction requests.
    """
    model_id: str
    model_version: str
    features: List[float]
