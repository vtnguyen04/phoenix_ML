from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceScore(BaseModel):
    """
    Value Object representing the confidence score of a prediction.
    Must be between 0.0 and 1.0.
    Using Pydantic for automatic validation.
    """
    model_config = ConfigDict(frozen=True)
    
    value: Annotated[float, Field(ge=0.0, le=1.0)]

    def __lt__(self, other: "ConfidenceScore") -> bool:
        return self.value < other.value

    def __gt__(self, other: "ConfidenceScore") -> bool:
        return self.value > other.value