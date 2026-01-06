from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class FeatureVector(BaseModel):
    """
    Value Object representing the input feature vector for the model.
    Immutable and validated upon creation using Pydantic.
    """
    model_config = ConfigDict(
        frozen=True, 
        arbitrary_types_allowed=True, 
        unsafe_hash=True  # type: ignore
    )
    
    values: np.ndarray

    def __hash__(self) -> int:
        return hash(self.values.tobytes())

    @field_validator("values", mode="before")
    @classmethod
    def validate_values(cls, v: Any) -> np.ndarray:
        if isinstance(v, list):
            v = np.array(v, dtype=np.float32)
        
        if not isinstance(v, np.ndarray):
            raise ValueError("Values must be a numpy array or a list")
            
        if v.size == 0:
            raise ValueError("Feature vector cannot be empty")
            
        if not np.issubdtype(v.dtype, np.number):
            raise ValueError("Feature vector must contain numeric values")
            
        return v

    def to_list(self) -> list[float]:
        return list(self.values.tolist())  # Explicit cast to list

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureVector):
            return NotImplemented
        return np.array_equal(self.values, other.values)