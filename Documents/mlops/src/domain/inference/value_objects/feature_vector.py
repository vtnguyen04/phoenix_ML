from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np
from typing import Any, List

class FeatureVector(BaseModel):
    """
    Value Object representing the input feature vector for the model.
    Immutable and validated upon creation using Pydantic.
    """
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    
    values: np.ndarray

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

    def to_list(self) -> List[float]:
        return self.values.tolist()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FeatureVector):
            return NotImplemented
        return np.array_equal(self.values, other.values)