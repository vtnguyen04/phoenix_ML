import time
import numpy as np
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.entities.model import Model
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.confidence_score import ConfidenceScore

class MockInferenceEngine(InferenceEngine):
    """
    A mock implementation of InferenceEngine for testing and local development.
    """
    
    def __init__(self):
        self.loaded_models = {}

    async def load(self, model: Model) -> None:
        # Simulate loading latency
        self.loaded_models[model.unique_key] = model

    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        if model.unique_key not in self.loaded_models:
            raise RuntimeError(f"Model {model.unique_key} not loaded")
        
        start_time = time.time()
        
        # Mock logic: result is just the mean of the features
        result = float(np.mean(features.values))
        confidence = ConfidenceScore(value=0.99)
        
        latency = (time.time() - start_time) * 1000
        
        return Prediction(
            model_id=model.id,
            model_version=model.version,
            result=result,
            confidence=confidence,
            latency_ms=latency
        )

    async def optimize(self, model: Model) -> None:
        pass
