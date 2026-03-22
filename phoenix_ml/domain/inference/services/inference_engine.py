from abc import ABC, abstractmethod

from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.value_objects.feature_vector import FeatureVector


class InferenceEngine(ABC):
    """
    Interface for ML Inference Engines (ONNX, TensorRT, etc.)
    Defined in Domain layer to follow Dependency Inversion Principle.
    """

    @abstractmethod
    async def load(self, model: Model) -> None:
        """Load a model into the engine"""
        pass

    @abstractmethod
    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        """Run inference using the loaded model"""
        pass

    @abstractmethod
    async def batch_predict(
        self, model: Model, features_list: list[FeatureVector]
    ) -> list[Prediction]:
        """Run batch inference for multiple inputs"""
        pass

    @abstractmethod
    async def optimize(self, model: Model) -> None:
        """Apply engine-specific optimizations (e.g., quantization)"""
        pass
