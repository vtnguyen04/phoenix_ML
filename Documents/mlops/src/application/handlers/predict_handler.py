from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.inference_engine import InferenceEngine
from src.application.dto.prediction_request import PredictCommand
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.value_objects.feature_vector import FeatureVector

class PredictHandler:
    """
    Application Service that handles prediction commands.
    Coordinates between Domain and Infrastructure.
    """
    
    def __init__(
        self, 
        model_repo: ModelRepository, 
        inference_engine: InferenceEngine
    ):
        self._model_repo = model_repo
        self._inference_engine = inference_engine

    async def execute(self, command: PredictCommand) -> Prediction:
        # 1. Fetch model metadata
        model = await self._model_repo.get_by_id(
            command.model_id, 
            command.model_version
        )
        
        if not model:
            raise ValueError(f"Model {command.model_id}:{command.model_version} not found")

        # 2. Prepare features (Value Object validation happens here)
        features = FeatureVector(values=command.features)

        # 3. Ensure model is loaded (Implementation specific)
        await self._inference_engine.load(model)

        # 4. Perform inference
        prediction = await self._inference_engine.predict(model, features)
        
        return prediction
