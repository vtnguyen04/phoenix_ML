from src.application.dto.prediction_request import PredictCommand
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.value_objects.feature_vector import FeatureVector


class PredictHandler:
    """
    Application Service that handles prediction commands.
    Coordinates between Domain and Infrastructure.
    """
    
    def __init__(
        self,
        model_repo: ModelRepository,
        inference_engine: InferenceEngine,
        feature_store: FeatureStore
    ) -> None:
        self._model_repo = model_repo
        self._inference_engine = inference_engine
        self._feature_store = feature_store

    async def execute(self, command: PredictCommand) -> Prediction:
        # 1. Fetch model metadata
        model = await self._model_repo.get_by_id(
            command.model_id,
            command.model_version
        )
        
        if not model:
            raise ValueError(
                f"Model {command.model_id}:{command.model_version} not found"
            )

        # 2. Resolve features
        feature_values = command.features
        
        if feature_values is None:
            if command.entity_id:
                # TODO: In real app, feature names should come from Model Metadata 
                # or Configuration specific to the model version.
                required_features = ["f1", "f2", "f3"] 
                feature_values = await self._feature_store.get_online_features(
                    command.entity_id, required_features
                )
                
            if feature_values is None:
                raise ValueError("No features provided and could not fetch from store")

        # 3. Prepare features (Value Object validation happens here)
        features = FeatureVector(values=feature_values)  # type: ignore

        # 4. Ensure model is loaded (Implementation specific)
        await self._inference_engine.load(model)

        # 5. Perform inference
        prediction = await self._inference_engine.predict(model, features)
        
        return prediction
