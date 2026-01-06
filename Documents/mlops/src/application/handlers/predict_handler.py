import time
from pathlib import Path

from src.application.commands.predict_command import PredictCommand
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.services.routing_strategy import (
    ABTestStrategy,
    RoutingStrategy,
)
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage
from src.infrastructure.monitoring.prometheus_metrics import (
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    PREDICTION_COUNT,
)


class PredictHandler:
    """
    Application Service that handles prediction commands.
    Coordinates between Domain and Infrastructure.
    """
    
    def __init__(
        self, 
        model_repo: ModelRepository, 
        inference_engine: InferenceEngine,
        feature_store: FeatureStore,
        artifact_storage: ArtifactStorage,
        routing_strategy: RoutingStrategy | None = None
    ) -> None:
        self._model_repo = model_repo
        self._inference_engine = inference_engine
        self._feature_store = feature_store
        self._artifact_storage = artifact_storage
        self._cache_dir = Path("/tmp/phoenix/model_cache")
        # Default to A/B testing with 50% split if not provided
        self._routing_strategy = routing_strategy or ABTestStrategy(0.5)

    async def execute(self, command: PredictCommand) -> Prediction:
        start_time = time.time()
        
        try:
            # 1. Determine Model Version (Routing Logic)
            model: Model | None = None
            
            if command.model_version and command.model_version != "latest":
                # Specific version requested
                model = await self._model_repo.get_by_id(
                    command.model_id, 
                    command.model_version
                )
            else:
                # Dynamic Routing
                candidates = await self._model_repo.get_active_versions(
                    command.model_id
                )
                if not candidates:
                     raise ValueError(
                         f"No active versions found for model {command.model_id}"
                     )
                
                context = {"user_id": command.entity_id} if command.entity_id else {}
                model = self._routing_strategy.select_model(candidates, context)

            if not model:
                raise ValueError(
                    f"Model {command.model_id}:{command.model_version} not found"
                )

            # 2. Ensure artifact is local
            local_model_path = self._cache_dir / model.id / model.version / "model.onnx"
            if not local_model_path.exists():
                await self._artifact_storage.download(model.uri, local_model_path)

            # 3. Resolve features
            feature_values = command.features
            
            if feature_values is None:
                if command.entity_id:
                    # TODO: Feature names should come from metadata
                    required_features = ["f1", "f2", "f3", "f4"] 
                    feature_values = await self._feature_store.get_online_features(
                        command.entity_id, required_features
                    )
                    
                if feature_values is None:
                    raise ValueError(
                        "No features provided and could not fetch from store"
                    )

            # 4. Prepare features
            # TODO: Fix type hint for numpy array conversion if needed
            features = FeatureVector(values=feature_values)  # type: ignore

            # 5. Ensure model is loaded
            await self._inference_engine.load(model)

            # 6. Perform inference
            prediction = await self._inference_engine.predict(model, features)
            
            # 7. Record Metrics
            latency = time.time() - start_time
            INFERENCE_LATENCY.labels(
                model_id=model.id, version=model.version
            ).observe(latency)
            
            PREDICTION_COUNT.labels(
                model_id=model.id, version=model.version, status="success"
            ).inc()
            
            MODEL_CONFIDENCE.labels(
                model_id=model.id, version=model.version
            ).observe(prediction.confidence.value)
            
            return prediction

        except Exception as e:
            PREDICTION_COUNT.labels(
                model_id=command.model_id, 
                version=command.model_version or "unknown", 
                status="error"
            ).inc()
            raise e