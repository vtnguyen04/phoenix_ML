from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.batch_manager import BatchManager
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.services.routing_strategy import RoutingStrategy
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage


@dataclass(frozen=True)
class PredictionRequest:
    model_id: str
    model_version: str | None = None
    entity_id: str | None = None
    features: list[float] | None = None


class InferenceService:
    """
    Domain Service that orchestrates the inference flow.
    Coordinates between routing, feature retrieval, and batch processing.
    """

    def __init__(  # noqa: PLR0913
        self,
        model_repo: ModelRepository,
        inference_engine: InferenceEngine,
        batch_manager: BatchManager,
        feature_store: FeatureStore,
        artifact_storage: ArtifactStorage,
        routing_strategy: RoutingStrategy,
        cache_dir: Path | None = None,
    ) -> None:
        self._model_repo = model_repo
        self._inference_engine = inference_engine
        self._batch_manager = batch_manager
        self._feature_store = feature_store
        self._artifact_storage = artifact_storage
        self._routing_strategy = routing_strategy
        self._cache_dir = cache_dir or Path("/tmp/phoenix/model_cache")

    async def predict(self, request: PredictionRequest) -> Prediction:
        """
        Coordinates the full prediction flow.
        """
        # 1. Select Model Version
        model = await self._select_model(
            request.model_id, request.model_version, request.entity_id
        )

        # 2. Ensure artifact is local
        local_model_path = self._cache_dir / model.id / model.version / "model.onnx"
        if not local_model_path.exists():
            await self._artifact_storage.download(model.uri, local_model_path)

        # 3. Ensure engine is loaded
        await self._inference_engine.load(model)

        # 4. Resolve Features
        feature_values = request.features
        if feature_values is None:
            if not request.entity_id:
                raise ValueError("No features provided and no entity_id for lookup")

            # TODO: Get required features from model metadata
            required_features = ["f1", "f2", "f3", "f4"]
            feature_values = await self._feature_store.get_online_features(
                request.entity_id, required_features
            )

            if feature_values is None:
                raise ValueError(f"Features not found for entity {request.entity_id}")

        feature_vector = FeatureVector(
            values=np.array(feature_values, dtype=np.float32)
        )

        # 5. Perform Prediction via BatchManager
        return await self._batch_manager.predict(model, feature_vector)

    async def _select_model(
        self, model_id: str, model_version: str | None, entity_id: str | None
    ) -> Model:
        model: Model | None = None
        if model_version and model_version != "latest":
            model = await self._model_repo.get_by_id(model_id, model_version)
        else:
            candidates = await self._model_repo.get_active_versions(model_id)
            if not candidates:
                raise ValueError(f"No active versions found for model {model_id}")

            context = {"user_id": entity_id} if entity_id else {}
            model = self._routing_strategy.select_model(candidates, context)

        if not model:
            raise ValueError(f"Model {model_id}:{model_version} not found")
        return model
