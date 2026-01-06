from pathlib import Path

from src.application.commands.load_model_command import LoadModelCommand
from src.domain.inference.repositories.model_repository import ModelRepository
from src.domain.inference.services.inference_engine import InferenceEngine
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)


class LoadModelHandler:
    """
    Orchestrates the loading of a model from storage to the inference engine.
    """
    def __init__(
        self,
        model_repo: ModelRepository,
        inference_engine: InferenceEngine,
        artifact_storage: LocalArtifactStorage
    ):
        self._model_repo = model_repo
        self._inference_engine = inference_engine
        self._artifact_storage = artifact_storage
        self._cache_dir = Path("/tmp/phoenix/model_cache") # Should inject config

    async def execute(self, command: LoadModelCommand) -> None:
        # 1. Fetch Model Metadata
        model = await self._model_repo.get_by_id(
            command.model_id, 
            command.model_version
        )
        if not model:
            raise ValueError(
                f"Model {command.model_id}:{command.model_version} not found"
            )

        # 2. Download Artifact if needed
        local_model_path = self._cache_dir / model.id / model.version / "model.onnx"
        if not local_model_path.exists():
            await self._artifact_storage.download(model.uri, local_model_path)
        
        # 3. Load into Engine
        await self._inference_engine.load(model)
