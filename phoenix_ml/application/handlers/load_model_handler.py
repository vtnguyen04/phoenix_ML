from phoenix_ml.application.commands.load_model_command import LoadModelCommand
from phoenix_ml.domain.inference.services.inference_engine import InferenceEngine
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository


class LoadModelHandler:
    """Application service for loading models into the inference engine."""

    def __init__(self, model_repo: ModelRepository, inference_engine: InferenceEngine) -> None:
        self._model_repo = model_repo
        self._inference_engine = inference_engine

    async def execute(self, command: LoadModelCommand) -> bool:
        model = await self._model_repo.get_by_id(command.model_id, command.model_version)

        if not model:
            raise ValueError(f"Model {command.model_id}:{command.model_version} not found")

        await self._inference_engine.load(model)

        return True
