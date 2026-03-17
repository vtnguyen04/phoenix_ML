from src.application.commands.load_model_command import LoadModelCommand
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.model_registry.repositories.model_repository import ModelRepository


class LoadModelHandler:
    """
    Application Service that handles model loading requests.
    Initializes the engine with the requested model.
    """

    def __init__(
        self, model_repo: ModelRepository, inference_engine: InferenceEngine
    ) -> None:
        self._model_repo = model_repo
        self._inference_engine = inference_engine

    async def execute(self, command: LoadModelCommand) -> bool:
        # 1. Fetch Model from Registry
        model = await self._model_repo.get_by_id(
            command.model_id, command.model_version
        )

        if not model:
            raise ValueError(
                f"Model {command.model_id}:{command.model_version} not found"
            )

        # 2. Trigger Engine Load
        await self._inference_engine.load(model)

        return True
