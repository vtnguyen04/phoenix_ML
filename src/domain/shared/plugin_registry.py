"""
PluginRegistry — Central registry for model-specific plugins.

Provides a single place to register and resolve ITrainer, IDataLoader,
IPreprocessor, and IPostprocessor implementations for each model.

The self-healing pipeline and inference engine use this registry
to discover the correct plugin for any registered model, making
the framework fully model-agnostic.

Usage::

    registry = PluginRegistry()

    # Register plugins for a YOLO object detection model
    registry.register_trainer("yolo-detect", YOLOTrainer())
    registry.register_data_loader("yolo-detect", COCODataLoader())
    registry.register_preprocessor("yolo-detect", ImagePreprocessor())
    registry.register_postprocessor("yolo-detect", DetectionPostprocessor())

    # Self-healing pipeline resolves the trainer automatically
    trainer = registry.get_trainer("yolo-detect")
    result = await trainer.train(config)
"""

import logging
from typing import Any

from src.domain.inference.services.processor_plugin import (
    ClassificationPostprocessor,
    IPostprocessor,
    IPreprocessor,
    PassthroughPreprocessor,
)
from src.domain.training.services.trainer_plugin import ITrainer, TrainResult

logger = logging.getLogger(__name__)


class _DefaultTrainer(ITrainer):
    """Fallback trainer that raises a helpful error."""

    async def train(self, config: dict[str, Any]) -> TrainResult:
        msg = (
            "No trainer registered for this model. "
            "Implement ITrainer and register it with the PluginRegistry."
        )
        raise NotImplementedError(msg)

    async def validate(self, model_path: str, data_path: str) -> dict[str, float]:
        return {}


class PluginRegistry:
    """Central registry for model-specific plugin implementations.

    Each model can have its own set of plugins for training,
    data loading, preprocessing, and postprocessing. When a plugin
    is not registered for a specific model, sensible defaults are used.
    """

    def __init__(self) -> None:
        self._trainers: dict[str, ITrainer] = {}
        self._preprocessors: dict[str, IPreprocessor] = {}
        self._postprocessors: dict[str, IPostprocessor] = {}

        # Defaults for models without registered plugins
        self._default_preprocessor = PassthroughPreprocessor()
        self._default_postprocessor = ClassificationPostprocessor()
        self._default_trainer = _DefaultTrainer()

    # --- Trainer ---
    def register_trainer(self, model_id: str, trainer: ITrainer) -> None:
        """Register a trainer for a specific model."""
        self._trainers[model_id] = trainer
        logger.info("📌 Registered trainer for model '%s': %s", model_id, type(trainer).__name__)

    def get_trainer(self, model_id: str) -> ITrainer:
        """Get the trainer for a model, or default."""
        return self._trainers.get(model_id, self._default_trainer)

    def has_trainer(self, model_id: str) -> bool:
        """Check if a custom trainer is registered for a model."""
        return model_id in self._trainers

    # --- Preprocessor ---
    def register_preprocessor(self, model_id: str, preprocessor: IPreprocessor) -> None:
        """Register a preprocessor for a specific model."""
        self._preprocessors[model_id] = preprocessor
        logger.info(
            "📌 Registered preprocessor for '%s': %s",
            model_id,
            type(preprocessor).__name__,
        )

    def get_preprocessor(self, model_id: str) -> IPreprocessor:
        """Get preprocessor for a model, or PassthroughPreprocessor."""
        return self._preprocessors.get(model_id, self._default_preprocessor)

    # --- Postprocessor ---
    def register_postprocessor(self, model_id: str, postprocessor: IPostprocessor) -> None:
        """Register a postprocessor for a specific model."""
        self._postprocessors[model_id] = postprocessor
        logger.info(
            "📌 Registered postprocessor for '%s': %s",
            model_id,
            type(postprocessor).__name__,
        )

    def get_postprocessor(self, model_id: str) -> IPostprocessor:
        """Get postprocessor for a model, or ClassificationPostprocessor."""
        return self._postprocessors.get(model_id, self._default_postprocessor)

    # --- Introspection ---
    @property
    def registered_models(self) -> list[str]:
        """List all model IDs with at least one registered plugin."""
        models: set[str] = set()
        models.update(self._trainers.keys())
        models.update(self._preprocessors.keys())
        models.update(self._postprocessors.keys())
        return sorted(models)

    def summary(self) -> dict[str, dict[str, str]]:
        """Return a summary of all registered plugins per model."""
        result: dict[str, dict[str, str]] = {}
        for model_id in self.registered_models:
            result[model_id] = {
                "trainer": type(self.get_trainer(model_id)).__name__,
                "preprocessor": type(self.get_preprocessor(model_id)).__name__,
                "postprocessor": type(self.get_postprocessor(model_id)).__name__,
            }
        return result
