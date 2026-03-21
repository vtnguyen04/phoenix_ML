# ruff: noqa: PLC0415
"""
DataLoader Registry — Unified plugin resolution.

Resolves the correct DataLoader for a model_id by reading
model_configs/<model_id>.yaml → data_loader field.

Default mapping:
    task_type: classification | regression → TabularDataLoader
    task_type: image_classification        → ImageDataLoader

Custom loaders can be registered via:
    1. YAML config: `data_loader: my_package.MyLoader`
    2. Programmatic: `DataLoaderRegistry.register("my-model", MyLoader)`
"""

import importlib
import logging
from pathlib import Path
from typing import Any

from src.domain.training.services.data_loader_plugin import IDataLoader

logger = logging.getLogger(__name__)

# Default mappings: task_type → loader class path
_TASK_TYPE_DEFAULTS: dict[str, str] = {
    "classification": "src.infrastructure.data_loaders.tabular_loader.TabularDataLoader",
    "regression": "src.infrastructure.data_loaders.tabular_loader.TabularDataLoader",
    "image_classification": "src.infrastructure.data_loaders.image_loader.ImageDataLoader",
}


class DataLoaderRegistry:
    """Plugin registry for DataLoaders.

    Follows the Strategy + Registry pattern:
    - Register loaders by model_id or task_type
    - Resolve at runtime from model_configs YAML
    - Fallback to task_type defaults
    """

    _registry: dict[str, type[IDataLoader]] = {}

    @classmethod
    def register(cls, model_id: str, loader_cls: type[IDataLoader]) -> None:
        """Register a custom DataLoader for a specific model."""
        cls._registry[model_id] = loader_cls
        logger.info("Registered DataLoader %s for model %s", loader_cls.__name__, model_id)

    @classmethod
    def get(cls, model_id: str) -> type[IDataLoader] | None:
        """Get a registered DataLoader by model_id."""
        return cls._registry.get(model_id)

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (for testing)."""
        cls._registry.clear()


def _import_class(dotted_path: str) -> type[IDataLoader]:
    """Import a class from a dotted module path (e.g. 'my.module.MyClass')."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, IDataLoader)):
        raise TypeError(f"{dotted_path} is not a subclass of IDataLoader")
    return cls


def _load_model_config(model_id: str) -> dict[str, Any]:
    """Load model config YAML for a given model_id."""
    import yaml  # type: ignore[import-untyped]

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    config_path = project_root / "model_configs" / f"{model_id}.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def resolve_data_loader(model_id: str) -> IDataLoader:
    """Resolve the correct DataLoader for a model_id.

    Resolution order:
        1. Programmatic registry (DataLoaderRegistry)
        2. YAML config: model_configs/<model_id>.yaml → data_loader field
        3. YAML config: task_type → default loader
        4. Fallback: TabularDataLoader

    Returns:
        Instantiated IDataLoader ready to use.
    """
    # 1. Check programmatic registry
    registered = DataLoaderRegistry.get(model_id)
    if registered is not None:
        logger.debug("Using registered DataLoader for %s: %s", model_id, registered.__name__)
        return registered()

    # 2. Check YAML config
    config = _load_model_config(model_id)

    # 2a. Explicit data_loader class in YAML
    loader_path = config.get("data_loader", "")
    if loader_path:
        try:
            loader_cls = _import_class(loader_path)
            logger.info(
                "Resolved DataLoader from YAML for %s: %s",
                model_id,
                loader_path,
            )
            return loader_cls()
        except Exception as e:
            logger.warning(
                "Failed to load data_loader '%s' from config: %s. Falling back.",
                loader_path,
                e,
            )

    # 2b. Resolve by task_type → default loader
    task_type = config.get("task_type", "classification")
    default_path = _TASK_TYPE_DEFAULTS.get(task_type)
    if default_path:
        try:
            loader_cls = _import_class(default_path)
            logger.debug(
                "Resolved DataLoader by task_type=%s for %s: %s",
                task_type,
                model_id,
                loader_cls.__name__,
            )
            return loader_cls()
        except Exception as e:
            logger.warning("Failed to load default loader for task_type=%s: %s", task_type, e)

    # 3. Ultimate fallback
    from src.infrastructure.data_loaders.tabular_loader import TabularDataLoader

    logger.debug("Falling back to TabularDataLoader for %s", model_id)
    return TabularDataLoader()
