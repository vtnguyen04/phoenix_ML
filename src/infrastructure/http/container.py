import asyncio
import logging
import os
from pathlib import Path

from src.config import get_settings
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.services.batch_manager import BatchConfig, BatchManager
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.domain.monitoring.services.model_evaluator import ModelEvaluator
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.feature_store.redis_feature_store import RedisFeatureStore
from src.infrastructure.messaging.kafka_producer import KafkaProducer
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.shared.utils.model_generator import generate_simple_onnx

logger = logging.getLogger(__name__)
settings = get_settings()

artifact_storage = LocalArtifactStorage(base_dir=Path("/tmp/phoenix/remote_storage"))
inference_engine = ONNXInferenceEngine(cache_dir=Path("/tmp/phoenix/model_cache"))
batch_config = BatchConfig(max_batch_size=16, max_wait_time_ms=10)
batch_manager = BatchManager(inference_engine, config=batch_config)
kafka_producer = KafkaProducer(bootstrap_servers=settings.KAFKA_URL)
drift_calculator = DriftCalculator()
model_evaluator = ModelEvaluator()

feature_store: FeatureStore
if settings.USE_REDIS:
    feature_store = RedisFeatureStore(redis_url=settings.REDIS_URL)
else:
    feature_store = InMemoryFeatureStore()

shutdown_event = asyncio.Event()


def find_project_root() -> Path:
    """Find root by searching for pyproject.toml upwards from this file."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_model_exists() -> Path:
    """Ensures model artifact exists, generating one if in CI/test context."""
    root = find_project_root()
    model_path = root / "models" / "credit_risk" / "v1" / "model.onnx"

    if model_path.exists():
        return model_path.absolute()

    is_ci = os.getenv("GITHUB_ACTIONS")
    is_test = "test" in str(Path.cwd())

    if is_ci or is_test:
        logger.warning(
            "🧪 CI/Test context. Generating valid ONNX model at %s", model_path
        )
        generate_simple_onnx(model_path)
        return model_path.absolute()

    msg = f"Model not found at {model_path} and not in CI environment."
    raise FileNotFoundError(msg)
