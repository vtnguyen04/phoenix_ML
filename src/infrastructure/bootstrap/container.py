import asyncio
import logging
import os
from collections.abc import Callable
from pathlib import Path

from src.config import get_settings
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.services.batch_manager import BatchConfig, BatchManager
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.monitoring.services.drift_calculator import DriftCalculator
from src.domain.monitoring.services.metrics_publisher import MetricsPublisher
from src.domain.monitoring.services.model_evaluator import (
    IModelEvaluator,
    get_evaluator,
)
from src.domain.shared.domain_events import (
    DriftDetected,
    DriftScorePublished,
    ModelRetrained,
    PredictionCompleted,
)
from src.domain.shared.event_bus import DomainEventBus
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.feature_store.redis_feature_store import RedisFeatureStore
from src.infrastructure.messaging.kafka_producer import KafkaProducer
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.infrastructure.ml_engines.tensorrt_executor import TensorRTExecutor
from src.infrastructure.ml_engines.triton_client import TritonInferenceClient
from src.infrastructure.monitoring.prometheus_metrics_publisher import (
    PrometheusMetricsPublisher,
)
from src.shared.utils.model_generator import generate_simple_onnx

logger = logging.getLogger(__name__)
settings = get_settings()

artifact_storage = LocalArtifactStorage(base_dir=Path(settings.ARTIFACT_STORAGE_DIR))

# ── Engine Factory Registry (OCP: add new engines via dict entry) ─────
_ENGINE_FACTORIES: dict[str, Callable[[], InferenceEngine]] = {
    "onnx": lambda: ONNXInferenceEngine(cache_dir=Path(settings.CACHE_DIR)),
    "tensorrt": lambda: TensorRTExecutor(cache_dir=Path(settings.CACHE_DIR)),
    "triton": lambda: TritonInferenceClient(
        triton_url=getattr(settings, "TRITON_URL", "http://localhost:8000"),
    ),
}

engine_type = getattr(settings, "INFERENCE_ENGINE", "onnx").lower()
_engine_factory = _ENGINE_FACTORIES.get(engine_type, _ENGINE_FACTORIES["onnx"])
inference_engine: InferenceEngine = _engine_factory()

batch_config = BatchConfig(
    max_batch_size=settings.BATCH_MAX_SIZE,
    max_wait_time_ms=settings.BATCH_MAX_WAIT_MS,
)
batch_manager = BatchManager(inference_engine, config=batch_config)
kafka_producer = KafkaProducer(bootstrap_servers=settings.KAFKA_URL)

# ── Kafka Consumer (consumes inference-events for downstream processing) ──
from src.infrastructure.messaging.kafka_consumer import KafkaConsumer  # noqa: E402

kafka_consumer = KafkaConsumer(
    bootstrap_servers=settings.KAFKA_URL,
    group_id="phoenix-ml-consumers",
)

# ── Prediction Cache ──────────────────────────────────────────────
from src.infrastructure.cache.prediction_cache import PredictionCache  # noqa: E402

prediction_cache = PredictionCache(default_ttl_seconds=300, max_size=10_000)

drift_calculator = DriftCalculator()

# ── Evaluator Factory (uses existing get_evaluator from model_evaluator) ──
default_task_type = getattr(settings, "DEFAULT_TASK_TYPE", "classification")
model_evaluator: IModelEvaluator = get_evaluator(default_task_type)

# ── MetricsPublisher (Adapter Pattern) ────────────────────────────
metrics_publisher: MetricsPublisher = PrometheusMetricsPublisher()

# ── Domain Event Bus (Observer Pattern) ───────────────────────────
#    Subscribers react to domain events independently.
#    Adding new side-effects = register a subscriber. Zero handler changes.
event_bus = DomainEventBus()

# Subscribe MetricsPublisher to domain events
event_bus.subscribe(
    PredictionCompleted,
    lambda e: (
        metrics_publisher.record_prediction(e.model_id, e.version, e.status),
        metrics_publisher.record_latency(e.model_id, e.version, e.latency),
        metrics_publisher.record_confidence(e.model_id, e.version, e.confidence)
        if e.status == "success"
        else None,
    ),
)

event_bus.subscribe(
    DriftScorePublished,
    lambda e: (
        metrics_publisher.publish_drift_score(e.model_id, e.feature_name, e.method, e.score)
    ),
)

event_bus.subscribe(
    DriftDetected, lambda e: (metrics_publisher.record_drift_detected(e.model_id, e.feature_name))
)

event_bus.subscribe(
    ModelRetrained,
    lambda e: (metrics_publisher.publish_model_metrics(e.model_id, e.version, e.metrics)),
)

# ── Feature Store Factory Registry (OCP) ──────────────────────────
_FEATURE_STORE_FACTORIES: dict[str, Callable[[], FeatureStore]] = {
    "redis": lambda: RedisFeatureStore(redis_url=settings.REDIS_URL),
    "memory": lambda: InMemoryFeatureStore(),
}

_fs_key = "redis" if settings.USE_REDIS else "memory"
feature_store: FeatureStore = _FEATURE_STORE_FACTORIES[_fs_key]()

# --- Plugin Registry (model-agnostic plugin resolution) ---
from src.domain.shared.plugin_registry import PluginRegistry  # noqa: E402

plugin_registry = PluginRegistry()

shutdown_event = asyncio.Event()


def find_project_root() -> Path:
    """Find root by searching for pyproject.toml upwards from this file."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_model_exists(
    model_id: str | None = None,
    version: str | None = None,
) -> Path:
    """Ensures model artifact exists, generating one if in CI/test context."""
    model_id = model_id or settings.DEFAULT_MODEL_ID
    version = version or settings.DEFAULT_MODEL_VERSION

    # Normalize model_id for filesystem (e.g. my-model -> my_model)
    fs_model_id = model_id.replace("-", "_")

    root = find_project_root()
    model_path = root / "models" / fs_model_id / version / "model.onnx"

    if model_path.exists():
        return model_path.absolute()

    is_ci = os.getenv("GITHUB_ACTIONS")
    is_test = "test" in str(Path.cwd())

    if is_ci or is_test:
        # Resolve feature count from model config so the generated
        # stub model has the correct input dimensions.
        n_features = 4  # default fallback
        try:
            from src.infrastructure.bootstrap.model_config_loader import (  # noqa: PLC0415
                load_all_model_configs,
            )

            cfgs = load_all_model_configs(root / settings.MODEL_CONFIG_DIR)
            if model_id in cfgs and cfgs[model_id].feature_names:
                n_features = len(cfgs[model_id].feature_names)
        except Exception:
            pass
        logger.warning(
            "🧪 CI/Test context. Generating valid ONNX model at %s (n_features=%d)",
            model_path,
            n_features,
        )
        generate_simple_onnx(model_path, n_features=n_features)
        return model_path.absolute()

    msg = f"Model not found at {model_path} and not in CI environment."
    raise FileNotFoundError(msg)
