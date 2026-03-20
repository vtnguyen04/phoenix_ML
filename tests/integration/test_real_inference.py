"""
Integration Test: Real Model Inference.

Tests the full prediction pipeline with a REAL ONNX model trained on
the German Credit dataset and REAL feature records.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.inference.entities.model import Model
from src.domain.inference.services.batch_manager import BatchConfig, BatchManager
from src.domain.inference.services.inference_service import InferenceService
from src.domain.inference.services.routing_strategy import SingleModelStrategy
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.domain.shared.event_bus import DomainEventBus
from src.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)
from src.infrastructure.feature_store.in_memory_feature_store import (
    InMemoryFeatureStore,
)
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository


def _find_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


ROOT = _find_root()
MODEL_PATH = ROOT / "models" / "credit_risk" / "v1" / "model.onnx"
FEATURES_PATH = ROOT / "data" / "reference_features.json"
METRICS_PATH = ROOT / "models" / "credit_risk" / "v1" / "metrics.json"

# Skip if real model doesn't exist (e.g. in CI without training step)
requires_real_model = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Real ONNX model not found — run scripts/train_model.py first",
)
requires_real_features = pytest.mark.skipif(
    not FEATURES_PATH.exists(),
    reason="Real features not found — run scripts/seed_features.py first",
)


@pytest.fixture
def real_model() -> Model:
    """Create a model entity pointing to the real trained ONNX model."""
    return Model(
        id="credit-risk",
        version="v1",
        uri=f"local://{MODEL_PATH.absolute()}",
        framework="onnx",
        metadata={
            "features": [
                "duration",
                "credit_amount",
                "installment_commitment",
                "residence_since",
                "age",
                "existing_credits",
                "num_dependents",
                "checking_status",
                "credit_history",
                "purpose",
                "savings_status",
                "employment",
                "personal_status",
                "other_parties",
                "property_magnitude",
                "other_payment_plans",
                "housing",
                "job",
                "own_telephone",
                "foreign_worker",
                "credit_per_month",
                "age_credit_ratio",
                "installment_credit_ratio",
                "age_employment_score",
                "credit_risk_density",
                "duration_installment",
                "checking_savings_interact",
                "age_checking_interact",
                "credit_existing_interact",
                "log_credit_amount",
            ],
            "role": "champion",
        },
    )


@pytest.fixture
def onnx_engine(tmp_path: Path) -> ONNXInferenceEngine:
    """ONNX engine with cache pointing to the real model location."""
    import shutil  # noqa: PLC0415

    cache_dir = tmp_path / "model_cache"
    model_cache_path = cache_dir / "credit-risk" / "v1" / "model.onnx"
    model_cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MODEL_PATH, model_cache_path)
    return ONNXInferenceEngine(cache_dir=cache_dir)


@pytest.fixture
def feature_store_with_real_data() -> InMemoryFeatureStore:
    """Feature store populated with real German Credit dataset records."""
    fs = InMemoryFeatureStore()
    if FEATURES_PATH.exists():
        with open(FEATURES_PATH) as f:
            records = json.load(f)
        for rec in records:
            # Synchronous helper: directly add to store
            fs._store[rec["entity_id"]] = rec["features"]
    return fs


@requires_real_model
class TestRealModelInference:
    """Tests using the REAL trained ONNX model."""

    async def test_predict_with_real_model(
        self,
        real_model: Model,
        onnx_engine: ONNXInferenceEngine,
    ) -> None:
        """Verify real model produces meaningful predictions."""
        fv = FeatureVector(values=np.array([float(i) * 0.1 for i in range(30)], dtype=np.float32))
        await onnx_engine.load(real_model)
        prediction = await onnx_engine.predict(real_model, fv)

        # Real model should produce a valid class (0 or 1)
        assert prediction.result in (0, 1)
        assert 0.0 <= prediction.confidence.value <= 1.0
        assert prediction.latency_ms >= 0
        assert prediction.model_id == "credit-risk"

    @requires_real_features
    async def test_predict_with_real_features(
        self,
        real_model: Model,
        onnx_engine: ONNXInferenceEngine,
        feature_store_with_real_data: InMemoryFeatureStore,
    ) -> None:
        """End-to-end: real model + real feature store data."""
        model_repo = InMemoryModelRepository()
        await model_repo.save(real_model)
        await model_repo.update_stage("credit-risk", "v1", "champion")

        batch_manager = BatchManager(
            onnx_engine, config=BatchConfig(max_batch_size=1, max_wait_time_ms=5)
        )
        service = InferenceService(
            model_repo=model_repo,
            inference_engine=onnx_engine,
            batch_manager=batch_manager,
            feature_store=feature_store_with_real_data,
            artifact_storage=LocalArtifactStorage(base_dir=Path("/tmp/test")),
            routing_strategy=SingleModelStrategy(),
        )
        handler = PredictHandler(service, DomainEventBus())

        try:
            # Predict for 5 real customers
            for i in range(5):
                command = PredictCommand(
                    model_id="credit-risk",
                    entity_id=f"customer-{i:04d}",
                )
                prediction = await handler.execute(command)
                assert prediction.result in (0, 1)
                assert 0.0 <= prediction.confidence.value <= 1.0
        finally:
            await batch_manager.stop()

    async def test_batch_predict_real_data(
        self,
        real_model: Model,
        onnx_engine: ONNXInferenceEngine,
    ) -> None:
        """Verify batch prediction works with the real model."""
        await onnx_engine.load(real_model)

        # Create a batch of feature vectors
        vectors = [
            FeatureVector(
                values=np.array([np.random.normal() for _ in range(30)], dtype=np.float32)
            )
            for _ in range(10)
        ]

        predictions = await onnx_engine.batch_predict(real_model, vectors)
        assert len(predictions) == 10
        for pred in predictions:
            assert pred.result in (0, 1)
            assert 0.0 <= pred.confidence.value <= 1.0

    @requires_real_features
    async def test_real_metrics_match_training(self) -> None:
        """Verify the saved metrics file contains real training results."""
        if not METRICS_PATH.exists():
            pytest.skip("metrics.json not found")

        with open(METRICS_PATH) as f:
            metrics = json.load(f)

        assert metrics["dataset"] == "german-credit-openml"
        assert 0.5 < metrics["accuracy"] < 1.0
        assert 0.5 < metrics["f1_score"] < 1.0
        assert metrics["train_samples"] > 0
        assert metrics["test_samples"] > 0
        assert metrics["n_features"] == 30


@requires_real_model
@requires_real_features
class TestRealDriftDetection:
    """Tests drift detection with real data distributions."""

    async def test_no_drift_with_same_distribution(self) -> None:
        """Reference data vs itself should show NO drift."""
        from src.domain.monitoring.services.drift_calculator import DriftCalculator  # noqa: PLC0415

        ref_path = ROOT / "data" / "reference_data.json"
        if not ref_path.exists():
            ref_path = ROOT / "models" / "data" / "reference_data.json"

        with open(ref_path) as f:
            data = json.load(f)

        distributions = data["reference_distributions"]
        income_data = distributions["credit_amount"]

        calculator = DriftCalculator()

        # Same distribution → no drift
        report = calculator.calculate_drift(
            feature_name="credit_amount",
            reference_data=income_data[:400],
            current_data=income_data[400:],
            test_type="ks",
        )
        # With same underlying distribution, drift should not be detected
        assert report.p_value > 0.01

    async def test_drift_detected_with_shifted_data(self) -> None:
        """Shifted data should trigger drift detection."""
        from src.domain.monitoring.services.drift_calculator import DriftCalculator  # noqa: PLC0415

        ref_path = ROOT / "data" / "reference_data.json"
        if not ref_path.exists():
            ref_path = ROOT / "models" / "data" / "reference_data.json"

        with open(ref_path) as f:
            data = json.load(f)

        income_data = data["reference_distributions"]["credit_amount"]

        calculator = DriftCalculator()

        # Create heavily shifted production data
        shifted_data = [x + 5.0 for x in income_data[:100]]

        report = calculator.calculate_drift(
            feature_name="credit_amount",
            reference_data=income_data,
            current_data=shifted_data,
            test_type="ks",
        )
        assert report.drift_detected is True
