"""Tests for training bounded context."""

from unittest.mock import AsyncMock

import pytest

from phoenix_ml.domain.training.entities.training_config import TrainingConfig
from phoenix_ml.domain.training.entities.training_job import (
    TrainingJob,
    TrainingMetrics,
    TrainingStatus,
)
from phoenix_ml.domain.training.services.hyperparameter_optimizer import (
    GridSearchStrategy,
    HyperparameterOptimizer,
    RandomSearchStrategy,
    SearchSpace,
)
from phoenix_ml.domain.training.services.training_service import TrainingService

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def config() -> TrainingConfig:
    return TrainingConfig(
        dataset_path="data/german_credit.parquet",
        model_type="xgboost",
        epochs=50,
        batch_size=16,
        learning_rate=0.01,
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.save = AsyncMock()
    repo.get_by_id = AsyncMock(return_value=None)
    repo.get_by_model_id = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def training_service(mock_repo: AsyncMock) -> TrainingService:
    return TrainingService(repository=mock_repo)


# ── TrainingJob Lifecycle ─────────────────────────────────────────────


class TestTrainingJobLifecycle:
    def test_new_job_is_pending(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        assert job.status == TrainingStatus.PENDING
        assert job.is_terminal is False

    def test_start_transitions_to_running(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        assert job.status == TrainingStatus.RUNNING
        assert job.started_at is not None

    def test_complete_transitions_to_completed(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        metrics = TrainingMetrics(accuracy=0.92, f1_score=0.90)
        job.complete(metrics=metrics, artifact_path="/models/v2/model.onnx")

        assert job.status == TrainingStatus.COMPLETED
        assert job.is_terminal is True
        assert job.metrics is not None
        assert job.metrics.accuracy == 0.92
        assert job.model_artifact_path == "/models/v2/model.onnx"

    def test_fail_transitions_to_failed(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        job.fail("OOM on GPU")

        assert job.status == TrainingStatus.FAILED
        assert job.error_message == "OOM on GPU"
        assert job.is_terminal is True

    def test_cancel_from_pending(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.cancel()
        assert job.status == TrainingStatus.CANCELLED

    def test_cannot_start_completed_job(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        job.complete(TrainingMetrics(), "/path")
        with pytest.raises(ValueError, match="completed"):
            job.start()

    def test_cannot_complete_pending_job(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        with pytest.raises(ValueError, match="pending"):
            job.complete(TrainingMetrics(), "/path")

    def test_duration_calculated(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        assert job.duration_seconds is None  # Not started

        job.start()
        assert job.duration_seconds is not None
        assert job.duration_seconds >= 0


# ── TrainingConfig ────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_config_is_frozen(self, config: TrainingConfig) -> None:
        with pytest.raises(AttributeError):
            config.epochs = 200  # type: ignore[misc]

    def test_with_hyperparameters(self, config: TrainingConfig) -> None:
        new_config = config.with_hyperparameters(max_depth="6", subsample="0.8")
        assert new_config.get_hyperparameter("max_depth") == "6"
        assert new_config.get_hyperparameter("subsample") == "0.8"
        # Original unchanged
        assert config.get_hyperparameter("max_depth") == ""

    def test_get_hyperparameter_default(self, config: TrainingConfig) -> None:
        assert config.get_hyperparameter("nonexistent", "default") == "default"


# ── TrainingService ───────────────────────────────────────────────────


class TestTrainingService:
    async def test_create_job(
        self, training_service: TrainingService, config: TrainingConfig, mock_repo: AsyncMock
    ) -> None:
        job = await training_service.create_job("credit-risk", config)
        assert job.model_id == "credit-risk"
        assert job.status == TrainingStatus.PENDING
        mock_repo.save.assert_called_once()

    async def test_create_job_empty_model_id_raises(
        self, training_service: TrainingService, config: TrainingConfig
    ) -> None:
        with pytest.raises(ValueError, match="model_id"):
            await training_service.create_job("", config)

    async def test_start_job(
        self, training_service: TrainingService, config: TrainingConfig, mock_repo: AsyncMock
    ) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        mock_repo.get_by_id.return_value = job

        result = await training_service.start_job(job.job_id)
        assert result.status == TrainingStatus.RUNNING

    async def test_complete_job_emits_event(
        self, training_service: TrainingService, config: TrainingConfig, mock_repo: AsyncMock
    ) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        mock_repo.get_by_id.return_value = job

        metrics = TrainingMetrics(accuracy=0.95, f1_score=0.93)
        result = await training_service.complete_job(job.job_id, metrics, "/models/v2/model.onnx")

        assert result.status == TrainingStatus.COMPLETED
        assert len(training_service.events) == 1
        event = training_service.events[0]
        assert event.model_id == "credit-risk"
        assert event.accuracy == 0.95

    async def test_fail_job(
        self, training_service: TrainingService, config: TrainingConfig, mock_repo: AsyncMock
    ) -> None:
        job = TrainingJob(model_id="credit-risk", config=config)
        job.start()
        mock_repo.get_by_id.return_value = job

        result = await training_service.fail_job(job.job_id, "CUDA OOM")
        assert result.status == TrainingStatus.FAILED
        assert result.error_message == "CUDA OOM"

    async def test_job_not_found_raises(
        self, training_service: TrainingService, mock_repo: AsyncMock
    ) -> None:
        mock_repo.get_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await training_service.start_job("nonexistent")

    async def test_create_job_empty_dataset_path_raises(
        self, training_service: TrainingService
    ) -> None:
        empty_config = TrainingConfig(dataset_path="")
        with pytest.raises(ValueError, match="dataset_path"):
            await training_service.create_job("credit-risk", empty_config)

    async def test_complete_job_not_found_raises(
        self, training_service: TrainingService, mock_repo: AsyncMock
    ) -> None:
        mock_repo.get_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await training_service.complete_job("nope", TrainingMetrics(), "/p")

    async def test_fail_job_not_found_raises(
        self, training_service: TrainingService, mock_repo: AsyncMock
    ) -> None:
        mock_repo.get_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await training_service.fail_job("nope", "err")

    async def test_get_history(
        self, training_service: TrainingService, config: TrainingConfig, mock_repo: AsyncMock
    ) -> None:
        mock_repo.get_by_model_id.return_value = []
        result = await training_service.get_history("credit-risk")
        assert result == []
        mock_repo.get_by_model_id.assert_called_once_with("credit-risk", 10)


class TestTrainingJobEdgeCases:
    """Cover error branches for fail/cancel on terminal jobs."""

    def test_cannot_fail_completed_job(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="m", config=config)
        job.start()
        job.complete(TrainingMetrics(), "/p")
        with pytest.raises(ValueError, match="completed"):
            job.fail("err")

    def test_cannot_cancel_completed_job(self, config: TrainingConfig) -> None:
        job = TrainingJob(model_id="m", config=config)
        job.start()
        job.complete(TrainingMetrics(), "/p")
        with pytest.raises(ValueError, match="completed"):
            job.cancel()


# ── HyperparameterOptimizer ──────────────────────────────────────────


class TestHyperparameterOptimizer:
    def test_grid_search_generates_all_combinations(self, config: TrainingConfig) -> None:
        space = SearchSpace(
            learning_rates=[0.01, 0.1],
            batch_sizes=[16, 32],
            epochs_options=[50, 100],
            extra={},
        )
        optimizer = HyperparameterOptimizer(GridSearchStrategy())
        configs = optimizer.generate_trials(config, space, max_trials=100)

        assert len(configs) == 8  # 2 x 2 x 2
        learning_rates = {c.learning_rate for c in configs}
        assert learning_rates == {0.01, 0.1}

    def test_grid_search_respects_max_trials(self, config: TrainingConfig) -> None:
        space = SearchSpace(
            learning_rates=[0.01, 0.1],
            batch_sizes=[16, 32],
            epochs_options=[50, 100],
            extra={},
        )
        optimizer = HyperparameterOptimizer(GridSearchStrategy())
        configs = optimizer.generate_trials(config, space, max_trials=3)

        assert len(configs) == 3

    def test_random_search_generates_requested_count(self, config: TrainingConfig) -> None:
        space = SearchSpace(
            learning_rates=[0.001, 0.01, 0.1],
            batch_sizes=[8, 16, 32, 64],
            epochs_options=[50, 100, 200],
            extra={},
        )
        optimizer = HyperparameterOptimizer(RandomSearchStrategy(seed=42))
        configs = optimizer.generate_trials(config, space, max_trials=5)

        assert len(configs) == 5

    def test_random_search_is_reproducible(self, config: TrainingConfig) -> None:
        space = SearchSpace(
            learning_rates=[0.001, 0.01, 0.1],
            batch_sizes=[8, 16, 32],
            epochs_options=[50, 100],
            extra={},
        )
        opt1 = HyperparameterOptimizer(RandomSearchStrategy(seed=42))
        opt2 = HyperparameterOptimizer(RandomSearchStrategy(seed=42))

        c1 = opt1.generate_trials(config, space, max_trials=5)
        c2 = opt2.generate_trials(config, space, max_trials=5)

        for a, b in zip(c1, c2, strict=True):
            assert a.learning_rate == b.learning_rate
            assert a.batch_size == b.batch_size

    def test_invalid_max_trials_raises(self, config: TrainingConfig) -> None:
        space = SearchSpace(learning_rates=[0.01], batch_sizes=[16], epochs_options=[50], extra={})
        optimizer = HyperparameterOptimizer()
        with pytest.raises(ValueError, match="max_trials"):
            optimizer.generate_trials(config, space, max_trials=0)
