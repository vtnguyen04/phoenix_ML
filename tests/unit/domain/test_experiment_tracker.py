"""Tests for ExperimentTracker — comprehensive MLflow tracking.

All tests patch _HAS_MLFLOW=False to avoid connecting to a real MLflow server.
"""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from phoenix_ml.domain.training.services.experiment_tracker import ExperimentTracker


@pytest.fixture
def tracker() -> Generator[ExperimentTracker, None, None]:
    # Patch _HAS_MLFLOW to False to avoid real MLflow calls
    with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
        t = ExperimentTracker(tracking_uri="http://nonexistent:5000")
        yield t


class TestExperimentTracker:
    def test_start_and_end_run(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            run = tracker.start_run(
                run_name="test-run",
                model_type="xgboost",
                hyperparameters={"n_estimators": 100, "max_depth": 5},
            )
            assert run.experiment_name == "phoenix-experiments"
            assert run.model_type == "xgboost"
            assert run.hyperparameters["n_estimators"] == 100
            assert run.start_time != ""

            tracker.log_metrics(run, {"accuracy": 0.95, "f1": 0.92})
            assert run.metrics["accuracy"] == 0.95

            result = tracker.end_run(run)
            assert result.end_time != ""
            assert result.duration_seconds >= 0

    def test_compare_runs(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            run1 = tracker.start_run("r1", "xgb", {"lr": 0.1})
            tracker.log_metrics(run1, {"accuracy": 0.9})
            tracker.end_run(run1)

            run2 = tracker.start_run("r2", "xgb", {"lr": 0.01})
            tracker.log_metrics(run2, {"accuracy": 0.95})
            tracker.end_run(run2)

            comparison = tracker.compare_runs("accuracy")
            assert len(comparison) == 2
            assert comparison[0]["metric"] == 0.95  # Best first

    def test_get_best_run(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            run1 = tracker.start_run("r1", "xgb", {"lr": 0.1})
            tracker.log_metrics(run1, {"accuracy": 0.85})
            tracker.end_run(run1)

            run2 = tracker.start_run("r2", "rf", {"n_trees": 200})
            tracker.log_metrics(run2, {"accuracy": 0.92})
            tracker.end_run(run2)

            best = tracker.get_best_run("accuracy")
            assert best is not None
            assert best.metrics["accuracy"] == 0.92
            assert best.model_type == "rf"

    def test_system_info_captured(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            run = tracker.start_run("sys-test", "test", {})
            assert "python_version" in run.system_info
            assert "os" in run.system_info
            tracker.end_run(run)

    def test_data_lineage(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            lineage = {"source": "credit_risk.csv", "rows": 1000, "version": "v2"}
            run = tracker.start_run("lineage-test", "xgb", {}, data_lineage=lineage)
            assert run.data_lineage["source"] == "credit_risk.csv"
            tracker.end_run(run)

    def test_save_local_log(self, tracker: ExperimentTracker, tmp_path: Path) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            run = tracker.start_run("save-test", "xgb", {"a": 1})
            tracker.log_metrics(run, {"loss": 0.1})
            tracker.end_run(run)

            log_path = str(tmp_path / "log.json")
            tracker.save_local_log(log_path)

            data = json.loads(Path(log_path).read_text())
            assert len(data) == 1
            assert data[0]["metrics"]["loss"] == 0.1

    def test_flatten_dict(self) -> None:
        flat = ExperimentTracker._flatten_dict(
            {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        )
        assert flat["a"] == "1"
        assert flat["b.c"] == "2"
        assert flat["b.d.e"] == "3"

    def test_no_runs_best_returns_none(self, tracker: ExperimentTracker) -> None:
        with patch("phoenix_ml.domain.training.services.experiment_tracker._HAS_MLFLOW", False):
            assert tracker.get_best_run() is None
