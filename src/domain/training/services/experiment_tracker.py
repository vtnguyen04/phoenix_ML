"""Enhanced Experiment Tracker — comprehensive MLflow tracking.

Tracks: hyperparameters, metrics, artifacts, tags, system info, data lineage.
Supports comparison between experiments and auto-logging.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try MLflow import
try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False
    logger.info("MLflow not installed, experiment tracking will use local file logging")


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    run_id: str
    experiment_name: str
    model_type: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    data_lineage: dict[str, Any] = field(default_factory=dict)
    system_info: dict[str, str] = field(default_factory=dict)


class ExperimentTracker:
    """Enhanced MLflow experiment tracker.

    Beyond basic logging, this tracks:
    - Full hyperparameter search space & selected values
    - Data lineage (source, version, quality report digest)
    - System info (Python version, OS, CPU, RAM)
    - Training duration and resource usage
    - Artifact versioning with tags
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "phoenix-experiments",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._runs: list[ExperimentRun] = []  # Local backup

        if _HAS_MLFLOW:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

    def start_run(
        self,
        run_name: str,
        model_type: str,
        hyperparameters: dict[str, Any],
        tags: dict[str, str] | None = None,
        data_lineage: dict[str, Any] | None = None,
    ) -> ExperimentRun:
        """Start a new experiment run with comprehensive tracking."""
        run_id = f"run_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        start = datetime.now(UTC).isoformat()

        system_info = {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        }

        run = ExperimentRun(
            run_id=run_id,
            experiment_name=self._experiment_name,
            model_type=model_type,
            hyperparameters=hyperparameters,
            metrics={},
            tags=tags or {},
            start_time=start,
            data_lineage=data_lineage or {},
            system_info=system_info,
        )

        if _HAS_MLFLOW:
            mlflow.start_run(run_name=run_name)
            # Log hyperparameters (flatten nested dicts)
            flat_params = self._flatten_dict(hyperparameters)
            mlflow.log_params(flat_params)
            # Log tags
            for k, v in (tags or {}).items():
                mlflow.set_tag(k, v)
            mlflow.set_tag("model_type", model_type)
            mlflow.set_tag("phoenix.run_id", run_id)
            # Log system info
            for k, v in system_info.items():
                mlflow.set_tag(f"system.{k}", v)
            # Log data lineage
            if data_lineage:
                mlflow.log_params(
                    {f"data.{k}": str(v)[:250] for k, v in data_lineage.items()}
                )

        logger.info("Started experiment run: %s (%s)", run_name, run_id)
        return run

    def log_metrics(
        self,
        run: ExperimentRun,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics during or after training."""
        run.metrics.update(metrics)
        if _HAS_MLFLOW:
            mlflow.log_metrics(metrics, step=step)
        logger.info("Logged metrics: %s", metrics)

    def log_artifact(self, run: ExperimentRun, filepath: str) -> None:
        """Log artifact (model file, plot, config, etc)."""
        run.artifacts.append(filepath)
        if _HAS_MLFLOW and Path(filepath).exists():
            mlflow.log_artifact(filepath)

    def end_run(self, run: ExperimentRun) -> ExperimentRun:
        """End experiment run, compute duration, store locally."""
        run.end_time = datetime.now(UTC).isoformat()
        if run.start_time:
            start_dt = datetime.fromisoformat(run.start_time)
            end_dt = datetime.fromisoformat(run.end_time)
            run.duration_seconds = (end_dt - start_dt).total_seconds()

        if _HAS_MLFLOW:
            mlflow.log_metric("duration_seconds", run.duration_seconds)
            mlflow.end_run()

        self._runs.append(run)
        logger.info(
            "Ended run %s: %d metrics, %.1fs duration",
            run.run_id,
            len(run.metrics),
            run.duration_seconds,
        )
        return run

    def compare_runs(self, metric_name: str = "accuracy") -> list[dict[str, Any]]:
        """Compare all runs sorted by a metric (descending)."""
        results = []
        for run in self._runs:
            results.append({
                "run_id": run.run_id,
                "model_type": run.model_type,
                "metric": run.metrics.get(metric_name),
                "hyperparameters": run.hyperparameters,
                "duration_seconds": run.duration_seconds,
            })
        return sorted(results, key=lambda x: x.get("metric") or 0, reverse=True)

    def get_best_run(self, metric_name: str = "accuracy") -> ExperimentRun | None:
        """Get run with the best value for the given metric."""
        if not self._runs:
            return None
        return max(
            self._runs,
            key=lambda r: r.metrics.get(metric_name, 0),
        )

    def save_local_log(self, path: str = "experiments/experiment_log.json") -> None:
        """Save all runs to a local JSON file (backup if MLflow unavailable)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for run in self._runs:
            data.append({
                "run_id": run.run_id,
                "experiment_name": run.experiment_name,
                "model_type": run.model_type,
                "hyperparameters": run.hyperparameters,
                "metrics": run.metrics,
                "tags": run.tags,
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration_seconds": run.duration_seconds,
                "data_lineage": run.data_lineage,
                "system_info": run.system_info,
            })
        p.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Saved %d experiment runs to %s", len(data), path)

    @staticmethod
    def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
        """Flatten nested dict for MLflow log_params."""
        items: dict[str, str] = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(ExperimentTracker._flatten_dict(v, key))
            else:
                items[key] = str(v)[:250]  # MLflow has 250 char limit
        return items
