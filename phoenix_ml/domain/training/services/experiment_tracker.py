"""MLflow experiment tracker.

Logs hyperparameters, metrics, artifacts, and system info to MLflow
for each training run. Provides both manual API and automatic pipeline
integration via ``tracked_run()``.

Dependencies:
    - ``mlflow`` (optional): falls back to local JSON if unavailable.
    - ``psutil`` (optional): for system resource logging.
    - ``pynvml`` (optional): for GPU metrics.
    - ``matplotlib``, ``scikit-learn`` (optional): for artifact generation.

Configuration:
    - ``tracking_uri``: MLflow server URI (default ``http://localhost:5000``).
    - ``experiment_name``: MLflow experiment name (default ``phoenix-experiments``).

Error handling:
    - Missing MLflow: all tracking silently falls back to local JSON.
    - Missing psutil/pynvml: system metrics are skipped.
    - Artifact generation failures are logged and do not raise.
"""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Try MLflow import
try:
    import mlflow
    import mlflow.onnx

    _HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False
    logger.info("MLflow not installed, experiment tracking will use local file logging")

# Try psutil for system metrics
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


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
    """Tracks training experiment runs in MLflow.

    Provides ``start_run()``, ``log_metrics()``, ``log_artifact()``,
    ``end_run()`` for manual control, and ``tracked_run()`` for automatic
    pipeline wrapping.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment to log runs under.

    State:
        Maintains an in-memory list of ``ExperimentRun`` objects as local
        backup. Can be persisted via ``save_local_log()``.
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

    # ── Tracked Pipeline Run (AUTO) ──────────────────────────────

    async def tracked_run(
        self,
        pipeline: Any,
        context: Any,
    ) -> Any:
        """Auto-wrap a TrainingPipeline.run() with full MLflow tracking.

        This is the main entry point for framework-level tracking.
        User writes zero MLflow code — everything auto-logged.
        """
        from phoenix_ml.domain.training.pipeline import PipelineContext  # noqa: PLC0415

        assert isinstance(context, PipelineContext)

        # Read config for hyperparams
        hyperparams = {
            "model_id": context.model_id,
            "version": context.version,
            "train_script": context.train_script,
            "data_path": context.data_path,
        }
        hyperparams.update(context.config)

        tags = {
            "pipeline_steps": " → ".join(s.name for s in pipeline.steps),
            "framework": "phoenix-ml",
        }

        data_lineage = {
            "data_path": context.data_path,
            "model_path": str(context.model_path),
        }

        # Start tracked run
        run = self.start_run(
            run_name=f"{context.model_id}-{context.version}",
            model_type=hyperparams.get("model_type", "unknown"),
            hyperparameters=hyperparams,
            tags=tags,
            data_lineage=data_lineage,
        )

        # Execute the pipeline (call _execute_steps directly to avoid recursion)
        start = time.monotonic()
        context = await pipeline._execute_steps(context)
        duration = time.monotonic() - start

        # Auto-log metrics from context
        if context.metrics:
            self.log_metrics(run, context.metrics)

        # Log duration
        self.log_metrics(run, {"training_duration_seconds": duration})

        # Log system resources
        self._log_system_resources(run)

        # Auto-generate artifacts from metrics (if classification results present)
        if context.metrics and "accuracy" in context.metrics:
            self._auto_log_classification_artifacts(run, context)

        # Log ONNX model if exists
        model_path = context.artifacts.get("model") or str(context.model_path)
        if model_path and Path(model_path).exists():
            self.log_artifact(run, model_path)

        # End run
        self.end_run(run)

        return context

    # ── Manual API ───────────────────────────────────────────────

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

        system_info = self._collect_system_info()

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

        logger.info("🚀 Started experiment run: %s (%s)", run_name, run_id)
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
            # MLflow requires numeric values
            safe_metrics = {
                k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float))
                and not (isinstance(v, float) and math.isnan(v))
            }
            if safe_metrics:
                mlflow.log_metrics(safe_metrics, step=step)
        logger.info("📊 Logged metrics: %s", list(metrics.keys()))

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
            "✅ Ended run %s: %d metrics, %.1fs duration",
            run.run_id,
            len(run.metrics),
            run.duration_seconds,
        )
        return run

    # ── Auto-Artifact Generation ─────────────────────────────────

    def auto_log_artifacts(
        self,
        run: ExperimentRun,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        """Auto-generate and log classification artifacts to MLflow.

        Generates:
        - Confusion matrix (PNG)
        - Classification report (text)
        - ROC curve (PNG, if probabilities available)
        """
        import tempfile  # noqa: PLC0415

        tmpdir = tempfile.mkdtemp(prefix="phoenix_artifacts_")

        # 1. Confusion matrix
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        self._generate_confusion_matrix(y_true, y_pred, cm_path, class_names)
        self.log_artifact(run, cm_path)

        # 2. Classification report
        report_path = os.path.join(tmpdir, "classification_report.txt")
        self._generate_classification_report(y_true, y_pred, report_path, class_names)
        self.log_artifact(run, report_path)

        # 3. ROC curve
        if y_proba is not None:
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            self._generate_roc_curve(y_true, y_proba, roc_path)
            self.log_artifact(run, roc_path)

        # 4. Feature importance (if available in metrics)
        logger.info("📎 Auto-logged %d artifacts", len(run.artifacts))

    def _auto_log_classification_artifacts(
        self,
        run: ExperimentRun,
        context: Any,
    ) -> None:
        """Auto-log artifacts from PipelineContext metrics (best-effort)."""
        # If context has y_test/y_pred stored, use them
        # Otherwise, log a metrics summary artifact
        import tempfile  # noqa: PLC0415

        tmpdir = tempfile.mkdtemp(prefix="phoenix_artifacts_")

        # Metrics summary JSON
        summary_path = os.path.join(tmpdir, "metrics_summary.json")
        Path(summary_path).write_text(
            json.dumps(context.metrics, indent=2, default=str)
        )
        self.log_artifact(run, summary_path)

        # Training config
        config_path = os.path.join(tmpdir, "training_config.json")
        Path(config_path).write_text(
            json.dumps(context.to_dict(), indent=2, default=str)
        )
        self.log_artifact(run, config_path)

    # ── Artifact Generators ──────────────────────────────────────

    @staticmethod
    def _generate_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        class_names: list[str] | None = None,
    ) -> None:
        """Generate confusion matrix PNG via matplotlib."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            from sklearn.metrics import ConfusionMatrixDisplay  # noqa: PLC0415

            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay.from_predictions(
                y_true, y_pred,
                display_labels=class_names,
                cmap="Blues",
                ax=ax,
            )
            ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info("📊 Confusion matrix saved: %s", output_path)
        except Exception as e:
            logger.warning("Could not generate confusion matrix: %s", e)

    @staticmethod
    def _generate_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        class_names: list[str] | None = None,
    ) -> None:
        """Generate text classification report."""
        try:
            from sklearn.metrics import classification_report  # noqa: PLC0415

            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
            )
            Path(output_path).write_text(report)
            logger.info("📊 Classification report saved: %s", output_path)
        except Exception as e:
            logger.warning("Could not generate classification report: %s", e)

    @staticmethod
    def _generate_roc_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_path: str,
    ) -> None:
        """Generate ROC curve PNG."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            from sklearn.metrics import RocCurveDisplay  # noqa: PLC0415

            fig, ax = plt.subplots(figsize=(8, 6))

            # Handle binary vs multi-class
            _binary_class_count = 2
            if y_proba.ndim == _binary_class_count and y_proba.shape[1] == _binary_class_count:
                proba = y_proba[:, 1]
            elif y_proba.ndim == 1:
                proba = y_proba
            else:
                logger.info("Multi-class ROC not supported, skipping")
                return

            RocCurveDisplay.from_predictions(
                y_true, proba, ax=ax,
                name="Model",
            )
            ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_path, dpi=150)
            plt.close(fig)
            logger.info("📊 ROC curve saved: %s", output_path)
        except Exception as e:
            logger.warning("Could not generate ROC curve: %s", e)

    # ── System Resources ─────────────────────────────────────────

    def _log_system_resources(self, run: ExperimentRun) -> None:
        """Log CPU/Memory/GPU usage at end of training."""
        if not _HAS_PSUTIL:
            return

        resources: dict[str, float] = {
            "system.cpu_percent": psutil.cpu_percent(interval=None),
            "system.memory_percent": psutil.virtual_memory().percent,
            "system.memory_used_gb": round(
                psutil.virtual_memory().used / (1024**3), 2
            ),
        }

        # GPU (if pynvml available)
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            resources["system.gpu_percent"] = float(util.gpu)
            resources["system.gpu_memory_used_gb"] = round(mem.used / (1024**3), 2)
        except Exception:
            pass  # No GPU or pynvml not available

        self.log_metrics(run, resources)

    @staticmethod
    def _collect_system_info() -> dict[str, str]:
        """Collect static system info for tags."""
        info = {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        }
        if _HAS_PSUTIL:
            info["cpu_count"] = str(psutil.cpu_count())
            info["memory_total_gb"] = str(
                round(psutil.virtual_memory().total / (1024**3), 1)
            )
        return info

    # ── Comparison & Best Run ────────────────────────────────────

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
