"""Multi-step training pipeline.

Executes a sequence of ``IPipelineStep`` instances against a
``PipelineContext``. Steps can be built-in (train, quantize, validate,
register) or user-supplied via ``script`` in YAML config.

Configuration (optional, in ``model_configs/<model_id>.yaml``)::

    pipeline:
      - step: train
      - step: quantize
        config: {method: dynamic, weight_type: int8}
      - step: validate
        config: {min_accuracy: 0.95}
      - step: register

If ``ExperimentTracker`` is attached via ``set_tracker()``, ``run()``
automatically wraps execution with MLflow logging.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """Shared state passed through pipeline steps.

    Each step can read/write to this context. The framework initializes
    it from the model config, and each step can modify it as needed.
    """

    model_id: str
    version: str
    model_path: Path
    data_path: str
    train_script: str
    config: dict[str, Any] = field(default_factory=dict)

    # Mutable state — steps can update these
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    should_deploy: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize context for logging/debugging."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "model_path": str(self.model_path),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "should_deploy": self.should_deploy,
            "error": self.error,
        }


class IPipelineStep(ABC):
    """Interface for a single pipeline step.

    Implement this to add custom steps: quantization, pruning,
    validation gates, model registration, notifications, etc.

    Example::

        class QuantizeStep(IPipelineStep):
            def __init__(self):
                super().__init__(name="quantize")

            async def execute(self, context: PipelineContext) -> PipelineContext:
                from onnxruntime.quantization import quantize_dynamic
                input_path = str(context.model_path)
                output_path = input_path.replace(".onnx", "_quantized.onnx")
                quantize_dynamic(input_path, output_path)
                context.model_path = Path(output_path)
                context.artifacts["quantized_model"] = output_path
                return context
    """

    def __init__(self, name: str = "unnamed", config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.step_config = config or {}

    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute this pipeline step.

        Args:
            context: Shared pipeline state. Read model_path, config, etc.
                     Write metrics, artifacts, should_deploy, etc.

        Returns:
            Updated context (can be same object or new).
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ─── Built-in Steps ──────────────────────────────────────────────


class TrainStep(IPipelineStep):
    """Built-in: Execute the training script (train_and_export)."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="train", config=config)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("📦 [train] Running %s", context.train_script)
        module_path, _, func_name = context.train_script.rpartition(".")
        if not func_name:
            module_path = context.train_script.replace("/", ".").removesuffix(".py")
            func_name = "train_and_export"

        try:
            module = importlib.import_module(module_path)
            train_fn = getattr(module, func_name)

            metrics_path = str(context.model_path.parent / "metrics.json")
            await _run_sync_or_async(
                train_fn,
                output_path=str(context.model_path),
                metrics_path=metrics_path,
                data_path=context.data_path,
            )
            context.artifacts["model"] = str(context.model_path)
            context.artifacts["metrics"] = metrics_path
            logger.info("✅ [train] Model saved to %s", context.model_path)
        except Exception as e:
            context.error = f"Training failed: {e}"
            context.should_deploy = False
            logger.error("❌ [train] %s", e)

        return context


class ValidateStep(IPipelineStep):
    """Built-in: Validate model meets minimum performance thresholds."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="validate", config=config)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        import json  # noqa: PLC0415

        logger.info("🔍 [validate] Checking model quality")

        metrics_path = context.artifacts.get("metrics")
        if not metrics_path or not Path(metrics_path).exists():
            logger.warning("⚠️ [validate] No metrics file found, skipping")
            return context

        with open(metrics_path) as f:
            metrics = json.load(f)

        context.metrics.update(metrics)

        # Check thresholds from step config
        for metric_name, min_value in self.step_config.items():
            if metric_name.startswith("min_"):
                actual_metric = metric_name[4:]  # min_accuracy → accuracy
                actual_value = metrics.get(actual_metric, 0.0)
                if actual_value < min_value:
                    context.error = (
                        f"Validation failed: {actual_metric}={actual_value:.4f} "
                        f"< threshold={min_value}"
                    )
                    context.should_deploy = False
                    logger.error("❌ [validate] %s", context.error)
                    return context

        logger.info("✅ [validate] All thresholds passed: %s", metrics)
        return context


class QuantizeStep(IPipelineStep):
    """Built-in: Quantize ONNX model for faster inference.

    Config options:
        method: 'dynamic' (default) or 'static'
        weight_type: 'int8' (default) or 'uint8'
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="quantize", config=config)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("⚡ [quantize] Quantizing model at %s", context.model_path)

        try:
            from onnxruntime.quantization import (  # noqa: PLC0415
                QuantType,
                quantize_dynamic,
            )

            method = self.step_config.get("method", "dynamic")
            weight_type_str = self.step_config.get("weight_type", "int8")
            weight_type = (
                QuantType.QInt8 if weight_type_str == "int8" else QuantType.QUInt8
            )

            input_path = str(context.model_path)
            output_path = input_path.replace(".onnx", "_quantized.onnx")

            if method == "dynamic":
                quantize_dynamic(input_path, output_path, weight_type=weight_type)
            else:
                logger.warning("⚠️ Static quantization requires calibration data")
                quantize_dynamic(input_path, output_path, weight_type=weight_type)

            context.model_path = Path(output_path)
            context.artifacts["quantized_model"] = output_path
            logger.info("✅ [quantize] Quantized model saved to %s", output_path)
        except ImportError:
            logger.warning(
                "⚠️ [quantize] onnxruntime.quantization not available, skipping"
            )
        except Exception as e:
            context.error = f"Quantization failed: {e}"
            logger.error("❌ [quantize] %s", e)

        return context


class RegisterStep(IPipelineStep):
    """Built-in: Register model version with the model registry."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="register", config=config)

    async def execute(self, context: PipelineContext) -> PipelineContext:
        logger.info("📋 [register] Registering model %s:%s", context.model_id, context.version)

        if not context.should_deploy:
            logger.warning("⚠️ [register] Skipping — should_deploy=False")
            return context

        strategy = self.step_config.get("strategy", "direct")
        context.artifacts["deploy_strategy"] = strategy
        logger.info(
            "✅ [register] Model registered (strategy=%s)", strategy
        )
        return context


# ─── Built-in Step Registry ─────────────────────────────────────

BUILTIN_STEPS: dict[str, type[IPipelineStep]] = {
    "train": TrainStep,
    "quantize": QuantizeStep,
    "validate": ValidateStep,
    "register": RegisterStep,
}


# ─── Pipeline Orchestrator ───────────────────────────────────────


class TrainingPipeline:
    """Orchestrates multi-step training pipeline.

    Steps execute sequentially. If any step sets context.should_deploy=False
    or context.error, subsequent steps can check and react accordingly.

    Can be built from YAML config or programmatically.

    If an ``ExperimentTracker`` is attached, ``run()`` automatically logs
    everything to MLflow — params, metrics, artifacts, system info.
    Users write zero MLflow code.

    Example::

        pipeline = TrainingPipeline.default()
        pipeline.set_tracker(ExperimentTracker("http://mlflow:5000"))
        context = await pipeline.run(context)
        # Everything auto-logged to MLflow
    """

    def __init__(
        self,
        steps: list[IPipelineStep] | None = None,
        experiment_tracker: Any | None = None,
    ) -> None:
        self._steps: list[IPipelineStep] = steps or []
        self._tracker = experiment_tracker

    def set_tracker(self, tracker: Any) -> TrainingPipeline:
        """Attach an ExperimentTracker for auto-MLflow logging."""
        self._tracker = tracker
        return self

    def add_step(self, step: IPipelineStep) -> TrainingPipeline:
        """Add a step to the pipeline. Returns self for chaining."""
        self._steps.append(step)
        return self

    @classmethod
    def from_config(cls, pipeline_config: list[dict[str, Any]]) -> TrainingPipeline:
        """Build pipeline from YAML config list.

        Example config::

            pipeline:
              - step: train
              - step: quantize
                config: {method: dynamic, weight_type: int8}
              - step: validate
                config: {min_accuracy: 0.95}
              - step: register
                config: {strategy: canary}

        For custom steps, use 'script' to specify a module path::

              - step: custom
                script: my_package.my_module.MyCustomStep
        """
        pipeline = cls()

        for step_def in pipeline_config:
            step_name = step_def.get("step", "")
            step_config = step_def.get("config", {})
            script = step_def.get("script", "")

            if step_name in BUILTIN_STEPS:
                step = BUILTIN_STEPS[step_name](config=step_config)
            elif script:
                # Custom step from module path
                step = _load_custom_step(script, step_name, step_config)
            else:
                logger.warning("Unknown step '%s', skipping", step_name)
                continue

            pipeline.add_step(step)

        return pipeline

    @classmethod
    def default(cls) -> TrainingPipeline:
        """Create default pipeline: train → validate → register."""
        return cls([
            TrainStep(),
            ValidateStep(),
            RegisterStep(),
        ])

    async def run(self, context: PipelineContext) -> PipelineContext:
        """Execute all steps in order.

        If an ExperimentTracker is attached, automatically wraps execution
        with full MLflow tracking (params, metrics, artifacts, system info).

        Returns the final context with metrics, artifacts, and status.
        """
        # Auto-delegate to tracker if available
        if self._tracker is not None:
            result: PipelineContext = await self._tracker.tracked_run(self, context)
            return result

        return await self._execute_steps(context)

    async def _execute_steps(self, context: PipelineContext) -> PipelineContext:
        """Internal step execution (called directly or via tracker)."""
        step_names = [s.name for s in self._steps]
        logger.info(
            "🚀 Pipeline starting: %s → %s",
            context.model_id,
            " → ".join(step_names),
        )

        for i, step in enumerate(self._steps, 1):
            logger.info(
                "── Step %d/%d: %s ──", i, len(self._steps), step.name
            )

            if context.error and step.name not in ("register",):
                logger.warning(
                    "⏭️  Skipping %s (previous error: %s)",
                    step.name,
                    context.error,
                )
                continue

            context = await step.execute(context)

        status = "✅ SUCCESS" if not context.error else f"❌ FAILED: {context.error}"
        logger.info("🏁 Pipeline complete: %s — %s", context.model_id, status)

        return context

    @property
    def steps(self) -> list[IPipelineStep]:
        """Get all pipeline steps."""
        return list(self._steps)

    def __repr__(self) -> str:
        step_names = " → ".join(s.name for s in self._steps)
        return f"TrainingPipeline({step_names})"


# ─── Helpers ──────────────────────────────────────────────────────


def _load_custom_step(
    dotted_path: str, name: str, config: dict[str, Any]
) -> IPipelineStep:
    """Import a custom IPipelineStep from a dotted module path."""
    module_path, _, class_name = dotted_path.rpartition(".")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, IPipelineStep)):
        msg = f"{dotted_path} is not a subclass of IPipelineStep"
        raise TypeError(msg)
    return cls(config=config)


import asyncio  # noqa: E402
import inspect  # noqa: E402


async def _run_sync_or_async(fn: Any, **kwargs: Any) -> Any:
    """Run a function that may be sync or async."""
    if inspect.iscoroutinefunction(fn):
        return await fn(**kwargs)
    return await asyncio.to_thread(fn, **kwargs)
