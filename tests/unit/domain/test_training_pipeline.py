"""Tests for TrainingPipeline — multi-step training orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from phoenix_ml.domain.training.pipeline import (
    IPipelineStep,
    PipelineContext,
    RegisterStep,
    TrainingPipeline,
    ValidateStep,
)

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def context() -> PipelineContext:
    return PipelineContext(
        model_id="test-model",
        version="v1",
        model_path=Path("/tmp/models/test/v1/model.onnx"),
        data_path="data/test/dataset.csv",
        train_script="my_training.train",
    )


# ── Custom step for testing ───────────────────────────────────────


class DummyStep(IPipelineStep):
    """Test step that records execution."""

    def __init__(self, name: str = "dummy", fail: bool = False) -> None:
        super().__init__(name=name)
        self.executed = False
        self._fail = fail

    async def execute(self, context: PipelineContext) -> PipelineContext:
        self.executed = True
        if self._fail:
            context.error = f"Step {self.name} failed"
            context.should_deploy = False
        context.metrics[self.name] = 1.0
        return context


# ── Tests ─────────────────────────────────────────────────────────


class TestPipelineContext:
    def test_to_dict(self, context: PipelineContext) -> None:
        d = context.to_dict()
        assert d["model_id"] == "test-model"
        assert d["should_deploy"] is True
        assert d["error"] is None

    def test_mutable_state(self, context: PipelineContext) -> None:
        context.metrics["accuracy"] = 0.95
        context.artifacts["model"] = "/path/to/model.onnx"
        assert context.metrics["accuracy"] == 0.95
        assert context.artifacts["model"] == "/path/to/model.onnx"


class TestIPipelineStep:
    @pytest.mark.asyncio
    async def test_custom_step_executes(self, context: PipelineContext) -> None:
        step = DummyStep(name="my-step")
        result = await step.execute(context)
        assert step.executed
        assert result.metrics["my-step"] == 1.0

    @pytest.mark.asyncio
    async def test_custom_step_failure(self, context: PipelineContext) -> None:
        step = DummyStep(name="failing", fail=True)
        result = await step.execute(context)
        assert result.should_deploy is False
        assert result.error is not None and "failing" in result.error


class TestTrainingPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_executes_all_steps(self, context: PipelineContext) -> None:
        step1 = DummyStep(name="step1")
        step2 = DummyStep(name="step2")
        step3 = DummyStep(name="step3")

        pipeline = TrainingPipeline([step1, step2, step3])
        result = await pipeline.run(context)

        assert step1.executed
        assert step2.executed
        assert step3.executed
        assert result.error is None

    @pytest.mark.asyncio
    async def test_pipeline_skips_after_error(self, context: PipelineContext) -> None:
        step1 = DummyStep(name="fail-step", fail=True)
        step2 = DummyStep(name="should-skip")

        pipeline = TrainingPipeline([step1, step2])
        result = await pipeline.run(context)

        assert step1.executed
        assert not step2.executed
        assert result.should_deploy is False

    @pytest.mark.asyncio
    async def test_add_step_chaining(self, context: PipelineContext) -> None:
        pipeline = TrainingPipeline()
        pipeline.add_step(DummyStep("a")).add_step(DummyStep("b"))
        assert len(pipeline.steps) == 2

    def test_default_pipeline_has_3_steps(self) -> None:
        pipeline = TrainingPipeline.default()
        names = [s.name for s in pipeline.steps]
        assert names == ["train", "validate", "register"]

    def test_from_config(self) -> None:
        config: list[dict[str, Any]] = [
            {"step": "train"},
            {"step": "quantize", "config": {"method": "dynamic"}},
            {"step": "validate", "config": {"min_accuracy": 0.9}},
            {"step": "register", "config": {"strategy": "canary"}},
        ]
        pipeline = TrainingPipeline.from_config(config)
        names = [s.name for s in pipeline.steps]
        assert names == ["train", "quantize", "validate", "register"]

    def test_from_config_skips_unknown(self) -> None:
        config: list[dict[str, Any]] = [{"step": "unknown_step"}]
        pipeline = TrainingPipeline.from_config(config)
        assert len(pipeline.steps) == 0

    def test_repr(self) -> None:
        pipeline = TrainingPipeline.default()
        assert "train" in repr(pipeline)
        assert "validate" in repr(pipeline)


class TestValidateStep:
    @pytest.mark.asyncio
    async def test_validate_passes_threshold(
        self, context: PipelineContext, tmp_path: Path,
    ) -> None:
        import json
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"accuracy": 0.97, "f1_score": 0.95}))
        context.artifacts["metrics"] = str(metrics_path)

        step = ValidateStep(config={"min_accuracy": 0.95})
        result = await step.execute(context)
        assert result.should_deploy is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_validate_fails_threshold(
        self, context: PipelineContext, tmp_path: Path,
    ) -> None:
        import json
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"accuracy": 0.80}))
        context.artifacts["metrics"] = str(metrics_path)

        step = ValidateStep(config={"min_accuracy": 0.95})
        result = await step.execute(context)
        assert result.should_deploy is False
        assert result.error is not None and "accuracy=0.8" in result.error

    @pytest.mark.asyncio
    async def test_validate_no_metrics_file(self, context: PipelineContext) -> None:
        step = ValidateStep(config={"min_accuracy": 0.95})
        result = await step.execute(context)
        assert result.should_deploy is True  # Skip, not fail


class TestRegisterStep:
    @pytest.mark.asyncio
    async def test_register_records_strategy(self, context: PipelineContext) -> None:
        step = RegisterStep(config={"strategy": "canary"})
        result = await step.execute(context)
        assert result.artifacts["deploy_strategy"] == "canary"

    @pytest.mark.asyncio
    async def test_register_skips_when_not_deployable(self, context: PipelineContext) -> None:
        context.should_deploy = False
        step = RegisterStep()
        result = await step.execute(context)
        assert "deploy_strategy" not in result.artifacts
