"""Tests for ModelConfig value object and ModelConfigLoader."""

import json
from pathlib import Path

import pytest

from src.domain.inference.value_objects.model_config import ModelConfig
from src.infrastructure.bootstrap.model_config_loader import (
    _dict_to_model_config,
    load_all_model_configs,
    load_features_from_metrics,
    load_model_config,
)

# ─── ModelConfig Value Object ────────────────────────────────────


class TestModelConfig:
    def test_creation_defaults(self) -> None:
        config = ModelConfig(model_id="my-model")
        assert config.model_id == "my-model"
        assert config.version == "v1"
        assert config.framework == "onnx"
        assert config.task_type == "classification"
        assert config.feature_names == ()
        assert config.has_named_features is False

    def test_creation_with_features(self) -> None:
        config = ModelConfig(
            model_id="credit-risk",
            version="v2",
            framework="tensorrt",
            task_type="classification",
            feature_names=("age", "income", "debt"),
            dataset_name="german-credit",
        )
        assert config.has_named_features is True
        assert len(config.feature_names) == 3
        assert config.dataset_name == "german-credit"

    def test_frozen_immutability(self) -> None:
        config = ModelConfig(model_id="test")
        with pytest.raises(AttributeError):
            config.model_id = "changed"  # type: ignore[misc]

    def test_fs_model_id(self) -> None:
        config = ModelConfig(model_id="credit-risk")
        assert config.fs_model_id == "credit_risk"

        config2 = ModelConfig(model_id="simple")
        assert config2.fs_model_id == "simple"

    def test_with_version(self) -> None:
        config = ModelConfig(model_id="m1", version="v1", framework="onnx")
        new_config = config.with_version("v2")
        assert new_config.version == "v2"
        assert new_config.model_id == "m1"
        assert new_config.framework == "onnx"
        # Original unchanged
        assert config.version == "v1"

    def test_get_metadata(self) -> None:
        config = ModelConfig(
            model_id="m1",
            metadata=(("role", "champion"), ("accuracy", 0.95)),
        )
        meta = config.get_metadata()
        assert meta == {"role": "champion", "accuracy": 0.95}

    def test_supports_various_task_types(self) -> None:
        """Framework supports any ML problem type via task_type."""
        for task in [
            "classification",
            "regression",
            "object_detection",
            "recommendation",
            "timeseries",
            "nlp",
            "embedding",
            "custom",
        ]:
            config = ModelConfig(model_id=f"{task}-model", task_type=task)
            assert config.task_type == task


# ─── ModelConfigLoader ───────────────────────────────────────────


class TestModelConfigLoader:
    def test_dict_to_model_config(self) -> None:
        data = {
            "model_id": "yolo-detect",
            "version": "v3",
            "framework": "pytorch",
            "task_type": "object_detection",
            "feature_names": [],
            "dataset_name": "coco-2017",
            "train_script": "examples/yolo/train.py",
        }
        config = _dict_to_model_config(data)
        assert config.model_id == "yolo-detect"
        assert config.task_type == "object_detection"
        assert config.framework == "pytorch"
        assert config.has_named_features is False

    def test_load_model_config_json(self, tmp_path: Path) -> None:
        config_data = {
            "model_id": "test-model",
            "version": "v1",
            "framework": "onnx",
            "task_type": "classification",
            "feature_names": ["f1", "f2", "f3"],
        }
        config_file = tmp_path / "test.json"
        config_file.write_text(json.dumps(config_data))

        config = load_model_config(config_file)
        assert config.model_id == "test-model"
        assert config.feature_names == ("f1", "f2", "f3")
        assert config.has_named_features is True

    def test_load_model_config_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_config(Path("/nonexistent/config.json"))

    def test_load_model_config_unsupported_format(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "config.xml"
        bad_file.write_text("<xml/>")
        with pytest.raises(ValueError, match="Unsupported config format"):
            load_model_config(bad_file)

    def test_load_all_model_configs_empty_dir(self, tmp_path: Path) -> None:
        configs = load_all_model_configs(tmp_path)
        assert configs == {}

    def test_load_all_model_configs_nonexistent_dir(self) -> None:
        configs = load_all_model_configs(Path("/nonexistent/dir"))
        assert configs == {}

    def test_load_all_model_configs_multiple(self, tmp_path: Path) -> None:
        for name, model_id in [("a.json", "model-a"), ("b.json", "model-b")]:
            (tmp_path / name).write_text(
                json.dumps(
                    {
                        "model_id": model_id,
                        "version": "v1",
                    }
                )
            )

        configs = load_all_model_configs(tmp_path)
        assert "model-a" in configs
        assert "model-b" in configs

    def test_load_features_from_metrics(self, tmp_path: Path) -> None:
        metrics = {
            "accuracy": 0.95,
            "all_features": ["age", "income", "score"],
        }
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps(metrics))

        features = load_features_from_metrics(metrics_file)
        assert features == ["age", "income", "score"]

    def test_load_features_from_metrics_missing(self) -> None:
        features = load_features_from_metrics(Path("/nonexistent.json"))
        assert features == []

    def test_load_features_from_metrics_no_key(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(json.dumps({"accuracy": 0.9}))

        features = load_features_from_metrics(metrics_file)
        assert features == []
