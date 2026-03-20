"""Tests for model_config_loader — YAML/JSON config loading."""

import json
from pathlib import Path

import pytest

from src.infrastructure.http.model_config_loader import (
    _dict_to_model_config,
    load_all_model_configs,
    load_features_from_metrics,
    load_model_config,
)


class TestDictToModelConfig:
    def test_full_config(self) -> None:
        data = {
            "model_id": "credit-risk",
            "version": "v2",
            "framework": "onnx",
            "model_path": "models/credit_risk/v2/model.onnx",
            "task_type": "classification",
            "feature_names": ["f1", "f2"],
            "metadata": {"author": "team"},
            "dataset_name": "german-credit",
            "train_script": "examples/credit_risk/train.py",
        }
        config = _dict_to_model_config(data)
        assert config.model_id == "credit-risk"
        assert config.version == "v2"
        assert config.framework == "onnx"
        assert config.task_type == "classification"
        assert config.feature_names == ("f1", "f2")
        assert config.dataset_name == "german-credit"

    def test_defaults(self) -> None:
        config = _dict_to_model_config({})
        assert config.model_id == ""
        assert config.version == "v1"
        assert config.framework == "onnx"
        assert config.task_type == "classification"

    def test_metadata_non_dict(self) -> None:
        config = _dict_to_model_config({"metadata": "invalid"})
        assert config.metadata == ()


class TestLoadModelConfig:
    def test_load_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "test.yaml"
        path.write_text("model_id: my-model\nversion: v1\nframework: onnx\n")
        config = load_model_config(path)
        assert config.model_id == "my-model"

    def test_load_json(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        path.write_text(json.dumps({"model_id": "json-model", "version": "v3"}))
        config = load_model_config(path)
        assert config.model_id == "json-model"
        assert config.version == "v3"

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_config(tmp_path / "nonexistent.yaml")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        path = tmp_path / "config.txt"
        path.write_text("model_id: test")
        with pytest.raises(ValueError, match="Unsupported"):
            load_model_config(path)


class TestLoadAllModelConfigs:
    def test_loads_multiple(self, tmp_path: Path) -> None:
        (tmp_path / "a.yaml").write_text("model_id: model-a\nversion: v1\n")
        (tmp_path / "b.json").write_text(json.dumps({"model_id": "model-b"}))
        configs = load_all_model_configs(tmp_path)
        assert "model-a" in configs
        assert "model-b" in configs

    def test_skips_empty_model_id(self, tmp_path: Path) -> None:
        (tmp_path / "empty.yaml").write_text("version: v1\n")
        configs = load_all_model_configs(tmp_path)
        assert len(configs) == 0

    def test_missing_dir(self, tmp_path: Path) -> None:
        configs = load_all_model_configs(tmp_path / "nope")
        assert configs == {}


class TestLoadFeaturesFromMetrics:
    def test_loads_features(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"all_features": ["f1", "f2", "f3"]}))
        features = load_features_from_metrics(path)
        assert features == ["f1", "f2", "f3"]

    def test_missing_file(self, tmp_path: Path) -> None:
        features = load_features_from_metrics(tmp_path / "nope.json")
        assert features == []

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json")
        features = load_features_from_metrics(path)
        assert features == []

    def test_no_all_features_key(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps({"accuracy": 0.95}))
        features = load_features_from_metrics(path)
        assert features == []
