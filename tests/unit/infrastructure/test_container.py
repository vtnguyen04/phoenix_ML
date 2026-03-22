"""Tests for container utility functions."""

from pathlib import Path
from unittest.mock import patch

import pytest

from phoenix_ml.infrastructure.bootstrap.container import (
    ensure_model_exists,
    find_project_root,
)

_MOD = "phoenix_ml.infrastructure.bootstrap.container"


class TestFindProjectRoot:
    def test_finds_pyproject_toml(self) -> None:
        root = find_project_root()
        assert (root / "pyproject.toml").exists()

    def test_returns_path(self) -> None:
        root = find_project_root()
        assert isinstance(root, Path)


class TestEnsureModelExists:
    def test_returns_path_when_model_exists(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "models" / "test_model" / "v1"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_bytes(b"fake")

        with patch(f"{_MOD}.find_project_root", return_value=tmp_path):
            result = ensure_model_exists("test-model", "v1")
            assert result.exists()

    def test_raises_when_model_missing_and_not_ci(self, tmp_path: Path) -> None:
        with (
            patch(f"{_MOD}.find_project_root", return_value=tmp_path),
            patch("os.getenv", return_value=None),
        ):
            with pytest.raises(FileNotFoundError, match="not found"):
                ensure_model_exists("nonexistent-model", "v1")

    def test_generates_in_ci(self, tmp_path: Path) -> None:
        with (
            patch(f"{_MOD}.find_project_root", return_value=tmp_path),
            patch("os.getenv", return_value="true"),
            patch(f"{_MOD}.generate_simple_onnx") as mock_gen,
        ):
            result = ensure_model_exists("ci-model", "v1")
            mock_gen.assert_called_once()
            assert result is not None
