"""
Tests for model_generator — generates ONNX models for CI/testing purposes.
"""

from pathlib import Path

import onnx

from phoenix_ml.shared.utils.model_generator import generate_simple_onnx


class TestModelGenerator:
    """Unit tests for ONNX model generation utility."""

    def test_generates_valid_onnx(self, tmp_path: Path) -> None:
        output = tmp_path / "model.onnx"
        generate_simple_onnx(output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_onnx_is_valid_model(self, tmp_path: Path) -> None:
        output = tmp_path / "model.onnx"
        generate_simple_onnx(output)
        model = onnx.load(str(output))
        onnx.checker.check_model(model)

    def test_onnx_has_correct_input_shape(self, tmp_path: Path) -> None:
        output = tmp_path / "model.onnx"
        generate_simple_onnx(output)
        model = onnx.load(str(output))
        inputs = model.graph.input
        assert len(inputs) == 1
        shape = inputs[0].type.tensor_type.shape
        # Second dim should be 4 (default n_features)
        assert shape.dim[1].dim_value == 4

    def test_onnx_has_correct_output_shape(self, tmp_path: Path) -> None:
        output = tmp_path / "model.onnx"
        generate_simple_onnx(output)
        model = onnx.load(str(output))
        outputs = model.graph.output
        assert len(outputs) == 1
        shape = outputs[0].type.tensor_type.shape
        assert shape.dim[1].dim_value == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "dirs" / "model.onnx"
        generate_simple_onnx(output)
        assert output.exists()

    def test_custom_n_features(self, tmp_path: Path) -> None:
        output = tmp_path / "model.onnx"
        generate_simple_onnx(output, n_features=30)
        model = onnx.load(str(output))
        shape = model.graph.input[0].type.tensor_type.shape
        assert shape.dim[1].dim_value == 30
