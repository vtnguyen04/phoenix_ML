import logging
from pathlib import Path

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)


def generate_simple_onnx(output_path: Path, n_features: int = 4) -> None:
    """Generates a simple ONNX model (Sigmoid(X*W+B)) for testing purposes.

    Args:
        output_path: Where to save the generated ONNX model.
        n_features: Number of input features for the model.
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])

    W = helper.make_tensor(
        "W",
        TensorProto.FLOAT,
        [n_features, 2],
        [0.1 * (i + 1) for i in range(n_features * 2)],
    )
    B = helper.make_tensor("B", TensorProto.FLOAT, [2], [0.1, 0.2])

    matmul_node = helper.make_node("MatMul", ["input", "W"], ["matmul_out"])
    add_node = helper.make_node("Add", ["matmul_out", "B"], ["add_out"])
    sigmoid_node = helper.make_node("Sigmoid", ["add_out"], ["output"])

    graph = helper.make_graph(
        [matmul_node, add_node, sigmoid_node], "simple-model", [X], [Y], [W, B]
    )

    opset = helper.make_operatorsetid("", 15)
    model = helper.make_model(
        graph, producer_name="phoenix-ml", ir_version=8, opset_imports=[opset]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    logger.info("ONNX model generated at %s (n_features=%d)", output_path, n_features)


if __name__ == "__main__":
    import sys

    _MIN_ARGS_WITH_FEATURES = 3
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phoenix/test_model.onnx"
    features = int(sys.argv[2]) if len(sys.argv) >= _MIN_ARGS_WITH_FEATURES else 4
    generate_simple_onnx(Path(path), n_features=features)

