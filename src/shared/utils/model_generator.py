import logging
from pathlib import Path

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)


def generate_simple_onnx(output_path: Path) -> None:
    """Generates a simple ONNX model (Sigmoid(X*W+B)) for testing purposes."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 2])

    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 2], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
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
    logger.info("ONNX model generated at %s", output_path)


if __name__ == "__main__":
    generate_simple_onnx(Path("/tmp/phoenix/remote_storage/demo/v1/model.onnx"))
