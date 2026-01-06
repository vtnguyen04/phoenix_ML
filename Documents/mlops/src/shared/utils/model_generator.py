from pathlib import Path

import onnx
from onnx import TensorProto, helper


def generate_simple_onnx(output_path: Path) -> None:
    # Create inputs and outputs
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 2])

    # Create weights and bias
    W = helper.make_tensor(
        'W', 
        TensorProto.FLOAT, 
        [4, 2], 
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    B = helper.make_tensor('B', TensorProto.FLOAT, [2], [0.1, 0.2])

    # Create nodes
    # Y = X * W + B
    matmul_node = helper.make_node('MatMul', ['input', 'W'], ['matmul_out'])
    add_node = helper.make_node('Add', ['matmul_out', 'B'], ['output'])

    # Create graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        'simple-model',
        [X],
        [Y],
        [W, B]
    )

    # Create model with explicit Opset 15 and IR version 8 (stable)
    opset = helper.make_operatorsetid('', 15)
    model = helper.make_model(
        graph, 
        producer_name='phoenix-ml', 
        ir_version=8,
        opset_imports=[opset]
    )
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"✅ Simple ONNX model generated at {output_path}")

if __name__ == "__main__":
    generate_simple_onnx(Path("/tmp/phoenix/remote_storage/demo/v1/model.onnx"))