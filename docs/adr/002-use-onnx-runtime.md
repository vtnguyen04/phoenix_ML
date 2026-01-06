# ADR 002: Standardization on ONNX Runtime

## Status
Accepted

## Context
Deploying ML models directly via heavy training frameworks (PyTorch, TensorFlow) leads to bloated Docker images (>3GB), slow startup times, and suboptimal inference latency. We need a unified runtime that is lightweight, fast, and framework-agnostic.

## Decision
We chose **ONNX (Open Neural Network Exchange)** as the standard model format and **ONNX Runtime** as the execution engine.

## Consequences
### Positive
*   **Performance**: ONNX Runtime offers graph optimizations (operator fusion, constant folding) yielding 2x-10x speedups over native frameworks.
*   **Interoperability**: Supports models trained in PyTorch, TensorFlow, Scikit-Learn, and XGBoost.
*   **Efficiency**: Significantly reduced container size and memory footprint.

### Negative
*   **Pipeline Complexity**: Adds an extra "export/conversion" step in the MLOps pipeline.
*   **Operator Support**: Some cutting-edge or custom layers in PyTorch might not yet be supported by the ONNX standard.