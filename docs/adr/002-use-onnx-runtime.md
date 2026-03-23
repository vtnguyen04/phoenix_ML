# ADR 002: Standardization on ONNX Runtime

## Status
✅ **Accepted** — March 2026

## Context

ML platform needs to serve models trained with various frameworks: scikit-learn, XGBoost, PyTorch, TensorFlow. Each framework has its own inference API, dependencies, and performance characteristics.

### Problem

- Managing multiple runtime dependencies is complex
- Inconsistent inference API across frameworks
- Hard to optimize performance when each framework has different batching/threading strategies
- Deployment size increases when bundling multiple ML frameworks

## Decision

Standardize on **ONNX Runtime** as the unified inference engine:

### Architecture

```
sklearn model → skl2onnx → model.onnx → ONNXInferenceEngine
XGBoost model → onnxmltools → model.onnx → ONNXInferenceEngine
PyTorch model → torch.onnx.export → model.onnx → ONNXInferenceEngine
TensorFlow → tf2onnx → model.onnx → ONNXInferenceEngine
```

### Implementation

3 engine implementations, all sharing the `InferenceEngine` interface:

| Engine | File | Purpose | Fallback |
|--------|------|----------|----------|
| `ONNXInferenceEngine` | `onnx_engine.py` | Default CPU/GPU inference | — |
| `TensorRTExecutor` | `tensorrt_executor.py` | High-performance GPU (FP16) | CPU via ONNX Runtime |
| `TritonInferenceClient` | `triton_client.py` | NVIDIA Triton Server | Mock predictions |

### Engine Selection

```bash
# Via environment variable
INFERENCE_ENGINE=onnx       # Default
INFERENCE_ENGINE=tensorrt   # GPU optimization
INFERENCE_ENGINE=triton     # Distributed serving
```

Factory pattern trong `container.py`:
```python
_ENGINE_FACTORIES = {
    "onnx": lambda: ONNXInferenceEngine(cache_dir=...),
    "tensorrt": lambda: TensorRTExecutor(cache_dir=...),
    "triton": lambda: TritonInferenceClient(triton_url=...),
}
```

## Consequences

### Positive
- ✅ Single dependency (`onnxruntime`) instead of multiple ML frameworks
- ✅ Consistent interface for all model types
- ✅ Hardware acceleration automatic (CPU, CUDA, TensorRT, DirectML)
- ✅ Smaller model size (ONNX protobuf more compact than pickle)
- ✅ Cross-platform: train on GPU, deploy on CPU/edge
- ✅ ONNX ecosystem: onnxruntime-gpu, onnxruntime-web, onnxruntime-mobile

### Negative
- ❌ Conversion step required (sklearn → ONNX) — but automated in training scripts
- ❌ Some model operations not supported (custom layers) — rare
- ❌ Debug difficulty: ONNX graph more opaque than native framework

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai)
- [sklearn-onnx Converter](https://onnx.ai/sklearn-onnx/)