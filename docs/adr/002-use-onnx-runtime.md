# ADR 002: Standardization on ONNX Runtime

## Status
✅ **Accepted** — March 2026

## Context

ML platform cần serve models trained bằng nhiều frameworks khác nhau: scikit-learn, XGBoost, PyTorch, TensorFlow. Mỗi framework có API inference riêng, dependencies riêng, performance characteristics riêng.

### Vấn đề

- Quản lý multiple runtime dependencies phức tạp
- Inconsistent inference API giữa frameworks
- Khó optimize performance khi mỗi framework có batching/threading strategy khác nhau
- Deployment size tăng khi bundle nhiều ML frameworks

## Decision

Standardize trên **ONNX Runtime** như unified inference engine:

### Architecture

```
sklearn model → skl2onnx → model.onnx → ONNXInferenceEngine
XGBoost model → onnxmltools → model.onnx → ONNXInferenceEngine
PyTorch model → torch.onnx.export → model.onnx → ONNXInferenceEngine
TensorFlow → tf2onnx → model.onnx → ONNXInferenceEngine
```

### Implementation

3 engine implementations, tất cả share interface `InferenceEngine`:

| Engine | File | Mục đích | Fallback |
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
- ✅ Single dependency (`onnxruntime`) thay vì nhiều ML frameworks
- ✅ Consistent interface cho tất cả model types
- ✅ Hardware acceleration tự động (CPU, CUDA, TensorRT, DirectML)
- ✅ Model size giảm (ONNX protobuf compact hơn pickle)
- ✅ Cross-platform: train trên GPU, deploy trên CPU/edge
- ✅ ONNX ecosystem: onnxruntime-gpu, onnxruntime-web, onnxruntime-mobile

### Negative
- ❌ Conversion step cần thiết (sklearn → ONNX) — nhưng automated trong training scripts
- ❌ Một số model operations không support (custom layers) — rare
- ❌ Debug difficulty: ONNX graph opaque hơn native framework

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai)
- [sklearn-onnx Converter](https://onnx.ai/sklearn-onnx/)