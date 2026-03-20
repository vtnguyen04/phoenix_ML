"""
Memory Benchmark — Profiles peak memory during inference bursts.

Uses tracemalloc to measure memory allocation during model loading
and concurrent prediction runs.
"""

import asyncio
import tracemalloc
from pathlib import Path

import numpy as np

from src.domain.inference.entities.model import Model
from src.domain.inference.value_objects.feature_vector import FeatureVector
from src.infrastructure.ml_engines.onnx_engine import ONNXInferenceEngine
from src.shared.utils.model_generator import generate_simple_onnx


def _format_bytes(size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(size) < 1024:  # noqa: PLR2004
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} TB"


async def run_memory_benchmark(n_predictions: int = 100) -> dict[str, str]:
    """
    Profile memory usage during inference.

    Returns dict with baseline, post-load, peak, and delta metrics.
    """
    # Setup: ensure model exists
    cache_dir = Path("/tmp/bench_model_cache")
    model_path = cache_dir / "bench" / "v1" / "model.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        generate_simple_onnx(model_path)

    engine = ONNXInferenceEngine(cache_dir=cache_dir)
    model = Model(
        id="bench",
        version="v1",
        uri="local:///tmp/bench_model_cache",
        framework="onnx",
    )

    # Start profiling
    tracemalloc.start()

    tracemalloc.take_snapshot()  # baseline snapshot
    baseline_size, _ = tracemalloc.get_traced_memory()

    # Load model
    await engine.load(model)
    post_load_size, _ = tracemalloc.get_traced_memory()

    # Run predictions
    for _ in range(n_predictions):
        fv = FeatureVector(
            values=np.random.default_rng(42).normal(size=30).astype(np.float32).tolist()
        )
        await engine.predict(model, fv)

    _, peak_size = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results = {
        "baseline": _format_bytes(baseline_size),
        "post_model_load": _format_bytes(post_load_size),
        "peak": _format_bytes(peak_size),
        "delta_load": _format_bytes(post_load_size - baseline_size),
        "delta_predict": _format_bytes(peak_size - post_load_size),
        "n_predictions": str(n_predictions),
    }

    print("=== Memory Benchmark ===")  # noqa: T201
    for k, v in results.items():
        print(f"  {k}: {v}")  # noqa: T201

    return results


if __name__ == "__main__":
    asyncio.run(run_memory_benchmark())
