"""
Latency Benchmark for Phoenix ML Platform.

Measures p50/p95/p99 latency for the predict endpoint.

Usage:
    python -m benchmarks.latency_benchmark [--url URL] [--requests N] [--warmup W]
"""

import argparse
import asyncio
import statistics
import time
from typing import Any

import httpx


async def run_benchmark(base_url: str, total_requests: int, warmup: int) -> dict[str, Any]:
    """Run latency benchmark against the predict endpoint."""
    payload: dict[str, Any] = {
        "model_id": "credit-risk",
        "model_version": "v1",
        "entity_id": "customer-benchmark",
    }
    """Run latency benchmark against the predict endpoint."""
    latencies: list[float] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Warm-up phase
        print(f"⏳ Warming up ({warmup} requests)...")
        for _ in range(warmup):
            await client.post("/predict", json=payload)

        # Measurement phase
        print(f"📊 Measuring ({total_requests} requests)...")
        for i in range(total_requests):
            start = time.perf_counter()
            resp = await client.post("/predict", json=payload)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            if resp.status_code != 200:  # noqa: PLR2004
                print(f"  ⚠️  Request {i} failed: {resp.status_code}")

    latencies.sort()
    results = {
        "total_requests": total_requests,
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "p50_ms": round(latencies[int(len(latencies) * 0.50)], 2),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)], 2),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 2),
        "stdev_ms": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0,
    }

    print("\n" + "=" * 50)
    print("  📈 LATENCY BENCHMARK RESULTS")
    print("=" * 50)
    for key, value in results.items():
        print(f"  {key:>20}: {value}")
    print("=" * 50)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix ML Latency Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--requests", type=int, default=100, help="Number of measurement requests")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warm-up requests")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.requests, args.warmup))


if __name__ == "__main__":
    main()
