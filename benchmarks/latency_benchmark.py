"""
Latency Benchmark for Phoenix ML Platform.

Measures p50, p95, p99 latency under varying concurrency levels.

Usage:
    python -m benchmarks.latency_benchmark [--url URL] [--requests N]
"""

import argparse
import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np


async def _timed_request(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> float | None:
    try:
        start = time.perf_counter()
        resp = await client.post("/predict", json=payload)
        elapsed = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:  # noqa: PLR2004
            return elapsed
    except httpx.HTTPError:
        pass
    return None


async def run_latency_test(
    base_url: str,
    n_requests: int,
    concurrency: int,
) -> dict[str, Any]:
    payload = {
        "model_id": "credit-risk",
        "model_version": "v1",
        "entity_id": "customer-benchmark",
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for _ in range(3):
            await client.post("/predict", json=payload)

        sem = asyncio.Semaphore(concurrency)

        async def bounded_request() -> float | None:
            async with sem:
                return await _timed_request(client, payload)

        print(f"🚀 Sending {n_requests} requests (concurrency={concurrency})...")
        start = time.perf_counter()
        tasks = [bounded_request() for _ in range(n_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    latencies = [r for r in results if r is not None]
    errors = sum(1 for r in results if r is None)

    if not latencies:
        print("❌ All requests failed")
        return {"error": "All requests failed"}

    arr = np.array(latencies)
    output = {
        "total_requests": n_requests,
        "successful": len(latencies),
        "errors": errors,
        "concurrency": concurrency,
        "total_time_s": round(total_time, 2),
        "rps": round(len(latencies) / total_time, 2),
        "latency_ms": {
            "min": round(float(np.min(arr)), 2),
            "p50": round(float(np.percentile(arr, 50)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "p99": round(float(np.percentile(arr, 99)), 2),
            "max": round(float(np.max(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
        },
    }

    print("\n" + "=" * 50)
    print("  📊 LATENCY BENCHMARK RESULTS")
    print("=" * 50)
    print(f"  {'requests':>20}: {output['total_requests']}")
    print(f"  {'successful':>20}: {output['successful']}")
    print(f"  {'errors':>20}: {output['errors']}")
    print(f"  {'rps':>20}: {output['rps']}")
    print(f"  {'p50 (ms)':>20}: {output['latency_ms']['p50']}")
    print(f"  {'p95 (ms)':>20}: {output['latency_ms']['p95']}")
    print(f"  {'p99 (ms)':>20}: {output['latency_ms']['p99']}")
    print(f"  {'max (ms)':>20}: {output['latency_ms']['max']}")
    print("=" * 50)

    with open("benchmarks/latency_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("📁 Results saved to benchmarks/latency_results.json")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix ML Latency Benchmark")
    parser.add_argument("--url", default="http://localhost:8001", help="Base URL")
    parser.add_argument("--requests", type=int, default=500, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_latency_test(args.url, args.requests, args.concurrency))


if __name__ == "__main__":
    main()
