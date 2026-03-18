"""
Throughput Benchmark for Phoenix ML Platform.

Measures requests-per-second under concurrent load.

Usage:
    python -m benchmarks.throughput_benchmark [--url URL] [--duration D]
"""

import argparse
import asyncio
import time
from typing import Any

import httpx


async def _worker(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    results: list[bool],
    stop_event: asyncio.Event,
) -> None:
    """Individual worker that sends requests in a loop until stopped."""
    while not stop_event.is_set():
        try:
            resp = await client.post("/predict", json=payload)
            results.append(resp.status_code == 200)  # noqa: PLR2004
        except httpx.HTTPError:
            results.append(False)


async def run_benchmark(base_url: str, duration_s: int, concurrency: int) -> dict[str, Any]:
    """Run throughput benchmark with concurrent workers."""
    payload = {
        "model_id": "credit-risk",
        "model_version": "v1",
        "entity_id": "customer-benchmark",
    }

    results: list[bool] = []
    stop_event = asyncio.Event()

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Warm-up
        print("⏳ Warming up...")
        for _ in range(5):
            await client.post("/predict", json=payload)

        # Spawn workers
        print(f"🚀 Running {concurrency} concurrent workers for {duration_s}s...")
        start = time.perf_counter()
        workers = [
            asyncio.create_task(_worker(client, payload, results, stop_event))
            for _ in range(concurrency)
        ]

        await asyncio.sleep(duration_s)
        stop_event.set()

        await asyncio.gather(*workers, return_exceptions=True)
        elapsed = time.perf_counter() - start

    total = len(results)
    successes = sum(results)
    failures = total - successes
    rps = total / elapsed if elapsed > 0 else 0

    output = {
        "duration_s": round(elapsed, 2),
        "concurrency": concurrency,
        "total_requests": total,
        "successes": successes,
        "failures": failures,
        "rps": round(rps, 2),
        "success_rate": round(successes / total * 100, 2) if total else 0,
    }

    print("\n" + "=" * 50)
    print("  🚀 THROUGHPUT BENCHMARK RESULTS")
    print("=" * 50)
    for key, value in output.items():
        print(f"  {key:>20}: {value}")
    print("=" * 50)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Phoenix ML Throughput Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--duration", type=int, default=10, help="Test duration in seconds")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.duration, args.concurrency))


if __name__ == "__main__":
    main()
