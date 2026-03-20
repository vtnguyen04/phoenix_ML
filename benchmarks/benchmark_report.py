"""
Consolidated Benchmark Report Generator.

Runs latency and throughput benchmarks, then generates a markdown report
with tables and summary statistics.
"""

import asyncio
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

REPORT_PATH = Path(__file__).parent / "RESULTS.md"


async def _measure_latency(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    n_requests: int = 100,
) -> list[float]:
    """Send n_requests sequentially and collect latencies in ms."""
    latencies: list[float] = []
    for _ in range(n_requests):
        start = time.perf_counter()
        resp = await client.post(url, json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:  # noqa: PLR2004
            latencies.append(elapsed_ms)
    return latencies


async def _measure_throughput(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    duration_seconds: float = 10.0,
    concurrency: int = 10,
) -> dict[str, float]:
    """Measure sustained RPS over a fixed duration."""
    total_requests = 0
    errors = 0
    end_time = time.perf_counter() + duration_seconds

    async def _worker() -> tuple[int, int]:
        nonlocal total_requests, errors
        count = 0
        errs = 0
        while time.perf_counter() < end_time:
            try:
                resp = await client.post(url, json=payload)
                count += 1
                if resp.status_code != 200:  # noqa: PLR2004
                    errs += 1
            except httpx.HTTPError:
                errs += 1
        return count, errs

    tasks = [asyncio.create_task(_worker()) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    total_requests = sum(r[0] for r in results)
    errors = sum(r[1] for r in results)

    return {
        "total_requests": total_requests,
        "errors": errors,
        "duration_seconds": duration_seconds,
        "rps": total_requests / duration_seconds,
    }


def _generate_report(
    latencies: dict[int, list[float]],
    throughput: dict[str, float],
) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# Phoenix ML — Benchmark Results",
        "",
        f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Latency (ms)",
        "",
        "| Concurrency | P50 | P95 | P99 | Mean | Min | Max |",
        "|-------------|-----|-----|-----|------|-----|-----|",
    ]

    for conc, lats in sorted(latencies.items()):
        if not lats:
            continue
        lines.append(
            f"| {conc} | {statistics.median(lats):.2f} | "
            f"{statistics.quantiles(lats, n=20)[-1]:.2f} | "
            f"{sorted(lats)[int(len(lats) * 0.99)]:.2f} | "
            f"{statistics.mean(lats):.2f} | "
            f"{min(lats):.2f} | {max(lats):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Throughput",
            "",
            f"- **Total Requests**: {throughput.get('total_requests', 0)}",
            f"- **Errors**: {throughput.get('errors', 0)}",
            f"- **Duration**: {throughput.get('duration_seconds', 0):.1f}s",
            f"- **RPS**: {throughput.get('rps', 0):.1f}",
            "",
        ]
    )

    return "\n".join(lines)


async def run_benchmark(
    host: str = "localhost",
    port: int = 8000,
) -> None:
    """Run full benchmark suite and write RESULTS.md."""
    base_url = f"http://{host}:{port}"
    payload = {"model_id": "credit-risk", "entity_id": "customer-0001"}

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        # Latency at varying concurrency (sequential per level)
        latency_results: dict[int, list[float]] = {}
        for conc in (1, 5, 10):
            lats = await _measure_latency(client, "/predict", payload, n_requests=50)
            latency_results[conc] = lats

        # Throughput
        throughput = await _measure_throughput(
            client, "/predict", payload, duration_seconds=10.0, concurrency=10
        )

    report = _generate_report(latency_results, throughput)
    REPORT_PATH.write_text(report)
    print(f"Report saved to {REPORT_PATH}")  # noqa: T201
    print(report)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(run_benchmark())
