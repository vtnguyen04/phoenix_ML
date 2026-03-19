# Phoenix ML — Benchmark Results

*Generated: 2026-03-19 13:25 (ONNX Runtime, CPU, 30-feature credit risk model)*

## Inference Latency (ms)

| Concurrency | P50   | P95   | P99   | Mean  | Min   | Max   |
|-------------|-------|-------|-------|-------|-------|-------|
| 1           | 0.116 | 0.133 | 0.528 | 0.116 | 0.082 | 0.926 |
| 10          | 0.090 | 0.134 | 0.163 | 0.099 | 0.083 | 0.224 |
| 50          | 0.090 | 0.118 | 0.131 | 0.095 | 0.087 | 0.143 |

**Key takeaway:** Sub-millisecond P99 at all concurrency levels. Cache warming improves latency at higher concurrency.

## Throughput

| Metric | Value |
|--------|-------|
| Total Requests | 47,514 |
| Duration | 5.00s |
| **RPS** | **9,502.8** |
| Errors | 0 |

**Key takeaway:** ~9,500 predictions/second on a single CPU thread with ONNX Runtime.

## Memory Profile

| Phase | Memory |
|-------|--------|
| Baseline | 0 B |
| Post Model Load | 88.9 KB |
| Peak (100 predictions) | 987.9 KB |
| Delta (model load) | 88.9 KB |
| Delta (predictions) | 899.0 KB |

**Key takeaway:** Lightweight footprint — under 1 MB peak for 100 sequential predictions. Suitable for containerized deployments with tight memory limits.

---

## How to Reproduce

```bash
# Memory benchmark (standalone, no server needed)
PYTHONPATH=. uv run python benchmarks/memory_benchmark.py

# Latency + throughput (requires running server)
uv run python benchmarks/benchmark_report.py --host localhost --port 8000

# Load testing with Locust
uv run locust -f benchmarks/locustfile.py --host http://localhost:8000
```
