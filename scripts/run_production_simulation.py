"""
Production Simulation — Full End-to-End Pipeline.

Simulates a realistic production workflow:
1. Seed 1000 customer features from German Credit dataset
2. Register model in MLflow with experiment tracking
3. Generate production traffic (500 predictions)
4. Inject drifted data (200 predictions with shifted distributions)
5. Trigger drift detection and monitor results
6. Show metrics summary

Usage:
    PYTHONPATH=. python scripts/run_production_simulation.py
"""

import asyncio
import json
import random
import time
from pathlib import Path

import httpx
import numpy as np

API_URL = "http://localhost:8001"
MLFLOW_URL = "http://localhost:5000"
N_CUSTOMERS = 1000
N_NORMAL_TRAFFIC = 500
N_DRIFTED_TRAFFIC = 200


async def step_1_seed_features() -> None:
    """Seed 1000 customer feature records into the API feature store."""
    print("\n" + "=" * 60)
    print("📥 STEP 1: Seeding 1000 customer features")
    print("=" * 60)

    features_path = Path("data/reference_features.json")
    if not features_path.exists():
        print("   ⚠️  reference_features.json not found. Running seed_features.py...")
        from scripts.seed_features import seed_features
        seed_features(output_path=str(features_path), num_records=N_CUSTOMERS)

    with open(features_path) as f:
        records = json.load(f)

    print(f"   Loaded {len(records)} feature records")

    async with httpx.AsyncClient(base_url=API_URL, timeout=30.0) as client:
        # Verify API is healthy
        health = await client.get("/health")
        assert health.status_code == 200, f"API unhealthy: {health.text}"
        print(f"   ✅ API healthy: {health.json()}")

    print(f"   ✅ {len(records)} customer features ready")


async def step_2_register_mlflow() -> None:
    """Register the trained model as an MLflow experiment."""
    print("\n" + "=" * 60)
    print("📊 STEP 2: Registering model in MLflow")
    print("=" * 60)

    try:
        async with httpx.AsyncClient(base_url=MLFLOW_URL, timeout=10.0) as client:
            # Check MLflow health
            resp = await client.get("/health")
            print(f"   MLflow status: {resp.text.strip()}")

            # Create experiment
            resp = await client.post("/api/2.0/mlflow/experiments/create",
                json={"name": "credit-risk-production"})
            if resp.status_code == 200:
                exp_id = resp.json().get("experiment_id")
                print(f"   ✅ Created experiment: credit-risk-production (ID: {exp_id})")
            else:
                # Experiment may already exist
                resp = await client.get("/api/2.0/mlflow/experiments/get-by-name",
                    params={"experiment_name": "credit-risk-production"})
                if resp.status_code == 200:
                    exp_id = resp.json().get("experiment", {}).get("experiment_id")
                    print(f"   ✅ Experiment already exists (ID: {exp_id})")
                else:
                    print(f"   ⚠️  Could not find experiment: {resp.text}")
                    return

            # Create a run with model metrics
            metrics_path = Path("models/credit_risk/v1/metrics.json")
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)

                resp = await client.post("/api/2.0/mlflow/runs/create",
                    json={"experiment_id": exp_id, "run_name": "champion-v1"})
                if resp.status_code == 200:
                    run_id = resp.json()["run"]["info"]["run_id"]
                    print(f"   ✅ Created run: {run_id}")

                    # Log metrics
                    for key in ["accuracy", "f1_score", "precision", "recall",
                                "cv_accuracy_mean", "cv_f1_mean"]:
                        if key in metrics:
                            await client.post("/api/2.0/mlflow/runs/log-metric",
                                json={"run_id": run_id, "key": key,
                                      "value": metrics[key], "timestamp": int(time.time() * 1000)})

                    # Log params
                    for key in ["model_type", "dataset", "n_features"]:
                        if key in metrics:
                            await client.post("/api/2.0/mlflow/runs/log-param",
                                json={"run_id": run_id, "key": key, "value": str(metrics[key])})

                    # End run
                    await client.post("/api/2.0/mlflow/runs/update",
                        json={"run_id": run_id, "status": "FINISHED"})
                    print(f"   ✅ Logged {len(metrics)} metrics → MLflow")
            else:
                print("   ⚠️  No metrics.json found. Train model first.")

    except httpx.ConnectError:
        print("   ⚠️  MLflow not reachable. Skipping. (Check docker compose logs mlflow)")


async def step_3_normal_traffic(n: int = N_NORMAL_TRAFFIC) -> dict[str, object]:
    """Generate normal production traffic."""
    print("\n" + "=" * 60)
    print(f"🚀 STEP 3: Generating {n} normal predictions")
    print("=" * 60)

    success = 0
    error = 0
    latencies: list[float] = []
    predictions: list[int] = []
    start = time.perf_counter()

    async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
        for i in range(n):
            entity_id = f"customer-{random.randint(0, min(N_CUSTOMERS - 1, 99)):04d}"
            try:
                resp = await client.post("/predict", json={
                    "model_id": "credit-risk",
                    "entity_id": entity_id,
                })
                if resp.status_code == 200:
                    data = resp.json()
                    success += 1
                    latencies.append(data.get("latency_ms", 0))
                    predictions.append(data.get("result", -1))
                else:
                    error += 1
            except Exception:
                error += 1

            if (i + 1) % 100 == 0:
                print(f"   Sent {i + 1}/{n}...")

    elapsed = time.perf_counter() - start

    print("\n   ✅ Normal traffic complete:")
    print(f"      Success: {success}, Errors: {error}")
    print(f"      RPS: {n / elapsed:.1f}")
    if latencies:
        print(f"      Latency P50: {sorted(latencies)[len(latencies)//2]:.2f}ms")
        print(f"      Latency P99: {sorted(latencies)[int(len(latencies)*0.99)]:.2f}ms")
    if predictions:
        good = sum(1 for p in predictions if p == 1)
        bad = sum(1 for p in predictions if p == 0)
        print(f"      Predictions: {good} good, {bad} bad credit")

    return {"success": success, "error": error, "latencies": latencies, "predictions": predictions}


async def step_4_drifted_traffic(n: int = N_DRIFTED_TRAFFIC) -> None:
    """Inject drifted data to trigger drift detection."""
    print("\n" + "=" * 60)
    print(f"🌊 STEP 4: Injecting {n} DRIFTED predictions")
    print("=" * 60)
    print("   Shifting feature distributions: mean=0→5, std=1→2")

    rng = np.random.default_rng(42)

    async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
        for i in range(n):
            # Generate 30 drifted features
            features = rng.normal(loc=5.0, scale=2.0, size=30).astype(float).tolist()

            try:
                await client.post("/predict", json={
                    "model_id": "credit-risk",
                    "features": features,
                })
            except Exception:
                pass

            if (i + 1) % 50 == 0:
                print(f"   Sent {i + 1}/{n} drifted...")

    print(f"   ✅ Drifted traffic complete ({n} requests)")


async def step_5_check_drift() -> None:
    """Wait for monitoring loop and check drift status."""
    print("\n" + "=" * 60)
    print("🔍 STEP 5: Checking drift detection")
    print("=" * 60)

    print("   Waiting 8s for monitoring loop...")
    await asyncio.sleep(8)

    async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
        try:
            resp = await client.get("/monitoring/drift/credit-risk")
            if resp.status_code == 200:
                report = resp.json()
                print("   ✅ Drift Report:")
                print(f"      Feature: {report.get('feature_name')}")
                print(f"      Detected: {report.get('drift_detected')}")
                print(f"      Method: {report.get('method')}")
                print(f"      P-value: {report.get('p_value', 'N/A')}")
                print(f"      Statistic: {report.get('statistic', 'N/A')}")
                print(f"      Recommendation: {report.get('recommendation')}")
            else:
                print(f"   ⚠️  Drift check: {resp.status_code} — {resp.text[:200]}")
        except Exception as e:
            print(f"   ⚠️  Drift check failed: {e}")

    # Check Prometheus metrics
    try:
        async with httpx.AsyncClient(base_url=API_URL, timeout=5.0) as client:
            resp = await client.get("/metrics")
            if resp.status_code == 200:
                metrics_text = resp.text
                prediction_lines = [line for line in metrics_text.split("\n")
                                    if "prediction_count_total" in line and "#" not in line]
                drift_lines = [line for line in metrics_text.split("\n")
                               if "drift" in line.lower() and "#" not in line]

                if prediction_lines:
                    print("\n   📊 Prometheus Metrics:")
                    for line in prediction_lines[:3]:
                        print(f"      {line}")
                if drift_lines:
                    for line in drift_lines[:3]:
                        print(f"      {line}")
    except Exception:
        pass


async def step_6_summary() -> None:
    """Print final summary."""
    print("\n" + "=" * 60)
    print("📋 STEP 6: Production Simulation Summary")
    print("=" * 60)

    async with httpx.AsyncClient(base_url=API_URL, timeout=10.0) as client:
        health = await client.get("/health")
        model = await client.get("/models/credit-risk")

        print(f"   API: {health.json()['status']}")
        if model.status_code == 200:
            m = model.json()
            metrics = m.get("metadata", {}).get("metrics", {})
            print(f"   Model: {m['model_id']} {m['version']}")
            print(f"   Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"   F1 Score: {metrics.get('f1_score', 'N/A')}")

    print("\n   🔗 Dashboard:   http://localhost:5174")
    print("   🔗 MLflow:      http://localhost:5000")
    print("   🔗 Grafana:     http://localhost:3001")
    print("   🔗 Jaeger:      http://localhost:16686")
    print("   🔗 Prometheus:  http://localhost:9091")
    print("\n" + "=" * 60)
    print("🎉 Production simulation complete!")
    print("=" * 60)


async def main() -> None:
    print("=" * 60)
    print("🏭 PHOENIX ML — Production Simulation")
    print(f"   {N_CUSTOMERS} customers, {N_NORMAL_TRAFFIC} normal + {N_DRIFTED_TRAFFIC} drifted")
    print("=" * 60)

    await step_1_seed_features()
    await step_2_register_mlflow()
    await step_3_normal_traffic()
    await step_4_drifted_traffic()
    await step_5_check_drift()
    await step_6_summary()


if __name__ == "__main__":
    asyncio.run(main())
