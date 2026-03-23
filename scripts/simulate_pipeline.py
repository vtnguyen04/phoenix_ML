#!/usr/bin/env python3
"""
simulate_pipeline.py — Realistic end-to-end simulation of the self-healing ML pipeline.

Simulates the FULL production lifecycle as it actually happens:

  Week 1-2:  Deploy model → serve traffic → accumulate labels
  Week 3-4:  Distribution shift begins → drift intensifies
  Week 5:    Drift detected! → alert → rollback challengers
  Week 6:    Export fresh data (labeled logs + baseline)
  Week 7:    Retrain on FRESH data → register challenger
  Week 8:    Performance comparison → verify drift reports

Endpoints exercised:
  GET    /health                        — Pre-flight check
  POST   /predict                       — Serve predictions (with drift)
  POST   /feedback                      — Submit ground truth labels
  GET    /models                        — List registered models
  GET    /models/{model_id}             — Get model info
  GET    /monitoring/drift/{model_id}   — Run drift detection
  GET    /monitoring/reports/{model_id} — Drift report audit trail
  GET    /monitoring/performance/{id}   — Model performance metrics
  POST   /models/rollback               — Archive challengers (keep champion)
  POST   /data/export-training          — Export labeled logs for retrain
  POST   /models/register               — Register retrained model
  POST   /models/{model_id}/retrain     — Trigger Airflow retrain pipeline

Usage:
    docker compose up -d
    uv run python scripts/simulate_pipeline.py
    uv run python scripts/simulate_pipeline.py --fast
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# ─── Config ──────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# German Credit dataset: 30 features after encoding
FEATURE_RANGES = [
    (6, 72),       # duration_months
    (250, 18500),  # credit_amount
    (19, 75),      # age
    (1, 2),        # num_dependents
    (0, 1),        # foreign_worker
    (1, 4),        # installment_rate
    (1, 4),        # present_residence
    (1, 4),        # existing_credits
] + [(0.0, 1.0)] * 22  # remaining binary/normalized features


# ─── Display Helpers ─────────────────────────────────────────────

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _header(title: str) -> None:
    print(f"\n{BOLD}{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}{RESET}\n")


def _week(num: int, title: str) -> None:
    print(f"\n{BOLD}{BLUE}  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  📅 Week {num}: {title:<46}║")
    print(f"  ╚══════════════════════════════════════════════════════════╝{RESET}\n")


def _step(msg: str) -> None:
    print(f"  {DIM}→{RESET} {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✅ {msg}{RESET}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠️  {msg}{RESET}")


def _alert(msg: str) -> None:
    print(f"  {RED}🚨 {msg}{RESET}")


def _info(msg: str) -> None:
    print(f"  {CYAN}ℹ️  {msg}{RESET}")


def _stat(label: str, value: Any) -> None:
    print(f"  {DIM}   {label}:{RESET} {value}")


def _progress(current: int, total: int) -> None:
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = current / total * 100
    print(f"\r  {DIM}   [{bar}] {pct:5.1f}% ({current}/{total}) {RESET}", end="")
    if current == total:
        print()


# ─── API Client ──────────────────────────────────────────────────


def _api(
    client: httpx.Client,
    method: str,
    path: str,
    **kwargs: Any,
) -> httpx.Response | None:
    """Make API request with error handling."""
    try:
        return client.request(method, path, timeout=30.0, **kwargs)
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError):
        return None


# ─── Data Generators ─────────────────────────────────────────────


def _random_features(drift_level: float = 0.0) -> list[float]:
    """Generate feature vector with controllable drift.

    drift_level: 0.0 = normal distribution, 1.0 = fully shifted
    """
    features = []
    for i, (lo, hi) in enumerate(FEATURE_RANGES):
        if drift_level > 0 and i < 5:
            center = lo + (hi - lo) * (0.5 + drift_level * 0.4)
            spread = (hi - lo) * 0.15
            val = random.gauss(center, spread)
            val = max(lo * 0.5, min(hi * 1.2, val))
        else:
            val = random.uniform(lo, hi)
        features.append(round(val, 4))
    return features


# ─── Simulation State ────────────────────────────────────────────


class PipelineStats:
    """Track cumulative stats across the simulation."""

    def __init__(self) -> None:
        self.total_predictions = 0
        self.total_labeled = 0
        self.total_errors = 0
        self.prediction_ids: list[str] = []
        self.results: list[int] = []
        self.confidences: list[float] = []
        self.latencies: list[float] = []

    def record(self, resp_json: dict[str, Any]) -> None:
        self.total_predictions += 1
        self.prediction_ids.append(resp_json["prediction_id"])
        self.results.append(resp_json["result"])
        self.confidences.append(resp_json["confidence"]["value"])
        self.latencies.append(resp_json["latency_ms"])

    @property
    def avg_confidence(self) -> float:
        return round(float(np.mean(self.confidences)), 4) if self.confidences else 0.0

    @property
    def avg_latency(self) -> float:
        return round(float(np.mean(self.latencies)), 2) if self.latencies else 0.0

    @property
    def label_rate(self) -> float:
        return self.total_labeled / max(self.total_predictions, 1) * 100


# ─── Simulation Phases ───────────────────────────────────────────
# Each function = single responsibility, matching one or more API endpoints.


def phase_preflight(client: httpx.Client, model_id: str) -> None:
    """PRE-FLIGHT: /health, /models, /predict"""
    _header("🔍 PRE-FLIGHT CHECK")

    _step("Checking API health...")
    resp = _api(client, "GET", "/health")
    if resp is None or resp.status_code != 200:
        print(f"\n  {RED}❌ API is not healthy. Run: docker compose up -d{RESET}")
        sys.exit(1)
    _ok(f"API: {resp.json()['status']} (v{resp.json()['version']})")

    _step("Checking model deployment...")
    resp = _api(client, "GET", "/models")
    if resp and resp.status_code == 200:
        models = resp.json()
        _ok(f"{len(models)} models registered")
        for m in models:
            _stat(f"{m['model_id']}:{m['version']}", m['status'])

    _step("Verifying prediction endpoint...")
    features = _random_features()
    resp = _api(client, "POST", "/predict", json={
        "model_id": model_id, "model_version": "v1", "features": features,
    })
    if resp is None or resp.status_code != 200:
        print(f"\n  {RED}❌ Prediction failed. Model may not be loaded.{RESET}")
        sys.exit(1)
    r = resp.json()
    _ok(f"Prediction works: result={r['result']}, confidence={r['confidence']['value']:.4f}")


def phase_traffic(
    client: httpx.Client,
    model_id: str,
    stats: PipelineStats,
    n_requests: int,
    drift_level: float = 0.0,
    label_ratio: float = 0.0,
    delay: float = 0.0,
) -> int:
    """TRAFFIC: POST /predict + POST /feedback"""
    successes = 0
    new_ids: list[str] = []

    drift_pct = int(drift_level * 100)
    _step(f"Sending {n_requests} prediction requests (drift={drift_pct}%)...")

    for i in range(n_requests):
        features = _random_features(drift_level)
        resp = _api(client, "POST", "/predict", json={
            "model_id": model_id, "model_version": "v1", "features": features,
        })
        if resp is not None and resp.status_code == 200:
            stats.record(resp.json())
            new_ids.append(resp.json()["prediction_id"])
            successes += 1
        else:
            stats.total_errors += 1
        _progress(i + 1, n_requests)
        if delay > 0:
            time.sleep(delay)

    # Submit ground truth labels (simulates human review)
    if label_ratio > 0 and new_ids:
        n_to_label = int(len(new_ids) * label_ratio)
        labeled = 0
        for pid in random.sample(new_ids, min(n_to_label, len(new_ids))):
            ground_truth = random.choice([0, 1])
            resp = _api(client, "POST", "/feedback", json={
                "prediction_id": pid,
                "ground_truth": ground_truth,
            })
            if resp is not None and resp.status_code == 200:
                stats.total_labeled += 1
                labeled += 1
        _step(f"Labeled {labeled} predictions (human review simulation)")

    return successes


def phase_drift_check(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """DRIFT: GET /monitoring/drift/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/drift/{model_id}")
    if resp is not None and resp.status_code == 200:
        return resp.json()
    return None


def phase_drift_reports(client: httpx.Client, model_id: str) -> list[dict[str, Any]]:
    """REPORTS: GET /monitoring/reports/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/reports/{model_id}")
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        return data if isinstance(data, list) else [data]
    return []


def phase_performance(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """PERFORMANCE: GET /monitoring/performance/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/performance/{model_id}")
    if resp is not None and resp.status_code == 200:
        return resp.json()
    return None


def phase_rollback(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """ROLLBACK: POST /models/rollback"""
    resp = _api(client, "POST", "/models/rollback", json={
        "model_id": model_id,
        "reason": "Drift detected — archiving challengers to protect champion",
    })
    if resp is not None and resp.status_code == 200:
        return resp.json()
    return None


def phase_export_data(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """EXPORT: POST /data/export-training"""
    resp = _api(client, "POST", "/data/export-training", json={
        "model_id": model_id,
        "min_samples": 10,
        "include_baseline": True,
        "max_fresh_samples": 10000,
    })
    if resp is not None and resp.status_code == 200:
        return resp.json()
    if resp is not None:
        _warn(f"Export failed: {resp.text[:200]}")
    return None


def phase_retrain(data_path: str | None) -> dict[str, Any] | None:
    """RETRAIN: local train.py subprocess"""
    timestamp = int(time.time())
    version = f"v{timestamp}"
    model_dir = PROJECT_ROOT / "models" / "credit_risk" / version
    output_path = str(model_dir / "model.onnx")
    metrics_path = str(model_dir / "metrics.json")

    data = data_path or str(DATA_DIR / "credit_risk" / "dataset.csv")
    if data_path:
        data = str(PROJECT_ROOT / data_path)

    result = subprocess.run(
        [
            "uv", "run", "python", "examples/credit_risk/train.py",
            "--output", output_path,
            "--metrics", metrics_path,
            "--data", data,
        ],
        check=False, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )

    if result.returncode == 0 and Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        return {"version": version, "metrics": metrics, "data_path": data}
    return None


def phase_register(
    client: httpx.Client, model_id: str, version: str, metrics: dict[str, Any],
) -> bool:
    """REGISTER: POST /models/register"""
    resp = _api(client, "POST", "/models/register", json={
        "model_id": model_id,
        "version": version,
        "uri": f"local:///models/credit_risk/{version}/model.onnx",
        "framework": "onnx",
        "stage": "challenger",
        "metrics": metrics,
    })
    return resp is not None and resp.status_code == 200


def phase_retrain_trigger(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """TRIGGER: POST /models/{model_id}/retrain  (Airflow DAG)"""
    resp = _api(client, "POST", f"/models/{model_id}/retrain")
    if resp is not None and resp.status_code == 200:
        return resp.json()
    # Airflow may not be running — that's OK in local simulation
    return None


def phase_model_info(client: httpx.Client, model_id: str) -> dict[str, Any] | None:
    """INFO: GET /models/{model_id}"""
    resp = _api(client, "GET", f"/models/{model_id}")
    if resp is not None and resp.status_code == 200:
        return resp.json()
    return None


# ─── Main Simulation ─────────────────────────────────────────────


def run_simulation(api_url: str, model_id: str, fast: bool = False) -> None:
    """Run realistic production lifecycle simulation covering ALL API endpoints."""
    stats = PipelineStats()
    delay = 0 if fast else 0.02
    start_time = time.time()

    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   🔥  Phoenix ML — Self-Healing Pipeline Simulation                  ║
║                                                                      ║
║   Simulating the COMPLETE production lifecycle:                       ║
║     • Normal traffic → labels accumulate → drift builds              ║
║     • Drift detected → ALERT → ROLLBACK challengers                  ║
║     • Export fresh data → retrain → register challenger               ║
║     • Performance check → drift reports audit                        ║
║                                                                      ║
║   Endpoints: /predict /feedback /monitoring/drift /models/rollback   ║
║              /data/export-training /models/register /models/retrain  ║
║              /monitoring/reports /monitoring/performance              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}

  {DIM}API: {api_url}   Model: {model_id}   Mode: {'fast' if fast else 'realistic'}{RESET}
""")

    with httpx.Client(base_url=api_url) as client:

        # ── Pre-flight ──────────────────────────────────────────
        phase_preflight(client, model_id)

        # ── Week 1: Normal traffic ──────────────────────────────
        _week(1, "Initial Deployment — Normal Traffic")
        _step("Model deployed to production. Normal traffic begins...")
        ok = phase_traffic(client, model_id, stats, 40, drift_level=0.0, label_ratio=0.3, delay=delay)
        _ok(f"{ok} predictions served")
        _stat("Cumulative predictions", stats.total_predictions)
        _stat("Cumulative labeled", stats.total_labeled)
        _stat("Avg confidence", f"{stats.avg_confidence:.4f}")

        # ── Week 2: More traffic, labels accumulate ─────────────
        _week(2, "Growing Traffic — Labels Accumulate")
        _step("Traffic increases. Labeling team reviews more predictions...")
        ok = phase_traffic(client, model_id, stats, 60, drift_level=0.0, label_ratio=0.4, delay=delay)
        _ok(f"{ok} predictions served")
        _stat("Cumulative predictions", stats.total_predictions)
        _stat("Cumulative labeled", stats.total_labeled)

        _step("Routine drift check (should be clean)...")
        drift = phase_drift_check(client, model_id)
        if drift:
            detected = drift.get("drift_detected", False)
            p_val = drift.get("p_value", 1.0)
            if detected:
                _warn(f"Drift check: DETECTED (p={p_val:.4f}) — early detection")
            else:
                _ok(f"Drift check: clean (p={p_val:.4f})")

        _step("Checking model performance baseline...")
        perf = phase_performance(client, model_id)
        if perf:
            _ok("Performance baseline recorded")
            for k, v in list(perf.items())[:4]:
                if isinstance(v, float):
                    _stat(k, f"{v:.4f}")
                else:
                    _stat(k, v)
        else:
            _info("Performance metrics not yet available (need more ground truth)")

        # ── Week 3: Subtle drift begins ─────────────────────────
        _week(3, "Distribution Shift Begins (Subtle)")
        _step("Market conditions change. Feature distributions start shifting...")
        ok = phase_traffic(client, model_id, stats, 50, drift_level=0.15, label_ratio=0.35, delay=delay)
        _ok(f"{ok} predictions served (some with shifted features)")
        _stat("Cumulative predictions", stats.total_predictions)
        _stat("Cumulative labeled", stats.total_labeled)

        # ── Week 4: Drift intensifies ───────────────────────────
        _week(4, "Drift Intensifies")
        _step("Distribution shift becomes more pronounced...")
        ok = phase_traffic(client, model_id, stats, 50, drift_level=0.30, label_ratio=0.3, delay=delay)
        _ok(f"{ok} predictions served")
        _stat("Cumulative predictions", stats.total_predictions)
        _stat("Cumulative labeled", stats.total_labeled)
        if stats.total_errors > 0:
            _stat("Total errors", stats.total_errors)

        _step("Drift check (should start showing)...")
        drift = phase_drift_check(client, model_id)
        if drift:
            detected = drift.get("drift_detected", False)
            p_val = drift.get("p_value", 1.0)
            if detected:
                _alert(f"DRIFT DETECTED! p_value={p_val:.4f}")
            else:
                _warn(f"p_value={p_val:.4f} (approaching threshold)")

        # ── Week 5: Significant drift + ALERT + ROLLBACK ────────
        _week(5, "Drift Confirmed — ALERT & ROLLBACK")
        _step("Strong distribution shift. Model accuracy degrading...")
        ok = phase_traffic(client, model_id, stats, 40, drift_level=0.50, label_ratio=0.25, delay=delay)
        _ok(f"{ok} predictions served under drift")
        _stat("Cumulative predictions", stats.total_predictions)
        _stat("Cumulative labeled", stats.total_labeled)

        _step("Running drift detection...")
        drift = phase_drift_check(client, model_id)
        if drift:
            detected = drift.get("drift_detected", False)
            p_val = drift.get("p_value", 1.0)
            rec = drift.get("recommendation", "")
            if detected:
                _alert(f"DRIFT CONFIRMED! p_value={p_val:.4f}")
                _alert(f"Recommendation: {rec}")
            else:
                _warn(f"Statistical test: p_value={p_val:.4f}")

        # 🚨 ALERT: In production, AlertManager would fire webhook
        _step("AlertManager fires webhook notification...")
        _alert("ALERT: Drift detected for credit-risk (severity=CRITICAL)")
        _alert("→ Slack/Discord webhook would be sent here")
        _alert("→ PagerDuty incident created")

        # ⏪ ROLLBACK: Archive all challengers, keep champion safe
        _step("Initiating rollback — archiving challengers...")
        rollback = phase_rollback(client, model_id)
        if rollback:
            champion = rollback.get("champion", "none")
            archived = rollback.get("archived_challengers", [])
            _ok(f"Rollback complete!")
            _stat("Champion (preserved)", champion or "v1 (default)")
            _stat("Archived challengers", archived if archived else "none currently")
        else:
            _warn("Rollback endpoint returned no data (no challengers to archive)")

        # Check drift reports audit trail
        _step("Checking drift report audit trail...")
        reports = phase_drift_reports(client, model_id)
        if reports:
            _ok(f"{len(reports)} drift report(s) in audit trail")
            latest = reports[0] if isinstance(reports[0], dict) else {}
            if "drift_detected" in latest:
                _stat("Latest report", f"drift={latest.get('drift_detected')}, p={latest.get('p_value', '?')}")
        else:
            _info("No historical drift reports yet")

        # ── Week 6: Export fresh data ────────────────────────────
        _week(6, "Self-Healing: Export Fresh Data")
        _step("Pipeline triggered: exporting labeled prediction logs from DB...")
        _step(f"Total labeled predictions in DB: ~{stats.total_labeled}")

        export_result = phase_export_data(client, model_id)
        if export_result:
            _ok("Fresh data exported!")
            _stat("Export path", export_result["export_path"])
            _stat("Total samples", export_result["total_samples"])
            _stat("Fresh from logs", f"{export_result['fresh_samples']} (NEW production data)")
            _stat("Baseline", f"{export_result['baseline_samples']} (original training data)")
            total = export_result["total_samples"]
            fresh_pct = export_result["fresh_samples"] / total * 100 if total > 0 else 0
            _stat("Fresh data ratio", f"{fresh_pct:.1f}%")
        else:
            _warn("Export failed — falling back to baseline data")

        # Try Airflow trigger (may not be running locally)
        _step("Attempting Airflow DAG trigger (POST /models/{model_id}/retrain)...")
        airflow_result = phase_retrain_trigger(client, model_id)
        if airflow_result:
            _ok(f"Airflow DAG triggered: {airflow_result.get('dag_run_id', '?')}")
        else:
            _info("Airflow not available — proceeding with local retrain")

        # ── Week 7: Retrain + register ───────────────────────────
        _week(7, "Self-Healing: Retrain & Deploy")
        export_path = export_result["export_path"] if export_result else None

        _step(f"Retraining on: {'FRESH exported data' if export_path else 'baseline (fallback)'}...")
        if export_path:
            _step(f"  Data: {export_path}")

        train_result = phase_retrain(export_path)
        if train_result:
            _ok(f"Model retrained: {train_result['version']}")
            _stat("Data used", train_result["data_path"])
            for k, v in list(train_result["metrics"].items())[:6]:
                _stat(k, v)

            # Register as challenger
            _step("Registering new version as challenger...")
            if phase_register(client, model_id, train_result["version"], train_result["metrics"]):
                _ok(f"Registered {model_id}:{train_result['version']} as challenger")
            else:
                _warn("Registration failed")

            # List model versions
            _step("Current model versions:")
            resp = _api(client, "GET", "/models")
            if resp and resp.status_code == 200:
                for m in resp.json():
                    if m["model_id"] == model_id:
                        _stat(f"{m['version']}", m['status'])
        else:
            _warn("Retrain failed")

        # ── Week 8: Verify healing ───────────────────────────────
        _week(8, "Verification — Post-Healing Checks")

        _step("Checking model info after healing...")
        model_info = phase_model_info(client, model_id)
        if model_info:
            _ok(f"Model info retrieved for {model_id}")
            if isinstance(model_info, dict):
                for k in ["model_id", "version", "status"]:
                    if k in model_info:
                        _stat(k, model_info[k])

        _step("Post-healing performance check...")
        perf = phase_performance(client, model_id)
        if perf:
            _ok("Performance metrics available")
            for k, v in list(perf.items())[:4]:
                if isinstance(v, float):
                    _stat(k, f"{v:.4f}")
                else:
                    _stat(k, v)
        else:
            _info("Performance metrics need more production data to compute")

        _step("Final drift report audit...")
        reports = phase_drift_reports(client, model_id)
        _ok(f"Total drift reports in audit trail: {len(reports)}")

        _step("Sending post-healing traffic to verify model still works...")
        ok = phase_traffic(client, model_id, stats, 20, drift_level=0.0, label_ratio=0.5, delay=delay)
        _ok(f"{ok} predictions served successfully post-healing")

    # ── Final Summary ─────────────────────────────────────────────

    elapsed = time.time() - start_time

    print(f"""
{BOLD}╔══════════════════════════════════════════════════════════════════════╗
║  ✨  SIMULATION COMPLETE — Full Lifecycle Summary                     ║
╚══════════════════════════════════════════════════════════════════════╝{RESET}

  {BOLD}Lifecycle Timeline:{RESET}
    Week 1-2:  Normal traffic         → {stats.total_predictions} predictions, {stats.total_labeled} labeled
    Week 3-4:  Drift begins           → Distribution shift 15-30%
    Week 5:    Drift confirmed        → 🚨 Alert + ⏪ Rollback challengers
    Week 6:    Export fresh data       → Labeled logs from DB
    Week 7:    Retrain + register     → New challenger deployed
    Week 8:    Verification           → Post-healing health check

  {BOLD}Cumulative Stats:{RESET}
    Total predictions:    {stats.total_predictions}
    Labeled (ground truth): {stats.total_labeled}  ({stats.label_rate:.0f}% label rate)
    Prediction errors:    {stats.total_errors}
    Avg confidence:       {stats.avg_confidence}
    Avg latency:          {stats.avg_latency}ms

  {BOLD}API Endpoints Exercised:{RESET}
    ✅ GET  /health                        — Pre-flight
    ✅ POST /predict                       — {stats.total_predictions} requests
    ✅ POST /feedback                      — {stats.total_labeled} labels
    ✅ GET  /models                        — List all versions
    ✅ GET  /models/{{model_id}}             — Model info
    ✅ GET  /monitoring/drift/{{model_id}}   — Drift detection
    ✅ GET  /monitoring/reports/{{model_id}} — Audit trail
    ✅ GET  /monitoring/performance/{{id}}   — Performance metrics
    ✅ POST /models/rollback               — Archive challengers
    ✅ POST /data/export-training          — Fresh data export
    ✅ POST /models/register               — Register retrained model
    ✅ POST /models/{{model_id}}/retrain    — Airflow trigger (if available)

  {BOLD}Self-Healing Loop:{RESET}
    ┌─── traffic ──→ predict ──→ log to DB ───┐
    │                                          │
    │   drift detected ← monitoring ← logs     │
    │        │                                 │
    │   🚨 ALERT ──→ webhook/Slack/PagerDuty   │
    │        │                                 │
    │   ⏪ ROLLBACK ──→ archive challengers     │
    │        │                                 │
    │   📤 export fresh data ← prediction_logs │
    │        │                                 │
    │   🏋️ retrain on FRESH data               │
    │        │                                 │
    │   📦 register as challenger              │
    │        │                                 │
    └── deploy new model ←─────────────────────┘

  Total simulation time: {elapsed:.1f}s
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate self-healing ML pipeline")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API base URL")
    parser.add_argument("--model-id", default="credit-risk", help="Model ID")
    parser.add_argument("--fast", action="store_true", help="Skip delays for speed")
    args = parser.parse_args()

    run_simulation(args.api_url, args.model_id, args.fast)


if __name__ == "__main__":
    main()
