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
from dataclasses import dataclass
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
    border = "══════════════════════════════════════════════════════════"
    print(f"\n{BOLD}{BLUE}  ╔{border}╗")
    print(f"  ║  📅 Week {num}: {title:<46}║")
    print(f"  ╚{border}╝{RESET}\n")


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
    end = "\n" if current == total else ""
    print(
        f"\r  {DIM}   [{bar}] {pct:5.1f}% ({current}/{total}) {RESET}",
        end=end,
    )


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
        if self.confidences:
            return round(float(np.mean(self.confidences)), 4)
        return 0.0

    @property
    def avg_latency(self) -> float:
        if self.latencies:
            return round(float(np.mean(self.latencies)), 2)
        return 0.0

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
        print(f"\n  {RED}❌ API not healthy. docker compose up -d{RESET}")
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
        "model_id": model_id, "model_version": "v1",
        "features": features,
    })
    if resp is None or resp.status_code != 200:
        print(f"\n  {RED}❌ Prediction failed.{RESET}")
        sys.exit(1)
    r = resp.json()
    conf = r['confidence']['value']
    _ok(f"Prediction works: result={r['result']}, conf={conf:.4f}")


@dataclass
class TrafficConfig:
    """Bundle traffic simulation parameters."""

    drift_level: float = 0.0
    label_ratio: float = 0.0
    delay: float = 0.0


def phase_traffic(
    client: httpx.Client,
    model_id: str,
    stats: PipelineStats,
    n_requests: int,
    cfg: TrafficConfig | None = None,
) -> int:
    """TRAFFIC: POST /predict + POST /feedback"""
    if cfg is None:
        cfg = TrafficConfig()
    successes = 0
    new_ids: list[str] = []

    drift_pct = int(cfg.drift_level * 100)
    _step(f"Sending {n_requests} requests (drift={drift_pct}%)...")

    for i in range(n_requests):
        features = _random_features(cfg.drift_level)
        resp = _api(client, "POST", "/predict", json={
            "model_id": model_id, "model_version": "v1",
            "features": features,
        })
        if resp is not None and resp.status_code == 200:
            stats.record(resp.json())
            new_ids.append(resp.json()["prediction_id"])
            successes += 1
        else:
            stats.total_errors += 1
        _progress(i + 1, n_requests)
        if cfg.delay > 0:
            time.sleep(cfg.delay)

    # Submit ground truth labels (simulates human review)
    if cfg.label_ratio > 0 and new_ids:
        n_to_label = int(len(new_ids) * cfg.label_ratio)
        labeled = 0
        sample_size = min(n_to_label, len(new_ids))
        for pid in random.sample(new_ids, sample_size):
            ground_truth = random.choice([0, 1])
            resp = _api(client, "POST", "/feedback", json={
                "prediction_id": pid,
                "ground_truth": ground_truth,
            })
            if resp is not None and resp.status_code == 200:
                stats.total_labeled += 1
                labeled += 1
        _step(f"Labeled {labeled} predictions (human review)")

    return successes


def phase_drift_check(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """DRIFT: GET /monitoring/drift/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/drift/{model_id}")
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    return None


def phase_drift_reports(
    client: httpx.Client, model_id: str,
) -> list[dict[str, Any]]:
    """REPORTS: GET /monitoring/reports/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/reports/{model_id}")
    if resp is not None and resp.status_code == 200:
        data = resp.json()
        return data if isinstance(data, list) else [data]
    return []


def phase_performance(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """PERFORMANCE: GET /monitoring/performance/{model_id}"""
    resp = _api(client, "GET", f"/monitoring/performance/{model_id}")
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    return None


def phase_rollback(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """ROLLBACK: POST /models/rollback"""
    reason = "Drift detected — archiving challengers"
    resp = _api(client, "POST", "/models/rollback", json={
        "model_id": model_id, "reason": reason,
    })
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    return None


def phase_export_data(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """EXPORT: POST /data/export-training"""
    resp = _api(client, "POST", "/data/export-training", json={
        "model_id": model_id,
        "min_samples": 10,
        "include_baseline": True,
        "max_fresh_samples": 10000,
    })
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    if resp is not None:
        _warn(f"Export failed: {resp.text[:200]}")
    return None


def phase_retrain(
    data_path: str | None,
) -> dict[str, Any] | None:
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
        check=False, capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )

    if result.returncode == 0 and Path(metrics_path).exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        return {
            "version": version,
            "metrics": metrics,
            "data_path": data,
        }
    return None


def phase_register(
    client: httpx.Client, model_id: str,
    version: str, metrics: dict[str, Any],
) -> bool:
    """REGISTER: POST /models/register"""
    uri = f"local:///models/credit_risk/{version}/model.onnx"
    resp = _api(client, "POST", "/models/register", json={
        "model_id": model_id, "version": version,
        "uri": uri, "framework": "onnx",
        "stage": "challenger", "metrics": metrics,
    })
    return resp is not None and resp.status_code == 200


def phase_retrain_trigger(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """TRIGGER: POST /models/{model_id}/retrain (Airflow DAG)"""
    resp = _api(client, "POST", f"/models/{model_id}/retrain")
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    return None


def phase_model_info(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    """INFO: GET /models/{model_id}"""
    resp = _api(client, "GET", f"/models/{model_id}")
    if resp is not None and resp.status_code == 200:
        result: dict[str, Any] = resp.json()
        return result
    return None


# ─── Week Simulations ────────────────────────────────────────────


def _sim_week1(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(1, "Initial Deployment — Normal Traffic")
    _step("Model deployed. Normal traffic begins...")
    tc = TrafficConfig(drift_level=0.0, label_ratio=0.3, delay=delay)
    ok = phase_traffic(client, model_id, stats, 40, tc)
    _ok(f"{ok} predictions served")
    _stat("Cumulative predictions", stats.total_predictions)
    _stat("Cumulative labeled", stats.total_labeled)
    _stat("Avg confidence", f"{stats.avg_confidence:.4f}")


def _sim_week2(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(2, "Growing Traffic — Labels Accumulate")
    _step("Traffic increases. Labeling team reviews...")
    tc = TrafficConfig(drift_level=0.0, label_ratio=0.4, delay=delay)
    ok = phase_traffic(client, model_id, stats, 60, tc)
    _ok(f"{ok} predictions served")
    _stat("Cumulative predictions", stats.total_predictions)
    _stat("Cumulative labeled", stats.total_labeled)

    _step("Routine drift check (should be clean)...")
    drift = phase_drift_check(client, model_id)
    if drift:
        detected = drift.get("drift_detected", False)
        p_val = drift.get("p_value", 1.0)
        if detected:
            _warn(f"Drift check: DETECTED (p={p_val:.4f})")
        else:
            _ok(f"Drift check: clean (p={p_val:.4f})")

    _step("Checking model performance baseline...")
    perf = phase_performance(client, model_id)
    if perf:
        _ok("Performance baseline recorded")
        for k, v in list(perf.items())[:4]:
            fmt = f"{v:.4f}" if isinstance(v, float) else v
            _stat(k, fmt)
    else:
        _info("Performance not yet available")


def _sim_week3(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(3, "Distribution Shift Begins (Subtle)")
    _step("Market conditions change. Distributions shifting...")
    tc = TrafficConfig(drift_level=0.15, label_ratio=0.35, delay=delay)
    ok = phase_traffic(client, model_id, stats, 50, tc)
    _ok(f"{ok} predictions served (some shifted)")
    _stat("Cumulative predictions", stats.total_predictions)
    _stat("Cumulative labeled", stats.total_labeled)


def _sim_week4(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(4, "Drift Intensifies")
    _step("Distribution shift becomes more pronounced...")
    tc = TrafficConfig(drift_level=0.30, label_ratio=0.3, delay=delay)
    ok = phase_traffic(client, model_id, stats, 50, tc)
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


def _sim_week5(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(5, "Drift Confirmed — ALERT & ROLLBACK")
    _step("Strong distribution shift. Accuracy degrading...")
    tc = TrafficConfig(drift_level=0.50, label_ratio=0.25, delay=delay)
    ok = phase_traffic(client, model_id, stats, 40, tc)
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

    _step("AlertManager fires webhook notification...")
    _alert(f"ALERT: Drift for {model_id} (severity=CRITICAL)")
    _alert("→ Slack/Discord webhook would be sent here")
    _alert("→ PagerDuty incident created")

    _step("Initiating rollback — archiving challengers...")
    rollback = phase_rollback(client, model_id)
    if rollback:
        champion = rollback.get("champion", "none")
        archived = rollback.get("archived_challengers", [])
        _ok("Rollback complete!")
        _stat("Champion (preserved)", champion or "v1 (default)")
        _stat(
            "Archived challengers",
            archived if archived else "none currently",
        )
    else:
        _warn("Rollback: no challengers to archive")

    _step("Checking drift report audit trail...")
    reports = phase_drift_reports(client, model_id)
    if reports:
        _ok(f"{len(reports)} drift report(s) in audit trail")
        latest = reports[0] if isinstance(reports[0], dict) else {}
        if "drift_detected" in latest:
            p = latest.get("p_value", "?")
            d = latest.get("drift_detected")
            _stat("Latest report", f"drift={d}, p={p}")
    else:
        _info("No historical drift reports yet")


def _sim_week6(
    client: httpx.Client, model_id: str,
) -> dict[str, Any] | None:
    _week(6, "Self-Healing: Export Fresh Data")
    _step("Exporting labeled prediction logs from DB...")

    log_count = "~unknown"
    perf_url = f"/monitoring/performance/{model_id}"
    resp = _api(client, "GET", perf_url)
    if resp and resp.status_code == 200:
        total = resp.json().get("total_predictions", "?")
        log_count = f"~{total}"
    _step(f"Total labeled predictions in DB: {log_count}")

    export_result = phase_export_data(client, model_id)
    if export_result:
        _ok("Fresh data exported!")
        _stat("Export path", export_result["export_path"])
        _stat("Total samples", export_result["total_samples"])
        fresh = export_result.get("fresh_samples", 0)
        baseline = export_result.get("baseline_samples", 0)
        total = export_result.get("total_samples", 0)
        _stat("Fresh from logs", f"{fresh} (NEW production data)")
        _stat("Baseline", f"{baseline} (original training data)")
        if total > 0:
            ratio = fresh / total * 100
            _stat("Fresh data ratio", f"{ratio:.1f}%")
    else:
        _warn("Export failed or insufficient labeled data")
        _info("Will fall back to baseline dataset for retrain")

    _step("Attempting Airflow DAG trigger...")
    airflow_result = phase_retrain_trigger(client, model_id)
    if airflow_result:
        dag_id = airflow_result.get("dag_run_id", "?")
        _ok(f"Airflow DAG triggered: {dag_id}")
    else:
        _info("Airflow not available — local retrain instead")

    return export_result


def _sim_week7(
    client: httpx.Client, model_id: str,
    export_result: dict[str, Any] | None,
) -> None:
    _week(7, "Self-Healing: Retrain & Deploy")
    export_path = (
        export_result["export_path"] if export_result else None
    )

    src = "FRESH exported data" if export_path else "baseline"
    _step(f"Retraining on: {src}...")
    if export_path:
        _step(f"  Data: {export_path}")

    train_result = phase_retrain(export_path)
    if train_result:
        _ok(f"Model retrained: {train_result['version']}")
        _stat("Data used", train_result["data_path"])
        for k, v in list(train_result["metrics"].items())[:6]:
            _stat(k, v)

        _step("Registering new version as challenger...")
        ver = train_result["version"]
        metrics = train_result["metrics"]
        if phase_register(client, model_id, ver, metrics):
            _ok(f"Registered {model_id}:{ver} as challenger")
        else:
            _warn("Registration failed")

        _step("Current model versions:")
        resp = _api(client, "GET", "/models")
        if resp and resp.status_code == 200:
            for m in resp.json():
                if m["model_id"] == model_id:
                    _stat(f"{m['version']}", m["status"])
    else:
        _warn("Retrain failed")


def _sim_week8(
    client: httpx.Client, model_id: str,
    stats: PipelineStats, delay: float,
) -> None:
    _week(8, "Verification — Post-Healing Checks")

    _step("Checking model info after healing...")
    model_info = phase_model_info(client, model_id)
    if model_info and isinstance(model_info, dict):
        _ok(f"Model info retrieved for {model_id}")
        for k in ["model_id", "version", "status"]:
            if k in model_info:
                _stat(k, model_info[k])

    _step("Post-healing performance check...")
    perf = phase_performance(client, model_id)
    if perf:
        _ok("Performance metrics available")
        for k, v in list(perf.items())[:4]:
            fmt = f"{v:.4f}" if isinstance(v, float) else v
            _stat(k, fmt)
    else:
        _info("Performance metrics need more data")

    _step("Final drift report audit...")
    reports = phase_drift_reports(client, model_id)
    _ok(f"Total drift reports: {len(reports)}")

    _step("Post-healing traffic to verify model works...")
    tc = TrafficConfig(drift_level=0.0, label_ratio=0.5, delay=delay)
    ok = phase_traffic(client, model_id, stats, 20, tc)
    _ok(f"{ok} predictions served successfully post-healing")


# ─── Summary & Banner ────────────────────────────────────────────


def _print_summary(stats: PipelineStats, elapsed: float) -> None:
    """Print final simulation summary."""
    preds = stats.total_predictions
    labeled = stats.total_labeled
    rate = stats.label_rate
    errs = stats.total_errors
    conf = stats.avg_confidence
    lat = stats.avg_latency
    print(f"""
{BOLD}═══════════════════════════════════════════════════════
  ✨  SIMULATION COMPLETE — Full Lifecycle Summary
══════════════════════════════════════════════════════={RESET}

  {BOLD}Lifecycle Timeline:{RESET}
    Week 1-2:  Normal traffic    → {preds} preds, {labeled} labeled
    Week 3-4:  Drift begins      → Shift 15-30%
    Week 5:    Drift confirmed   → 🚨 Alert + ⏪ Rollback
    Week 6:    Export fresh data  → Labeled logs from DB
    Week 7:    Retrain + register → New challenger deployed
    Week 8:    Verification      → Post-healing check

  {BOLD}Cumulative Stats:{RESET}
    Total predictions:    {preds}
    Labeled (ground truth): {labeled}  ({rate:.0f}% label rate)
    Prediction errors:    {errs}
    Avg confidence:       {conf}
    Avg latency:          {lat}ms

  Total simulation time: {elapsed:.1f}s
""")


def _print_banner(
    api_url: str, model_id: str, fast: bool,
) -> None:
    mode = "fast" if fast else "standard"
    print(f"""
{BOLD}═══════════════════════════════════════════════════════
  🔬  Phoenix ML — Self-Healing Pipeline Simulation
═══════════════════════════════════════════════════════{RESET}

  Full 8-week production lifecycle:
    Week 1-2: Normal traffic + labeling
    Week 3-4: Distribution shift (subtle → strong)
    Week 5:   Drift confirmed → alert + rollback
    Week 6:   Export fresh data from prediction_logs
    Week 7:   Retrain on fresh data + register model
    Week 8:   Post-healing verification

  API: {api_url}   Model: {model_id}   Mode: {mode}
""")


# ─── Main Simulation ─────────────────────────────────────────────


def run_simulation(
    api_url: str, model_id: str, fast: bool = False,
) -> None:
    """Run realistic production lifecycle simulation."""
    stats = PipelineStats()
    delay = 0 if fast else 0.02
    start_time = time.time()

    _print_banner(api_url, model_id, fast)

    with httpx.Client(base_url=api_url) as client:
        phase_preflight(client, model_id)
        _sim_week1(client, model_id, stats, delay)
        _sim_week2(client, model_id, stats, delay)
        _sim_week3(client, model_id, stats, delay)
        _sim_week4(client, model_id, stats, delay)
        _sim_week5(client, model_id, stats, delay)
        export_result = _sim_week6(client, model_id)
        _sim_week7(client, model_id, export_result)
        _sim_week8(client, model_id, stats, delay)

    _print_summary(stats, time.time() - start_time)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate self-healing ML pipeline",
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8001",
        help="API base URL",
    )
    parser.add_argument(
        "--model-id", default="credit-risk",
        help="Model ID",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip delays for speed",
    )
    args = parser.parse_args()

    run_simulation(args.api_url, args.model_id, args.fast)


if __name__ == "__main__":
    main()
