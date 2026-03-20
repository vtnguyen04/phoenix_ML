"""
End-to-end test for Alert, Anomaly Detection, and Rollback flows.

Tests all three monitoring flows with simulated data to verify:
1. AlertManager: fires when metric breaches threshold, respects cooldown
2. AnomalyDetector: detects prediction anomalies, latency spikes, error rates
3. RollbackManager: triggers rollback when challenger error_rate or latency exceeds thresholds

Usage:
    python scripts/test_monitoring_flows.py
"""

import asyncio
import logging
import sys
from unittest.mock import AsyncMock

# Add project root to path
sys.path.insert(0, ".")

from src.domain.monitoring.services.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
)
from src.domain.monitoring.services.anomaly_detector import AnomalyDetector, AnomalyType
from src.domain.monitoring.services.rollback_manager import (
    ChallengerMetrics,
    RollbackManager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[tuple[str, bool]] = []


def report(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed))
    status = PASS if passed else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"  {status} {name}{suffix}")


# ─────────────────────────── 1. AlertManager ───────────────────────────
def test_alert_manager() -> None:
    print("\n═══ 1. AlertManager ═══")

    am = AlertManager()

    # Register rules
    am.register_rule(AlertRule(
        name="high_drift",
        metric="drift_score",
        threshold=0.3,
        severity=AlertSeverity.CRITICAL,
        comparison="gt",
        cooldown_seconds=0,   # disable cooldown for test
        description="Drift score exceeds 0.3",
    ))
    am.register_rule(AlertRule(
        name="low_accuracy",
        metric="accuracy",
        threshold=0.7,
        severity=AlertSeverity.WARNING,
        comparison="lt",
        cooldown_seconds=0,
        description="Accuracy below 70%",
    ))
    am.register_rule(AlertRule(
        name="high_latency",
        metric="latency_ms",
        threshold=100.0,
        severity=AlertSeverity.WARNING,
        comparison="gt",
        cooldown_seconds=0,
        description="Latency exceeds 100ms",
    ))

    # Test 1: Drift above threshold → should fire
    alerts = am.evaluate("drift_score", 0.85, model_id="credit-risk")
    report(
        "drift_score=0.85 > 0.3 → fires CRITICAL alert",
        len(alerts) == 1 and alerts[0].severity == AlertSeverity.CRITICAL,
        f"fired={len(alerts)}",
    )

    # Test 2: Drift below threshold → should NOT fire
    alerts = am.evaluate("drift_score", 0.1, model_id="credit-risk")
    report(
        "drift_score=0.1 < 0.3 → no alert",
        len(alerts) == 0,
        f"fired={len(alerts)}",
    )

    # Test 3: Accuracy below threshold → should fire WARNING
    alerts = am.evaluate("accuracy", 0.55, model_id="house-price")
    report(
        "accuracy=0.55 < 0.7 → fires WARNING",
        len(alerts) == 1 and alerts[0].severity == AlertSeverity.WARNING,
        f"fired={len(alerts)}",
    )

    # Test 4: Accuracy above threshold → no alert
    alerts = am.evaluate("accuracy", 0.85, model_id="house-price")
    report("accuracy=0.85 > 0.7 → no alert", len(alerts) == 0)

    # Test 5: Latency above threshold → should fire
    alerts = am.evaluate("latency_ms", 250.0, model_id="fraud-detection")
    report(
        "latency=250ms > 100ms → fires WARNING",
        len(alerts) == 1,
        f"model_id={alerts[0].model_id if alerts else '?'}",
    )

    # Test 6: Unknown metric → no rules match → no alert
    alerts = am.evaluate("unknown_metric", 999.0)
    report("unknown_metric → no alert", len(alerts) == 0)

    # Test 7: Cooldown test
    am2 = AlertManager()
    am2.register_rule(AlertRule(
        name="cooldown_test",
        metric="test",
        threshold=1.0,
        severity=AlertSeverity.INFO,
        comparison="gt",
        cooldown_seconds=9999,  # very long cooldown
    ))
    fire1 = am2.evaluate("test", 5.0)
    fire2 = am2.evaluate("test", 5.0)  # should be blocked by cooldown
    report(
        "cooldown blocks re-firing",
        len(fire1) == 1 and len(fire2) == 0,
        f"fire1={len(fire1)}, fire2={len(fire2)}",
    )

    # Test 8: Active alerts accumulate
    total = am.get_active_alerts()
    report(
        "active alerts accumulate correctly",
        len(total) >= 3,
        f"total_active={len(total)}",
    )

    # Test 9: Clear alerts
    am.clear_alerts()
    report("clear_alerts() empties list", len(am.get_active_alerts()) == 0)

    # Test 10: Alert.to_dict() serialization
    if total:
        d = total[0].to_dict()
        required_keys = {"rule_name", "severity", "status", "metric_value", "threshold", "message", "model_id", "timestamp"}
        report(
            "Alert.to_dict() has all required keys",
            required_keys.issubset(d.keys()),
            f"keys={list(d.keys())}",
        )


# ─────────────────────────── 2. AnomalyDetector ───────────────────────────
def test_anomaly_detector() -> None:
    print("\n═══ 2. AnomalyDetector ═══")

    ad = AnomalyDetector(z_score_threshold=2.0, latency_multiplier=2.0, error_rate_threshold=0.05)

    # Test 1: Normal confidence scores → no anomaly
    normal_scores = [0.85, 0.87, 0.83, 0.86, 0.84, 0.88, 0.82, 0.85, 0.86, 0.84]
    report1 = ad.detect_prediction_anomaly(normal_scores)
    report(
        "normal confidence → no anomaly",
        not report1.is_anomalous,
        f"score={report1.score:.2f}, type={report1.anomaly_type.value}",
    )

    # Test 2: Scores far from known baseline → anomaly detected
    anomalous_scores = [0.85, 0.86, 0.84, 0.85, 0.83, 0.86, 0.84, 0.85, 0.86, 0.84]
    report2 = ad.detect_prediction_anomaly(
        anomalous_scores, baseline_mean=0.5, baseline_std=0.05
    )
    report(
        "scores far from baseline → anomaly detected",
        report2.is_anomalous,
        f"score={report2.score:.2f}, detail={report2.detail}",
    )

    # Test 3: Empty data → no anomaly
    report3 = ad.detect_prediction_anomaly([])
    report("empty data → no anomaly", not report3.is_anomalous and report3.sample_size == 0)

    # Test 4: Single value → no anomaly (zero variance)
    report4 = ad.detect_prediction_anomaly([0.5, 0.5, 0.5])
    report(
        "constant values → zero variance, no anomaly",
        not report4.is_anomalous,
        report4.detail,
    )

    # Test 5: Latency spike detection — normal
    normal_latencies = [2.0, 3.0, 2.5, 2.8, 3.2, 2.1, 2.9, 3.0, 2.7, 2.3]
    report5 = ad.detect_latency_spike(normal_latencies, baseline_p99_ms=5.0)
    report(
        "normal latency → no spike",
        not report5.is_anomalous,
        f"p99={report5.score:.1f}ms vs threshold={report5.threshold:.1f}ms",
    )

    # Test 6: Latency spike — with spikes
    spike_latencies = [2.0, 3.0, 150.0, 200.0, 2.5, 180.0, 2.8, 3.2, 170.0, 190.0]
    report6 = ad.detect_latency_spike(spike_latencies, baseline_p99_ms=5.0)
    report(
        "high latency → spike detected",
        report6.is_anomalous,
        f"p99={report6.score:.1f}ms vs threshold={report6.threshold:.1f}ms",
    )

    # Test 7: Error rate — normal
    report7 = ad.detect_error_rate(total_requests=1000, error_count=10)  # 1%
    report(
        "1% error rate → normal",
        not report7.is_anomalous,
        f"rate={report7.score:.2%}",
    )

    # Test 8: Error rate — high
    report8 = ad.detect_error_rate(total_requests=100, error_count=15)  # 15%
    report(
        "15% error rate → anomaly",
        report8.is_anomalous,
        f"rate={report8.score:.2%}",
    )

    # Test 9: Error rate — zero requests
    report9 = ad.detect_error_rate(total_requests=0, error_count=0)
    report("0 requests → no anomaly", not report9.is_anomalous and report9.sample_size == 0)

    # Test 10: Anomaly type classification
    report(
        "prediction anomaly type correct",
        report2.anomaly_type == AnomalyType.PREDICTION_ANOMALY,
    )
    report(
        "latency spike type correct",
        report6.anomaly_type == AnomalyType.LATENCY_SPIKE,
    )
    report(
        "error rate type correct",
        report8.anomaly_type == AnomalyType.ERROR_RATE,
    )


# ─────────────────────────── 3. RollbackManager ───────────────────────────
def test_rollback_manager() -> None:
    print("\n═══ 3. RollbackManager ═══")

    mock_repo = AsyncMock()
    mock_repo.update_stage = AsyncMock(return_value=None)

    rm = RollbackManager(
        model_repo=mock_repo,
        error_rate_threshold=0.10,   # 10%
        latency_threshold_ms=500.0,  # 500ms
        min_requests=10,             # lowered for test
    )

    async def run_rollback_tests() -> None:
        # Test 1: Insufficient data → no rollback
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="credit-risk",
            challenger_version="v2",
            champion_version="v1",
            total_requests=5,
            error_count=0,
            avg_latency_ms=10.0,
        ))
        report(
            "insufficient data (5/10) → no rollback",
            not decision.should_rollback,
            decision.reason,
        )

        # Test 2: Healthy challenger → no rollback
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="fraud-detection",
            challenger_version="v2",
            champion_version="v1",
            total_requests=100,
            error_count=2,  # 2% error rate
            avg_latency_ms=15.0,
        ))
        report(
            "healthy challenger (2% error, 15ms) → no rollback",
            not decision.should_rollback,
            f"error_rate={decision.error_rate:.2%}, latency={decision.avg_latency_ms:.1f}ms",
        )

        # Test 3: High error rate → rollback triggered
        mock_repo.reset_mock()
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="credit-risk",
            challenger_version="v3",
            champion_version="v1",
            total_requests=100,
            error_count=25,  # 25% error rate
            avg_latency_ms=20.0,
        ))
        report(
            "25% error rate > 10% → rollback TRIGGERED",
            decision.should_rollback,
            decision.reason,
        )
        report(
            "rollback calls update_stage(archived)",
            mock_repo.update_stage.called,
            f"called with: {mock_repo.update_stage.call_args}",
        )

        # Test 4: High latency → rollback triggered
        mock_repo.reset_mock()
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="house-price",
            challenger_version="v2",
            champion_version="v1",
            total_requests=50,
            error_count=1,  # 2% — below threshold
            avg_latency_ms=800.0,  # way above 500ms
        ))
        report(
            "800ms latency > 500ms → rollback TRIGGERED",
            decision.should_rollback,
            decision.reason,
        )

        # Test 5: Edge case — exactly at threshold
        mock_repo.reset_mock()
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="image-class",
            challenger_version="v2",
            champion_version="v1",
            total_requests=100,
            error_count=10,  # exactly 10%
            avg_latency_ms=500.0,  # exactly at threshold
        ))
        report(
            "exactly at threshold (10%, 500ms) → no rollback (gt, not gte)",
            not decision.should_rollback,
            f"error_rate={decision.error_rate:.2%}",
        )

        # Test 6: Zero requests → no rollback
        decision = await rm.evaluate_challenger(ChallengerMetrics(
            model_id="m1",
            challenger_version="v2",
            champion_version="v1",
            total_requests=0,
            error_count=0,
            avg_latency_ms=0.0,
        ))
        report("0 requests → no rollback", not decision.should_rollback)

        # Test 7: Rollback preserves correct model versions
        mock_repo.reset_mock()
        await rm.evaluate_challenger(ChallengerMetrics(
            model_id="fraud-detection",
            challenger_version="v99",
            champion_version="v1",
            total_requests=100,
            error_count=50,  # 50% error rate
            avg_latency_ms=10.0,
        ))
        if mock_repo.update_stage.called:
            call_args = mock_repo.update_stage.call_args
            report(
                "rollback archives correct challenger version",
                call_args[0] == ("fraud-detection", "v99", "archived"),
                f"args={call_args[0]}",
            )
        else:
            report("rollback archives correct challenger version", False, "update_stage not called")

    asyncio.run(run_rollback_tests())


# ─────────────────────────── 4. Integration: Alert + Anomaly ───────────────────────────
def test_integration() -> None:
    print("\n═══ 4. Integration: Alert + Anomaly → Alert firing ═══")

    # Setup alert manager with anomaly-driven rules
    am = AlertManager()
    am.register_rule(AlertRule(
        name="anomaly_confidence",
        metric="anomaly_score",
        threshold=2.5,
        severity=AlertSeverity.CRITICAL,
        comparison="gt",
        cooldown_seconds=0,
    ))
    am.register_rule(AlertRule(
        name="high_error_rate",
        metric="error_rate",
        threshold=0.05,
        severity=AlertSeverity.WARNING,
        comparison="gt",
        cooldown_seconds=0,
    ))

    # Detect anomaly with known baseline
    ad = AnomalyDetector(z_score_threshold=2.0)
    anomalous_scores = [0.85, 0.86, 0.84, 0.85, 0.83, 0.86, 0.84, 0.85, 0.86, 0.84]
    anomaly_report = ad.detect_prediction_anomaly(
        anomalous_scores, baseline_mean=0.5, baseline_std=0.05
    )

    # Feed anomaly score into alert manager
    alerts = am.evaluate("anomaly_score", anomaly_report.score, model_id="credit-risk")
    report(
        "anomaly score → triggers CRITICAL alert",
        len(alerts) == 1 and anomaly_report.is_anomalous,
        f"anomaly_score={anomaly_report.score:.2f}, alerts_fired={len(alerts)}",
    )

    # Detect high error rate
    error_report = ad.detect_error_rate(total_requests=200, error_count=30)  # 15%
    alerts2 = am.evaluate("error_rate", error_report.score, model_id="house-price")
    report(
        "high error rate anomaly → triggers WARNING alert",
        len(alerts2) == 1 and error_report.is_anomalous,
        f"error_rate={error_report.score:.2%}",
    )


# ─────────────────────────── 5. API Integration ───────────────────────────
def test_api_integration() -> None:
    print("\n═══ 5. Live API Integration ═══")

    import requests

    api_url = "http://localhost:8001"

    # Test predict endpoint works
    try:
        resp = requests.post(
            f"{api_url}/predict",
            json={"model_id": "credit-risk", "features": [1.0] * 30},
            timeout=5,
        )
        report(
            "POST /predict works",
            resp.status_code == 200,
            f"status={resp.status_code}, latency={resp.json().get('latency_ms', '?')}ms",
        )
    except Exception as e:
        report("POST /predict works", False, str(e))

    # Test drift endpoint works
    try:
        resp = requests.get(f"{api_url}/monitoring/drift/credit-risk", timeout=5)
        report(
            "GET /monitoring/drift works",
            resp.status_code == 200,
            f"status={resp.status_code}",
        )
    except Exception as e:
        report("GET /monitoring/drift works", False, str(e))

    # Test performance endpoint
    try:
        resp = requests.get(f"{api_url}/monitoring/performance/credit-risk", timeout=5)
        data = resp.json()
        report(
            "GET /monitoring/performance works",
            resp.status_code == 200 and "total_predictions" in str(data),
            f"predictions={data.get('total_predictions', '?')}",
        )
    except Exception as e:
        report("GET /monitoring/performance works", False, str(e))

    # Test rollback endpoint
    try:
        resp = requests.post(
            f"{api_url}/models/rollback",
            json={"model_id": "credit-risk"},
            timeout=5,
        )
        # Rollback may fail if no challenger — that's expected
        report(
            "POST /models/rollback endpoint exists",
            resp.status_code in (200, 400, 404, 422),
            f"status={resp.status_code}",
        )
    except Exception as e:
        report("POST /models/rollback endpoint exists", False, str(e))


# ─────────────────────────── MAIN ───────────────────────────
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Phoenix ML – Monitoring Flows E2E Test                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    test_alert_manager()
    test_anomaly_detector()
    test_rollback_manager()
    test_integration()
    test_api_integration()

    # Summary
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    failed = total - passed

    print(f"\n{'═' * 50}")
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    print(f"{'═' * 50}")

    if failed > 0:
        print("\n  Failed tests:")
        for name, ok in results:
            if not ok:
                print(f"    ❌ {name}")
        sys.exit(1)
    else:
        print("  ✅ All monitoring flows verified!\n")
        sys.exit(0)
