# Monitoring & Alerting Guide

This guide covers how Phoenix ML's monitoring stack works end-to-end and how to configure it.

---

## Architecture Overview

```
Predictions → Prometheus Metrics → Grafana Dashboards
     ↓
Redis (prediction logs) → Drift Detection (every 30s)
     ↓                           ↓
AnomalyDetector              DriftCalculator
     ↓                           ↓
AlertManager ← rules ←   MonitoringService
     ↓                           ↓
AlertNotifier             Airflow Self-Healing
(Slack/PagerDuty)        (alert → rollback → retrain → deploy)
```

---

## 1. Drift Detection

Drift detection compares current prediction feature distributions against a reference baseline using statistical tests.

### How It Works

1. **Reference data** is saved during training (`reference_features.json`)
2. **Monitoring loop** runs every `MONITORING_INTERVAL_SECONDS` (default: 30s)
3. **KS test** (Kolmogorov-Smirnov) compares distributions
4. If `p-value < DRIFT_THRESHOLD` (default: 0.05) → drift detected
5. Drift triggers the Airflow `self_healing_pipeline`

### Configuration

```python
# src/config/monitoring.py
MONITORING_INTERVAL_SECONDS = 30     # Check frequency
DRIFT_THRESHOLD = 0.05               # KS test p-value threshold
```

### Manual Drift Check

```bash
# Trigger drift detection for a specific model
curl http://localhost:8001/monitoring/drift/credit-risk

# Response:
# {
#   "drift_detected": true,
#   "method": "ks_2samp",
#   "p_value": 0.00001,
#   "statistic": 0.875,
#   "threshold": 0.05,
#   "sample_size": 100
# }

# View historical drift reports
curl http://localhost:8001/monitoring/reports/credit-risk?limit=10
```

### Simulating Drift

```bash
# Send drifted data (shifted features) to trigger detection
uv run python scripts/simulate_drift.py
```

---

## 2. Anomaly Detection

The `AnomalyDetector` monitors three types of anomalies in real-time:

| Type | Method | Threshold |
|------|--------|-----------|
| **Prediction Anomaly** | Z-score on confidence | `z_score > 3.0` |
| **Latency Spike** | P99 vs baseline × multiplier | `p99 > baseline × 3.0` |
| **Error Rate** | Error count / total | `rate > 5%` |

### Configuration

```python
# src/config/monitoring.py
ANOMALY_Z_SCORE_THRESHOLD = 3.0      # For confidence score anomalies
ANOMALY_LATENCY_MULTIPLIER = 3.0     # P99 latency spike multiplier
ANOMALY_ERROR_RATE_THRESHOLD = 0.05  # 5% error rate threshold
```

### Usage in Code

```python
from src.domain.monitoring.services.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(z_score_threshold=3.0)

# Check prediction confidence for anomalies
report = detector.detect_prediction_anomaly(
    confidence_scores=[0.85, 0.87, 0.02, 0.84, ...]
)
# report.is_anomalous, report.score, report.detail

# Check latency spikes
report = detector.detect_latency_spike(
    latencies=[2.0, 3.0, 150.0, 2.5, ...],
    baseline_p99_ms=5.0
)

# Check error rate
report = detector.detect_error_rate(
    total_requests=1000,
    error_count=80
)
```

---

## 3. Alert System

The `AlertManager` evaluates configurable rules and fires alerts with severity levels.

### Alert Severity Levels

| Level | When |
|-------|------|
| `CRITICAL` | Severe drift, system failures |
| `WARNING` | Degraded accuracy, latency issues |
| `INFO` | Informational (non-actionable) |

### Creating Alert Rules

```python
from src.domain.monitoring.services.alert_manager import (
    AlertManager, AlertRule, AlertSeverity
)

manager = AlertManager()

# Fire when drift score exceeds 0.3
manager.register_rule(AlertRule(
    name="high_drift",
    metric="drift_score",
    threshold=0.3,
    severity=AlertSeverity.CRITICAL,
    comparison="gt",           # gt (>), lt (<), gte (>=), lte (<=)
    cooldown_seconds=300,      # Don't re-fire for 5 minutes
    description="Drift score exceeds 0.3",
))

# Fire when accuracy drops below 70%
manager.register_rule(AlertRule(
    name="low_accuracy",
    metric="accuracy",
    threshold=0.7,
    severity=AlertSeverity.WARNING,
    comparison="lt",
    cooldown_seconds=600,
))
```

### Alert Lifecycle

1. Metric value evaluated against all matching rules
2. If threshold breached → alert created with severity
3. Alert stored in active list
4. **Cooldown** prevents duplicate alerts for the same rule
5. Alert can be sent via `AlertNotifier` (Slack, PagerDuty, webhook)

### Notification Channels

Set the `ALERT_WEBHOOK_URL` environment variable to receive Slack-compatible notifications:

```bash
# In .env or docker-compose.yaml
ALERT_WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/xxx
```

---

## 4. Rollback System

The `RollbackManager` automatically archives challenger models that perform poorly.

### Rollback Criteria

| Metric | Threshold | What Happens |
|--------|-----------|-------------|
| Error Rate | > 10% | Challenger → `archived` |
| Avg Latency | > 500ms | Challenger → `archived` |
| Insufficient Data | < 50 requests | No action (wait for more data) |

### Configuration

```python
# src/config/monitoring.py
ROLLBACK_ERROR_RATE_THRESHOLD = 0.10   # 10%
ROLLBACK_LATENCY_THRESHOLD_MS = 500.0  # 500 ms
ROLLBACK_MIN_REQUESTS = 50             # Minimum before evaluation
```

### Manual Rollback

```bash
# Archive all challengers for a model
curl -X POST http://localhost:8001/models/rollback \
  -H "Content-Type: application/json" \
  -d '{"model_id": "credit-risk"}'

# Response:
# {"model_id": "credit-risk", "champion": "v1", "archived_challengers": ["v3"]}
```

### Automatic Rollback (via Airflow)

The self-healing DAG runs rollback as step 2 of 5:

```
send_alert → rollback_challenger → train_model → log_mlflow → register_model
```

---

## 5. Airflow Self-Healing Pipeline

When drift is detected, the platform automatically triggers an Airflow DAG that heals the system.

### Pipeline Tasks

| Step | Task | Action |
|------|------|--------|
| 1 | `send_alert` | Sends Slack/webhook notification |
| 2 | `rollback_challenger` | Archives poor-performing challengers via API |
| 3 | `train_model` | Retrains model using `model_configs/*.yaml` |
| 4 | `log_mlflow` | Logs metrics + params to MLflow |
| 5 | `register_model` | Registers new challenger via API |

### Configuration

- **DAG location**: `dags/retrain_pipeline.py`
- **Schedule**: `None` (triggered by API, not scheduled)
- **Max active runs**: `1` (deduplicates concurrent triggers)
- **Airflow UI**: http://localhost:8080 (admin/admin)

### Manually Trigger

```bash
# Via Airflow API
curl -X POST http://localhost:8080/api/v1/dags/self_healing_pipeline/dagRuns \
  -u admin:admin \
  -H "Content-Type: application/json" \
  -d '{
    "conf": {
      "model_id": "credit-risk",
      "drift_score": 0.9,
      "reason": "Manual trigger for testing"
    }
  }'
```

---

## 6. Prometheus Metrics

Phoenix ML exposes these metrics at `GET /metrics`:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `prediction_count_total` | Counter | model_id, version | Total predictions served |
| `inference_latency_seconds` | Histogram | model_id, version | P50/P95/P99 latency |
| `model_confidence` | Gauge | model_id | Current confidence score |
| `feature_drift_score` | Gauge | model_id, feature | Drift score per feature |
| `drift_detected_events_total` | Counter | model_id | Number of drift events |

### Custom Metrics

Add in `src/infrastructure/monitoring/prometheus_metrics.py`:

```python
from prometheus_client import Counter, Histogram

MY_METRIC = Counter(
    "my_custom_metric_total",
    "Description of my metric",
    ["label1", "label2"],
)

# Use in code:
MY_METRIC.labels(label1="value", label2="value").inc()
```

---

## 7. Grafana Dashboards

Pre-provisioned dashboard at http://localhost:3001:

- **Throughput (RPS)** per model
- **P99 Latency** over time
- **Feature Drift Scores**

### Customizing

1. Edit dashboard in Grafana UI
2. Export JSON via Grafana → Dashboard Settings → JSON Model
3. Save to `grafana/provisioning/dashboards/`
4. Restart: `docker compose restart grafana`

---

## 8. Performance Monitoring

Real-time performance data via API:

```bash
curl http://localhost:8001/monitoring/performance/credit-risk

# {
#   "model_id": "credit-risk",
#   "total_predictions": 204,
#   "metrics": {
#     "avg_latency_ms": 2.2,
#     "avg_confidence": 0.67
#   }
# }
```
