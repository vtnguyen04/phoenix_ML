# Self-Healing ML: Automated Drift Detection and Recovery

*How Phoenix ML detects model degradation in production and automatically recovers — no human intervention required.*

---

## The Problem: Models Decay in Production

Every ML model deployed to production faces the same reality: **data distributions shift over time**. Customer behavior changes, market conditions evolve, and the patterns your model learned during training gradually become stale.

Traditional approaches require manual monitoring and human-triggered retraining — a process that can take days or weeks, during which the model serves increasingly unreliable predictions.

Phoenix ML solves this with a **self-healing pipeline** that autonomously detects, diagnoses, and recovers from model drift.

## The Self-Healing Architecture

```
Prediction → Log → Drift Detection → Alert → Auto-Retrain → Model Swap
     ↑                                                          |
     └──────────────────────────────────────────────────────────┘
```

### Step 1: Continuous Monitoring

Every prediction is logged with its input features, enabling statistical comparison against the training distribution:

```python
class DriftCalculator:
    def calculate_drift(
        self, feature_name, reference_data, current_data,
        threshold=0.05, test_type="ks"
    ) -> DriftReport:
```

### Step 2: Multi-Method Drift Detection

We employ four complementary statistical tests:

| Method | Best For | How It Works |
|--------|----------|-------------|
| **Kolmogorov-Smirnov** | Continuous features | Compares empirical CDFs; p-value < 0.05 = drift |
| **Population Stability Index** | Distribution shifts | PSI > 0.25 = significant drift |
| **Wasserstein Distance** | Magnitude of shift | Earth mover's distance vs. reference std |
| **Chi-Squared** | Categorical features | Tests independence of observed vs. expected |

Using multiple methods reduces false positives. A single test might flag noise as drift; when two or more agree, confidence is high.

### Step 3: Severity-Based Recommendations

The system generates actionable recommendations based on drift severity:

```
No drift    → "No action needed. Distribution remains stable."
Moderate    → "WARNING: Drift detected in {feature}. Scheduling auto-retraining."
Severe      → "CRITICAL: Severe drift in {feature}. Immediate retraining required."
```

### Step 4: Automatic Retraining

When drift is confirmed, the `RetrainHandler` triggers a new training job:

1. **Create Training Job** — PENDING state with current best hyperparameters
2. **Execute Training** — using recent production data
3. **Evaluate** — compare new model metrics against the current champion
4. **Promote or Rollback** — if the new model is better, swap it in; otherwise, keep the current champion

### Step 5: Safe Model Promotion

The model registry handles the lifecycle:
```
staging → champion (current model retires to "retired")
```

The `RollbackManager` can instantly revert to the previous champion if the new model underperforms in production.

## Circuit Breaker Protection

During retraining, the circuit breaker pattern protects against cascading failures:

```
CLOSED → (failures exceed threshold) → OPEN → (timeout) → HALF_OPEN → (success) → CLOSED
```

When open, all inference requests are routed to a fallback model, ensuring zero downtime.

## Results

In testing with the German Credit dataset:
- **Drift detection latency**: < 100ms for 1000-sample comparison
- **False positive rate**: < 1% with KS + PSI dual confirmation
- **Recovery time**: New model trained and promoted within minutes
- **Zero-downtime**: Circuit breaker ensures 100% availability during transitions

---

*Self-healing ML isn't magic — it's disciplined monitoring, statistical rigor, and automated orchestration.*
