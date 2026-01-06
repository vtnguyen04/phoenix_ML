# ADR 004: Unified Observability with Prometheus and Grafana

## Status
Accepted

## Context
A critical challenge in MLOps is the detection of **"Silent Failures"**. Unlike traditional software where failures are binary (up/down), ML models can silently degrade in quality (Data Drift) while still returning successful HTTP 200 responses. We need a unified system to monitor:
1.  **System Health**: Latency, CPU/Memory usage, and Throughput.
2.  **Model Health**: Confidence distributions and Prediction outcomes.
3.  **Statistical Health**: Feature distribution shifts over time.

## Decision
We have implemented a comprehensive observability stack based on **Prometheus** (Metrics Aggregator) and **Grafana** (Visualization). 

### Key Design Elements:
*   **Instrumentation**: Custom metrics are exported via the `/metrics` endpoint using the Prometheus ASGI middleware.
*   **Metric Types**:
    *   `prediction_count_total` (Counter): Track throughput across different model versions (v1 vs v2).
    *   `inference_latency_seconds` (Histogram): Track p50, p95, and p99 latencies to ensure SLA compliance.
    *   `feature_drift_score` (Gauge): Real-time output from the `MonitoringService` statistical tests.
*   **Dashboard-as-Code**: Grafana is configured using **Provisioning files**. All dashboards and data sources are stored in Git (`grafana/provisioning/`), ensuring the monitoring environment is reproducible and version-controlled.

## Consequences

### Positive
*   **Sub-second Visibility**: Real-time insight into the behavior of models under production load.
*   **A/B Test Monitoring**: Instant visual comparison between Champion and Challenger models.
*   **Proactive Alerting**: Enables the creation of alerts based on statistical drift thresholds before users notice accuracy drops.
*   **Portability**: The entire stack can be moved from Docker Compose to Kubernetes (Prometheus Operator) with minimal configuration changes.

### Negative
*   **Instrumentation Burden**: Developers must manually add metrics logic to new service handlers.
*   **Storage Management**: High-cardinality metrics (e.g., tracking metrics per unique `entity_id`) can bloat Prometheus storage. We mitigate this by only tracking aggregate-level metrics.

## Alternatives Considered
*   **ELK Stack (Elasticsearch, Logstash, Kibana)**: Rejected as the primary monitoring tool because log-based monitoring is more expensive and slower than metric-based monitoring for real-time latency/throughput tracking.
*   **CloudWatch**: Rejected to avoid cloud vendor lock-in and to maintain a local-first development experience.
