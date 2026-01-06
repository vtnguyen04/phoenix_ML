from prometheus_client import Counter, Gauge, Histogram

# --- INFERENCE METRICS ---
PREDICTION_COUNT = Counter(
    "prediction_count", 
    "Total number of predictions",
    ["model_id", "version", "status"]
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time spent processing inference request",
    ["model_id", "version"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

MODEL_CONFIDENCE = Histogram(
    "model_confidence",
    "Confidence score of predictions",
    ["model_id", "version"],
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0)
)

# --- DRIFT METRICS ---
DRIFT_SCORE = Gauge(
    "feature_drift_score",
    "Drift score (KS/PSI) for a specific feature",
    ["model_id", "feature_name", "method"]
)

DRIFT_DETECTED_COUNT = Counter(
    "drift_detected_events",
    "Number of times significant drift was detected",
    ["model_id", "feature_name"]
)
