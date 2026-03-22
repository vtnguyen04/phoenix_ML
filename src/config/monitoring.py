"""Monitoring, drift, anomaly, rollback, and alert settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class MonitoringSettings(BaseSettings):
    """All monitoring-related thresholds and intervals."""

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Monitoring loop
    MONITORING_INTERVAL_SECONDS: int = 30
    MONITORING_MIN_DATA_POINTS: int = 5

    # Drift detection thresholds
    DRIFT_PSI_NO_DRIFT: float = 0.1
    DRIFT_PSI_MODERATE: float = 0.25
    DRIFT_WASSERSTEIN_FACTOR: float = 0.5
    DRIFT_CRITICAL_THRESHOLD: float = 0.5
    DRIFT_CHI2_BINS: int = 10

    # Anomaly detection
    ANOMALY_EPSILON: float = 1e-10
    ANOMALY_RATIO_THRESHOLD: float = 0.1

    # Rollback
    ROLLBACK_ERROR_RATE_THRESHOLD: float = 0.10
    ROLLBACK_LATENCY_THRESHOLD_MS: float = 500.0
    ROLLBACK_MIN_REQUESTS: int = 50

    # Alert
    ALERT_COOLDOWN_SECONDS: float = 300.0

    # A/B test thresholds (Cohen's effect size)
    AB_TEST_CONFIDENCE_LEVEL: float = 0.95
    AB_TEST_NEGLIGIBLE_EFFECT: float = 0.2
    AB_TEST_LARGE_EFFECT: float = 0.8

    # Data validation
    DATA_VALIDATION_NULL_THRESHOLD: float = 0.3
    DATA_VALIDATION_OUTLIER_MULTIPLIER: float = 1.5
    DATA_VALIDATION_DUPLICATE_THRESHOLD: float = 0.5

    # Input validation
    INPUT_MAX_FEATURES: int = 10_000
    INPUT_MAX_BODY_SIZE_MB: int = 10

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60

    # Audit log
    AUDIT_LOG_FILE: str = "logs/audit.jsonl"
