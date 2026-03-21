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
