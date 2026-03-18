"""
AlertManager Domain Service.
Evaluates alert rules against monitoring metrics and dispatches
notifications when thresholds are breached.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    FIRING = "firing"
    RESOLVED = "resolved"


@dataclass
class AlertRule:
    name: str
    metric: str
    threshold: float
    severity: AlertSeverity
    comparison: str = "gt"  # "gt", "lt", "gte", "lte"
    cooldown_seconds: float = 300.0
    description: str = ""


@dataclass
class Alert:
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    metric_value: float
    threshold: float
    message: str
    model_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "message": self.message,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat(),
        }


class AlertManager:
    """
    Evaluates alert rules against incoming metric values and fires alerts
    when thresholds are breached with cooldown support.
    """

    def __init__(self) -> None:
        self._rules: list[AlertRule] = []
        self._last_fired: dict[str, float] = {}
        self._active_alerts: list[Alert] = []

    def register_rule(self, rule: AlertRule) -> None:
        self._rules.append(rule)
        logger.info("Registered alert rule: %s", rule.name)

    def evaluate(self, metric_name: str, value: float, model_id: str = "") -> list[Alert]:
        """Evaluate all rules for a given metric and return any firing alerts."""
        fired: list[Alert] = []
        now = datetime.now(UTC).timestamp()

        for rule in self._rules:
            if rule.metric != metric_name:
                continue

            breached = self._check_threshold(value, rule.threshold, rule.comparison)

            if not breached:
                continue

            # Check cooldown
            last = self._last_fired.get(rule.name, 0.0)
            if (now - last) < rule.cooldown_seconds:
                continue

            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                status=AlertStatus.FIRING,
                metric_value=value,
                threshold=rule.threshold,
                message=f"[{rule.severity.value.upper()}] {rule.name}: "
                f"{metric_name}={value:.4f} breached threshold {rule.threshold} "
                f"({rule.comparison}). {rule.description}",
                model_id=model_id,
            )

            self._last_fired[rule.name] = now
            self._active_alerts.append(alert)
            fired.append(alert)
            logger.warning("🚨 Alert FIRED: %s", alert.message)

        return fired

    def get_active_alerts(self) -> list[Alert]:
        return list(self._active_alerts)

    def clear_alerts(self) -> None:
        self._active_alerts.clear()

    @staticmethod
    def _check_threshold(value: float, threshold: float, comparison: str) -> bool:
        if comparison == "gt":
            return value > threshold
        if comparison == "lt":
            return value < threshold
        if comparison == "gte":
            return value >= threshold
        if comparison == "lte":
            return value <= threshold
        return False
