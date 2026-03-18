import pytest

from src.domain.monitoring.services.alert_manager import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
)


@pytest.fixture
def alert_manager() -> AlertManager:
    manager = AlertManager()
    manager.register_rule(
        AlertRule(
            name="high_drift",
            metric="drift_score",
            threshold=0.25,
            severity=AlertSeverity.CRITICAL,
            comparison="gt",
            cooldown_seconds=0,  # disable cooldown for tests
            description="Drift exceeds safe threshold",
        )
    )
    manager.register_rule(
        AlertRule(
            name="high_error_rate",
            metric="error_rate",
            threshold=0.10,
            severity=AlertSeverity.WARNING,
            comparison="gt",
            cooldown_seconds=0,
        )
    )
    return manager


def test_alert_fires_when_threshold_breached(alert_manager: AlertManager) -> None:
    alerts = alert_manager.evaluate("drift_score", 0.35, model_id="model-1")
    assert len(alerts) == 1
    assert alerts[0].rule_name == "high_drift"
    assert alerts[0].severity == AlertSeverity.CRITICAL
    assert alerts[0].status == AlertStatus.FIRING


def test_no_alert_when_below_threshold(alert_manager: AlertManager) -> None:
    alerts = alert_manager.evaluate("drift_score", 0.10, model_id="model-1")
    assert len(alerts) == 0


def test_no_alert_for_unrelated_metric(alert_manager: AlertManager) -> None:
    alerts = alert_manager.evaluate("cpu_usage", 99.0, model_id="model-1")
    assert len(alerts) == 0


def test_alert_to_dict(alert_manager: AlertManager) -> None:
    alerts = alert_manager.evaluate("error_rate", 0.15, model_id="model-2")
    assert len(alerts) == 1
    d = alerts[0].to_dict()
    assert d["severity"] == "warning"
    assert d["model_id"] == "model-2"
    assert "timestamp" in d


def test_cooldown_prevents_duplicate_alerts() -> None:
    manager = AlertManager()
    manager.register_rule(
        AlertRule(
            name="cooldown_test",
            metric="latency",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            comparison="gt",
            cooldown_seconds=9999,  # very long cooldown
        )
    )
    first = manager.evaluate("latency", 200.0)
    assert len(first) == 1

    # Should be suppressed by cooldown
    second = manager.evaluate("latency", 200.0)
    assert len(second) == 0


def test_get_active_alerts(alert_manager: AlertManager) -> None:
    alert_manager.evaluate("drift_score", 0.35, model_id="model-1")
    alert_manager.evaluate("error_rate", 0.20, model_id="model-2")
    assert len(alert_manager.get_active_alerts()) == 2

    alert_manager.clear_alerts()
    assert len(alert_manager.get_active_alerts()) == 0
