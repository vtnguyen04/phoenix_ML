"""Tests for AlertNotifier webhook delivery."""

from unittest.mock import AsyncMock, MagicMock

from src.domain.monitoring.services.alert_manager import (
    Alert,
    AlertSeverity,
    AlertStatus,
)
from src.infrastructure.monitoring.alert_notifier import AlertNotifier


async def test_notify_posts_to_webhook() -> None:
    notifier = AlertNotifier(webhook_url="http://test.local/hook")
    alert = Alert(
        rule_name="drift-alarm",
        severity=AlertSeverity.WARNING,
        status=AlertStatus.FIRING,
        metric_value=0.03,
        threshold=0.05,
        message="Drift detected",
        model_id="credit-risk",
    )
    mock_resp = MagicMock(status_code=200)
    notifier._client = MagicMock()
    notifier._client.post = AsyncMock(return_value=mock_resp)

    result = await notifier.notify(alert)
    assert result is True
    notifier._client.post.assert_called_once()


async def test_notify_without_webhook_returns_false() -> None:
    notifier = AlertNotifier(webhook_url=None)
    alert = Alert(
        rule_name="test",
        severity=AlertSeverity.INFO,
        status=AlertStatus.FIRING,
        metric_value=0.0,
        threshold=0.0,
        message="test",
    )
    result = await notifier.notify(alert)
    assert result is False
