"""Abstract interface for alert notification channels.

Implementations deliver alerts to specific backends (Slack, PagerDuty,
email, webhooks). Configured in ``container.py``.
"""

from abc import ABC, abstractmethod

from phoenix_ml.domain.monitoring.services.alert_manager import Alert


class AlertNotifier(ABC):
    """
    Domain port for sending alert notifications.

    Implementations in infrastructure layer:
        - SlackAlertNotifier
        - PagerDutyAlertNotifier
        - WebhookAlertNotifier
        - LogAlertNotifier (default/development)
    """

    @abstractmethod
    async def notify(self, alert: Alert) -> bool:
        """
        Send an alert notification.

        Returns:
            True if notification was sent successfully, False otherwise.
        """
