"""Webhook-based alert notifier adapter.

Sends JSON alert payloads via HTTP POST to configured webhook URLs.
"""

import logging
from typing import Any

import httpx

from phoenix_ml.domain.monitoring.services.alert_manager import Alert

logger = logging.getLogger(__name__)

HTTP_OK = 200
HTTP_NO_CONTENT = 204


class AlertNotifier:
    """
    Dispatches Alert objects to external webhook endpoints.
    Supports Slack-compatible payloads and generic JSON webhooks.
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._client = httpx.AsyncClient(timeout=5.0)

    async def notify(self, alert: Alert) -> bool:
        """Send an alert notification via webhook. Returns True if successful."""
        if not self._webhook_url:
            logger.info("No webhook URL configured. Alert logged only: %s", alert.message)
            return False

        payload = self._build_payload(alert)

        try:
            resp = await self._client.post(self._webhook_url, json=payload)
            if resp.status_code in (HTTP_OK, HTTP_NO_CONTENT):
                logger.info("✅ Alert sent successfully to webhook: %s", alert.rule_name)
                return True

            logger.warning(
                "⚠️ Webhook returned status %d for alert %s",
                resp.status_code,
                alert.rule_name,
            )
            return False
        except httpx.RequestError as e:
            logger.error("❌ Failed to send alert webhook: %s", e)
            return False

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build a Slack-compatible webhook payload."""
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        emoji = severity_emoji.get(alert.severity.value, "📢")

        return {
            "text": f"{emoji} *{alert.rule_name}* ({alert.severity.value.upper()})",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} *{alert.rule_name}*\n"
                            f"Severity: `{alert.severity.value}`\n"
                            f"Model: `{alert.model_id or 'N/A'}`\n"
                            f"Value: `{alert.metric_value:.4f}` "
                            f"(threshold: `{alert.threshold}`)\n"
                            f"{alert.message}"
                        ),
                    },
                }
            ],
        }

    async def close(self) -> None:
        await self._client.aclose()
