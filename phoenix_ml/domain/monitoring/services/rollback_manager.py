"""
RollbackManager Domain Service.
Monitors canary/challenger deployment health and automatically
rolls back to the champion model when error rate exceeds thresholds.
"""

import logging
from dataclasses import dataclass

from phoenix_ml.config import get_settings
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class ChallengerMetrics:
    model_id: str
    challenger_version: str
    champion_version: str
    total_requests: int
    error_count: int
    avg_latency_ms: float


@dataclass
class RollbackDecision:
    should_rollback: bool
    reason: str
    model_id: str
    challenger_version: str
    champion_version: str
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0


class RollbackManager:
    """
    Evaluates challenger model health metrics and decides whether
    to rollback to the champion version.
    """

    def __init__(
        self,
        model_repo: ModelRepository,
        error_rate_threshold: float = _settings.ROLLBACK_ERROR_RATE_THRESHOLD,
        latency_threshold_ms: float = _settings.ROLLBACK_LATENCY_THRESHOLD_MS,
        min_requests: int = _settings.ROLLBACK_MIN_REQUESTS,
    ) -> None:
        self._model_repo = model_repo
        self._error_rate_threshold = error_rate_threshold
        self._latency_threshold_ms = latency_threshold_ms
        self._min_requests = min_requests

    async def evaluate_challenger(self, metrics: ChallengerMetrics) -> RollbackDecision:
        """
        Evaluate challenger deployment health and decide if rollback is needed.
        """
        if metrics.total_requests < self._min_requests:
            return RollbackDecision(
                should_rollback=False,
                reason=(
                    f"Insufficient data: {metrics.total_requests}/{self._min_requests} requests"
                ),
                model_id=metrics.model_id,
                challenger_version=metrics.challenger_version,
                champion_version=metrics.champion_version,
            )

        error_rate = (
            metrics.error_count / metrics.total_requests if metrics.total_requests > 0 else 0.0
        )

        if error_rate > self._error_rate_threshold:
            decision = RollbackDecision(
                should_rollback=True,
                reason=(
                    f"Error rate {error_rate:.2%} exceeds threshold "
                    f"{self._error_rate_threshold:.2%}"
                ),
                model_id=metrics.model_id,
                challenger_version=metrics.challenger_version,
                champion_version=metrics.champion_version,
                error_rate=error_rate,
                avg_latency_ms=metrics.avg_latency_ms,
            )
            logger.warning("🚨 ROLLBACK triggered: %s", decision.reason)
            await self._execute_rollback(
                metrics.model_id,
                metrics.challenger_version,
                metrics.champion_version,
            )
            return decision

        if metrics.avg_latency_ms > self._latency_threshold_ms:
            decision = RollbackDecision(
                should_rollback=True,
                reason=(
                    f"Avg latency {metrics.avg_latency_ms:.1f}ms exceeds "
                    f"threshold {self._latency_threshold_ms:.1f}ms"
                ),
                model_id=metrics.model_id,
                challenger_version=metrics.challenger_version,
                champion_version=metrics.champion_version,
                error_rate=error_rate,
                avg_latency_ms=metrics.avg_latency_ms,
            )
            logger.warning("🚨 ROLLBACK triggered: %s", decision.reason)
            await self._execute_rollback(
                metrics.model_id,
                metrics.challenger_version,
                metrics.champion_version,
            )
            return decision

        return RollbackDecision(
            should_rollback=False,
            reason="Challenger is healthy",
            model_id=metrics.model_id,
            challenger_version=metrics.challenger_version,
            champion_version=metrics.champion_version,
            error_rate=error_rate,
            avg_latency_ms=metrics.avg_latency_ms,
        )

    async def _execute_rollback(
        self, model_id: str, challenger_version: str, champion_version: str
    ) -> None:
        """Demote challenger and ensure champion remains active."""
        try:
            await self._model_repo.update_stage(model_id, challenger_version, "archived")
            logger.info(
                "✅ Rolled back %s: challenger %s → archived, champion %s remains active",
                model_id,
                challenger_version,
                champion_version,
            )
        except Exception as e:
            logger.error("❌ Rollback execution failed: %s", e)
