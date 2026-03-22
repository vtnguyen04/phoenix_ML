"""
Batch Prediction Handler — processes multiple predictions efficiently.

Uses the existing PredictHandler internally but collects results
without making separate HTTP calls.
"""

import asyncio
import logging
import time
from typing import Any

from phoenix_ml.application.commands.batch_predict_command import BatchPredictCommand
from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.application.handlers.predict_handler import PredictHandler

logger = logging.getLogger(__name__)


class BatchPredictHandler:
    """Handles batch prediction requests.

    Delegates to PredictHandler for each item, running concurrently
    with asyncio.gather for throughput.
    """

    def __init__(self, predict_handler: PredictHandler) -> None:
        self._predict_handler = predict_handler

    async def handle(self, command: BatchPredictCommand) -> dict[str, Any]:
        """Execute batch predictions.

        Returns:
            Dict with predictions list, total count, and aggregate latency.
        """
        start = time.perf_counter()

        tasks = []
        for i, features in enumerate(command.batch):
            entity_id = (
                command.entity_ids[i]
                if command.entity_ids and i < len(command.entity_ids)
                else None
            )
            cmd = PredictCommand(
                model_id=command.model_id,
                model_version=command.model_version,
                features=features,
                entity_id=entity_id,
            )
            tasks.append(self._predict_handler.execute(cmd))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        predictions = []
        errors = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append({"index": i, "error": str(result)})
            else:
                predictions.append(result)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Batch prediction: %d/%d successful in %.1fms",
            len(predictions),
            len(command.batch),
            elapsed_ms,
        )

        return {
            "predictions": predictions,
            "total": len(command.batch),
            "successful": len(predictions),
            "errors": errors,
            "batch_latency_ms": round(elapsed_ms, 2),
        }
