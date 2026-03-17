import time

from src.application.commands.predict_command import PredictCommand
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.services.inference_service import (
    InferenceService,
    PredictionRequest,
)
from src.infrastructure.monitoring.prometheus_metrics import (
    INFERENCE_LATENCY,
    MODEL_CONFIDENCE,
    PREDICTION_COUNT,
)


class PredictHandler:
    """
    Application Service that handles prediction commands.
    Delegates orchestration to InferenceService.
    """

    def __init__(self, inference_service: InferenceService) -> None:
        self._inference_service = inference_service

    async def execute(self, command: PredictCommand) -> Prediction:
        start_time = time.time()

        try:
            request = PredictionRequest(
                model_id=command.model_id,
                model_version=command.model_version,
                entity_id=command.entity_id,
                features=command.features,
            )
            prediction = await self._inference_service.predict(request)

            # Record Metrics
            latency = time.time() - start_time
            INFERENCE_LATENCY.labels(
                model_id=prediction.model_id, version=prediction.model_version
            ).observe(latency)

            PREDICTION_COUNT.labels(
                model_id=prediction.model_id,
                version=prediction.model_version,
                status="success",
            ).inc()

            MODEL_CONFIDENCE.labels(
                model_id=prediction.model_id, version=prediction.model_version
            ).observe(prediction.confidence.value)

            return prediction

        except Exception as e:
            PREDICTION_COUNT.labels(
                model_id=command.model_id,
                version=command.model_version or "unknown",
                status="error",
            ).inc()
            raise e
