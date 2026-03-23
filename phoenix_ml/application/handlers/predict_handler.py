import time

from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.domain.inference.entities.prediction import Prediction
from phoenix_ml.domain.inference.services.inference_service import (
    InferenceService,
    PredictionRequest,
)
from phoenix_ml.domain.shared.domain_events import PredictionCompleted
from phoenix_ml.domain.shared.event_bus import DomainEventBus


class PredictHandler:
    """Application service for prediction requests.

    Delegates to ``InferenceService`` and emits ``PredictionCompleted``
    events via the domain event bus.
    """

    def __init__(
        self,
        inference_service: InferenceService,
        event_bus: DomainEventBus,
    ) -> None:
        self._inference_service = inference_service
        self._event_bus = event_bus

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

            # Emit domain event — subscribers handle metrics, logging, Kafka, etc.
            self._event_bus.publish(
                PredictionCompleted(
                    model_id=prediction.model_id,
                    version=prediction.model_version,
                    latency=time.time() - start_time,
                    confidence=prediction.confidence.value,
                    status="success",
                )
            )

            return prediction

        except Exception as e:
            self._event_bus.publish(
                PredictionCompleted(
                    model_id=command.model_id,
                    version=command.model_version or "unknown",
                    latency=time.time() - start_time,
                    confidence=0.0,
                    status="error",
                )
            )
            raise e
