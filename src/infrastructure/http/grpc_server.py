"""
gRPC Server for Phoenix ML Inference Service.

Provides a high-performance gRPC interface alongside the existing FastAPI
HTTP server. Both servers share the same domain layer and infrastructure.

Uses compiled proto stubs from inference.proto for proper gRPC registration.
"""

import logging
import time
import uuid
from concurrent import futures
from typing import Any

import grpc

from src.application.commands.predict_command import PredictCommand
from src.application.handlers.predict_handler import PredictHandler
from src.domain.feature_store.repositories.feature_store import FeatureStore
from src.domain.inference.services.batch_manager import BatchManager
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.services.inference_service import InferenceService
from src.domain.inference.services.routing_strategy import SingleModelStrategy
from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.infrastructure.http.proto import inference_pb2, inference_pb2_grpc

logger = logging.getLogger(__name__)


# ── Fallback classes (used when proto stubs are unavailable, e.g. tests) ──


class PredictRequest:
    """Mirrors proto PredictRequest."""

    def __init__(
        self,
        model_id: str = "",
        model_version: str = "",
        entity_id: str = "",
        features: list[float] | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.entity_id = entity_id
        self.features = features or []


class PredictResponse:
    """Mirrors proto PredictResponse."""

    def __init__(self) -> None:
        self.prediction_id: str = ""
        self.model_id: str = ""
        self.version: str = ""
        self.result: list[float] = []
        self.confidence: float = 0.0
        self.latency_ms: float = 0.0


class HealthCheckResponse:
    """Mirrors proto HealthCheckResponse."""

    SERVING = 1
    NOT_SERVING = 2

    def __init__(self) -> None:
        self.status: int = self.SERVING


# ── Logging Interceptor ──────────────────────────────────────────────


class LoggingInterceptor(grpc.aio.ServerInterceptor):
    """gRPC server interceptor that logs request metadata and latency."""

    async def intercept_service(
        self,
        continuation: Any,
        handler_call_details: grpc.HandlerCallDetails,
    ) -> Any:
        method = handler_call_details.method
        start = time.perf_counter()
        logger.info("gRPC request: %s", method)

        handler = await continuation(handler_call_details)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("gRPC response: %s (%.2fms)", method, elapsed_ms)
        return handler


# ── InferenceServicer ────────────────────────────────────────────────


class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):
    """
    gRPC service implementation for the Phoenix Inference API.

    Inherits from the generated InferenceServiceServicer stub and wraps
    the same PredictHandler used by the FastAPI server for consistency.
    """

    def __init__(self, predict_handler: PredictHandler) -> None:
        self._handler = predict_handler

    async def Predict(  # noqa: N802
        self, request: Any, context: Any
    ) -> Any:
        """Handle a Predict RPC."""
        try:
            command = PredictCommand(
                model_id=request.model_id,
                model_version=request.model_version or None,
                entity_id=request.entity_id or None,
                features=list(request.features) if request.features else None,
            )
            prediction = await self._handler.execute(command)

            response = inference_pb2.PredictResponse()
            response.prediction_id = str(uuid.uuid4())
            response.model_id = prediction.model_id
            response.version = prediction.model_version
            raw = prediction.result
            if isinstance(raw, list):
                response.result[:] = [float(v) for v in raw]
            else:
                response.result.append(float(raw))
            response.confidence = prediction.confidence.value
            response.latency_ms = round(prediction.latency_ms, 2)
            return response

        except ValueError as e:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()
        except Exception as e:
            logger.exception("gRPC Predict failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return inference_pb2.PredictResponse()

    async def HealthCheck(  # noqa: N802
        self, request: Any, context: Any
    ) -> Any:
        """Handle a HealthCheck RPC."""
        response = inference_pb2.HealthCheckResponse()
        response.status = inference_pb2.HealthCheckResponse.SERVING
        return response


# ── Server Factory ───────────────────────────────────────────────────


def create_grpc_server(  # noqa: PLR0913
    model_repo: ModelRepository,
    inference_engine: InferenceEngine,
    batch_manager: BatchManager,
    feature_store: FeatureStore,
    artifact_storage: ArtifactStorage,
    port: int = 50051,
    max_workers: int = 10,
) -> grpc.aio.Server:
    """
    Create and configure a gRPC async server.

    Returns the server instance (call ``await server.start()`` to run).
    """
    inference_service = InferenceService(
        model_repo=model_repo,
        inference_engine=inference_engine,
        batch_manager=batch_manager,
        feature_store=feature_store,
        artifact_storage=artifact_storage,
        routing_strategy=SingleModelStrategy(),
    )

    handler = PredictHandler(inference_service)
    servicer = InferenceServicer(handler)

    interceptors = [LoggingInterceptor()]
    server = grpc.aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=interceptors,
    )

    # Register the servicer with the generated proto stubs
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{port}")

    logger.info("gRPC server configured on port %d with logging interceptor", port)
    return server
