import asyncio
import time
from dataclasses import dataclass
from typing import Any

from src.domain.inference.entities.model import Model
from src.domain.inference.entities.prediction import Prediction
from src.domain.inference.services.inference_engine import InferenceEngine
from src.domain.inference.value_objects.feature_vector import FeatureVector


@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time_ms: float = 10.0
    enabled: bool = True


class BatchManager:
    """
    Manages dynamic batching of inference requests to optimize throughput.
    Accumulates requests over a short window and executes them as a single batch.
    """

    def __init__(self, engine: InferenceEngine, config: BatchConfig | None = None) -> None:
        self._engine = engine
        self._config = config or BatchConfig()
        self._queues: dict[
            str, asyncio.Queue[tuple[FeatureVector, asyncio.Future[Prediction]]]
        ] = {}
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    async def predict(self, model: Model, features: FeatureVector) -> Prediction:
        """
        Submits a prediction request to the batcher.
        Returns a future that will be resolved once the batch is processed.
        """
        if not self._config.enabled:
            return await self._engine.predict(model, features)

        async with self._lock:
            if model.unique_key not in self._queues:
                self._queues[model.unique_key] = asyncio.Queue()
                self._running_tasks[model.unique_key] = asyncio.create_task(
                    self._batch_worker(model),
                    name=f"batch_worker_{model.unique_key}",
                )

        future: asyncio.Future[Prediction] = asyncio.Future()
        await self._queues[model.unique_key].put((features, future))
        return await future

    async def _batch_worker(self, model: Model) -> None:  # noqa: PLR0912
        """
        Worker loop that pulls requests from the queue and processes them in batches.
        """
        queue = self._queues[model.unique_key]

        try:
            while True:
                batch_items: list[tuple[FeatureVector, asyncio.Future[Prediction]]] = []

                try:
                    item = await queue.get()
                except asyncio.CancelledError:
                    break

                batch_items.append(item)

                deadline = time.time() + (self._config.max_wait_time_ms / 1000.0)

                while len(batch_items) < self._config.max_batch_size:
                    wait_time = deadline - time.time()
                    if wait_time <= 0:
                        break

                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=wait_time)
                        batch_items.append(item)
                    except TimeoutError:
                        break
                    except asyncio.CancelledError:
                        raise

                if batch_items:
                    features_list = [item[0] for item in batch_items]
                    futures = [item[1] for item in batch_items]

                    padded_features_list = self._pad_batch(features_list)

                    try:
                        predictions = await self._engine.batch_predict(model, padded_features_list)

                        predictions = predictions[: len(features_list)]

                        for i, prediction in enumerate(predictions):
                            if not futures[i].done():
                                futures[i].set_result(prediction)

                    except Exception as e:
                        for future in futures:
                            if not future.done():
                                future.set_exception(e)
                    finally:
                        for _ in range(len(batch_items)):
                            queue.task_done()
        except asyncio.CancelledError:
            pass
        finally:
            while not queue.empty():
                _, future = queue.get_nowait()
                if not future.done():
                    future.set_exception(RuntimeError("Batch worker shutting down"))
                queue.task_done()

    async def stop(self) -> None:
        """Cancel all worker tasks and cleanup"""
        try:
            async with self._lock:
                for task in self._running_tasks.values():
                    task.cancel()

                if self._running_tasks:
                    await asyncio.gather(*self._running_tasks.values(), return_exceptions=True)
        except RuntimeError:
            # Event loop may already be closed when a module-level
            # singleton is reused across tests with different loops.
            pass
        finally:
            self._running_tasks.clear()
            self._queues.clear()

    def _pad_batch(self, features_list: list[FeatureVector]) -> list[FeatureVector]:
        """
        Pad batch to the nearest power of 2 for optimal GPU execution.
        """
        current_size = len(features_list)
        if current_size == 0:
            return features_list

        # Find next power of 2
        power = 1
        while power < current_size:
            power *= 2

        if power == current_size:
            return features_list

        # Pad with copies of the last element
        padding_size = power - current_size
        padding = [features_list[-1]] * padding_size
        return features_list + padding
