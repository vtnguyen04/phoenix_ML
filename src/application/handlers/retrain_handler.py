import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.application.commands.trigger_retrain_command import TriggerRetrainCommand
from src.domain.inference.entities.model import Model
from src.domain.model_registry.repositories.model_repository import ModelRepository
from src.domain.monitoring.services.model_evaluator import ModelEvaluator
from src.infrastructure.monitoring.prometheus_metrics import (
    MODEL_ACCURACY,
    MODEL_F1_SCORE,
    MODEL_PRECISION,
    MODEL_RECALL,
)

logger = logging.getLogger(__name__)


class RetrainHandler:
    """
    Application Service that handles model retraining triggers.
    Executes training, evaluates results, and promotes if better.
    """

    def __init__(
        self,
        project_root: Path,
        model_repo: ModelRepository,
        evaluator: ModelEvaluator,
    ) -> None:
        self._project_root = project_root
        self._model_repo = model_repo
        self._evaluator = evaluator

    async def execute(self, command: TriggerRetrainCommand) -> bool:
        """
        Full retraining loop: Train -> Evaluate -> Compare -> Promote.
        """
        logger.info(
            "🚀 [RETRAIN] Starting intelligent retraining for %s",
            command.model_id,
        )

        # 1. Define paths
        version = f"v{int(datetime.now(UTC).timestamp())}"
        model_dir = self._project_root / "models" / command.model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "model.onnx"
        metrics_path = model_dir / "metrics.json"

        # 2. Run Training Script
        script_path = self._project_root / "scripts" / "train_model.py"
        success = await self._run_training(script_path, model_path, metrics_path)
        if not success:
            return False

        # 3. Load Metrics
        with open(metrics_path) as f:
            challenger_metrics = json.load(f)

        # 4. Compare with Champion
        champion = await self._model_repo.get_champion(command.model_id)
        should_promote = True

        if champion and "metrics" in champion.metadata:
            champion_metrics = champion.metadata["metrics"]
            should_promote = self._evaluator.is_better(champion_metrics, challenger_metrics)
            logger.info(
                "📊 Comparison: Challenger F1=%s vs Champion F1=%s",
                challenger_metrics["f1_score"],
                champion_metrics.get("f1_score", 0),
            )

        # 5. Register and potentially Promote
        role = "champion" if should_promote else "challenger"
        new_model = Model(
            id=command.model_id,
            version=version,
            uri=f"local://{model_path}",
            framework="onnx",
            metadata={"metrics": challenger_metrics, "role": role},
            created_at=datetime.now(UTC),
            is_active=True,
        )

        await self._model_repo.save(new_model)

        if should_promote:
            await self._model_repo.update_stage(command.model_id, version, "champion")
            logger.info("👑 Model %s:%s promoted to CHAMPION", command.model_id, version)

        # 6. Update Prometheus
        self._update_prometheus(command.model_id, version, challenger_metrics)

        return True

    async def _run_training(self, script: Path, output: Path, metrics: Path) -> bool:
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script),
                "--output",
                str(output),
                "--metrics",
                str(metrics),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            return process.returncode == 0
        except Exception as e:
            logger.error("❌ Training failed: %s", e)
            return False

    def _update_prometheus(self, model_id: str, version: str, metrics: dict[str, float]) -> None:
        MODEL_ACCURACY.labels(model_id=model_id, version=version).set(metrics["accuracy"])
        MODEL_F1_SCORE.labels(model_id=model_id, version=version).set(metrics["f1_score"])
        MODEL_PRECISION.labels(model_id=model_id, version=version).set(metrics["precision"])
        MODEL_RECALL.labels(model_id=model_id, version=version).set(metrics["recall"])
