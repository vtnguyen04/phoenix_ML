import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from phoenix_ml.application.commands.trigger_retrain_command import TriggerRetrainCommand
from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.model_registry.repositories.model_repository import ModelRepository
from phoenix_ml.domain.monitoring.services.model_evaluator import IModelEvaluator
from phoenix_ml.domain.shared.domain_events import ModelRetrained
from phoenix_ml.domain.shared.event_bus import DomainEventBus

logger = logging.getLogger(__name__)


class RetrainHandler:
    """Application service for model retraining.

    Runs the training script, evaluates results against the current
    champion, and promotes the challenger if metrics improve. Emits
    ``ModelRetrained`` events via the domain event bus.
    """

    def __init__(
        self,
        project_root: Path,
        model_repo: ModelRepository,
        evaluator: IModelEvaluator,
        event_bus: DomainEventBus,
    ) -> None:
        self._project_root = project_root
        self._model_repo = model_repo
        self._evaluator = evaluator
        self._event_bus = event_bus

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
            primary = self._evaluator.primary_metric()
            logger.info(
                "📊 Comparison: Challenger %s=%s vs Champion %s=%s",
                primary,
                challenger_metrics.get(primary, "N/A"),
                primary,
                champion_metrics.get(primary, "N/A"),
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

        # 6. Emit domain event (Observer Pattern)
        self._event_bus.publish(
            ModelRetrained(
                model_id=command.model_id,
                version=version,
                metrics=challenger_metrics,
                promoted=should_promote,
            )
        )

        return True

    async def _run_training(self, script: Path, output: Path, metrics: Path) -> bool:
        try:
            logger.info("🏋️ Running training: %s --output %s --metrics %s", script, output, metrics)
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
            if stdout:
                logger.info("🏋️ Training stdout: %s", stdout.decode()[-500:])
            if stderr:
                logger.warning("🏋️ Training stderr: %s", stderr.decode()[-500:])
            if process.returncode != 0:
                logger.error("❌ Training script failed with code %s", process.returncode)
                return False
            logger.info("✅ Training script completed successfully")
            return True
        except Exception as e:
            logger.error("❌ Training failed: %s", e, exc_info=True)
            return False
