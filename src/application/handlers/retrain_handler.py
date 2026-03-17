import asyncio
import logging
import sys
from pathlib import Path

from src.application.commands.trigger_retrain_command import TriggerRetrainCommand

logger = logging.getLogger(__name__)


class RetrainHandler:
    """
    Application Service that handles model retraining triggers.
    Executes the training script as a separate process.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self._project_root = (
            project_root or Path(__file__).resolve().parent.parent.parent.parent
        )

    async def execute(self, command: TriggerRetrainCommand) -> bool:
        """
        Executes the retraining logic by running scripts/train_model.py.
        """
        logger.info(
            "🚀 [RETRAIN] Starting real retraining for model %s. Reason: %s",
            command.model_id,
            command.reason,
        )

        script_path = self._project_root / "scripts" / "train_model.py"
        if not script_path.exists():
            logger.error("❌ Retraining script not found at %s", script_path)
            return False

        # In a real production system, use a task queue like Celery or a K8s Job.
        try:
            # Prepare the output path for the new model version
            output_model_path = (
                self._project_root / "models" / "credit_risk" / "v1" / "model.onnx"
            )

            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                "--output",
                str(output_model_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(
                    "✅ [RETRAIN] Successfully retrained model %s", command.model_id
                )
                if stdout:
                    logger.debug("Retrain Stdout: %s", stdout.decode())
                return True

            logger.error(
                "❌ [RETRAIN] Retraining failed with code %s", process.returncode
            )
            logger.error("Retrain Stderr: %s", stderr.decode())
            return False

        except Exception as e:
            logger.exception("❌ [RETRAIN] Unexpected error: %s", e)
            return False
