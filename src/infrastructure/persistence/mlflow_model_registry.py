import mlflow
import mlflow.onnx
from typing import List, Optional
from src.domain.inference.entities.model import Model
from src.domain.model_registry.repositories.model_repository import ModelRepository

class MlflowModelRegistry(ModelRepository):
    """
    Infrastructure implementation of ModelRepository using MLflow.
    Used for experiment tracking and artifact versioning.
    """

    def __init__(self, tracking_uri: str, experiment_name: str = "phoenix-models"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    async def save(self, model: Model) -> None:
        """Logs the model artifact to MLflow registry."""
        with mlflow.start_run(run_name=f"load-{model.id}-{model.version}"):
            mlflow.log_params(model.metadata or {})
            # We assume URI is local for now, MLflow will upload it
            local_path = model.uri.replace("local://", "")
            mlflow.onnx.log_model(
                onnx_model=None, # In real case, we load it or use path
                artifact_path="model",
                registered_model_name=model.id
            )
            mlflow.log_metric("accuracy", model.metadata.get("metrics", {}).get("accuracy", 0.0))

    async def get_by_id(self, model_id: str, version: str) -> Optional[Model]:
        # Implementation to fetch from MLflow Model Registry
        pass

    async def get_active_versions(self, model_id: str) -> List[Model]:
        pass

    async def get_champion(self, model_id: str) -> Optional[Model]:
        pass

    async def update_stage(self, model_id: str, version: str, stage: str) -> None:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_id,
            version=version,
            stage=stage.capitalize()
        )
