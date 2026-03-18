from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import mlflow
import mlflow.onnx

from src.domain.inference.entities.model import Model, ModelStage
from src.domain.model_registry.repositories.model_repository import ModelRepository


class _RunData(Protocol):
    metrics: dict[str, float]


class _Run(Protocol):
    data: _RunData


class _ModelVersion(Protocol):
    name: str
    version: str  # MLflow's version is numeric string
    source: str
    current_stage: str
    run_id: str | None
    creation_timestamp: int | None  # ms since epoch


class _MlflowClient(Protocol):
    def get_model_version(self, name: str, version: str) -> _ModelVersion: ...

    def get_run(self, run_id: str) -> _Run: ...

    def search_model_versions(self, filter_string: str) -> list[_ModelVersion]: ...

    def transition_model_version_stage(self, *, name: str, version: str, stage: str) -> None: ...


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
        """
        Logs an ONNX artifact into the MLflow Model Registry.

        Notes:
        - MLflow assigns its own numeric model version. Our domain `Model.version`
          (e.g. "v1" or a timestamp) is stored as a tag for traceability.
        """
        client = self._client()
        self._ensure_supported_framework(model.framework)

        local_path = self._require_local_path(model.uri)
        desired_stage = self._map_role_to_mlflow_stage((model.metadata or {}).get("role"))

        with mlflow.start_run(run_name=f"register-{model.id}-{model.version}"):
            mlflow.log_param("phoenix_model_id", model.id)
            mlflow.log_param("phoenix_model_version", model.version)
            mlflow.log_params(model.metadata or {})

            onnx_model = self._load_onnx(local_path)
            mlflow.onnx.log_model(
                onnx_model=onnx_model,
                artifact_path="model",
                registered_model_name=model.id,
            )

            self._log_numeric_metrics((model.metadata or {}).get("metrics"))

        # Best-effort: if caller provided a role, promote the latest registered version.
        if desired_stage is not None:
            latest = self._latest_mlflow_version(client, model.id)
            if latest is not None:
                client.transition_model_version_stage(
                    name=model.id, version=latest.version, stage=desired_stage
                )

    async def get_by_id(self, model_id: str, version: str) -> Model | None:
        """
        Fetch a model version.

        `version` can be either:
        - MLflow numeric version (e.g. "7")
        - Phoenix semantic version/tag (e.g. "v1") stored as `phoenix_model_version`
        """
        client = self._client()

        if version.isdigit():
            mv = self._safe_get_model_version(client, model_id, version)
            return self._to_entity(client, mv) if mv else None

        mv = self._find_by_phoenix_version(client, model_id, version)
        return self._to_entity(client, mv) if mv else None

    async def get_active_versions(self, model_id: str) -> list[Model]:
        client = self._client()
        versions = client.search_model_versions(f"name='{model_id}'")
        active = [mv for mv in versions if mv.current_stage.lower() != "archived"]
        return [self._to_entity(client, mv) for mv in active]

    async def get_champion(self, model_id: str) -> Model | None:
        client = self._client()
        for mv in client.search_model_versions(f"name='{model_id}'"):
            if mv.current_stage.lower() == "production":
                return self._to_entity(client, mv)
        return None

    async def update_stage(self, model_id: str, version: str, stage: str) -> None:
        client = self._client()
        mapped = self._map_role_to_mlflow_stage(stage) or stage.capitalize()
        mv_version = version

        if not mv_version.isdigit():
            mv = self._find_by_phoenix_version(client, model_id, version)
            if mv is None:
                raise ValueError(f"MLflow model version not found for {model_id}:{version}")
            mv_version = mv.version

        client.transition_model_version_stage(name=model_id, version=mv_version, stage=mapped)

    @staticmethod
    def _require_local_path(uri: str) -> Path:
        if uri.startswith("local://"):
            return Path(uri.removeprefix("local://"))
        raise ValueError("MlflowModelRegistry.save expects uri to be local://<path-to-onnx>")

    @staticmethod
    def _map_role_to_mlflow_stage(role: object) -> str | None:
        if not isinstance(role, str):
            return None
        normalized = role.strip().lower()
        return {
            "champion": "Production",
            "production": "Production",
            "prod": "Production",
            "challenger": "Staging",
            "staging": "Staging",
            "stage": "Staging",
            "archived": "Archived",
            "retired": "Archived",
        }.get(normalized)

    @staticmethod
    def _mlflow_stage_to_domain(stage: str) -> ModelStage:
        normalized = stage.strip().lower()
        if normalized == "production":
            return ModelStage.PRODUCTION
        if normalized == "staging":
            return ModelStage.STAGING
        if normalized == "archived":
            return ModelStage.ARCHIVED
        return ModelStage.DEVELOPMENT

    def _to_entity(self, client: _MlflowClient, mv: _ModelVersion) -> Model:
        metrics = self._safe_run_metrics(client, mv.run_id)
        created_at = self._safe_created_at(mv.creation_timestamp)
        stage = mv.current_stage or "None"

        return Model(
            id=mv.name,
            version=mv.version,  # MLflow numeric version
            uri=mv.source,
            framework="onnx",
            stage=self._mlflow_stage_to_domain(stage),
            metadata={
                "metrics": metrics,
                "role": stage.lower(),
                "phoenix_version": self._safe_phoenix_version(client, mv.run_id),
            },
            created_at=created_at,
            is_active=stage.lower() != "archived",
        )

    @staticmethod
    def _client() -> _MlflowClient:
        return mlflow.tracking.MlflowClient()

    @staticmethod
    def _ensure_supported_framework(framework: str) -> None:
        if framework.strip().lower() != "onnx":
            raise ValueError(f"Unsupported framework for MLflow registry: {framework}")

    @staticmethod
    def _load_onnx(path: Path):  # returns onnx.ModelProto
        import onnx  # type: ignore # noqa: PLC0415

        return onnx.load(str(path))

    @staticmethod
    def _log_numeric_metrics(metrics_obj: object) -> None:
        if not isinstance(metrics_obj, dict):
            return
        for k, v in metrics_obj.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(str(k), float(v))

    @staticmethod
    def _safe_get_model_version(
        client: _MlflowClient, model_id: str, version: str
    ) -> _ModelVersion | None:
        try:
            return client.get_model_version(name=model_id, version=version)
        except Exception:
            return None

    @staticmethod
    def _safe_run_metrics(client: _MlflowClient, run_id: str | None) -> dict[str, float]:
        if not run_id:
            return {}
        try:
            run = client.get_run(run_id)
            return dict(run.data.metrics or {})
        except Exception:
            return {}

    @staticmethod
    def _safe_phoenix_version(client: _MlflowClient, run_id: str | None) -> str | None:
        """
        Best-effort: derive phoenix semantic version from run params.
        This stays optional and never breaks core registry operations.
        """
        if not run_id:
            return None
        try:
            run = client.get_run(run_id)
            params = getattr(run.data, "params", None)
            if isinstance(params, dict):
                v = params.get("phoenix_model_version")
                return str(v) if v is not None else None
        except Exception:
            return None
        return None

    @staticmethod
    def _safe_created_at(creation_timestamp_ms: int | None) -> datetime:
        if not creation_timestamp_ms:
            return datetime.now(UTC)
        try:
            return datetime.fromtimestamp(creation_timestamp_ms / 1000, tz=UTC)
        except Exception:
            return datetime.now(UTC)

    @staticmethod
    def _latest_mlflow_version(client: _MlflowClient, model_id: str) -> _ModelVersion | None:
        versions = client.search_model_versions(f"name='{model_id}'")
        numeric = [mv for mv in versions if mv.version.isdigit()]
        if not numeric:
            return None
        return max(numeric, key=lambda mv: int(mv.version))

    @staticmethod
    def _find_by_phoenix_version(
        client: _MlflowClient, model_id: str, phoenix_version: str
    ) -> _ModelVersion | None:
        """
        Locate a model version using the run param `phoenix_model_version`.
        """
        for mv in client.search_model_versions(f"name='{model_id}'"):
            try:
                run = client.get_run(mv.run_id) if mv.run_id else None
            except Exception:
                continue
            if run is None:
                continue
            params = getattr(run.data, "params", None)
            if not isinstance(params, dict):
                continue
            if str(params.get("phoenix_model_version", "")) == phoenix_version:
                return mv
        return None
