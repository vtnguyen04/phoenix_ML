# Phoenix ML Platform — Detailed Project Structure

> Documentation describing **all directories and files** in the project (excluding `__pycache__`, `data/`, `models/`, `.venv/`, `.git/`).
> Architecture: **Layered modular architecture** with clear separation of concerns

---

## 📁 High-Level Structure

```
phoenix_ML/
├── phoenix_ml/                    ← Backend source code (Python)
│   ├── config/             ← App settings (reads from .env)
│   ├── domain/             ← Pure business logic (no framework dependencies)
│   ├── application/        ← Use cases, commands, handlers (CQRS)
│   ├── infrastructure/     ← Adapters: DB, Kafka, HTTP, gRPC, ML engines
│   └── shared/             ← Utilities, exceptions, data ingestion
├── frontend/               ← React TypeScript dashboard
├── tests/                  ← Unit / Integration / E2E tests
├── dags/                   ← Airflow DAGs
├── examples/               ← Training scripts per model
├── model_configs/          ← YAML config per model
├── scripts/                ← Utility scripts
├── benchmarks/             ← Performance benchmarks
├── deploy/                 ← Helm charts (Kubernetes)
├── docs/                   ← MkDocs documentation site
├── grafana/                ← Grafana dashboards & provisioning
├── notebooks/              ← Jupyter demo notebooks
├── alembic/                ← Database migrations
├── .github/workflows/      ← CI/CD pipelines
└── [root config files]     ← Docker, pyproject, env, etc.
```

---

## 🔧 Root — Configuration Files

| File | Purpose |
|------|-----------|
| `pyproject.toml` | Python project config: dependencies, tool settings (ruff, mypy, pytest). Managed by `uv` |
| `Makefile` | Shortcuts for Docker/dev: `make up`, `make down`, `make test`, `make build`, `make clean`, `make ps` |
| `Dockerfile` | Build backend API image: Python 3.13 + FastAPI + ONNX Runtime |
| `Dockerfile.airflow` | Build Airflow worker image: base airflow + project dependencies |
| `Dockerfile.frontend` | Build frontend image: Node.js + Vite dev server |
| `docker-compose.yaml` | **14 Docker services**: api, frontend, postgres, redis, kafka, zookeeper, mlflow, prometheus, grafana, jaeger, kafka-ui, minio |
| `docker-compose.airflow.yaml` | **Airflow services**: webserver, scheduler, init, postgres-airflow |
| `docker-entrypoint.sh` | Docker entrypoint: fix volume permissions → run app under user `phoenix` |
| `.env` | Environment variables: `DATABASE_URL`, `REDIS_URL`, `KAFKA_URL`, `MLFLOW_TRACKING_URI`, ports |
| `.env.example` | Template for `.env` — used when freshly cloned |
| `prometheus.yml` | Prometheus config: scrape interval, target (API `/metrics` endpoint) |
| `mkdocs.yml` | MkDocs site config: navigation, Material theme, plugins (search, mermaid) |
| `alembic.ini` | Alembic config: database URL, migration directory |
| `.gitignore` | Files ignored by Git: `__pycache__`, `.venv`, `node_modules`, `*.pyc`, `models/`, logs, etc. |
| `.dockerignore` | Files ignored by Docker during image build |
| `README.md` | Project overview: features, architecture, tech stack, getting started |
| `QUICKSTART.md` | Quick start guide: clone → setup → run |
| `CONTRIBUTING.md` | Contributing guide: branching, commit convention, CI checks |
| `project.md` | Detailed internal documentation (72KB) — design, architecture decisions |

<h3>CI/CD</h3>

**Path:** `.github/workflows/`

| File | Purpose |
|------|-----------|
| `.github/workflows/ci.yaml` | **CI pipeline**: `ruff check` → `mypy` → `pytest` → build Docker image. Runs on every push/PR |
| `.github/workflows/docs.yaml` | **Docs pipeline**: auto build + deploy MkDocs to GitHub Pages |

---

## 📂 Backend Source Code

**Path:** `phoenix_ml/`

<h3>`__init__.py`</h3>

**Path:** `phoenix_ml/__init__.py`
Package init.

<h3>`py.typed`</h3>

**Path:** `phoenix_ml/py.typed`
PEP 561 marker — allows mypy to type-check this package when used as library.

---

<h3>Application Configuration</h3>

**Path:** `phoenix_ml/config/`

Reads environment variables from `.env` and returns Pydantic Settings objects.

| File | Purpose |
|------|-----------|
| `__init__.py` | Exports `get_settings()` — merges all settings into a single singleton |
| `app.py` | **AppSettings**: `APP_VERSION`, `DEFAULT_MODEL_ID`, `DEFAULT_MODEL_VERSION`, `MODEL_CONFIG_DIR` |
| `inference.py` | **InferenceSettings**: `BATCH_MAX_SIZE`, `BATCH_MAX_WAIT_MS`, `CACHE_DIR`, `INFERENCE_ENGINE` (onnx/tensorrt/triton) |
| `infrastructure.py` | **InfraSettings**: `DATABASE_URL`, `REDIS_URL`, `KAFKA_URL`, `MLFLOW_TRACKING_URI`, `ARTIFACT_STORAGE_DIR` |
| `monitoring.py` | **MonitoringSettings**: `MONITORING_INTERVAL_SECONDS`, `DRIFT_THRESHOLD`, `USE_REDIS` |

**Usage:**
```python
from phoenix_ml.config import get_settings
settings = get_settings()
print(settings.DATABASE_URL)   # postgresql+asyncpg://...
print(settings.KAFKA_URL)      # kafka:9092
```

---

<h3>Domain Layer (Business Logic)</h3>

**Path:** `phoenix_ml/domain/`

> Pure Python layer, does **NOT import** any framework (FastAPI, SQLAlchemy, Kafka, etc.)
> Contains entities, value objects, domain services, and repository interfaces (ABCs).

<h4>`__init__.py`</h4>

**Path:** `phoenix_ml/domain/__init__.py`
Package init.

---

<h4>Bounded Context: Inference</h4>

**Path:** `phoenix_ml/domain/inference/`

**Responsibility**: Managing models, performing predictions, routing traffic, circuit breaking.

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h5>Entities</h5>

**Path:** `phoenix_ml/domain/inference/entities/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `model.py` | **Model entity**: `id`, `version`, `uri`, `framework`, `stage` (PRODUCTION/STAGING/ARCHIVED/DEVELOPMENT), `metadata`, `is_active`, `created_at`. **ModelStage enum**. Property `unique_key` = `"{id}:{version}"` |
| `prediction.py` | **Prediction entity**: `model_id`, `model_version`, `result`, `confidence` (ConfidenceScore), `latency_ms`. Result of a single inference |

<h5>Value Objects</h5>

**Path:** `phoenix_ml/domain/inference/value_objects/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `confidence_score.py` | **ConfidenceScore**: immutable, validates value ∈ [0.0, 1.0]. Raises ValueError if out of range |
| `feature_vector.py` | **FeatureVector**: wraps numpy `ndarray`, validates dtype float32, property `dimension` |
| `latency_budget.py` | **LatencyBudget**: inference time limit (milliseconds), method `is_exceeded(elapsed)` |
| `model_config.py` | **ModelConfig dataclass**: per-model configuration from YAML — `model_id`, `version`, `feature_names`, `data_loader`, `train_script`, `monitoring_drift_test`, `has_named_features` |
| `model_version.py` | **ModelVersion**: parses semantic version string (v1, v2.1), supports comparison operators |

<h5>Events</h5>

**Path:** `phoenix_ml/domain/inference/events/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `model_loaded.py` | **ModelLoaded event**: `model_id`, `version`, `timestamp` — emitted when a model is loaded into the engine |
| `prediction_made.py` | **PredictionMade event**: `model_id`, `result`, `confidence`, `latency` — emitted after each prediction |

<h5>Services</h5>

**Path:** `phoenix_ml/domain/inference/services/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `inference_engine.py` | **InferenceEngine ABC**: ML engine interface. Methods: `load(model)`, `predict(model, features)`, `batch_predict(model, features_list)`, `optimize(model)` |
| `inference_service.py` | **InferenceService**: main orchestrator — receives request → resolves model (routing) → gets features → runs inference → returns Prediction |
| `batch_manager.py` | **BatchManager** + **BatchConfig**: automatically groups concurrent requests into 1 batch → calls `batch_predict()` once → splits results. Configure `max_batch_size`, `max_wait_time_ms` |
| `routing_strategy.py` | **RoutingStrategy ABC** + 4 implementations: `SingleModelStrategy` (100% champion), `ABTestStrategy` (split by ratio), `CanaryStrategy` (small % to challenger), `ShadowStrategy` (mirror traffic, only returns champion) |
| `circuit_breaker.py` | **CircuitBreaker**: 3 states (CLOSED → OPEN → HALF_OPEN). Automatically stops inference when error rate exceeds threshold, self-recovers after timeout |
| `request_pipeline.py` | **RequestPipeline**: Chain of Responsibility — runs ordered middleware steps (logging, validation, caching) before/after inference |
| `processor_plugin.py` | **IPreprocessor ABC**: transforms raw input → model features. **IPostprocessor ABC**: transforms model output → API response. Built-in: `PassthroughPreprocessor`, `ClassificationPostprocessor` (binary/multi-class) |

---

<h4>Bounded Context: Monitoring</h4>

**Path:** `phoenix_ml/domain/monitoring/`

**Responsibility**: Detecting data drift, evaluating model performance, automatic alerting and rollback.

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h5>Entities</h5>

**Path:** `phoenix_ml/domain/monitoring/entities/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `drift_report.py` | **DriftReport entity**: `model_id`, `feature_name`, `method` (ks/psi/chi2), `score`, `is_drifted`, `threshold`, `timestamp` |

<h5>Repositories</h5>

**Path:** `phoenix_ml/domain/monitoring/repositories/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `drift_report_repository.py` | **DriftReportRepository ABC**: `save(report)`, `get_by_model(model_id, limit)` |
| `prediction_log_repository.py` | **PredictionLogRepository ABC**: `log(command, prediction)`, `get_recent(model_id, limit)`, `update_ground_truth(pred_id, truth)` |

<h5>Services</h5>

**Path:** `phoenix_ml/domain/monitoring/services/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `drift_calculator.py` | **DriftCalculator**: calculates data drift. Methods: `calculate_ks(reference, current)`, `calculate_psi(ref, cur)`, `calculate_chi2(ref, cur)`. Returns `DriftResult(score, is_drifted)` |
| `alert_manager.py` | **AlertManager**: manages alert rules. **AlertRule**: `name`, `metric`, `threshold`, `severity` (INFO/WARNING/CRITICAL), `comparison` (gt/lt), `cooldown_seconds`. **Alert**: result of evaluating one rule |
| `alert_notifier.py` | **IAlertNotifier ABC**: notification sending interface. Method: `notify(alert) → bool` |
| `anomaly_detector.py` | **AnomalyDetector**: detects anomalies in metrics. Methods: `detect_zscore(values, threshold)`, `detect_iqr(values)`. Returns list of anomaly indices |
| `metrics_publisher.py` | **MetricsPublisher ABC**: metrics publishing interface. Methods: `record_prediction()`, `record_latency()`, `record_confidence()`, `publish_model_metrics()`, `publish_drift_score()`, `record_drift_detected()` |
| `model_evaluator.py` | **IModelEvaluator ABC** + `ClassificationEvaluator` (accuracy, f1, precision, recall) + `RegressionEvaluator` (rmse, mae, r2). Factory function `get_evaluator(task_type)` |
| `rollback_manager.py` | **RollbackManager**: automatic model rollback. Logic: check performance → if below threshold → archive challenger → restore champion |

---

<h4>Bounded Context: Training</h4>

**Path:** `phoenix_ml/domain/training/`

**Responsibility**: Training job management, plugin interfaces for data loading and training.

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h5>Entities</h5>

**Path:** `phoenix_ml/domain/training/entities/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `training_config.py` | **TrainingConfig**: training configuration — `model_id`, `data_path`, `hyperparameters`, `epochs`, `batch_size` |
| `training_job.py` | **TrainingJob entity**: `job_id`, `model_id`, `status` (PENDING/RUNNING/COMPLETED/FAILED), `started_at`, `finished_at`, `metrics` |

<h5>Events</h5>

**Path:** `phoenix_ml/domain/training/events/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `training_completed.py` | **TrainingCompleted event**: `model_id`, `version`, `metrics`, `duration` |

<h5>Repositories</h5>

**Path:** `phoenix_ml/domain/training/repositories/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `training_repository.py` | **TrainingRepository ABC**: `save(job)`, `get(job_id)`, `list_by_model(model_id)` |

<h5>Services</h5>

**Path:** `phoenix_ml/domain/training/services/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `training_service.py` | **TrainingService**: orchestrates training — load data → train → evaluate → save model |
| `data_loader_plugin.py` | **IDataLoader ABC**: plugin interface for dataset loading. Methods: `load(data_path)` → `(data, DatasetInfo)`, `split(data, test_size)` → `(train, test)`. **DatasetInfo dataclass**: `num_samples`, `num_features`, `feature_names`, `class_labels`, `data_format` |
| `trainer_plugin.py` | **ITrainer ABC**: plugin interface for training algorithms. Methods: `train(data, config)`, `evaluate(model, test_data)` |
| `hyperparameter_optimizer.py` | **HyperparameterOptimizer**: hyperparameter optimization — grid search, random search |

---

<h4>Bounded Context: Feature Store</h4>

**Path:** `phoenix_ml/domain/feature_store/`

**Responsibility**: Managing features for inference (online) and training (offline).

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h5>Entities</h5>

**Path:** `phoenix_ml/domain/feature_store/entities/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `feature_registry.py` | **FeatureRegistry**: manages metadata — feature name, type, source, lineage |

<h5>Repositories</h5>

**Path:** `phoenix_ml/domain/feature_store/repositories/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `feature_store.py` | **FeatureStore ABC**: online store interface. Methods: `get_online_features(entity_id, feature_names) → list[float]`, `add_features(entity_id, data)` |
| `offline_feature_store.py` | **OfflineFeatureStore ABC**: offline store interface for batch retrieval |

---

<h4>Bounded Context: Model Registry</h4>

**Path:** `phoenix_ml/domain/model_registry/`

**Responsibility**: Managing model artifacts, versioning, and staging.

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h5>Repositories</h5>

**Path:** `phoenix_ml/domain/model_registry/repositories/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `artifact_storage.py` | **ArtifactStorage ABC**: model file storage interface. Methods: `download(remote_uri, local_path)`, `upload(local_path, remote_uri)`, `exists(remote_uri)`, `delete(remote_uri)` |
| `model_repository.py` | **ModelRepository ABC**: model CRUD interface. Methods: `save(model)`, `get_by_id(id, version)`, `get_active_versions(id)`, `get_champion(id)`, `update_stage(id, version, stage)`, `list_all()` |

---

<h4>Shared Domain Kernel</h4>

**Path:** `phoenix_ml/domain/shared/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `domain_events.py` | Shared domain events: **PredictionCompleted** (`model_id`, `version`, `status`, `latency`, `confidence`), **DriftDetected**, **DriftScorePublished**, **ModelRetrained** |
| `event_bus.py` | **DomainEventBus**: Observer pattern — `subscribe(event_type, handler)`, `publish(event)`. Fully decouples modules |
| `plugin_registry.py` | **PluginRegistry**: registers + resolves preprocessor/postprocessor/data_loader plugins by `model_id`. Method `register_model(id, pre, post, loader)`, `resolve(id)` |

<h5>Interfaces</h5>

**Path:** `phoenix_ml/domain/shared/interfaces/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `message_producer.py` | **IMessageProducer ABC**: message queue producer interface. Method: `publish(topic, event)` |

---

<h3>Application Layer (Use Cases)</h3>

**Path:** `phoenix_ml/application/`

> Orchestrates domain objects, implements CQRS pattern (Command/Query separation).

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `decorators.py` | Utility decorators: `@timing` (measure + log execution time), `@retry` (auto-retry on failure with configurable attempts) |

<h4>Commands</h4>

**Path:** `phoenix_ml/application/commands/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `predict_command.py` | **PredictCommand**: input DTO — `model_id`, `model_version` (optional), `entity_id` (optional), `features` (optional list[float]) |
| `batch_predict_command.py` | **BatchPredictCommand**: `model_id`, `batch` (list of feature vectors), `model_version`, `entity_ids` |
| `load_model_command.py` | **LoadModelCommand**: `model_id`, `version` |
| `trigger_retrain_command.py` | **TriggerRetrainCommand**: `model_id`, `reason` |

<h4>Handlers</h4>

**Path:** `phoenix_ml/application/handlers/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `predict_handler.py` | **PredictHandler**: receives PredictCommand → resolves model → calls InferenceService.predict() → publishes PredictionCompleted event → returns Prediction |
| `batch_predict_handler.py` | **BatchPredictHandler**: wraps PredictHandler for batch — iterates through each item, collects results |
| `load_model_handler.py` | **LoadModelHandler**: loads model from registry → loads into inference engine |
| `retrain_handler.py` | **RetrainHandler**: triggers retraining workflow — validates model exists → kicks off training |
| `query_handlers.py` | 3 query handlers (CQRS read side): **GetModelQueryHandler** (queries model info), **GetDriftReportQueryHandler** (retrieves drift reports), **GetModelPerformanceQueryHandler** (computes performance metrics) |

<h4>Dto</h4>

**Path:** `phoenix_ml/application/dto/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `prediction_request.py` | **PredictionRequest DTO**: Pydantic model for HTTP request validation |
| `prediction_response.py` | **PredictionResponse DTO**: Pydantic model for HTTP response serialization |

<h4>Queries</h4>

**Path:** `phoenix_ml/application/queries/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Export query DTOs: **GetModelQuery** (`model_id`, `version`), **GetDriftReportQuery** (`model_id`, `limit`), **GetModelPerformanceQuery** (`model_id`) |

<h4>Services</h4>

**Path:** `phoenix_ml/application/services/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `monitoring_service.py` | **MonitoringService**: orchestrates drift detection flow — retrieves recent prediction logs → computes drift vs reference data → saves report → checks alert rules → fires alerts if needed |

---

<h3>Infrastructure Layer (Adapters)</h3>

**Path:** `phoenix_ml/infrastructure/`

> Implements ABCs from the domain layer. Connects to external systems: DB, Kafka, Redis, ML engines, HTTP, gRPC.

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h4>Dependency Injection & App Lifecycle</h4>

**Path:** `phoenix_ml/infrastructure/bootstrap/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `container.py` | **DI Container**: creates and wires all singletons — `inference_engine` (ONNX/TensorRT/Triton factory), `kafka_producer`, `kafka_consumer`, `feature_store` (Redis/Memory factory), `event_bus`, `metrics_publisher`, `drift_calculator`, `model_evaluator`, `plugin_registry`, `artifact_storage`, `batch_manager`. Function `ensure_model_exists()` auto-generates ONNX models for CI/test |
| `lifespan.py` | **FastAPI Lifespan**: STARTUP: create DB tables → seed all models from `model_configs/` → seed feature store → start gRPC server → start Kafka producer/consumer → start monitoring loop. SHUTDOWN: cancel tasks → stop gRPC/Kafka/batch_manager → dispose DB engine |
| `model_config_loader.py` | **load_all_model_configs(config_dir)**: scans all `.yaml` files in `model_configs/` → parses into `dict[str, ModelConfig]`. Function `load_features_from_metrics(path)`: reads feature names from `metrics.json` |

<h4>ML Inference Engines</h4>

**Path:** `phoenix_ml/infrastructure/ml_engines/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `onnx_engine.py` | **ONNXInferenceEngine** (implement InferenceEngine): production engine using ONNX Runtime. Loads `.onnx` → caches session → CPU inference via `asyncio.to_thread`. Supports sklearn ONNX (class probabilities) + multi-class + regression. **Default engine** |
| `tensorrt_executor.py` | **TensorRTExecutor** (implement InferenceEngine): high-performance GPU inference using ONNX Runtime TensorrtExecutionProvider, FP16 support, CPU fallback |
| `triton_client.py` | **TritonInferenceClient** (implement InferenceEngine): HTTP REST v2 client for NVIDIA Triton Inference Server. Calls `/v2/models/{id}/infer`. Falls back to mock if Triton offline |
| `mock_engine.py` | **MockInferenceEngine** (implement InferenceEngine): mock engine for testing — result = `mean(features)`, confidence = 0.99 |

<h4>Kafka</h4>

**Path:** `phoenix_ml/infrastructure/messaging/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `kafka_producer.py` | **KafkaProducer**: async event publishing to Kafka topics. Uses `AIOKafkaProducer`, JSON serialization. No-op fallback if Kafka offline |
| `kafka_consumer.py` | **KafkaConsumer**: async event consumption from Kafka topics. Uses `AIOKafkaConsumer`, JSON deserialization, per-message error handling, auto-commit offsets. No-op fallback if Kafka offline |

<h4>Feature Store Adapters</h4>

**Path:** `phoenix_ml/infrastructure/feature_store/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `redis_feature_store.py` | **RedisFeatureStore** (implement FeatureStore): production store using Redis. Key format `features:{entity_id}`, uses `HMGET`/`HSET` for hash storage |
| `in_memory_feature_store.py` | **InMemoryFeatureStore** (implement FeatureStore): in-memory dict store for testing/dev |
| `parquet_feature_store.py` | **ParquetFeatureStore** (implement OfflineFeatureStore): reads features from Parquet files for batch processing |

<h4>Dataset Loading</h4>

**Path:** `phoenix_ml/infrastructure/data_loaders/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init — exports `DataLoaderRegistry`, `TabularDataLoader`, `ImageDataLoader`, `resolve_data_loader` |
| `tabular_loader.py` | **TabularDataLoader** (implement IDataLoader): loads CSV/Parquet → pandas DataFrame. `split()` uses sklearn `train_test_split` |
| `image_loader.py` | **ImageDataLoader** (implement IDataLoader): loads NPZ archives or image directories (class_0/, class_1/...). Normalizes 0-255→0-1, flattens for ONNX, stratified split |
| `registry.py` | **DataLoaderRegistry**: registry pattern. `resolve_data_loader(model_id)` → resolves loader based on model config YAML's `data_loader` field. Default: `TabularDataLoader` |

<h4>FastAPI HTTP Server</h4>

**Path:** `phoenix_ml/infrastructure/http/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `fastapi_server.py` | Creates FastAPI app instance, includes routers, CORS middleware config |
| `routes.py` | **Main API Router** — 14 endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`, `POST /feedback`, `GET /models`, `GET /models/{id}`, `POST /models/register`, `POST /models/rollback`, `POST /models/{id}/retrain`, `GET /monitoring/drift/{id}`, `GET /monitoring/reports/{id}`, `GET /monitoring/performance/{id}`. Background task: logs prediction → Postgres + Kafka |
| `data_routes.py` | **Data Router** — 3 endpoints: `POST /data/ingest`, `POST /data/validate`, `POST /data/export-training`. Export training uses SRP helpers: `_fetch_labeled_logs()`, `_load_baseline_data()`, `_build_fresh_dataframe()`, `_merge_datasets()`, `_write_export_csv()` |
| `feature_routes.py` | **Feature Router**: `GET /features/{entity_id}` (gets features), `POST /features/{entity_id}` (adds features) |
| `dependencies.py` | FastAPI Depends: `get_predict_handler()` — injects PredictHandler with InferenceService, event_bus |

<h4>gRPC Server</h4>

**Path:** `phoenix_ml/infrastructure/grpc/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `grpc_server.py` | **InferenceServicer**: gRPC async server. RPCs: `Predict(PredictRequest) → PredictResponse`, `HealthCheck() → HealthCheckResponse`. **LoggingInterceptor**: logs method + latency. Factory `create_grpc_server()`. Port: `50051` |

<h5>Proto</h5>

**Path:** `phoenix_ml/infrastructure/grpc/proto/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `inference.proto` | Protocol Buffers definition: message `PredictRequest`, `PredictResponse`, `HealthCheckRequest`, `HealthCheckResponse`. Service `InferenceService` |
| `inference_pb2.py` | Generated protobuf Python code (from `protoc`) |
| `inference_pb2_grpc.py` | Generated gRPC stubs (servicer base class + client stub) |

<h4>Model Artifact Storage</h4>

**Path:** `phoenix_ml/infrastructure/artifact_storage/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `local_artifact_storage.py` | **LocalArtifactStorage** (implement ArtifactStorage): stores/reads model files on local filesystem |
| `s3_artifact_storage.py` | **S3ArtifactStorage** (implement ArtifactStorage): stores/reads model files on S3/MinIO via boto3. URI format: `s3://bucket/key` |

<h4>Observability Adapters</h4>

**Path:** `phoenix_ml/infrastructure/monitoring/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `prometheus_metrics.py` | Prometheus metric object declarations: `PREDICTION_COUNT` (Counter), `INFERENCE_LATENCY` (Histogram), `MODEL_CONFIDENCE` (Histogram), `DRIFT_SCORE` (Gauge), `DRIFT_DETECTED_COUNT` (Counter), `MODEL_ACCURACY`/`MODEL_F1_SCORE`/`MODEL_RMSE`/`MODEL_MAE`/`MODEL_R2`/`MODEL_PRIMARY_METRIC` (Gauges) |
| `prometheus_metrics_publisher.py` | **PrometheusMetricsPublisher** (implement MetricsPublisher): sets/increments Prometheus metrics. Metric mapping follows OCP — adding a new metric = adding a dict entry |
| `alert_notifier.py` | **AlertNotifier** (implement IAlertNotifier): sends alerts via HTTP webhook. Slack-compatible payload (blocks + emoji). Supports Slack, Discord, generic webhooks |
| `tracing.py` | **init_tracing()**: sets up OpenTelemetry TracerProvider + OTLP exporter sending traces → Jaeger. Functions `get_tracer()`, `shutdown_tracing()` |
| `in_memory_log_repo.py` | **InMemoryPredictionLogRepository** (implement PredictionLogRepository): stores prediction logs in RAM for testing |

<h4>Database Adapters</h4>

**Path:** `phoenix_ml/infrastructure/persistence/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `database.py` | SQLAlchemy async engine + session factory. `get_db()` → `AsyncSession`. `Base` declarative base for ORM |
| `models.py` | ORM models: **ModelORM** (table `models`), **PredictionLogORM** (table `prediction_logs`), **DriftReportORM** (table `drift_reports`) |
| `postgres_model_registry.py` | **PostgresModelRegistry** (implement ModelRepository): CRUD models in PostgreSQL. `update_stage("champion")` auto-demotes old champion → retired |
| `postgres_log_repo.py` | **PostgresPredictionLogRepository** (implement PredictionLogRepository): stores logs in PostgreSQL, queries recent predictions |
| `postgres_drift_repo.py` | **PostgresDriftReportRepository** (implement DriftReportRepository): stores drift reports in PostgreSQL |
| `mlflow_model_registry.py` | **MlflowModelRegistry** (implement ModelRepository): CRUD models via MLflow API. Logs ONNX artifacts, manages stages (Production/Staging/Archived), tracks metrics. Bi-directional mapping between MLflow numeric version ↔ Phoenix semantic version |
| `in_memory_model_repo.py` | **InMemoryModelRepository** (implement ModelRepository): stores models in dict for testing |

---

<h3>Shared Utilities</h3>

**Path:** `phoenix_ml/shared/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |

<h4>Exceptions</h4>

**Path:** `phoenix_ml/shared/exceptions/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Custom exceptions: **ModelNotFoundError**, **InferenceError**, **DriftDetectedError**, **ValidationError**, **ConfigurationError** |

<h4>Data Ingestion Pipeline</h4>

**Path:** `phoenix_ml/shared/ingestion/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `interfaces.py` | **IDataIngestor ABC**: interface — `ingest(source) → list[dict]` |
| `api_ingestor.py` | **APIDataIngestor** (implement IDataIngestor): ingests data from external HTTP APIs via httpx |
| `redis_ingestor.py` | **RedisDataIngestor** (implement IDataIngestor): ingests data from Redis streams/keys |
| `data_collector.py` | **DataCollector**: aggregates data from multiple ingestors, merges and deduplicates |
| `service.py` | **IngestionService**: orchestrates full ingestion pipeline — collect → validate → transform → store |

<h4>Utils</h4>

**Path:** `phoenix_ml/shared/utils/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `model_generator.py` | `generate_simple_onnx(path, n_features)`: generates a mock ONNX model (sklearn LogisticRegression → skl2onnx) for CI/testing. Used when model.onnx does not exist |

---

## 📂 React TypeScript Dashboard

**Path:** `frontend/`

<h3>Root Files</h3>

| File | Purpose |
|------|-----------|
| `index.html` | HTML template — mount point `<div id="root">` |
| `package.json` | NPM dependencies: react, recharts, vite, vitest, @testing-library |
| `vite.config.ts` | Vite dev server config: port 5173, proxy `/api` → backend `VITE_API_TARGET` |
| `tsconfig.json` | TypeScript config root — references app + node configs |
| `tsconfig.app.json` | TypeScript config for app code: strict mode, jsx react-jsx |
| `tsconfig.node.json` | TypeScript config for Node.js files (vite config) |
| `eslint.config.js` | ESLint config: react-hooks rules, typescript-eslint |
| `.gitignore` | Frontend-specific gitignore (dist, node_modules) |
| `README.md` | Frontend README |
| `public/vite.svg` | Vite logo favicon |

<h3>Phoenix Ml</h3>

**Path:** `frontend/phoenix_ml/`

| File | Purpose |
|------|-----------|
| `App.tsx` | **Main App component**: layout (Sidebar + content), state management, API calls on mount, renders all dashboard panels |
| `main.tsx` | Entry point: `ReactDOM.createRoot()` → render `<App />` |
| `index.css` | Global CSS styles: dark theme, gradients, animations, responsive grid |
| `config.ts` | Centralized config: `API_BASE_URL`, `SERVICES` array (11 services with name/port/icon/healthUrl), `ENDPOINTS` mapping, `CHART_CONFIG` |

<h3>Api</h3>

**Path:** `frontend/phoenix_ml/api/`

| File | Purpose |
|------|-----------|
| `mlService.ts` | **MLService class**: fetch wrapper for all API calls — `predict(modelId, features)`, `getModelInfo(modelId)`, `getDriftReport(modelId)`, `getPerformance(modelId)`, `getModels()`, `getPipelineStatus()` |

<h3>Dashboard</h3>

**Path:** `frontend/phoenix_ml/components/dashboard/`

| File | Purpose |
|------|-----------|
| `StatsGrid.tsx` | Grid of 4 stat cards: Accuracy, F1 Score, Avg Latency, Avg Confidence |
| `ModelInfoCard.tsx` | Card displaying model details: ID, version, framework, stage, feature count |
| `ModelComparison.tsx` | Champion vs Challenger comparison table: metrics side-by-side, promote/rollback actions |
| `PredictionPanel.tsx` | Interactive form: select entity → fill features → call `/predict` → display result + confidence + latency |
| `DriftPanel.tsx` | Displays drift detection report: score, method (KS/PSI/Chi2), status badge (OK/DRIFTED) |
| `PerformancePanel.tsx` | Chart displaying performance metrics over time (using Recharts) |
| `PipelineStatus.tsx` | Status of Airflow pipeline tasks: data ingestion, training, evaluation, deployment |
| `ServicesStatus.tsx` | Grid of 11 ServiceCards: displays all infrastructure services (API, DB, Redis, Kafka, MLflow, etc.) |
| `GrafanaEmbed.tsx` | Embeds Grafana dashboard iframe with configurable URL |

<h3>Layout</h3>

**Path:** `frontend/phoenix_ml/components/layout/`

| File | Purpose |
|------|-----------|
| `Sidebar.tsx` | Navigation sidebar: logo + nav links (Dashboard, Models, Monitoring, Pipeline) |

<h3>Ui</h3>

**Path:** `frontend/phoenix_ml/components/ui/`

| File | Purpose |
|------|-----------|
| `ServiceCard.tsx` | Single service card: icon + name + port + live health check (fetches healthUrl every 15s → 🟢/🔴/🟡) |
| `StatCard.tsx` | Single metric card: value + label + icon + trend indicator (up/down) |
| `StatusBadge.tsx` | Badge displaying model stage: champion (green), challenger (yellow), archived (gray) |
| `Spinner.tsx` | CSS loading spinner animation |
| `PredictionResult.tsx` | Panel displaying prediction result: class label + confidence bar + latency + model info |
| `EntitySelector.tsx` | Dropdown to select entity ID for feature lookup |

<h3>Tests (Vitest + React Testing Library)</h3>

**Path:** `frontend/phoenix_ml/test/`

| File | Purpose |
|------|-----------|
| `setup.ts` | Test environment setup (jsdom, cleanup) |
| `App.test.tsx` | Tests that App component renders correctly |
| `api/mlService.test.ts` | Tests MLService: mock fetch, verify API calls |
| `dashboard/StatsGrid.test.tsx` | Test StatsGrid renders 4 stat cards |
| `dashboard/ModelInfoCard.test.tsx` | Test ModelInfoCard displays model info |
| `dashboard/DriftPanel.test.tsx` | Test DriftPanel shows drift status |
| `dashboard/PredictionPanel.test.tsx` | Test PredictionPanel form + submit |
| `dashboard/PipelineStatus.test.tsx` | Test PipelineStatus renders tasks |
| `dashboard/ServicesStatus.test.tsx` | Test ServicesStatus shows 11 services |
| `dashboard/GrafanaEmbed.test.tsx` | Test GrafanaEmbed renders iframe |
| `layout/Sidebar.test.tsx` | Test Sidebar navigation links |
| `ui/ServiceCard.test.tsx` | Test ServiceCard render + health indicator |
| `ui/EntitySelector.test.tsx` | Test EntitySelector dropdown |
| `ui/PredictionResult.test.tsx` | Test PredictionResult display |
| `ui/Spinner.test.tsx` | Test Spinner renders |
| `ui/StatCard.test.tsx` | Test StatCard value + label |
| `ui/StatusBadge.test.tsx` | Test StatusBadge colors |

**Total: 16 test files, 104 tests**

---

## 📂 Backend Tests (pytest)

**Path:** `tests/`

<h3>Root</h3>

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `conftest.py` | Shared fixtures: mock inference engine, sample Model, FeatureVector, async DB session factory |

<h3>End-to-End Tests</h3>

**Path:** `tests/e2e/`

| File | Purpose |
|------|-----------|
| `test_full_flow.py` | Full flow: register model → predict → submit feedback → check drift |

<h3>Integration Tests</h3>

**Path:** `tests/integration/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_api_predict.py` | Tests `/predict` endpoint with real handler |
| `test_api_routes.py` | Tests all API routes: health, models, predict, monitoring |
| `test_circuit_breaker_integration.py` | Test circuit breaker open/close flow |
| `test_feature_store_integration.py` | Test feature store get/add flow |
| `test_full_self_healing_flow.py` | Test self-healing: drift detected → alert → rollback |
| `test_grpc_inference.py` | Test gRPC predict + health check |
| `test_model_registry_integration.py` | Test model registry CRUD |
| `test_monitoring_pipeline.py` | Test monitoring: log predictions → check drift → save report |
| `test_real_inference.py` | Tests inference with real ONNX model |
| `test_real_model_inference.py` | Test end-to-end inference pipeline |

<h3>Application Layer Unit Tests</h3>

**Path:** `tests/unit/application/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_predict_handler.py` | Test PredictHandler: mock service → verify prediction + events |
| `test_predict_handler_routing.py` | Tests PredictHandler with routing strategies (A/B, canary) |
| `test_batch_predict_handler.py` | Test BatchPredictHandler multi-item batching |
| `test_load_model_handler.py` | Test LoadModelHandler loads correct model |
| `test_retrain_handler.py` | Test RetrainHandler trigger flow |
| `test_query_handlers.py` | Test GetModel, GetDriftReport, GetPerformance handlers |
| `test_monitoring_service.py` | Test MonitoringService drift detection + alerting |
| `test_dto_and_config.py` | Test DTOs validation + Settings loading |

<h3>Inference Domain Unit Tests</h3>

**Path:** `tests/unit/domain/inference/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_domain_models.py` | Test Model entity, Prediction entity, value objects |
| `test_events_and_latency.py` | Test domain events + LatencyBudget |
| `test_model_config.py` | Test ModelConfig parsing, validation |
| `test_model_version.py` | Test ModelVersion parsing, comparison |
| `test_routing_strategy.py` | Test all routing strategies: Single, A/B, Canary, Shadow |
| `test_circuit_breaker.py` | Test CircuitBreaker state transitions: CLOSED → OPEN → HALF_OPEN |
| `test_request_pipeline.py` | Test RequestPipeline middleware chain |
| `test_plugin_registry.py` | Test PluginRegistry register/resolve |

<h3>Monitoring Domain Unit Tests</h3>

**Path:** `tests/unit/domain/monitoring/`

| File | Purpose |
|------|-----------|
| `test_drift_calculator.py` | Test drift calculation: KS, PSI, chi2 |
| `test_alert_manager.py` | Test AlertManager: rules, evaluation, cooldown |
| `test_anomaly_detector.py` | Test AnomalyDetector: z-score, IQR |
| `test_model_evaluator.py` | Test ClassificationEvaluator + RegressionEvaluator metrics |
| `test_rollback_manager.py` | Test RollbackManager decision logic |

<h3>Other Domain Tests</h3>

**Path:** `tests/unit/domain/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_inference_service.py` | Test InferenceService orchestration |
| `test_feature_lineage.py` | Test FeatureRegistry lineage tracking |

<h3>Training Domain Unit Tests</h3>

**Path:** `tests/unit/domain/training/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_training_service.py` | Test TrainingService flow: load → train → evaluate → save |
| `test_data_loader_plugin.py` | Test IDataLoader interface + DatasetInfo |

<h3>Infrastructure Unit Tests</h3>

**Path:** `tests/unit/infrastructure/`

| File | Purpose |
|------|-----------|
| `test_onnx_engine.py` | Test ONNXInferenceEngine: load, predict, batch |
| `test_tensorrt_executor.py` | Test TensorRTExecutor |
| `test_triton_client.py` | Test TritonInferenceClient: HTTP calls + mock fallback |
| `test_kafka_producer.py` | Test KafkaProducer: publish + no-op fallback |
| `test_kafka_consumer.py` | Test KafkaConsumer: start/stop + no-op fallback |
| `test_redis_feature_store.py` | Test RedisFeatureStore: HMGET/HSET |
| `test_in_memory_feature_store.py` | Test InMemoryFeatureStore |
| `test_parquet_feature_store.py` | Test ParquetFeatureStore |
| `test_data_loaders.py` | Test TabularDataLoader + ImageDataLoader |
| `test_routes.py` | Test all HTTP routes: predict, models, monitoring |
| `test_feature_routes.py` | Test feature store routes |
| `test_container.py` | Test DI container wiring |
| `test_model_config_loader.py` | Test YAML config loading |
| `test_mlflow_registry.py` | Test MlflowModelRegistry: CRUD + stage management |
| `test_postgres_repos.py` | Test PostgreSQL repositories |
| `test_orm_models.py` | Test ORM model definitions |
| `test_in_memory_model_repo.py` | Test InMemoryModelRepository |
| `test_in_memory_log_repo.py` | Test InMemoryPredictionLogRepository |
| `test_local_artifact_storage.py` | Test LocalArtifactStorage: upload/download |
| `test_s3_artifact_storage.py` | Test S3ArtifactStorage: mock boto3 |
| `test_alert_notifier.py` | Test AlertNotifier: webhook + Slack payload |
| `test_tracing.py` | Test OpenTelemetry tracing setup |
| `test_model_generator.py` | Test generate_simple_onnx() |
| `test_data_ingestion.py` | Test data ingestion pipeline |

<h3>Shared Utils Tests</h3>

**Path:** `tests/unit/shared/`

| File | Purpose |
|------|-----------|
| `__init__.py` | Package init |
| `test_exceptions.py` | Test custom exceptions |
| `test_interfaces.py` | Test IDataIngestor interface |
| `test_api_ingestor.py` | Test APIDataIngestor |
| `test_redis_ingestor.py` | Test RedisDataIngestor |
| `test_data_collector.py` | Test DataCollector aggregation |
| `test_ingestion_service.py` | Test IngestionService pipeline |

**Total backend: 60 test files, ~460+ tests, 87% coverage**

---

## 📂 Model Configuration (YAML)

**Path:** `model_configs/`

| File | Model | Data Loader | Drift Test |
|------|-------|-------------|------------|
| `credit-risk.yaml` | Credit risk detection — GBClassifier, 30 features | `tabular` | `ks` |
| `fraud-detection.yaml` | Fraud detection — RandomForest, 12 features | `tabular` | `psi` |
| `house-price.yaml` | House price prediction — Ridge regression, 8 features | `tabular` | `chi2` |
| `image-class.yaml` | Image classification — MLP, 784 pixel features (28×28), 10 classes | `image` | `ks` |
| `sentiment.yaml` | Sentiment analysis — TF-IDF + LogisticRegression, text features | `text` | `psi` |

---

## 📂 Training Scripts

**Path:** `examples/`

| File | Purpose |
|------|-----------|
| `credit_risk/train.py` | Trains GBClassifier on `data/credit_risk/dataset.csv`. Output: `models/credit_risk/v1/model.onnx` + `metrics.json` |
| `fraud_detection/train.py` | Trains RandomForest classifier on `data/fraud_detection/dataset.csv`. Output: `models/fraud_detection/v1/model.onnx` + `metrics.json` |
| `house_price/train.py` | Trains Ridge regression on `data/house_price/dataset.csv`. Output: `models/house_price/v1/model.onnx` + `metrics.json` |
| `image_classification/train.py` | Trains MLP classifier on `data/image_class/dataset.npz`. Output: `models/image_class/v1/model.onnx` + `metrics.json` |
| `sentiment/train.py` | Trains TF-IDF + LogisticRegression on `data/sentiment/reviews.csv`. Output: `models/sentiment/v1/model.onnx` + `metrics.json` |

---

## 📂 Airflow DAGs

**Path:** `dags/`

| File | Purpose |
|------|-----------|
| `retrain_pipeline.py` | **DAG `phoenix_retrain_all`**: dynamic pipeline retraining all models. Tasks per model: `generate_data` → `train_model` → `log_mlflow` → `promote_model`. Reads config from `model_configs/*.yaml`, resolves training script path, logs metrics to MLflow |

---

## 📂 Utility Scripts

**Path:** `scripts/`

| File | Purpose |
|------|-----------|
| `generate_datasets.py` | Generates synthetic datasets for all models: `credit_risk/dataset.csv`, `fraud_detection/dataset.csv`, `house_price/dataset.csv`, `image_class/dataset.npz`, `sentiment/reviews.csv` |
| `seed_features.py` | Seeds feature store (Redis/Memory) with reference feature records |
| `simulate_pipeline.py` | **Full 8-week lifecycle simulation**: traffic → labels → drift → alert → rollback → export fresh data → retrain → register → verify. Exercises ALL 12 API endpoints |
| `simulate_traffic.py` | Simulates inference traffic: sends random predict requests continuously |
| `simulate_drift.py` | Simulates data drift: sends requests with shifted feature distributions |
| `train_challenger.py` | Trains challenger model version, registers in model registry |
| `run_production_simulation.py` | Full production simulation: traffic + drift + retrain + rollback |
| `test_monitoring_flows.py` | Tests monitoring integration: drift detection + alerting flows |
| `test_real_e2e.py` | End-to-end test with real Docker services running |

---

## 📂 Performance Benchmarks

**Path:** `benchmarks/`

| File | Purpose |
|------|-----------|
| `latency_benchmark.py` | Measures inference latency (p50, p95, p99, mean) |
| `throughput_benchmark.py` | Measures throughput (requests/second) with concurrent clients |
| `memory_benchmark.py` | Measures memory usage during load/predict (RSS, peak) |
| `load_test.py` | Load testing: ramps up concurrent users → measures response times |
| `locustfile.py` | Locust config: defines user behaviors for load testing |
| `benchmark_report.py` | Generates HTML/Markdown benchmark report from results |
| `RESULTS.md` | Benchmark results |

---

## 📂 Kubernetes Helm Chart

**Path:** `deploy/helm/phoenix-ml/`

| File | Purpose |
|------|-----------|
| `Chart.yaml` | Helm chart metadata: name, version, description |
| `values.yaml` | Default values: replica count, image tag, resources (CPU/memory limits), service ports |
| `templates/deployment.yaml` | K8s Deployment: container spec, health probes, env vars from ConfigMap/Secret |
| `templates/service.yaml` | K8s Service: ClusterIP for internal access |
| `templates/ingress.yaml` | K8s Ingress: external access rules, TLS config |
| `templates/hpa.yaml` | HorizontalPodAutoscaler: auto-scales based on CPU/custom metrics |

---

## 📂 Grafana Configuration

**Path:** `grafana/`

| File | Purpose |
|------|-----------|
| `dashboards/phoenix-ml.json` | Main dashboard JSON: panels for inference latency, prediction count, drift score, model accuracy, system health |
| `provisioning/dashboards/provider.yml` | Dashboard auto-provisioning: points Grafana to JSON files |
| `provisioning/dashboards/phoenix_dashboard.json` | Dashboard provisioning config |
| `provisioning/datasources/datasource.yml` | Auto-provision Prometheus datasource: URL `http://prometheus:9090` |

---

## 📂 MkDocs Documentation Site

**Path:** `docs/`

| File | Purpose |
|------|-----------|
| `index.md` | Homepage: project overview, quick links |
| `architecture/system-design.md` | System architecture: components, data flow, deployment diagram |
| `api/reference.md` | API reference: all endpoints with examples |
| `guides/customization.md` | Customization guide: adding new models, custom plugins |
| `guides/library-api.md` | Guide to using as Python library |
| `guides/monitoring.md` | Monitoring setup guide: Prometheus, Grafana, alerts |
| `guides/troubleshooting.md` | FAQ + troubleshooting guide |
| `deployment/docker-stack.md` | Docker deployment guide: compose, volumes, networking |
| `frontend/architecture.md` | Frontend architecture: React components, state, API integration |
| `blog/system-design-overview.md` | Blog: system design decisions |
| `blog/self-healing-ml.md` | Blog: self-healing ML pipeline |
| `blog/performance-optimization.md` | Blog: performance optimization techniques |
| `adr/002-use-onnx-runtime.md` | ADR #2: Why ONNX Runtime for inference |
| `adr/003-use-kafka-for-event-streaming.md` | ADR #3: Why Kafka for event streaming |
| `adr/004-observability-with-prometheus-grafana.md` | ADR #4: Why Prometheus + Grafana |
| `assets/icon.png` | Site icon |
| `assets/logo.png` | Site logo |
| `assets/mlops-pipeline.png` | Pipeline architecture diagram |
| `assets/system-architecture.png` | System architecture diagram |
| `assets/tech-stack.png` | Tech stack diagram |

---

## 📂 Jupyter Demo Notebooks

**Path:** `notebooks/`

| File | Purpose |
|------|-----------|
| `01_inference_demo.ipynb` | Demo: call predict API, visualize results |
| `02_drift_detection_demo.ipynb` | Demo: generate drift, detect, visualize scores |
| `03_self_healing_demo.ipynb` | Demo: full self-healing cycle (drift → alert → retrain → promote) |
| `04_ab_testing_demo.ipynb` | Demo: A/B testing setup, traffic routing, compare models |

---

## 📂 Database Migrations

**Path:** `alembic/`

| File | Purpose |
|------|-----------|
| `env.py` | Alembic environment: async SQLAlchemy engine, auto-detect model changes |
| `script.py.mako` | Template for auto-generated migration files |
| `versions/8e03d4fe1b79_initial_schema_*.py` | **Initial migration**: creates 3 tables — `models` (id, version, uri, framework, stage, metadata, metrics, is_active, created_at), `prediction_logs` (id, model_id, features, result, confidence, latency, ground_truth, timestamp), `drift_reports` (id, model_id, feature_name, method, score, is_drifted, threshold, timestamp) |

---

## 🚀 Running the Project

```bash
# 1. Clone & setup
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML
cp .env.example .env

# 2. Start all services (14 containers)
docker compose up -d
docker compose -f docker-compose.airflow.yaml up -d

# 3. Generate datasets & train models
uv run python scripts/generate_datasets.py
uv run python examples/credit_risk/train.py

# 4. Access services
#   Frontend:     http://localhost:5174
#   API Docs:     http://localhost:8000/docs
#   Airflow:      http://localhost:8080  (admin/admin)
#   MLflow:       http://localhost:5001
#   Grafana:      http://localhost:3001  (admin/admin)
#   Kafka UI:     http://localhost:8082
#   Prometheus:   http://localhost:9090
#   Jaeger:       http://localhost:16686

# 5. Run all tests & lints
uv run ruff check .                                  # Lint
uv run ruff format --check .                         # Format check
uv run mypy phoenix_ml/                                     # Type check
uv run pytest tests/                                 # Backend tests
docker exec phoenix-frontend npx tsc --noEmit        # Frontend type check
docker exec phoenix-frontend npx eslint phoenix_ml/          # Frontend lint
docker exec phoenix-frontend npx vitest run           # Frontend tests
```
