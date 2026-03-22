# Phoenix ML Platform — Cấu Trúc Dự Án Chi Tiết

> Tài liệu mô tả **toàn bộ thư mục và file** trong dự án (trừ `__pycache__`, `data/`, `models/`, `.venv/`, `.git/`).
> Kiến trúc: **Domain-Driven Design (DDD)** + Clean Architecture + CQRS

---

## 📁 Cấu trúc tổng quan

```
phoenix_ML/
├── src/                    ← Backend source code (Python)
│   ├── config/             ← App settings (đọc từ .env)
│   ├── domain/             ← Business logic thuần (không phụ thuộc framework)
│   ├── application/        ← Use cases, commands, handlers (CQRS)
│   ├── infrastructure/     ← Adapters: DB, Kafka, HTTP, gRPC, ML engines
│   └── shared/             ← Utilities, exceptions, data ingestion
├── frontend/               ← React TypeScript dashboard
├── tests/                  ← Unit / Integration / E2E tests
├── dags/                   ← Airflow DAGs
├── examples/               ← Training scripts cho từng model
├── model_configs/          ← YAML config cho từng model
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

## 🔧 Root — File cấu hình gốc

| File | Nhiệm vụ |
|------|-----------|
| `pyproject.toml` | Config Python project: dependencies, tool settings (ruff, mypy, pytest). Quản lý bằng `uv` |
| `Makefile` | Shortcuts cho Docker/dev: `make up`, `make down`, `make test`, `make build`, `make clean`, `make ps` |
| `Dockerfile` | Build image backend API: Python 3.13 + FastAPI + ONNX Runtime |
| `Dockerfile.airflow` | Build image Airflow worker: base airflow + project dependencies |
| `Dockerfile.frontend` | Build image frontend: Node.js + Vite dev server |
| `docker-compose.yaml` | **14 Docker services**: api, frontend, postgres, redis, kafka, zookeeper, mlflow, prometheus, grafana, jaeger, kafka-ui, minio |
| `docker-compose.airflow.yaml` | **Airflow services**: webserver, scheduler, init, postgres-airflow |
| `docker-entrypoint.sh` | Docker entrypoint: fix volume permissions → chạy app dưới user `phoenix` |
| `.env` | Biến môi trường thật: `DATABASE_URL`, `REDIS_URL`, `KAFKA_URL`, `MLFLOW_TRACKING_URI`, ports |
| `.env.example` | Template cho `.env` — dùng khi clone mới |
| `prometheus.yml` | Cấu hình Prometheus: scrape interval, target (API `/metrics` endpoint) |
| `mkdocs.yml` | Cấu hình MkDocs site: navigation, Material theme, plugins (search, mermaid) |
| `alembic.ini` | Cấu hình Alembic: database URL, migration directory |
| `dvc.yaml` | Pipeline DVC: `generate_datasets` → `train_<model>` cho từng model type |
| `dvc.lock` | Lock file DVC: hash checksum của data/model artifacts |
| `.dvcignore` | Files DVC bỏ qua |
| `.gitignore` | Files Git bỏ qua: `__pycache__`, `.venv`, `node_modules`, `*.pyc`, `models/`, logs, etc. |
| `.dockerignore` | Files Docker bỏ qua khi build image |
| `README.md` | Tổng quan dự án: features, architecture, tech stack, hướng dẫn chạy |
| `QUICKSTART.md` | Quick start guide: clone → setup → chạy nhanh |
| `CONTRIBUTING.md` | Hướng dẫn contribute: branching, commit convention, CI checks |
| `project.md` | Tài liệu chi tiết nội bộ (72KB) — thiết kế, architecture decisions |
| `PROJECT_STRUCTURE.md` | File này — mô tả cấu trúc dự án |

### `.dvc/` — DVC Config

| File | Nhiệm vụ |
|------|-----------|
| `.dvc/config` | DVC remote storage config |
| `.dvc/.gitignore` | DVC internal gitignore |

### `.github/workflows/` — CI/CD

| File | Nhiệm vụ |
|------|-----------|
| `.github/workflows/ci.yaml` | **CI pipeline**: `ruff check` → `mypy` → `pytest` → build Docker image. Chạy trên mỗi push/PR |
| `.github/workflows/docs.yaml` | **Docs pipeline**: auto build + deploy MkDocs lên GitHub Pages |

---

## 📂 `src/` — Backend Source Code

### `src/__init__.py`
Package init.

### `src/py.typed`
PEP 561 marker — cho phép mypy type-check package này khi dùng như library.

---

### `src/config/` — Cấu hình ứng dụng

Đọc biến môi trường từ `.env`, trả về Pydantic Settings objects.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Export hàm `get_settings()` — merge tất cả settings thành 1 singleton |
| `app.py` | **AppSettings**: `APP_VERSION`, `DEFAULT_MODEL_ID`, `DEFAULT_MODEL_VERSION`, `MODEL_CONFIG_DIR` |
| `inference.py` | **InferenceSettings**: `BATCH_MAX_SIZE`, `BATCH_MAX_WAIT_MS`, `CACHE_DIR`, `INFERENCE_ENGINE` (onnx/tensorrt/triton) |
| `infrastructure.py` | **InfraSettings**: `DATABASE_URL`, `REDIS_URL`, `KAFKA_URL`, `MLFLOW_TRACKING_URI`, `ARTIFACT_STORAGE_DIR` |
| `monitoring.py` | **MonitoringSettings**: `MONITORING_INTERVAL_SECONDS`, `DRIFT_THRESHOLD`, `USE_REDIS` |

**Cách dùng:**
```python
from src.config import get_settings
settings = get_settings()
print(settings.DATABASE_URL)   # postgresql+asyncpg://...
print(settings.KAFKA_URL)      # kafka:9092
```

---

### `src/domain/` — Domain Layer (Business Logic)

> Layer thuần Python, **KHÔNG import** bất kỳ framework nào (FastAPI, SQLAlchemy, Kafka, etc.)
> Chứa entities, value objects, domain services, repository interfaces (ABCs).

#### `src/domain/__init__.py`
Package init.

---

#### `src/domain/inference/` — Bounded Context: Inference (Suy luận ML)

**Nhiệm vụ**: Quản lý models, thực hiện predictions, routing traffic, circuit breaking.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

##### `src/domain/inference/entities/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `model.py` | **Model entity**: `id`, `version`, `uri`, `framework`, `stage` (PRODUCTION/STAGING/ARCHIVED/DEVELOPMENT), `metadata`, `is_active`, `created_at`. **ModelStage enum**. Property `unique_key` = `"{id}:{version}"` |
| `prediction.py` | **Prediction entity**: `model_id`, `model_version`, `result`, `confidence` (ConfidenceScore), `latency_ms`. Kết quả 1 lần inference |

##### `src/domain/inference/value_objects/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `confidence_score.py` | **ConfidenceScore**: immutable, validate giá trị ∈ [0.0, 1.0]. Raise ValueError nếu ngoài range |
| `feature_vector.py` | **FeatureVector**: wrap numpy `ndarray`, validate dtype float32, property `dimension` |
| `latency_budget.py` | **LatencyBudget**: giới hạn thời gian inference (milliseconds), method `is_exceeded(elapsed)` |
| `model_config.py` | **ModelConfig dataclass**: cấu hình 1 model từ YAML — `model_id`, `version`, `feature_names`, `data_loader`, `train_script`, `monitoring_drift_test`, `has_named_features` |
| `model_version.py` | **ModelVersion**: parse semantic version string (v1, v2.1), support comparison operators |

##### `src/domain/inference/events/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `model_loaded.py` | **ModelLoaded event**: `model_id`, `version`, `timestamp` — phát ra khi model được load vào engine |
| `prediction_made.py` | **PredictionMade event**: `model_id`, `result`, `confidence`, `latency` — phát ra sau mỗi prediction |

##### `src/domain/inference/services/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `inference_engine.py` | **InferenceEngine ABC**: interface cho ML engine. Methods: `load(model)`, `predict(model, features)`, `batch_predict(model, features_list)`, `optimize(model)` |
| `inference_service.py` | **InferenceService**: orchestrator chính — nhận request → resolve model (routing) → get features → run inference → trả Prediction |
| `batch_manager.py` | **BatchManager** + **BatchConfig**: tự động gom nhiều concurrent requests thành 1 batch → gọi `batch_predict()` 1 lần → split kết quả. Cấu hình `max_batch_size`, `max_wait_time_ms` |
| `routing_strategy.py` | **RoutingStrategy ABC** + 4 implementations: `SingleModelStrategy` (100% champion), `ABTestStrategy` (split theo ratio), `CanaryStrategy` (% nhỏ cho challenger), `ShadowStrategy` (mirror traffic, chỉ return champion) |
| `circuit_breaker.py` | **CircuitBreaker**: 3 states (CLOSED → OPEN → HALF_OPEN). Tự động ngắt inference khi error rate vượt threshold, tự phục hồi sau timeout |
| `request_pipeline.py` | **RequestPipeline**: Chain of Responsibility — chạy ordered list middleware steps (logging, validation, caching) trước/sau inference |
| `processor_plugin.py` | **IPreprocessor ABC**: transform raw input → model features. **IPostprocessor ABC**: transform model output → API response. Built-in: `PassthroughPreprocessor`, `ClassificationPostprocessor` (binary/multi-class) |

---

#### `src/domain/monitoring/` — Bounded Context: Monitoring

**Nhiệm vụ**: Phát hiện data drift, đánh giá model performance, tự động alert và rollback.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

##### `src/domain/monitoring/entities/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `drift_report.py` | **DriftReport entity**: `model_id`, `feature_name`, `method` (ks/psi/chi2), `score`, `is_drifted`, `threshold`, `timestamp` |

##### `src/domain/monitoring/repositories/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `drift_report_repository.py` | **DriftReportRepository ABC**: `save(report)`, `get_by_model(model_id, limit)` |
| `prediction_log_repository.py` | **PredictionLogRepository ABC**: `log(command, prediction)`, `get_recent(model_id, limit)`, `update_ground_truth(pred_id, truth)` |

##### `src/domain/monitoring/services/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `drift_calculator.py` | **DriftCalculator**: tính data drift. Methods: `calculate_ks(reference, current)`, `calculate_psi(ref, cur)`, `calculate_chi2(ref, cur)`. Trả `DriftResult(score, is_drifted)` |
| `alert_manager.py` | **AlertManager**: quản lý alert rules. **AlertRule**: `name`, `metric`, `threshold`, `severity` (INFO/WARNING/CRITICAL), `comparison` (gt/lt), `cooldown_seconds`. **Alert**: kết quả đánh giá 1 rule |
| `alert_notifier.py` | **IAlertNotifier ABC**: interface gửi notifications. Method: `notify(alert) → bool` |
| `anomaly_detector.py` | **AnomalyDetector**: phát hiện anomaly trong metrics. Methods: `detect_zscore(values, threshold)`, `detect_iqr(values)`. Trả list anomaly indices |
| `metrics_publisher.py` | **MetricsPublisher ABC**: interface publish metrics. Methods: `record_prediction()`, `record_latency()`, `record_confidence()`, `publish_model_metrics()`, `publish_drift_score()`, `record_drift_detected()` |
| `model_evaluator.py` | **IModelEvaluator ABC** + `ClassificationEvaluator` (accuracy, f1, precision, recall) + `RegressionEvaluator` (rmse, mae, r2). Factory function `get_evaluator(task_type)` |
| `rollback_manager.py` | **RollbackManager**: tự động rollback model. Logic: check performance → nếu dưới threshold → archive challenger → restore champion |

---

#### `src/domain/training/` — Bounded Context: Training

**Nhiệm vụ**: Quản lý training jobs, plugin interface cho data loading và training.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

##### `src/domain/training/entities/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `training_config.py` | **TrainingConfig**: cấu hình training — `model_id`, `data_path`, `hyperparameters`, `epochs`, `batch_size` |
| `training_job.py` | **TrainingJob entity**: `job_id`, `model_id`, `status` (PENDING/RUNNING/COMPLETED/FAILED), `started_at`, `finished_at`, `metrics` |

##### `src/domain/training/events/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `training_completed.py` | **TrainingCompleted event**: `model_id`, `version`, `metrics`, `duration` |

##### `src/domain/training/repositories/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `training_repository.py` | **TrainingRepository ABC**: `save(job)`, `get(job_id)`, `list_by_model(model_id)` |

##### `src/domain/training/services/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `training_service.py` | **TrainingService**: orchestrates training — load data → train → evaluate → save model |
| `data_loader_plugin.py` | **IDataLoader ABC**: plugin interface cho dataset loading. Methods: `load(data_path)` → `(data, DatasetInfo)`, `split(data, test_size)` → `(train, test)`. **DatasetInfo dataclass**: `num_samples`, `num_features`, `feature_names`, `class_labels`, `data_format` |
| `trainer_plugin.py` | **ITrainer ABC**: plugin interface cho training algorithms. Methods: `train(data, config)`, `evaluate(model, test_data)` |
| `hyperparameter_optimizer.py` | **HyperparameterOptimizer**: tối ưu hyperparameters — grid search, random search |

---

#### `src/domain/feature_store/` — Bounded Context: Feature Store

**Nhiệm vụ**: Quản lý features cho inference (online) và training (offline).

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

##### `src/domain/feature_store/entities/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `feature_registry.py` | **FeatureRegistry**: quản lý metadata — feature name, type, source, lineage |

##### `src/domain/feature_store/repositories/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `feature_store.py` | **FeatureStore ABC**: online store interface. Methods: `get_online_features(entity_id, feature_names) → list[float]`, `add_features(entity_id, data)` |
| `offline_feature_store.py` | **OfflineFeatureStore ABC**: offline store interface cho batch retrieval |

---

#### `src/domain/model_registry/` — Bounded Context: Model Registry

**Nhiệm vụ**: Quản lý model artifacts, versioning, staging.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

##### `src/domain/model_registry/repositories/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `artifact_storage.py` | **ArtifactStorage ABC**: interface lưu trữ model files. Methods: `download(remote_uri, local_path)`, `upload(local_path, remote_uri)`, `exists(remote_uri)`, `delete(remote_uri)` |
| `model_repository.py` | **ModelRepository ABC**: interface CRUD models. Methods: `save(model)`, `get_by_id(id, version)`, `get_active_versions(id)`, `get_champion(id)`, `update_stage(id, version, stage)`, `list_all()` |

---

#### `src/domain/shared/` — Shared Domain Kernel

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `domain_events.py` | Domain events dùng chung: **PredictionCompleted** (`model_id`, `version`, `status`, `latency`, `confidence`), **DriftDetected**, **DriftScorePublished**, **ModelRetrained** |
| `event_bus.py` | **DomainEventBus**: Observer pattern — `subscribe(event_type, handler)`, `publish(event)`. Decouple modules hoàn toàn |
| `plugin_registry.py` | **PluginRegistry**: đăng ký + resolve preprocessor/postprocessor/data_loader plugins theo `model_id`. Method `register_model(id, pre, post, loader)`, `resolve(id)` |

##### `src/domain/shared/interfaces/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `message_producer.py` | **IMessageProducer ABC**: interface cho message queue producers. Method: `publish(topic, event)` |

---

### `src/application/` — Application Layer (Use Cases)

> Orchestrates domain objects, implements CQRS pattern (Command/Query separation).

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `decorators.py` | Utility decorators: `@timing` (measure + log execution time), `@retry` (auto-retry on failure với configurable attempts) |

#### `src/application/commands/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `predict_command.py` | **PredictCommand**: input DTO — `model_id`, `model_version` (optional), `entity_id` (optional), `features` (optional list[float]) |
| `batch_predict_command.py` | **BatchPredictCommand**: `model_id`, `batch` (list of feature vectors), `model_version`, `entity_ids` |
| `load_model_command.py` | **LoadModelCommand**: `model_id`, `version` |
| `trigger_retrain_command.py` | **TriggerRetrainCommand**: `model_id`, `reason` |

#### `src/application/handlers/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `predict_handler.py` | **PredictHandler**: nhận PredictCommand → resolve model → gọi InferenceService.predict() → publish PredictionCompleted event → trả Prediction |
| `batch_predict_handler.py` | **BatchPredictHandler**: wrap PredictHandler cho batch — lặp qua từng item, collect results |
| `load_model_handler.py` | **LoadModelHandler**: load model từ registry → load vào inference engine |
| `retrain_handler.py` | **RetrainHandler**: trigger retraining workflow — validate model exists → kickoff training |
| `query_handlers.py` | 3 query handlers (CQRS read side): **GetModelQueryHandler** (truy vấn model info), **GetDriftReportQueryHandler** (lấy drift reports), **GetModelPerformanceQueryHandler** (tính performance metrics) |

#### `src/application/dto/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `prediction_request.py` | **PredictionRequest DTO**: Pydantic model cho HTTP request validation |
| `prediction_response.py` | **PredictionResponse DTO**: Pydantic model cho HTTP response serialization |

#### `src/application/queries/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Export query DTOs: **GetModelQuery** (`model_id`, `version`), **GetDriftReportQuery** (`model_id`, `limit`), **GetModelPerformanceQuery** (`model_id`) |

#### `src/application/services/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `monitoring_service.py` | **MonitoringService**: orchestrates drift detection flow — lấy recent prediction logs → tính drift vs reference data → lưu report → check alert rules → fire alerts nếu cần |

---

### `src/infrastructure/` — Infrastructure Layer (Adapters)

> Implement các ABC từ domain layer. Kết nối với external systems: DB, Kafka, Redis, ML engines, HTTP, gRPC.

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

#### `src/infrastructure/bootstrap/` — Dependency Injection & App Lifecycle

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `container.py` | **DI Container**: tạo và wire tất cả singletons — `inference_engine` (ONNX/TensorRT/Triton factory), `kafka_producer`, `kafka_consumer`, `feature_store` (Redis/Memory factory), `event_bus`, `metrics_publisher`, `drift_calculator`, `model_evaluator`, `plugin_registry`, `artifact_storage`, `batch_manager`. Hàm `ensure_model_exists()` tự generate ONNX model cho CI/test |
| `lifespan.py` | **FastAPI Lifespan**: STARTUP: create DB tables → seed all models từ `model_configs/` → seed feature store → start gRPC server → start Kafka producer/consumer → start monitoring loop. SHUTDOWN: cancel tasks → stop gRPC/Kafka/batch_manager → dispose DB engine |
| `model_config_loader.py` | **load_all_model_configs(config_dir)**: scan tất cả `.yaml` files trong `model_configs/` → parse thành `dict[str, ModelConfig]`. Hàm `load_features_from_metrics(path)`: đọc feature names từ `metrics.json` |

#### `src/infrastructure/ml_engines/` — ML Inference Engines

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `onnx_engine.py` | **ONNXInferenceEngine** (implement InferenceEngine): production engine dùng ONNX Runtime. Load `.onnx` → cache session → inference CPU via `asyncio.to_thread`. Hỗ trợ sklearn ONNX (class probabilities) + multi-class + regression. **Engine mặc định** |
| `tensorrt_executor.py` | **TensorRTExecutor** (implement InferenceEngine): high-performance GPU inference, dùng ONNX Runtime TensorrtExecutionProvider, FP16 support, fallback CPU |
| `triton_client.py` | **TritonInferenceClient** (implement InferenceEngine): HTTP REST v2 client cho NVIDIA Triton Inference Server. Gọi `/v2/models/{id}/infer`. Fallback mock nếu Triton offline |
| `mock_engine.py` | **MockInferenceEngine** (implement InferenceEngine): engine giả cho testing — result = `mean(features)`, confidence = 0.99 |

#### `src/infrastructure/messaging/` — Kafka

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `kafka_producer.py` | **KafkaProducer**: async publish events lên Kafka topics. Dùng `AIOKafkaProducer`, JSON serialization. No-op fallback nếu Kafka offline |
| `kafka_consumer.py` | **KafkaConsumer**: async consume events từ Kafka topics. Dùng `AIOKafkaConsumer`, JSON deserialization, per-message error handling, auto-commit offsets. No-op fallback nếu Kafka offline |

#### `src/infrastructure/feature_store/` — Feature Store Adapters

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `redis_feature_store.py` | **RedisFeatureStore** (implement FeatureStore): production store dùng Redis. Key format `features:{entity_id}`, dùng `HMGET`/`HSET` cho hash storage |
| `in_memory_feature_store.py` | **InMemoryFeatureStore** (implement FeatureStore): store trong dict Python cho testing/dev |
| `parquet_feature_store.py` | **ParquetFeatureStore** (implement OfflineFeatureStore): đọc features từ Parquet files cho batch processing |

#### `src/infrastructure/data_loaders/` — Dataset Loading

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init — export `DataLoaderRegistry`, `TabularDataLoader`, `ImageDataLoader`, `resolve_data_loader` |
| `tabular_loader.py` | **TabularDataLoader** (implement IDataLoader): load CSV/Parquet → pandas DataFrame. `split()` dùng sklearn `train_test_split` |
| `image_loader.py` | **ImageDataLoader** (implement IDataLoader): load NPZ archives hoặc image directories (class_0/, class_1/...). Normalization 0-255→0-1, flatten cho ONNX, stratified split |
| `registry.py` | **DataLoaderRegistry**: registry pattern. `resolve_data_loader(model_id)` → tìm loader dựa trên model config yaml's `data_loader` field. Mặc định: `TabularDataLoader` |

#### `src/infrastructure/http/` — FastAPI HTTP Server

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `fastapi_server.py` | Tạo FastAPI app instance, include routers, CORS middleware config |
| `routes.py` | **Main API Router** — 11 endpoints: `GET /health`, `POST /predict`, `POST /predict/batch`, `POST /feedback`, `GET /models`, `GET /models/{id}`, `POST /models/register`, `POST /models/rollback`, `GET /monitoring/drift/{id}`, `GET /monitoring/reports/{id}`, `GET /monitoring/performance/{id}`. Background task: log prediction → Postgres + Kafka |
| `feature_routes.py` | **Feature Router**: `GET /features/{entity_id}` (lấy features), `POST /features/{entity_id}` (thêm features) |
| `dependencies.py` | FastAPI Depends: `get_predict_handler()` — inject PredictHandler với InferenceService, event_bus |

#### `src/infrastructure/grpc/` — gRPC Server

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `grpc_server.py` | **InferenceServicer**: gRPC async server. RPCs: `Predict(PredictRequest) → PredictResponse`, `HealthCheck() → HealthCheckResponse`. **LoggingInterceptor**: log method + latency. Factory `create_grpc_server()`. Port: `50051` |

##### `src/infrastructure/grpc/proto/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `inference.proto` | Protocol Buffers definition: message `PredictRequest`, `PredictResponse`, `HealthCheckRequest`, `HealthCheckResponse`. Service `InferenceService` |
| `inference_pb2.py` | Generated protobuf Python code (from `protoc`) |
| `inference_pb2_grpc.py` | Generated gRPC stubs (servicer base class + client stub) |

#### `src/infrastructure/artifact_storage/` — Model Artifact Storage

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `local_artifact_storage.py` | **LocalArtifactStorage** (implement ArtifactStorage): lưu/đọc model files trên local filesystem |
| `s3_artifact_storage.py` | **S3ArtifactStorage** (implement ArtifactStorage): lưu/đọc model files trên S3/MinIO via boto3. URI format: `s3://bucket/key` |

#### `src/infrastructure/monitoring/` — Observability Adapters

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `prometheus_metrics.py` | Khai báo Prometheus metric objects: `PREDICTION_COUNT` (Counter), `INFERENCE_LATENCY` (Histogram), `MODEL_CONFIDENCE` (Histogram), `DRIFT_SCORE` (Gauge), `DRIFT_DETECTED_COUNT` (Counter), `MODEL_ACCURACY`/`MODEL_F1_SCORE`/`MODEL_RMSE`/`MODEL_MAE`/`MODEL_R2`/`MODEL_PRIMARY_METRIC` (Gauges) |
| `prometheus_metrics_publisher.py` | **PrometheusMetricsPublisher** (implement MetricsPublisher): set/inc Prometheus metrics. Metric mapping OCP pattern — thêm metric mới = thêm entry vào dict |
| `alert_notifier.py` | **AlertNotifier** (implement IAlertNotifier): gửi alerts qua HTTP webhook. Slack-compatible payload (blocks + emoji). Supports Slack, Discord, generic webhooks |
| `tracing.py` | **init_tracing()**: setup OpenTelemetry TracerProvider + OTLP exporter gửi traces → Jaeger. Hàm `get_tracer()`, `shutdown_tracing()` |
| `in_memory_log_repo.py` | **InMemoryPredictionLogRepository** (implement PredictionLogRepository): store prediction logs trong RAM cho testing |

#### `src/infrastructure/persistence/` — Database Adapters

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `database.py` | SQLAlchemy async engine + session factory. `get_db()` → `AsyncSession`. `Base` declarative base cho ORM |
| `models.py` | ORM models: **ModelORM** (table `models`), **PredictionLogORM** (table `prediction_logs`), **DriftReportORM** (table `drift_reports`) |
| `postgres_model_registry.py` | **PostgresModelRegistry** (implement ModelRepository): CRUD models trong PostgreSQL. `update_stage("champion")` tự demote old champion → retired |
| `postgres_log_repo.py` | **PostgresPredictionLogRepository** (implement PredictionLogRepository): lưu logs vào PostgreSQL, query recent predictions |
| `postgres_drift_repo.py` | **PostgresDriftReportRepository** (implement DriftReportRepository): lưu drift reports vào PostgreSQL |
| `mlflow_model_registry.py` | **MlflowModelRegistry** (implement ModelRepository): CRUD models qua MLflow API. Log ONNX artifacts, manage stages (Production/Staging/Archived), track metrics. Bi-directional mapping giữa MLflow numeric version ↔ Phoenix semantic version |
| `in_memory_model_repo.py` | **InMemoryModelRepository** (implement ModelRepository): store models trong dict cho testing |

---

### `src/shared/` — Shared Utilities

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |

#### `src/shared/exceptions/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Custom exceptions: **ModelNotFoundError**, **InferenceError**, **DriftDetectedError**, **ValidationError**, **ConfigurationError** |

#### `src/shared/ingestion/` — Data Ingestion Pipeline

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `interfaces.py` | **IDataIngestor ABC**: interface — `ingest(source) → list[dict]` |
| `api_ingestor.py` | **APIDataIngestor** (implement IDataIngestor): ingest data từ external HTTP APIs via httpx |
| `redis_ingestor.py` | **RedisDataIngestor** (implement IDataIngestor): ingest data từ Redis streams/keys |
| `data_collector.py` | **DataCollector**: aggregate data từ nhiều ingestors, merge và deduplicate |
| `service.py` | **IngestionService**: orchestrates full ingestion pipeline — collect → validate → transform → store |

#### `src/shared/utils/`

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `model_generator.py` | `generate_simple_onnx(path, n_features)`: tạo ONNX model giả (sklearn LogisticRegression → skl2onnx) cho CI/testing. Dùng khi model.onnx chưa tồn tại |

---

## 📂 `frontend/` — React TypeScript Dashboard

### Root Files

| File | Nhiệm vụ |
|------|-----------|
| `index.html` | HTML template — mount point `<div id="root">` |
| `package.json` | NPM dependencies: react, recharts, vite, vitest, @testing-library |
| `vite.config.ts` | Vite dev server config: port 5173, proxy `/api` → backend `VITE_API_TARGET` |
| `tsconfig.json` | TypeScript config root — references app + node configs |
| `tsconfig.app.json` | TypeScript config cho app code: strict mode, jsx react-jsx |
| `tsconfig.node.json` | TypeScript config cho Node.js files (vite config) |
| `eslint.config.js` | ESLint config: react-hooks rules, typescript-eslint |
| `.gitignore` | Frontend-specific gitignore (dist, node_modules) |
| `README.md` | Frontend README |
| `public/vite.svg` | Vite logo favicon |

### `frontend/src/`

| File | Nhiệm vụ |
|------|-----------|
| `App.tsx` | **Main App component**: layout (Sidebar + content), state management, API calls on mount, render all dashboard panels |
| `main.tsx` | Entry point: `ReactDOM.createRoot()` → render `<App />` |
| `index.css` | Global CSS styles: dark theme, gradients, animations, responsive grid |
| `config.ts` | Cấu hình tập trung: `API_BASE_URL`, `SERVICES` array (11 services với name/port/icon/healthUrl), `ENDPOINTS` mapping, `CHART_CONFIG` |

### `frontend/src/api/`

| File | Nhiệm vụ |
|------|-----------|
| `mlService.ts` | **MLService class**: fetch wrapper cho tất cả API calls — `predict(modelId, features)`, `getModelInfo(modelId)`, `getDriftReport(modelId)`, `getPerformance(modelId)`, `getModels()`, `getPipelineStatus()` |

### `frontend/src/components/dashboard/`

| File | Nhiệm vụ |
|------|-----------|
| `StatsGrid.tsx` | Grid 4 stat cards: Accuracy, F1 Score, Avg Latency, Avg Confidence |
| `ModelInfoCard.tsx` | Card hiển thị model details: ID, version, framework, stage, feature count |
| `ModelComparison.tsx` | Bảng so sánh Champion vs Challenger: metrics side-by-side, promote/rollback actions |
| `PredictionPanel.tsx` | Interactive form: chọn entity → điền features → gọi `/predict` → hiển thị result + confidence + latency |
| `DriftPanel.tsx` | Hiển thị drift detection report: score, method (KS/PSI/Chi2), status badge (OK/DRIFTED) |
| `PerformancePanel.tsx` | Chart hiển thị performance metrics over time (dùng Recharts) |
| `PipelineStatus.tsx` | Status của Airflow pipeline tasks: data ingestion, training, evaluation, deployment |
| `ServicesStatus.tsx` | Grid 11 ServiceCards: hiển thị tất cả infrastructure services (API, DB, Redis, Kafka, MLflow, etc.) |
| `GrafanaEmbed.tsx` | Embed Grafana dashboard iframe với configurable URL |

### `frontend/src/components/layout/`

| File | Nhiệm vụ |
|------|-----------|
| `Sidebar.tsx` | Navigation sidebar: logo + nav links (Dashboard, Models, Monitoring, Pipeline) |

### `frontend/src/components/ui/`

| File | Nhiệm vụ |
|------|-----------|
| `ServiceCard.tsx` | Card 1 service: icon + name + port + live health check (fetch healthUrl mỗi 15s → 🟢/🔴/🟡) |
| `StatCard.tsx` | Card 1 metric: value + label + icon + trend indicator (up/down) |
| `StatusBadge.tsx` | Badge con hiển thị model stage: champion (green), challenger (yellow), archived (gray) |
| `Spinner.tsx` | CSS loading spinner animation |
| `PredictionResult.tsx` | Panel hiển thị prediction result: class label + confidence bar + latency + model info |
| `EntitySelector.tsx` | Dropdown chọn entity ID cho feature lookup |

### `frontend/src/test/` — Tests (Vitest + React Testing Library)

| File | Nhiệm vụ |
|------|-----------|
| `setup.ts` | Test environment setup (jsdom, cleanup) |
| `App.test.tsx` | Test App component renders chính xác |
| `api/mlService.test.ts` | Test MLService: mock fetch, verify API calls |
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

**Tổng: 16 test files, 104 tests**

---

## 📂 `tests/` — Backend Tests (pytest)

### Root

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `conftest.py` | Shared fixtures: mock inference engine, sample Model, FeatureVector, async DB session factory |

### `tests/e2e/` — End-to-End Tests

| File | Nhiệm vụ |
|------|-----------|
| `test_full_flow.py` | Full flow: register model → predict → submit feedback → check drift |

### `tests/integration/` — Integration Tests

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `test_api_predict.py` | Test `/predict` endpoint với real handler |
| `test_api_routes.py` | Test tất cả API routes: health, models, predict, monitoring |
| `test_circuit_breaker_integration.py` | Test circuit breaker open/close flow |
| `test_feature_store_integration.py` | Test feature store get/add flow |
| `test_full_self_healing_flow.py` | Test self-healing: drift detected → alert → rollback |
| `test_grpc_inference.py` | Test gRPC predict + health check |
| `test_model_registry_integration.py` | Test model registry CRUD |
| `test_monitoring_pipeline.py` | Test monitoring: log predictions → check drift → save report |
| `test_real_inference.py` | Test inference với ONNX model thật |
| `test_real_model_inference.py` | Test end-to-end inference pipeline |

### `tests/unit/application/` — Application Layer Unit Tests

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `test_predict_handler.py` | Test PredictHandler: mock service → verify prediction + events |
| `test_predict_handler_routing.py` | Test PredictHandler với routing strategies (A/B, canary) |
| `test_batch_predict_handler.py` | Test BatchPredictHandler multi-item batching |
| `test_load_model_handler.py` | Test LoadModelHandler loads correct model |
| `test_retrain_handler.py` | Test RetrainHandler trigger flow |
| `test_query_handlers.py` | Test GetModel, GetDriftReport, GetPerformance handlers |
| `test_monitoring_service.py` | Test MonitoringService drift detection + alerting |
| `test_dto_and_config.py` | Test DTOs validation + Settings loading |

### `tests/unit/domain/inference/` — Inference Domain Unit Tests

| File | Nhiệm vụ |
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

### `tests/unit/domain/monitoring/` — Monitoring Domain Unit Tests

| File | Nhiệm vụ |
|------|-----------|
| `test_drift_calculator.py` | Test drift calculation: KS, PSI, chi2 |
| `test_alert_manager.py` | Test AlertManager: rules, evaluation, cooldown |
| `test_anomaly_detector.py` | Test AnomalyDetector: z-score, IQR |
| `test_model_evaluator.py` | Test ClassificationEvaluator + RegressionEvaluator metrics |
| `test_rollback_manager.py` | Test RollbackManager decision logic |

### `tests/unit/domain/` — Other Domain Tests

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `test_inference_service.py` | Test InferenceService orchestration |
| `test_feature_lineage.py` | Test FeatureRegistry lineage tracking |

### `tests/unit/domain/training/` — Training Domain Unit Tests

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `test_training_service.py` | Test TrainingService flow: load → train → evaluate → save |
| `test_data_loader_plugin.py` | Test IDataLoader interface + DatasetInfo |

### `tests/unit/infrastructure/` — Infrastructure Unit Tests

| File | Nhiệm vụ |
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

### `tests/unit/shared/` — Shared Utils Tests

| File | Nhiệm vụ |
|------|-----------|
| `__init__.py` | Package init |
| `test_exceptions.py` | Test custom exceptions |
| `test_interfaces.py` | Test IDataIngestor interface |
| `test_api_ingestor.py` | Test APIDataIngestor |
| `test_redis_ingestor.py` | Test RedisDataIngestor |
| `test_data_collector.py` | Test DataCollector aggregation |
| `test_ingestion_service.py` | Test IngestionService pipeline |

**Tổng backend: 60 test files, ~460+ tests, 87% coverage**

---

## 📂 `model_configs/` — Model Configuration (YAML)

| File | Model | Data Loader | Drift Test |
|------|-------|-------------|------------|
| `credit-risk.yaml` | Phát hiện rủi ro tín dụng — XGBoost, features: income, age, credit_score, etc. | `tabular` | `ks` |
| `fraud-detection.yaml` | Phát hiện gian lận — RandomForest, features: amount, time, v1-v28, etc. | `tabular` | `psi` |
| `house-price.yaml` | Dự đoán giá nhà — Regression, features: sqft, bedrooms, bathrooms, etc. | `tabular` | `chi2` |
| `image-class.yaml` | Phân loại ảnh — MLP, 784 pixel features (28×28), 10 classes | `image` | `ks` |

---

## 📂 `examples/` — Training Scripts

| File | Nhiệm vụ |
|------|-----------|
| `credit_risk/train.py` | Train XGBoost binary classifier trên `data/credit_risk/dataset.csv`. Output: `models/credit_risk/v1/model.onnx` + `metrics.json` |
| `fraud_detection/train.py` | Train RandomForest classifier trên `data/fraud_detection/dataset.csv`. Output: `models/fraud_detection/v1/model.onnx` + `metrics.json` |
| `house_price/train.py` | Train regression model trên `data/house_price/dataset.csv`. Output: `models/house_price/v1/model.onnx` + `metrics.json` |
| `image_classification/train.py` | Train MLP classifier trên `data/image_class/dataset.npz`. Output: `models/image_class/v1/model.onnx` + `metrics.json` |

---

## 📂 `dags/` — Airflow DAGs

| File | Nhiệm vụ |
|------|-----------|
| `retrain_pipeline.py` | **DAG `phoenix_retrain_all`**: dynamic pipeline retrain tất cả models. Tasks per model: `generate_data` → `train_model` → `log_mlflow` → `promote_model`. Đọc config từ `model_configs/*.yaml`, resolve training script path, log metrics to MLflow |

---

## 📂 `scripts/` — Utility Scripts

| File | Nhiệm vụ |
|------|-----------|
| `generate_datasets.py` | Generate synthetic datasets cho tất cả models: `credit_risk/dataset.csv`, `fraud_detection/dataset.csv`, `house_price/dataset.csv`, `image_class/dataset.npz` |
| `seed_features.py` | Seed feature store (Redis/Memory) với reference feature records |
| `simulate_traffic.py` | Simulate inference traffic: gửi random predict requests liên tục |
| `simulate_drift.py` | Simulate data drift: gửi requests với shifted feature distributions |
| `train_challenger.py` | Train challenger model version, register vào model registry |
| `run_production_simulation.py` | Full production simulation: traffic + drift + retrain + rollback |
| `test_monitoring_flows.py` | Test monitoring integration: drift detection + alerting flows |
| `test_real_e2e.py` | End-to-end test với Docker services chạy thật |

---

## 📂 `benchmarks/` — Performance Benchmarks

| File | Nhiệm vụ |
|------|-----------|
| `latency_benchmark.py` | Đo inference latency (p50, p95, p99, mean) |
| `throughput_benchmark.py` | Đo throughput (requests/second) với concurrent clients |
| `memory_benchmark.py` | Đo memory usage khi load/predict (RSS, peak) |
| `load_test.py` | Load testing: ramp up concurrent users → measure response times |
| `locustfile.py` | Locust config: define user behaviors cho load testing |
| `benchmark_report.py` | Generate HTML/Markdown benchmark report từ results |
| `RESULTS.md` | Kết quả benchmark đã chạy |

---

## 📂 `deploy/helm/phoenix-ml/` — Kubernetes Helm Chart

| File | Nhiệm vụ |
|------|-----------|
| `Chart.yaml` | Helm chart metadata: name, version, description |
| `values.yaml` | Default values: replica count, image tag, resources (CPU/memory limits), service ports |
| `templates/deployment.yaml` | K8s Deployment: container spec, health probes, env vars from ConfigMap/Secret |
| `templates/service.yaml` | K8s Service: ClusterIP cho internal access |
| `templates/ingress.yaml` | K8s Ingress: external access rules, TLS config |
| `templates/hpa.yaml` | HorizontalPodAutoscaler: auto-scale dựa trên CPU/custom metrics |

---

## 📂 `grafana/` — Grafana Configuration

| File | Nhiệm vụ |
|------|-----------|
| `dashboards/phoenix-ml.json` | Main dashboard JSON: panels cho inference latency, prediction count, drift score, model accuracy, system health |
| `provisioning/dashboards/provider.yml` | Dashboard auto-provisioning: point Grafana tới JSON files |
| `provisioning/dashboards/phoenix_dashboard.json` | Dashboard provisioning config |
| `provisioning/datasources/datasource.yml` | Auto-provision Prometheus datasource: URL `http://prometheus:9090` |

---

## 📂 `docs/` — MkDocs Documentation Site

| File | Nhiệm vụ |
|------|-----------|
| `index.md` | Homepage: project overview, quick links |
| `architecture/system-design.md` | System architecture: components, data flow, deployment diagram |
| `architecture/ddd-overview.md` | DDD architecture: bounded contexts, layers, dependency rules |
| `api/reference.md` | API reference: all endpoints with examples |
| `guides/customization.md` | Hướng dẫn customize: thêm model mới, custom plugins |
| `guides/library-api.md` | Hướng dẫn dùng như Python library |
| `guides/monitoring.md` | Hướng dẫn setup monitoring: Prometheus, Grafana, alerts |
| `guides/troubleshooting.md` | FAQ + troubleshooting guide |
| `deployment/docker-stack.md` | Docker deployment guide: compose, volumes, networking |
| `frontend/architecture.md` | Frontend architecture: React components, state, API integration |
| `blog/system-design-overview.md` | Blog: system design decisions |
| `blog/self-healing-ml.md` | Blog: self-healing ML pipeline |
| `blog/performance-optimization.md` | Blog: performance optimization techniques |
| `adr/001-use-ddd-architecture.md` | ADR #1: Why DDD + Clean Architecture |
| `adr/002-use-onnx-runtime.md` | ADR #2: Why ONNX Runtime for inference |
| `adr/003-use-kafka-for-event-streaming.md` | ADR #3: Why Kafka for event streaming |
| `adr/004-observability-with-prometheus-grafana.md` | ADR #4: Why Prometheus + Grafana |
| `adr/005-dvc-data-versioning.md` | ADR #5: Why DVC for data versioning |
| `assets/icon.png` | Site icon |
| `assets/logo.png` | Site logo |
| `assets/mlops-pipeline.png` | Pipeline architecture diagram |
| `assets/system-architecture.png` | System architecture diagram |
| `assets/tech-stack.png` | Tech stack diagram |

---

## 📂 `notebooks/` — Jupyter Demo Notebooks

| File | Nhiệm vụ |
|------|-----------|
| `01_inference_demo.ipynb` | Demo: call predict API, visualize results |
| `02_drift_detection_demo.ipynb` | Demo: generate drift, detect, visualize scores |
| `03_self_healing_demo.ipynb` | Demo: full self-healing cycle (drift → alert → retrain → promote) |
| `04_ab_testing_demo.ipynb` | Demo: A/B testing setup, traffic routing, compare models |

---

## 📂 `alembic/` — Database Migrations

| File | Nhiệm vụ |
|------|-----------|
| `env.py` | Alembic environment: async SQLAlchemy engine, auto-detect model changes |
| `script.py.mako` | Template cho auto-generated migration files |
| `versions/8e03d4fe1b79_initial_schema_*.py` | **Initial migration**: tạo 3 tables — `models` (id, version, uri, framework, stage, metadata, metrics, is_active, created_at), `prediction_logs` (id, model_id, features, result, confidence, latency, ground_truth, timestamp), `drift_reports` (id, model_id, feature_name, method, score, is_drifted, threshold, timestamp) |

---

## 🚀 Cách chạy dự án

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
uv run dvc repro

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
uv run mypy src/                                     # Type check
uv run pytest tests/                                 # Backend tests
docker exec phoenix-frontend npx tsc --noEmit        # Frontend type check
docker exec phoenix-frontend npx eslint src/          # Frontend lint
docker exec phoenix-frontend npx vitest run           # Frontend tests
```
