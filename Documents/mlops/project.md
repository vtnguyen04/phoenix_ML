# 🔥 PHOENIX ML PLATFORM
## Self-Healing Real-time ML Inference System

---

## 💡 TẠI SAO DỰ ÁN NÀY?

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Một platform HOÀN CHỈNH thể hiện TOÀN BỘ skill của Senior ML Engineer  │
│                                                                         │
│  ✅ System Design      ✅ Deep Learning      ✅ Algorithm              │
│  ✅ DDD/SOLID/KISS     ✅ Performance Opt    ✅ Production-Ready       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ SYSTEM ARCHITECTURE

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                           PHOENIX ML PLATFORM                                       │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────┐     ┌─────────────────────────────────────────────────────────┐  │
│  │   CLIENTS   │     │                    API GATEWAY                          │  │
│  │  ─────────  │────▶│  • Rate Limiting  • Auth  • Load Balancing  • Circuit  │  │
│  │  REST/gRPC  │     │                                               Breaker   │  │
│  └─────────────┘     └──────────────────────┬──────────────────────────────────┘  │
│                                              │                                     │
│         ┌────────────────────────────────────┼────────────────────────────────┐   │
│         │                                    ▼                                │   │
│         │  ┌─────────────────────────────────────────────────────────────┐   │   │
│         │  │                  INFERENCE SERVICE (Core Domain)             │   │   │
│         │  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐  │   │   │
│         │  │  │ Model Router  │  │ Batch Manager │  │ Feature Engine  │  │   │   │
│         │  │  │ (A/B Testing) │  │ (Dynamic      │  │ (Real-time      │  │   │   │
│         │  │  │               │  │  Batching)    │  │  Transform)     │  │   │   │
│         │  │  └───────┬───────┘  └───────┬───────┘  └────────┬────────┘  │   │   │
│         │  │          │                  │                   │           │   │   │
│         │  │          ▼                  ▼                   ▼           │   │   │
│         │  │  ┌─────────────────────────────────────────────────────┐   │   │   │
│         │  │  │              MODEL EXECUTOR (Anti-Corruption Layer) │   │   │   │
│         │  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │   │   │   │
│         │  │  │  │TensorRT │  │  ONNX   │  │ Triton  │  │  Custom │ │   │   │   │
│         │  │  │  │ Engine  │  │ Runtime │  │ Server  │  │  Engine │ │   │   │   │
│         │  │  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │   │   │   │
│         │  │  └─────────────────────────────────────────────────────┘   │   │   │
│         │  └─────────────────────────────────────────────────────────────┘   │   │
│         │                                                                     │   │
│  ┌──────┴──────────────────────────────────────────────────────────────────┐ │   │
│  │                                                                          │ │   │
│  │  ┌────────────────────┐  ┌────────────────────┐  ┌───────────────────┐  │ │   │
│  │  │   FEATURE STORE    │  │   MODEL REGISTRY   │  │  TRAINING ENGINE  │  │ │   │
│  │  │   ──────────────   │  │   ──────────────   │  │  ───────────────  │  │ │   │
│  │  │ • Offline Store    │  │ • Version Control  │  │ • Distributed     │  │ │   │
│  │  │   (Parquet/Delta)  │  │ • Metadata Store   │  │   Training        │  │ │   │
│  │  │ • Online Store     │  │ • Artifact Storage │  │ • Hyperparameter  │  │ │   │
│  │  │   (Redis Cluster)  │  │ • Model Lineage    │  │   Optimization    │  │ │   │
│  │  │ • Feature Server   │  │ • A/B Config       │  │ • Auto-Retraining │  │ │   │
│  │  └────────────────────┘  └────────────────────┘  └───────────────────┘  │ │   │
│  │                                                                          │ │   │
│  │                         SUPPORTING SUBDOMAINS                            │ │   │
│  └──────────────────────────────────────────────────────────────────────────┘ │   │
│                                                                                │   │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│  │                     SELF-HEALING SUBSYSTEM                               │  │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐   │  │   │
│  │  │   DRIFT     │  │   ANOMALY    │  │   AUTO      │  │   ROLLBACK   │   │  │   │
│  │  │  DETECTOR   │  │   DETECTOR   │  │  RETRAIN    │  │   MANAGER    │   │  │   │
│  │  │             │  │              │  │  TRIGGER    │  │              │   │  │   │
│  │  │ •Data Drift │  │ •Prediction  │  │ •Pipeline   │  │ •Canary      │   │  │   │
│  │  │ •Concept   │  │  Anomaly     │  │  Orchestr.  │  │  Deploy      │   │  │   │
│  │  │  Drift     │  │ •Latency     │  │ •Champion/  │  │ •Auto        │   │  │   │
│  │  │ •Feature   │  │  Spike       │  │  Challenger │  │  Rollback    │   │  │   │
│  │  │  Drift     │  │ •Error Rate  │  │             │  │              │   │  │   │
│  │  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └──────┬───────┘   │  │   │
│  │         │                │                 │                │           │  │   │
│  │         └────────────────┴────────┬────────┴────────────────┘           │  │   │
│  │                                   ▼                                      │  │   │
│  │                     ┌─────────────────────────┐                         │  │   │
│  │                     │    EVENT BUS (Kafka)    │                         │  │   │
│  │                     │  • Drift Events         │                         │  │   │
│  │                     │  • Retrain Triggers     │                         │  │   │
│  │                     │  • Deployment Events    │                         │  │   │
│  │                     └─────────────────────────┘                         │  │   │
│  └─────────────────────────────────────────────────────────────────────────┘  │   │
│                                                                                │   │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │   │
│  │                      OBSERVABILITY PLATFORM                              │  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │  │   │
│  │  │  Prometheus │  │   Grafana   │  │   Jaeger    │  │   Custom ML     │ │  │   │
│  │  │  + Metrics  │  │ Dashboards  │  │   Tracing   │  │   Dashboards    │ │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘ │  │   │
│  └─────────────────────────────────────────────────────────────────────────┘  │   │
│                                                                                │   │
└────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 CORE DOMAIN: MULTI-MODAL INFERENCE ENGINE

### Use Case: Real-time Content Understanding Platform

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTENT UNDERSTANDING                             │
│                                                                      │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│   │  IMAGE  │    │  TEXT   │    │  AUDIO  │    │  MULTI-MODAL    │  │
│   │         │    │         │    │         │    │   FUSION        │  │
│   │ •Safety │    │ •Toxic  │    │ •Speech │    │                 │  │
│   │ •NSFW   │    │ •Spam   │    │  to     │    │ •Cross-modal    │  │
│   │ •Object │    │ •Intent │    │  Text   │    │  Reasoning      │  │
│   │  Detect │    │ •NER    │    │ •Emotion│    │ •Unified        │  │
│   │ •OCR    │    │ •Embed  │    │         │    │  Embedding      │  │
│   └────┬────┘    └────┬────┘    └────┬────┘    └────────┬────────┘  │
│        │              │              │                  │           │
│        └──────────────┴──────────────┴──────────────────┘           │
│                               │                                      │
│                               ▼                                      │
│              ┌────────────────────────────────┐                     │
│              │     UNIFIED API RESPONSE       │                     │
│              │  {                             │                     │
│              │    "safety_score": 0.98,       │                     │
│              │    "categories": [...],        │                     │
│              │    "embeddings": [...],        │                     │
│              │    "latency_ms": 45            │                     │
│              │  }                             │                     │
│              └────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 DESIGN PATTERNS & PRINCIPLES

### 1️⃣ DDD (Domain-Driven Design)

```
src/
├── domain/                          # 🎯 CORE DOMAIN (Pure Business Logic)
│   ├── inference/
│   │   ├── entities/
│   │   │   ├── model.py            # Model Aggregate Root
│   │   │   ├── prediction.py       # Prediction Entity
│   │   │   └── feature_vector.py   # Value Object
│   │   ├── value_objects/
│   │   │   ├── model_version.py
│   │   │   ├── confidence_score.py
│   │   │   └── latency_budget.py
│   │   ├── services/
│   │   │   ├── inference_service.py      # Domain Service
│   │   │   └── routing_strategy.py       # Strategy Pattern
│   │   ├── repositories/
│   │   │   └── model_repository.py       # Interface only
│   │   └── events/
│   │       ├── prediction_made.py        # Domain Event
│   │       └── model_loaded.py
│   │
│   ├── training/                    # Separate Bounded Context
│   │   ├── entities/
│   │   ├── services/
│   │   └── events/
│   │
│   └── monitoring/                  # Another Bounded Context
│       ├── entities/
│       │   ├── drift_report.py
│       │   └── performance_metric.py
│       └── services/
│           └── drift_detector.py
│
├── application/                     # 🔄 APPLICATION LAYER
│   ├── commands/
│   │   ├── predict_command.py
│   │   ├── load_model_command.py
│   │   └── trigger_retrain_command.py
│   ├── queries/
│   │   ├── get_model_metrics_query.py
│   │   └── get_predictions_query.py
│   ├── handlers/
│   │   ├── predict_handler.py
│   │   └── retrain_handler.py
│   └── dto/
│       ├── prediction_request.py
│       └── prediction_response.py
│
├── infrastructure/                  # 🏗️ INFRASTRUCTURE LAYER
│   ├── persistence/
│   │   ├── redis_feature_store.py
│   │   ├── postgres_model_registry.py
│   │   └── s3_artifact_storage.py
│   ├── ml_engines/
│   │   ├── tensorrt_executor.py
│   │   ├── onnx_executor.py
│   │   └── triton_client.py
│   ├── messaging/
│   │   ├── kafka_producer.py
│   │   └── kafka_consumer.py
│   └── http/
│       ├── fastapi_server.py
│       └── grpc_server.py
│
└── shared/                          # 🔗 SHARED KERNEL
    ├── exceptions/
    ├── utils/
    └── interfaces/
```

### 2️⃣ SOLID Principles Implementation

```python
# ═══════════════════════════════════════════════════════════════════════
# S - SINGLE RESPONSIBILITY PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

# ❌ BAD: One class doing everything
class BadModelService:
    def load_model(self): ...
    def preprocess(self): ...
    def predict(self): ...
    def postprocess(self): ...
    def log_metrics(self): ...
    def cache_result(self): ...

# ✅ GOOD: Each class has one responsibility
class ModelLoader:
    """Only responsible for loading models"""
    def load(self, model_id: str) -> Model: ...

class Preprocessor:
    """Only responsible for preprocessing"""
    def transform(self, raw_input: RawInput) -> FeatureVector: ...

class Predictor:
    """Only responsible for prediction"""
    def predict(self, model: Model, features: FeatureVector) -> Prediction: ...


# ═══════════════════════════════════════════════════════════════════════
# O - OPEN/CLOSED PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

from abc import ABC, abstractmethod

class InferenceEngine(ABC):
    """Open for extension, closed for modification"""
    
    @abstractmethod
    def load(self, model_path: str) -> None: ...
    
    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray: ...
    
    @abstractmethod
    def optimize(self) -> None: ...

class TensorRTEngine(InferenceEngine):
    """Extend without modifying base class"""
    def load(self, model_path: str) -> None:
        self.engine = tensorrt.load(model_path)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.engine.infer(inputs)
    
    def optimize(self) -> None:
        self.engine.enable_fp16()
        self.engine.enable_dynamic_batching()

class ONNXEngine(InferenceEngine):
    """Another extension"""
    def load(self, model_path: str) -> None:
        self.session = onnxruntime.InferenceSession(model_path)
    
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": inputs})[0]
    
    def optimize(self) -> None:
        self.session.enable_graph_optimization()


# ═══════════════════════════════════════════════════════════════════════
# L - LISKOV SUBSTITUTION PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

class ModelRepository(ABC):
    @abstractmethod
    def save(self, model: Model) -> ModelVersion: ...
    
    @abstractmethod
    def load(self, model_id: str, version: ModelVersion) -> Model: ...
    
    @abstractmethod
    def list_versions(self, model_id: str) -> List[ModelVersion]: ...

class S3ModelRepository(ModelRepository):
    """Can substitute base class anywhere"""
    def save(self, model: Model) -> ModelVersion:
        # S3 implementation
        ...

class LocalModelRepository(ModelRepository):
    """Can substitute base class anywhere - useful for testing"""
    def save(self, model: Model) -> ModelVersion:
        # Local filesystem implementation
        ...

# Usage - can swap implementations without changing code
def get_model_service(repo: ModelRepository) -> ModelService:
    return ModelService(repo)  # Works with any implementation


# ═══════════════════════════════════════════════════════════════════════
# I - INTERFACE SEGREGATION PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

# ❌ BAD: Fat interface
class BadMLPipeline:
    def train(self): ...
    def evaluate(self): ...
    def deploy(self): ...
    def monitor(self): ...
    def retrain(self): ...

# ✅ GOOD: Segregated interfaces
class Trainable(Protocol):
    def train(self, data: Dataset) -> Model: ...

class Evaluatable(Protocol):
    def evaluate(self, model: Model, data: Dataset) -> Metrics: ...

class Deployable(Protocol):
    def deploy(self, model: Model) -> Endpoint: ...

class Monitorable(Protocol):
    def monitor(self, endpoint: Endpoint) -> HealthStatus: ...

# Classes implement only what they need
class TrainingPipeline(Trainable, Evaluatable):
    def train(self, data: Dataset) -> Model: ...
    def evaluate(self, model: Model, data: Dataset) -> Metrics: ...

class ServingPipeline(Deployable, Monitorable):
    def deploy(self, model: Model) -> Endpoint: ...
    def monitor(self, endpoint: Endpoint) -> HealthStatus: ...


# ═══════════════════════════════════════════════════════════════════════
# D - DEPENDENCY INVERSION PRINCIPLE
# ═══════════════════════════════════════════════════════════════════════

# High-level module depends on abstraction, not concrete implementation
class InferenceService:
    def __init__(
        self,
        model_loader: ModelLoader,           # Abstract
        feature_store: FeatureStore,         # Abstract
        cache: CacheService,                 # Abstract
        metrics: MetricsCollector,           # Abstract
    ):
        self._model_loader = model_loader
        self._feature_store = feature_store
        self._cache = cache
        self._metrics = metrics
    
    async def predict(self, request: PredictRequest) -> PredictResponse:
        # Uses abstractions, not concrete implementations
        features = await self._feature_store.get_features(request.entity_id)
        model = await self._model_loader.load(request.model_id)
        
        cached = await self._cache.get(request.cache_key)
        if cached:
            return cached
        
        result = model.predict(features)
        await self._metrics.record_prediction(result)
        
        return result

# Dependency Injection Container
class Container:
    @provider
    def inference_service(self) -> InferenceService:
        return InferenceService(
            model_loader=TensorRTModelLoader(),  # Concrete at composition root
            feature_store=RedisFeatureStore(),
            cache=RedisCache(),
            metrics=PrometheusMetrics(),
        )
```

### 3️⃣ Advanced Design Patterns

```python
# ═══════════════════════════════════════════════════════════════════════
# STRATEGY PATTERN - Model Routing
# ═══════════════════════════════════════════════════════════════════════

class RoutingStrategy(ABC):
    @abstractmethod
    def select_model(self, request: Request, models: List[Model]) -> Model: ...

class ABTestingStrategy(RoutingStrategy):
    def __init__(self, traffic_split: Dict[str, float]):
        self.traffic_split = traffic_split
    
    def select_model(self, request: Request, models: List[Model]) -> Model:
        # Consistent hashing for user-level consistency
        bucket = hash(request.user_id) % 100
        cumulative = 0
        for model_id, percentage in self.traffic_split.items():
            cumulative += percentage * 100
            if bucket < cumulative:
                return next(m for m in models if m.id == model_id)
        return models[0]  # Default

class CanaryStrategy(RoutingStrategy):
    def __init__(self, canary_percentage: float = 5.0):
        self.canary_percentage = canary_percentage
    
    def select_model(self, request: Request, models: List[Model]) -> Model:
        champion = next(m for m in models if m.is_champion)
        challenger = next((m for m in models if m.is_challenger), None)
        
        if challenger and random.random() < self.canary_percentage / 100:
            return challenger
        return champion

class ShadowStrategy(RoutingStrategy):
    """Route to champion, but also run shadow predictions"""
    async def select_and_shadow(
        self, request: Request, models: List[Model]
    ) -> Tuple[Model, asyncio.Task]:
        champion = next(m for m in models if m.is_champion)
        shadow = next((m for m in models if m.is_shadow), None)
        
        shadow_task = None
        if shadow:
            shadow_task = asyncio.create_task(
                self._run_shadow_prediction(shadow, request)
            )
        
        return champion, shadow_task


# ═══════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER PATTERN - Fault Tolerance
# ═══════════════════════════════════════════════════════════════════════

from enum import Enum
from dataclasses import dataclass
import time

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_requests: int = 3

class CircuitBreaker:
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_successes = 0
    
    async def execute(self, func: Callable, fallback: Callable) -> Any:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_successes = 0
            else:
                return await fallback()
        
        try:
            result = await func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if self.state == CircuitState.OPEN:
                return await fallback()
            raise
    
    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_successes += 1
            if self.half_open_successes >= self.config.half_open_requests:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN


# ═══════════════════════════════════════════════════════════════════════
# CHAIN OF RESPONSIBILITY - Request Pipeline
# ═══════════════════════════════════════════════════════════════════════

class RequestHandler(ABC):
    def __init__(self):
        self._next: Optional[RequestHandler] = None
    
    def set_next(self, handler: 'RequestHandler') -> 'RequestHandler':
        self._next = handler
        return handler
    
    async def handle(self, request: InferenceRequest) -> InferenceRequest:
        if self._next:
            return await self._next.handle(request)
        return request

class ValidationHandler(RequestHandler):
    async def handle(self, request: InferenceRequest) -> InferenceRequest:
        if not request.is_valid():
            raise ValidationError("Invalid request")
        return await super().handle(request)

class RateLimitHandler(RequestHandler):
    def __init__(self, rate_limiter: RateLimiter):
        super().__init__()
        self.rate_limiter = rate_limiter
    
    async def handle(self, request: InferenceRequest) -> InferenceRequest:
        if not await self.rate_limiter.acquire(request.client_id):
            raise RateLimitExceeded()
        return await super().handle(request)

class CacheHandler(RequestHandler):
    def __init__(self, cache: Cache):
        super().__init__()
        self.cache = cache
    
    async def handle(self, request: InferenceRequest) -> InferenceRequest:
        cached = await self.cache.get(request.cache_key)
        if cached:
            request.cached_result = cached
            return request
        return await super().handle(request)

class FeatureEnrichmentHandler(RequestHandler):
    def __init__(self, feature_store: FeatureStore):
        super().__init__()
        self.feature_store = feature_store
    
    async def handle(self, request: InferenceRequest) -> InferenceRequest:
        features = await self.feature_store.get_online_features(
            entity_id=request.entity_id,
            feature_names=request.required_features
        )
        request.features = features
        return await super().handle(request)

# Build the chain
def build_request_pipeline(container: Container) -> RequestHandler:
    validation = ValidationHandler()
    rate_limit = RateLimitHandler(container.rate_limiter)
    cache = CacheHandler(container.cache)
    feature = FeatureEnrichmentHandler(container.feature_store)
    
    validation.set_next(rate_limit).set_next(cache).set_next(feature)
    
    return validation
```

---

## ⚡ PERFORMANCE OPTIMIZATION

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION TECHNIQUES                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         MODEL OPTIMIZATION                               │   │
│  │                                                                          │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │   │
`│  │   │ Quantization│    │   Pruning   │    │ Distillation│                 │   │
│  │   │             │    │             │    │             │                 │   │
│  │   │ FP32→INT8   │    │ 40% sparse  │    │ Teacher →   │                 │   │
│  │   │ 4x smaller  │    │ 2x faster   │    │ Student     │                 │   │
│  │   │ 3x faster   │    │             │    │ 10x smaller │                 │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘                 │   │
│  │                                                                          │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │   │
│  │   │   TensorRT  │    │ONNX Runtime │    │   Triton    │                 │   │
│  │   │             │    │             │    │             │                 │   │
│  │   │ Graph Opt   │    │ Graph Opt   │    │ Dynamic     │                 │   │
│  │   │ Kernel Fuse │    │ Parallel    │    │ Batching    │                 │   │
│  │   │ Memory Pool │    │ Execution   │    │ Ensemble    │                 │   │
│  │   └─────────────┘    └─────────────┘    └─────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         SYSTEM OPTIMIZATION                              │   │
│  │                                                                          │   │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │   │                    DYNAMIC BATCHING                              │   │   │
│  │   │                                                                  │   │   │
│  │   │   Request 1 ──┐                                                  │   │   │
│  │   │   Request 2 ──┼──► Batch Queue ──► Batch (size=4) ──► GPU       │   │   │
│  │   │   Request 3 ──┤       │                                          │   │   │
│  │   │   Request 4 ──┘       │                                          │   │   │
│  │   │                       ▼                                          │   │   │
│  │   │              Timeout (10ms) or Max Batch Size                    │   │   │
│  │   └─────────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  │   ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐   │   │
│  │   │  Async Pipeline  │  │  Connection Pool │  │   Result Cache     │   │   │
│  │   │                  │  │                  │  │                    │   │   │
│  │   │ Preprocess ──┐   │  │ Redis: 100 conn  │  │ LRU Cache: 10GB    │   │   │
│  │   │ Inference ───┼── │  │ DB: 50 conn      │  │ TTL: 1 hour        │   │   │
│  │   │ Postprocess ─┘   │  │ gRPC: 200 conn   │  │ Hit rate: 85%      │   │   │
│  │   └──────────────────┘  └──────────────────┘  └────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Dynamic Batching Implementation

```python
import asyncio
from collections import deque
from dataclasses import dataclass
from typing import List, Any
import time

@dataclass
class BatchConfig:
    max_batch_size: int = 32
    max_wait_time_ms: float = 10.0
    preferred_batch_sizes: List[int] = None  # [1, 2, 4, 8, 16, 32]

class DynamicBatcher:
    def __init__(self, config: BatchConfig, executor: InferenceEngine):
        self.config = config
        self.executor = executor
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    async def start(self):
        self._running = True
        asyncio.create_task(self._batch_loop())
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        future = asyncio.Future()
        await self.queue.put((request, future))
        return await future
    
    async def _batch_loop(self):
        while self._running:
            batch = []
            futures = []
            deadline = time.time() + self.config.max_wait_time_ms / 1000
            
            # Collect requests
            while len(batch) < self.config.max_batch_size:
                timeout = max(0, deadline - time.time())
                try:
                    request, future = await asyncio.wait_for(
                        self.queue.get(), timeout=timeout
                    )
                    batch.append(request)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break
            
            if batch:
                await self._process_batch(batch, futures)
    
    async def _process_batch(
        self, 
        batch: List[InferenceRequest], 
        futures: List[asyncio.Future]
    ):
        try:
            # Pad to preferred batch size for GPU efficiency
            padded_batch = self._pad_batch(batch)
            
            # Run inference
            results = await self.executor.batch_predict(padded_batch)
            
            # Return results
            for i, future in enumerate(futures):
                future.set_result(results[i])
        except Exception as e:
            for future in futures:
                future.set_exception(e)
    
    def _pad_batch(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        if not self.config.preferred_batch_sizes:
            return batch
        
        current_size = len(batch)
        for preferred_size in self.config.preferred_batch_sizes:
            if preferred_size >= current_size:
                # Pad with dummy requests
                padding = [batch[-1]] * (preferred_size - current_size)
                return batch + padding
        
        return batch
```

---

## 🔄 SELF-HEALING: DRIFT DETECTION

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import stats

class DriftType(Enum):
    DATA_DRIFT = "data_drift"           # Input distribution changed
    CONCEPT_DRIFT = "concept_drift"     # P(Y|X) changed  
    PREDICTION_DRIFT = "prediction_drift"  # Output distribution changed

@dataclass
class DriftReport:
    drift_type: DriftType
    feature_name: Optional[str]
    drift_score: float
    p_value: float
    is_significant: bool
    reference_stats: Dict
    current_stats: Dict
    sample_size: int
    recommendation: str

class DriftDetector:
    def __init__(self, config: DriftConfig):
        self.config = config
        self.reference_distributions: Dict[str, np.ndarray] = {}
        self.statistical_tests = {
            "ks": self._ks_test,
            "psi": self._psi_test,
            "chi2": self._chi2_test,
            "wasserstein": self._wasserstein_test,
        }
    
    def set_reference(self, feature_name: str, data: np.ndarray):
        """Set reference distribution from training data"""
        self.reference_distributions[feature_name] = data
    
    def detect_drift(
        self, 
        feature_name: str, 
        current_data: np.ndarray,
        test_type: str = "ks"
    ) -> DriftReport:
        reference = self.reference_distributions.get(feature_name)
        if reference is None:
            raise ValueError(f"No reference distribution for {feature_name}")
        
        test_func = self.statistical_tests[test_type]
        drift_score, p_value = test_func(reference, current_data)
        
        is_significant = p_value < self.config.significance_level
        
        return DriftReport(
            drift_type=DriftType.DATA_DRIFT,
            feature_name=feature_name,
            drift_score=drift_score,
            p_value=p_value,
            is_significant=is_significant,
            reference_stats=self._compute_stats(reference),
            current_stats=self._compute_stats(current_data),
            sample_size=len(current_data),
            recommendation=self._generate_recommendation(
                is_significant, drift_score, feature_name
            )
        )
    
    def _ks_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Kolmogorov-Smirnov test for continuous features"""
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value
    
    def _psi_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray,
        n_bins: int = 10
    ) -> tuple[float, float]:
        """Population Stability Index"""
        # Create bins from reference
        bins = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        bins[0] = -np.inf
        bins[-1] = np.inf
        
        ref_counts = np.histogram(reference, bins=bins)[0] / len(reference)
        cur_counts = np.histogram(current, bins=bins)[0] / len(current)
        
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        ref_counts = np.clip(ref_counts, eps, 1)
        cur_counts = np.clip(cur_counts, eps, 1)
        
        psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
        
        # PSI thresholds: < 0.1 no drift, 0.1-0.25 moderate, > 0.25 significant
        if psi < 0.1:
            p_value = 0.5
        elif psi < 0.25:
            p_value = 0.1
        else:
            p_value = 0.01
        
        return psi, p_value
    
    def _wasserstein_test(
        self, 
        reference: np.ndarray, 
        current: np.ndarray
    ) -> tuple[float, float]:
        """Earth Mover's Distance"""
        distance = stats.wasserstein_distance(reference, current)
        
        # Bootstrap for p-value
        combined = np.concatenate([reference, current])
        n_ref = len(reference)
        
        bootstrap_distances = []
        for _ in range(1000):
            np.random.shuffle(combined)
            boot_ref = combined[:n_ref]
            boot_cur = combined[n_ref:]
            boot_dist = stats.wasserstein_distance(boot_ref, boot_cur)
            bootstrap_distances.append(boot_dist)
        
        p_value = np.mean(np.array(bootstrap_distances) >= distance)
        return distance, p_value
    
    def _compute_stats(self, data: np.ndarray) -> Dict:
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }
    
    def _generate_recommendation(
        self, 
        is_significant: bool, 
        drift_score: float,
        feature_name: str
    ) -> str:
        if not is_significant:
            return "No action needed. Continue monitoring."
        
        if drift_score > 0.5:
            return (
                f"CRITICAL: Severe drift detected in {feature_name}. "
                f"Immediate retraining recommended. "
                f"Consider investigating data pipeline for issues."
            )
        elif drift_score > 0.25:
            return (
                f"WARNING: Moderate drift detected in {feature_name}. "
                f"Schedule retraining. Monitor model performance closely."
            )
        else:
            return (
                f"NOTICE: Minor drift detected in {feature_name}. "
                f"Add to retraining queue. No immediate action required."
            )
```

---

## 📊 METRICS & MONITORING DASHBOARD

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     PHOENIX ML PLATFORM - DASHBOARD                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MODEL PERFORMANCE                          SYSTEM HEALTH                        │
│  ┌─────────────────────────────────────┐   ┌──────────────────────────────────┐ │
│  │  Accuracy     ████████████░ 94.2%   │   │  API Latency (p99)    45ms ✅    │ │
│  │  Precision    █████████████ 96.1%   │   │  Throughput           12.5k rps  │ │
│  │  Recall       ████████████░ 92.3%   │   │  Error Rate           0.01% ✅   │ │
│  │  F1 Score     ████████████░ 94.2%   │   │  GPU Utilization      78% ✅     │ │
│  └─────────────────────────────────────┘   │  Memory Usage         65% ✅     │ │
│                                             │  CPU Usage            45% ✅     │ │
│  DRIFT MONITORING                           └──────────────────────────────────┘ │
│  ┌─────────────────────────────────────┐                                        │
│  │  Feature: user_age                  │   A/B TEST STATUS                      │
│  │  PSI Score: 0.08 ✅ (< 0.1)         │   ┌──────────────────────────────────┐ │
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ → ▁▂▃▅▆▇█▇▆▅▄▃▂▁  │   │  Champion (v2.1)    90% traffic  │ │
│  │                                      │   │  Challenger (v2.2)  10% traffic  │ │
│  │  Feature: purchase_amount           │   │                                  │ │
│  │  PSI Score: 0.23 ⚠️ (approaching)   │   │  Challenger Performance:         │ │
│  │  ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁ → ▂▃▄▅▆▇█▇▇▆▅▄▃▂▁  │   │  +2.3% accuracy                  │ │
│  │                                      │   │  -5ms latency                    │ │
│  │  Feature: session_duration          │   │  Confidence: 95%                 │ │
│  │  PSI Score: 0.05 ✅ (stable)        │   │  Status: WINNING ✅              │ │
│  └─────────────────────────────────────┘   └──────────────────────────────────┘ │
│                                                                                  │
│  INFERENCE PIPELINE TRACE                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                             │ │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │ │
│  │  │ Gateway  │──▶│ Feature  │──▶│  Batch   │──▶│ Inference│──▶│   Post   │ │ │
│  │  │  2.1ms   │   │  Store   │   │ Manager  │   │  Engine  │   │ Process  │ │ │
│  │  └──────────┘   │  5.3ms   │   │  3.2ms   │   │  28.5ms  │   │  2.4ms   │ │ │
│  │                 └──────────┘   └──────────┘   └──────────┘   └──────────┘ │ │
│  │                                                                             │ │
│  │  Total Latency: 41.5ms (p50) | 45.2ms (p95) | 52.1ms (p99)                │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 TECH STACK

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNOLOGY STACK                                    │
├──────────────────┬──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  INFERENCE       │  • TensorRT / ONNX Runtime / Triton Inference Server         │
│                  │  • CUDA / cuDNN for GPU acceleration                          │
│                  │  • TorchScript / TorchServe                                   │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  BACKEND         │  • FastAPI (async HTTP) / gRPC (high-performance)            │
│                  │  • Python 3.11+ with asyncio                                  │
│                  │  • Pydantic v2 for validation                                 │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  DATA LAYER      │  • Redis Cluster (online features, caching)                  │
│                  │  • PostgreSQL (metadata, model registry)                      │
│                  │  • MinIO/S3 (model artifacts)                                │
│                  │  • Apache Kafka (event streaming)                             │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  MLOPS           │  • MLflow (experiment tracking)                              │
│                  │  • DVC (data versioning)                                      │
│                  │  • Feast (feature store)                                      │
│                  │  • Airflow/Prefect (pipeline orchestration)                  │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  OBSERVABILITY   │  • Prometheus + Grafana (metrics)                            │
│                  │  • Jaeger (distributed tracing)                               │
│                  │  • ELK Stack (logging)                                        │
│                  │  • Custom ML monitoring dashboards                            │
│                  │                                                               │
├──────────────────┼──────────────────────────────────────────────────────────────┤
│                  │                                                               │
│  INFRASTRUCTURE  │  • Kubernetes (container orchestration)                       │
│                  │  • Docker (containerization)                                  │
│                  │  • Terraform (IaC)                                            │
│                  │  • GitHub Actions (CI/CD)                                     │
│                  │                                                               │
└──────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## 📅 IMPLEMENTATION ROADMAP (4 MONTHS)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           4-MONTH IMPLEMENTATION PLAN                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  MONTH 1: CORE INFERENCE ENGINE                                                  │
│  ═══════════════════════════════                                                │
│  Week 1-2: ┌─────────────────────────────────────────────────────────────────┐  │
│            │ • Set up project structure (DDD architecture)                    │  │
│            │ • Implement domain entities and value objects                    │  │
│            │ • Create abstract interfaces for all components                  │  │
│            └─────────────────────────────────────────────────────────────────┘  │
│  Week 3-4: ┌─────────────────────────────────────────────────────────────────┐  │
│            │ • Implement TensorRT/ONNX inference engines                     │  │
│            │ • Build dynamic batching system                                  │  │
│            │ • Create FastAPI + gRPC servers                                  │  │
│            │ • Unit tests with 80%+ coverage                                  │  │
│            └─────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  MONTH 2: FEATURE STORE & MODEL REGISTRY                                        │
│  ═══════════════════════════════════════                                        │
│  Week 5-6: ┌─────────────────────────────────────────────────────────────────┐  │
│            │ • Build custom Feature Store (Redis + Parquet)                  │  │
│            │ • Implement online/offline feature serving                       │  │
│            │ • Feature versioning and lineage                                 │  │
│            └─────────────────────────────────────────────────────────────────┘  │
│  Week 7-8: ┌─────────────────────────────────────────────────────────────────┐  │
│            │ • Build Model Registry with versioning                          │  │
│            │ • Implement A/B testing framework                                │  │
│            │ • Create model deployment pipeline                               │  │
│            │ • Integration tests                                              │  │
│            └─────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  MONTH 3: SELF-HEALING & MONITORING                                             │
│  ══════════════════════════════════                                             │
│  Week 9-10: ┌────────────────────────────────────────────────────────────────┐  │
│             │ • Implement drift detection algorithms                         │  │
│             │ • Build auto-retraining trigger system                         │  │
│             │ • Create rollback mechanisms                                    │  │
│             └────────────────────────────────────────────────────────────────┘  │
│  Week 11-12: ┌───────────────────────────────────────────────────────────────┐  │
│              │ • Set up Prometheus + Grafana dashboards                     │  │
│              │ • Implement distributed tracing with Jaeger                   │  │
│              │ • Create alerting system                                      │  │
│              │ • Performance benchmarks                                      │  │
│              └───────────────────────────────────────────────────────────────┘  │
│                                                                                  │
│  MONTH 4: PRODUCTION-READY & DOCUMENTATION                                      │
│  ═════════════════════════════════════════                                      │
│  Week 13-14: ┌───────────────────────────────────────────────────────────────┐  │
│              │ • Kubernetes deployment (Helm charts)                         │  │
│              │ • CI/CD pipeline with GitHub Actions                          │  │
│              │ • Load testing and optimization                               │  │
│              └───────────────────────────────────────────────────────────────┘  │
│  Week 15-16: ┌───────────────────────────────────────────────────────────────┐  │
│              │ • Comprehensive documentation                                 │  │
│              │ • Architecture Decision Records (ADRs)                        │  │
│              │ • Demo video and presentation                                 │  │
│              │ • Blog posts about key components                             │  │
│              └───────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 TẠI SAO DỰ ÁN NÀY SẼ GIÚP BẠN NỔI BẬT?

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ĐIỂM NỔI BẬT TRÊN CV                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  🏆 THUẬT TOÁN & DEEP LEARNING                                                  │
│  ├── Drift detection algorithms (KS-test, PSI, Wasserstein)                     │
│  ├── Model optimization (Quantization, Pruning, Distillation)                   │
│  ├── Dynamic batching for GPU efficiency                                        │
│  └── Multi-modal model ensemble                                                  │
│                                                                                  │
│  🏗️ SYSTEM DESIGN                                                               │
│  ├── Microservices with clear bounded contexts                                   │
│  ├── Event-driven architecture (Kafka)                                           │
│  ├── High-availability with circuit breakers                                     │
│  └── Scalable to millions of requests/day                                        │
│                                                                                  │
│  💻 ENGINEERING EXCELLENCE                                                       │
│  ├── DDD with clear domain separation                                            │
│  ├── SOLID principles throughout codebase                                        │
│  ├── Design patterns (Strategy, Circuit Breaker, Chain of Responsibility)       │
│  └── 80%+ test coverage                                                          │
│                                                                                  │
│  ⚡ PERFORMANCE OPTIMIZATION                                                     │
│  ├── <50ms p99 latency                                                           │
│  ├── 10k+ RPS throughput                                                         │
│  ├── GPU memory optimization                                                     │
│  └── Intelligent caching strategies                                              │
│                                                                                  │
│  📊 PRODUCTION READINESS                                                         │
│  ├── Full observability (metrics, traces, logs)                                  │
│  ├── A/B testing and canary deployments                                          │
│  ├── Auto-healing with drift detection                                           │
│  └── Kubernetes-native deployment                                                │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 GITHUB REPOSITORY STRUCTURE

```
phoenix-ml-platform/
├── README.md                    # Comprehensive documentation
├── docs/
│   ├── architecture/
│   │   ├── SYSTEM_DESIGN.md
│   │   ├── DDD_OVERVIEW.md
│   │   └── adr/                 # Architecture Decision Records
│   ├── api/
│   └── deployment/
├── src/
│   ├── domain/
│   ├── application/
│   ├── infrastructure/
│   └── shared/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── benchmarks/
│   ├── latency_benchmark.py
│   └── throughput_benchmark.py
├── deployment/
│   ├── kubernetes/
│   ├── docker/
│   └── terraform/
├── notebooks/                   # Demo notebooks
├── .github/
│   └── workflows/               # CI/CD
└── Makefile
```

---


> **"Dự án này không chỉ là code - mà là PORTFOLIO của tôi"**

1. **Viết documentation như Senior**: ADRs, API docs, architecture diagrams
2. **Record demo video**: Giải thích system design decisions
3. **Viết blog posts**: Mỗi component = 1 technical blog post
4. **Benchmark & share**: So sánh với baseline, publish kết quả
5. **Open source**: Star, forks, và contributions sẽ là proof of quality

