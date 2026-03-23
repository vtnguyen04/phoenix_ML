# Library API Guide

Phoenix ML Platform can be used as a **Python library** — import directly, no Docker required.

## Installation

```bash
# Clone repository
git clone https://github.com/vtnguyen04/phoenix_ML.git
cd phoenix_ML

# Install dependencies
uv sync
```

## Quick Start — Programmatic Inference

```python
import asyncio
import numpy as np
from phoenix_ml.infrastructure.bootstrap.container import (
    inference_engine,
    feature_store,
)
from phoenix_ml.domain.inference.entities.model import Model
from phoenix_ml.domain.inference.value_objects.feature_vector import FeatureVector

async def main():
    # 1. Define model
    model = Model(
        id="credit-risk",
        version="v1",
        uri="local:///models/credit_risk/v1/model.onnx",
        framework="onnx",
        metadata={"role": "champion"},
        is_active=True,
    )
    
    # 2. Load model into engine
    await inference_engine.load(model)
    
    # 3. Create feature vector
    features = FeatureVector(values=np.array([0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0], dtype=np.float32))
    
    # 4. Run prediction
    prediction = await inference_engine.predict(model, features)
    
    print(f"Result: {prediction.result}")
    print(f"Confidence: {prediction.confidence.value:.4f}")
    print(f"Latency: {prediction.latency_ms:.2f} ms")

asyncio.run(main())
```

## Using the Full Application Stack

```python
import asyncio
from phoenix_ml.application.commands.predict_command import PredictCommand
from phoenix_ml.application.handlers.predict_handler import PredictHandler
from phoenix_ml.infrastructure.bootstrap.container import (
    inference_engine,
    event_bus,
    feature_store,
    plugin_registry,
)
from phoenix_ml.infrastructure.persistence.in_memory_model_repo import InMemoryModelRepository

async def main():
    # 1. Setup handler with DI
    model_repo = InMemoryModelRepository()
    handler = PredictHandler(
        model_repo=model_repo,
        inference_engine=inference_engine,
        event_bus=event_bus,
        feature_store=feature_store,
        plugin_registry=plugin_registry,
    )
    
    # 2. Register a model
    from phoenix_ml.domain.inference.entities.model import Model
    model = Model(id="credit-risk", version="v1", ...)
    await model_repo.save(model)
    
    # 3. Predict via command
    command = PredictCommand(
        model_id="credit-risk",
        features=[0.5, 1.2, 0.8, 3.4, 0.1, 2.5, 1.0]
    )
    prediction = await handler.handle(command)
    print(prediction)

asyncio.run(main())
```

## Feature Store API

```python
from phoenix_ml.infrastructure.feature_store.in_memory_feature_store import InMemoryFeatureStore

async def feature_example():
    store = InMemoryFeatureStore()
    
    # Add features for an entity
    await store.add_features("customer_42", {
        "income": 55000.0,
        "age": 32.0,
        "credit_score": 720.0,
    })
    
    # Retrieve features
    features = await store.get_online_features(
        entity_id="customer_42",
        feature_names=["income", "age", "credit_score"]
    )
    print(features)  # [55000.0, 32.0, 720.0]
```

## Drift Detection API

```python
from phoenix_ml.domain.monitoring.services.drift_calculator import DriftCalculator

calculator = DriftCalculator()

# Reference distribution (training data)
reference = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12]

# Current distribution (production data)
current = [0.5, 0.6, 0.55, 0.7, 0.65, 0.58, 0.62, 0.52]

# KS Test
result = calculator.calculate_ks(reference, current)
print(f"KS Score: {result.score:.4f}, Drifted: {result.is_drifted}")

# PSI
result = calculator.calculate_psi(reference, current)
print(f"PSI Score: {result.score:.4f}")

# Chi-squared
result = calculator.calculate_chi2(reference, current)
print(f"Chi2 Score: {result.score:.4f}")
```

## Model Evaluation API

```python
from phoenix_ml.domain.monitoring.services.model_evaluator import get_evaluator

# Classification
evaluator = get_evaluator("classification")
metrics = evaluator.evaluate(
    predictions=[1, 0, 1, 1, 0],
    ground_truth=[1, 0, 1, 0, 0]
)
print(metrics)  # {"accuracy": 0.8, "f1_score": 0.8, "precision": 1.0, "recall": 0.67}

# Regression
evaluator = get_evaluator("regression")
metrics = evaluator.evaluate(
    predictions=[2.5, 3.1, 4.0],
    ground_truth=[2.3, 3.0, 4.5]
)
print(metrics)  # {"rmse": 0.3, "mae": 0.27, "r2": 0.89}
```

## Alert Management API

```python
from phoenix_ml.domain.monitoring.services.alert_manager import AlertManager, AlertRule

manager = AlertManager()

# Define alert rules
rules = [
    AlertRule(
        name="high_drift",
        metric="drift_score",
        threshold=0.3,
        severity="CRITICAL",
        comparison="gt",  # greater than
        cooldown_seconds=300,  # 5min cooldown
    ),
    AlertRule(
        name="low_accuracy",
        metric="accuracy",
        threshold=0.8,
        severity="WARNING",
        comparison="lt",  # less than
    ),
]

# Evaluate rules against current metrics
alerts = manager.evaluate(rules, current_metrics={"drift_score": 0.45, "accuracy": 0.75})
for alert in alerts:
    print(f"ALERT: {alert.name} — {alert.severity}")
```

## Circuit Breaker API

```python
from phoenix_ml.domain.inference.services.circuit_breaker import CircuitBreaker

cb = CircuitBreaker(
    failure_threshold=5,     # Open after 5 failures
    recovery_timeout=30.0,   # Try recovery after 30s
    half_open_max_calls=3,   # Allow 3 test calls in half-open
)

# Check state before inference
if cb.is_available():
    try:
        result = await engine.predict(model, features)
        cb.record_success()
    except Exception:
        cb.record_failure()
else:
    print("Circuit OPEN — inference blocked")
```

## Batch Prediction

```python
from phoenix_ml.domain.inference.services.batch_manager import BatchManager, BatchConfig

config = BatchConfig(max_batch_size=32, max_wait_time_ms=10)
batch_manager = BatchManager(config=config, engine=inference_engine)

# Batch manager automatically collects requests
results = await batch_manager.submit(model, [features_1, features_2, features_3])
```

## Routing Strategies

```python
from phoenix_ml.domain.inference.services.routing_strategy import (
    SingleModelStrategy,
    ABTestStrategy,
    CanaryStrategy,
    ShadowStrategy,
)

# A/B Testing: 80% champion, 20% challenger
strategy = ABTestStrategy(traffic_ratio=0.8)
selected = strategy.select(champion_model, [challenger_model])

# Canary: 5% to new version
strategy = CanaryStrategy(canary_percentage=0.05)
selected = strategy.select(champion_model, [canary_model])
```

---
*Document Status: v4.0 — Updated March 2026*
