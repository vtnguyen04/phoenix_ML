import pytest

from src.domain.inference.entities.model import Model
from src.domain.inference.services.routing_strategy import (
    ABTestStrategy,
    SingleModelStrategy,
)


@pytest.fixture
def models() -> list[Model]:
    return [
        Model(id="m1", version="v1", uri="loc://m1", framework="onnx", is_active=True),
        Model(id="m2", version="v2", uri="loc://m2", framework="onnx", is_active=True),
    ]

def test_single_model_strategy(models: list[Model]) -> None:
    strategy = SingleModelStrategy()
    selected = strategy.select_model(models)
    assert selected.id == "m1"

def test_ab_test_strategy_always_champion(models: list[Model]) -> None:
    # 0% traffic to challenger
    strategy = ABTestStrategy(challenger_traffic_percentage=0.0)
    selected = strategy.select_model(models)
    assert selected.id == "m1" # Champion

def test_ab_test_strategy_always_challenger(models: list[Model]) -> None:
    # 100% traffic to challenger
    strategy = ABTestStrategy(challenger_traffic_percentage=1.0)
    selected = strategy.select_model(models)
    assert selected.id == "m2" # Challenger (last in list)

def test_ab_test_validation() -> None:
    with pytest.raises(ValueError):
        ABTestStrategy(challenger_traffic_percentage=1.5)
