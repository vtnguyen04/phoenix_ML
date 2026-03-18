import pytest

from src.domain.inference.entities.model import Model
from src.domain.inference.services.routing_strategy import (
    ABTestStrategy,
    CanaryStrategy,
    ShadowStrategy,
    SingleModelStrategy,
)


@pytest.fixture
def models() -> list[Model]:
    return [
        Model(id="m1", version="v1", uri="loc://m1", framework="onnx", is_active=True),
        Model(id="m2", version="v2", uri="loc://m2", framework="onnx", is_active=True),
    ]


@pytest.fixture
def role_models() -> list[Model]:
    return [
        Model(
            id="m1",
            version="v1",
            uri="loc://m1",
            framework="onnx",
            metadata={"role": "champion"},
        ),
        Model(
            id="m2",
            version="v2",
            uri="loc://m2",
            framework="onnx",
            metadata={"role": "challenger"},
        ),
    ]


def test_single_model_strategy(models: list[Model]) -> None:
    strategy = SingleModelStrategy()
    selected = strategy.select_model(models)
    assert selected.id == "m1"


def test_ab_test_strategy_always_champion(models: list[Model]) -> None:
    strategy = ABTestStrategy(challenger_traffic_percentage=0.0)
    selected = strategy.select_model(models)
    assert selected.id == "m1"


def test_ab_test_strategy_always_challenger(models: list[Model]) -> None:
    strategy = ABTestStrategy(challenger_traffic_percentage=1.0)
    selected = strategy.select_model(models)
    assert selected.id == "m2"


def test_ab_test_validation() -> None:
    with pytest.raises(ValueError):
        ABTestStrategy(challenger_traffic_percentage=1.5)


def test_canary_always_champion(role_models: list[Model]) -> None:
    strategy = CanaryStrategy(canary_percentage=0.0)
    selected = strategy.select_model(role_models)
    assert selected.id == "m1"


def test_canary_always_challenger(role_models: list[Model]) -> None:
    strategy = CanaryStrategy(canary_percentage=100.0)
    selected = strategy.select_model(role_models)
    assert selected.id == "m2"


def test_canary_validation() -> None:
    with pytest.raises(ValueError):
        CanaryStrategy(canary_percentage=150.0)


def test_canary_no_challenger(role_models: list[Model]) -> None:
    champion_only = [role_models[0]]
    strategy = CanaryStrategy(canary_percentage=100.0)
    selected = strategy.select_model(champion_only)
    assert selected.id == "m1"


def test_shadow_returns_champion(role_models: list[Model]) -> None:
    strategy = ShadowStrategy()
    selected = strategy.select_model(role_models)
    assert selected.id == "m1"
    assert strategy.shadow_model is not None
    assert strategy.shadow_model.id == "m2"


def test_shadow_no_challenger(role_models: list[Model]) -> None:
    champion_only = [role_models[0]]
    strategy = ShadowStrategy()
    selected = strategy.select_model(champion_only)
    assert selected.id == "m1"
    assert strategy.shadow_model is None
