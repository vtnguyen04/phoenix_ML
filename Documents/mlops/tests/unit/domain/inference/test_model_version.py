import pytest

from src.domain.inference.value_objects.model_version import ModelVersion


def test_model_version_parsing() -> None:
    major, minor, patch = 1, 2, 3
    v = ModelVersion.from_string(f"{major}.{minor}.{patch}")
    assert v.major == major
    assert v.minor == minor
    assert v.patch == patch
    assert str(v) == f"{major}.{minor}.{patch}"

def test_model_version_ordering() -> None:
    v1 = ModelVersion(1, 0, 0)
    v2 = ModelVersion(1, 0, 1)
    v3 = ModelVersion(2, 0, 0)

    assert v1 < v2
    assert v2 < v3
    assert v1 < v3

def test_invalid_version_string() -> None:
    with pytest.raises(ValueError):
        ModelVersion.from_string("1.2")
    
    with pytest.raises(ValueError):
        ModelVersion.from_string("v1.2.3")
