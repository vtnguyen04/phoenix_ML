"""Tests for FeatureLineage, FeatureRegistry, and FeatureMetadata."""

import pytest

from src.domain.feature_store.entities.feature_registry import (
    FeatureLineage,
    FeatureMetadata,
    FeatureRegistry,
    FeatureTransformation,
)

# ─── FeatureTransformation ───────────────────────────────────────


class TestFeatureTransformation:
    def test_creation(self) -> None:
        t = FeatureTransformation(name="standard_scaler", params={"mean": 0.0, "std": 1.0})
        assert t.name == "standard_scaler"
        assert t.params == {"mean": 0.0, "std": 1.0}

    def test_frozen(self) -> None:
        t = FeatureTransformation(name="log")
        with pytest.raises(AttributeError):
            t.name = "other"  # type: ignore[misc]


# ─── FeatureLineage ──────────────────────────────────────────────


class TestFeatureLineage:
    def test_creation_defaults(self) -> None:
        lineage = FeatureLineage(source="openml/credit-g")
        assert lineage.source == "openml/credit-g"
        assert lineage.version == "v1"
        assert lineage.transformations == []
        assert lineage.parent_features == []

    def test_add_transformation(self) -> None:
        lineage = FeatureLineage(source="raw_table")
        lineage.add_transformation("standard_scaler", {"with_mean": True})
        lineage.add_transformation("log_transform", description="Apply log1p")

        assert len(lineage.transformations) == 2
        assert lineage.transformations[0].name == "standard_scaler"
        assert lineage.transformations[1].description == "Apply log1p"

    def test_bump_version(self) -> None:
        lineage = FeatureLineage(source="db")
        assert lineage.version == "v1"
        new_version = lineage.bump_version()
        assert new_version == "v2"
        assert lineage.version == "v2"
        lineage.bump_version()
        assert lineage.version == "v3"

    def test_to_dict(self) -> None:
        lineage = FeatureLineage(
            source="kafka_stream",
            parent_features=["age", "income"],
        )
        lineage.add_transformation("ratio", {"numerator": "income", "denominator": "age"})
        d = lineage.to_dict()

        assert d["source"] == "kafka_stream"
        assert d["version"] == "v1"
        assert d["parent_features"] == ["age", "income"]
        assert len(d["transformations"]) == 1
        assert d["transformations"][0]["name"] == "ratio"
        assert "created_at" in d
        assert "updated_at" in d


# ─── FeatureMetadata ─────────────────────────────────────────────


class TestFeatureMetadata:
    def test_creation_with_lineage(self) -> None:
        lineage = FeatureLineage(source="openml")
        meta = FeatureMetadata(
            name="credit_amount",
            dtype="float64",
            description="Loan amount",
            owner="ml-team",
            lineage=lineage,
        )
        assert meta.lineage is not None
        assert meta.lineage.source == "openml"

    def test_to_dict_includes_lineage(self) -> None:
        lineage = FeatureLineage(source="db")
        meta = FeatureMetadata(
            name="age", dtype="int", description="Age", owner="team", lineage=lineage
        )
        d = meta.to_dict()
        assert d["lineage"] is not None
        assert d["lineage"]["source"] == "db"

    def test_to_dict_without_lineage(self) -> None:
        meta = FeatureMetadata(name="x", dtype="float", description="", owner="")
        d = meta.to_dict()
        assert d["lineage"] is None


# ─── FeatureRegistry ─────────────────────────────────────────────


class TestFeatureRegistry:
    def test_register_and_get(self) -> None:
        registry = FeatureRegistry()
        meta = FeatureMetadata(name="age", dtype="int", description="Customer age", owner="team")
        registry.register(meta)

        result = registry.get("age")
        assert result is not None
        assert result.name == "age"

    def test_get_missing(self) -> None:
        registry = FeatureRegistry()
        assert registry.get("nonexistent") is None

    def test_list_features_all(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureMetadata(name="a", dtype="int", description="", owner=""))
        registry.register(FeatureMetadata(name="b", dtype="float", description="", owner=""))
        assert len(registry.list_features()) == 2

    def test_list_features_by_tag(self) -> None:
        registry = FeatureRegistry()
        registry.register(
            FeatureMetadata(name="a", dtype="int", description="", owner="", tags=["credit"])
        )
        registry.register(
            FeatureMetadata(name="b", dtype="float", description="", owner="", tags=["personal"])
        )

        credit_features = registry.list_features(tag="credit")
        assert len(credit_features) == 1
        assert credit_features[0].name == "a"

    def test_get_lineage(self) -> None:
        registry = FeatureRegistry()
        lineage = FeatureLineage(source="openml")
        lineage.add_transformation("scaler")
        registry.register(
            FeatureMetadata(
                name="amount", dtype="float", description="", owner="", lineage=lineage
            )
        )

        result = registry.get_lineage("amount")
        assert result is not None
        assert result.source == "openml"
        assert len(result.transformations) == 1

    def test_get_lineage_missing(self) -> None:
        registry = FeatureRegistry()
        assert registry.get_lineage("nope") is None

    def test_list_all(self) -> None:
        registry = FeatureRegistry()
        registry.register(FeatureMetadata(name="x", dtype="int", description="", owner=""))
        all_features = registry.list_all()
        assert "x" in all_features
        assert all_features["x"]["dtype"] == "int"
