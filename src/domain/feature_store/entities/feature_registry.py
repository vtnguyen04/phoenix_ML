from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class FeatureTransformation:
    """Value Object representing a single transformation step."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""


@dataclass
class FeatureLineage:
    """
    Value Object tracking the full provenance of a feature.

    Records source, transformation chain, and version history
    for complete feature lineage (as required by project.md).
    """

    source: str
    transformations: list[FeatureTransformation] = field(default_factory=list)
    version: str = "v1"
    parent_features: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def add_transformation(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        description: str = "",
    ) -> None:
        """Record a transformation step in the lineage."""
        self.transformations.append(
            FeatureTransformation(name=name, params=params or {}, description=description)
        )
        self.updated_at = datetime.now(tz=UTC)

    def bump_version(self) -> str:
        """Increment version (v1 -> v2 -> v3...)."""
        current = int(self.version.lstrip("v"))
        self.version = f"v{current + 1}"
        self.updated_at = datetime.now(tz=UTC)
        return self.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "transformations": [
                {"name": t.name, "params": t.params, "description": t.description}
                for t in self.transformations
            ],
            "version": self.version,
            "parent_features": self.parent_features,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class FeatureMetadata:
    """
    Metadata registry entity for tracking individual features.
    """

    name: str
    dtype: str
    description: str
    owner: str
    version: str = "v1"
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    data_source: str | None = None
    lineage: FeatureLineage | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "description": self.description,
            "owner": self.owner,
            "version": self.version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "data_source": self.data_source,
            "lineage": self.lineage.to_dict() if self.lineage else None,
        }


class FeatureRegistry:
    """
    Registry for managing feature metadata and lineage.

    Provides lookup, registration, versioning, and lineage tracking
    for all features in the platform.
    """

    def __init__(self) -> None:
        self._features: dict[str, FeatureMetadata] = {}

    def register(self, metadata: FeatureMetadata) -> None:
        """Register or update a feature."""
        self._features[metadata.name] = metadata

    def get(self, name: str) -> FeatureMetadata | None:
        """Lookup feature metadata by name."""
        return self._features.get(name)

    def list_features(self, tag: str | None = None) -> list[FeatureMetadata]:
        """List all features, optionally filtered by tag."""
        if tag is None:
            return list(self._features.values())
        return [f for f in self._features.values() if tag in f.tags]

    def get_lineage(self, name: str) -> FeatureLineage | None:
        """Get lineage for a specific feature."""
        meta = self._features.get(name)
        return meta.lineage if meta else None

    def list_all(self) -> dict[str, dict[str, Any]]:
        """Export all features as dict."""
        return {name: meta.to_dict() for name, meta in self._features.items()}

