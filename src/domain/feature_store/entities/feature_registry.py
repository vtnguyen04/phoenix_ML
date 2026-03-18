from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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
    created_at: datetime = field(default_factory=datetime.utcnow)
    data_source: str | None = None

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
        }
