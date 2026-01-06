import re
from dataclasses import dataclass
from functools import total_ordering


@total_ordering
@dataclass(frozen=True)
class ModelVersion:
    """
    Value Object representing a semantic version of a model.
    Format: MAJOR.MINOR.PATCH (e.g., 1.0.0)
    """
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "ModelVersion":
        pattern = r"^(\d+)\.(\d+)\.(\d+)$"
        match = re.match(pattern, version_str)
        if not match:
            raise ValueError(
                f"Invalid version format: {version_str}. "
                "Expected 'MAJOR.MINOR.PATCH'"
            )
        
        return cls(
            major=int(match.group(1)),
            minor=int(match.group(2)),
            patch=int(match.group(3))
        )

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ModelVersion):
            return NotImplemented
        self_tuple = (self.major, self.minor, self.patch)
        other_tuple = (other.major, other.minor, other.patch)
        return self_tuple < other_tuple
