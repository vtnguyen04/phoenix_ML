from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    """
    Aggregate Root representing a Machine Learning Model.
    """
    model_config = ConfigDict(validate_assignment=True)

    id: str
    version: str
    uri: str
    framework: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    is_active: bool = True

    @property
    def unique_key(self) -> str:
        return f"{self.id}:{self.version}"

    def deactivate(self) -> None:
        self.is_active = False

    def activate(self) -> None:
        self.is_active = True