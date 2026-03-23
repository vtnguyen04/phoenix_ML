from abc import ABC, abstractmethod
from pathlib import Path


class ArtifactStorage(ABC):
    """Abstract interface for model artifact storage and retrieval."""

    @abstractmethod
    async def download(self, remote_uri: str, local_path: Path) -> Path:
        """Download artifact to local path for loading"""
        pass

    @abstractmethod
    async def upload(self, local_path: Path, remote_uri: str) -> None:
        """Upload artifact to remote storage"""
        pass
