import shutil
from pathlib import Path

from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage


class LocalArtifactStorage(ArtifactStorage):
    """
    Simulates remote storage using local filesystem.
    Useful for local development and testing.
    """
    
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def download(self, remote_uri: str, local_path: Path) -> Path:
        # Expected remote_uri format: "local://path/to/model"
        source_path = Path(remote_uri.replace("local://", ""))
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source artifact {source_path} not found")
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, local_path)
        return local_path

    async def upload(self, local_path: Path, remote_uri: str) -> None:
        target_path = Path(remote_uri.replace("local://", ""))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, target_path)
