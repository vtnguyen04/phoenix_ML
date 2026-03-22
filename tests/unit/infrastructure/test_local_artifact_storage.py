"""
Tests for LocalArtifactStorage — stores model artifacts on local filesystem.
"""

from pathlib import Path

import pytest

from phoenix_ml.infrastructure.artifact_storage.local_artifact_storage import (
    LocalArtifactStorage,
)


@pytest.fixture()
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture()
def storage(storage_dir: Path) -> LocalArtifactStorage:
    return LocalArtifactStorage(storage_dir)


class TestLocalArtifactStorage:
    """Unit tests for local-filesystem artifact storage."""

    @pytest.mark.asyncio()
    async def test_download_copies_file(
        self, storage: LocalArtifactStorage, tmp_path: Path
    ) -> None:
        source = tmp_path / "source_model.onnx"
        source.write_bytes(b"model-data-123")
        dest = tmp_path / "dest" / "model.onnx"

        result = await storage.download(f"local://{source}", dest)
        assert result == dest
        assert dest.read_bytes() == b"model-data-123"

    @pytest.mark.asyncio()
    async def test_download_creates_parent_dirs(
        self, storage: LocalArtifactStorage, tmp_path: Path
    ) -> None:
        source = tmp_path / "file.bin"
        source.write_bytes(b"data")
        dest = tmp_path / "a" / "b" / "c" / "file.bin"

        await storage.download(f"local://{source}", dest)
        assert dest.exists()

    @pytest.mark.asyncio()
    async def test_download_missing_source_raises(
        self, storage: LocalArtifactStorage, tmp_path: Path
    ) -> None:
        dest = tmp_path / "missing.onnx"
        with pytest.raises(FileNotFoundError, match="not found"):
            await storage.download("local:///nonexistent/file", dest)

    @pytest.mark.asyncio()
    async def test_upload_copies_file(self, storage: LocalArtifactStorage, tmp_path: Path) -> None:
        source = tmp_path / "local_model.onnx"
        source.write_bytes(b"upload-data")
        remote = f"local://{tmp_path / 'remote' / 'model.onnx'}"

        await storage.upload(source, remote)
        target = Path(remote.replace("local://", ""))
        assert target.read_bytes() == b"upload-data"

    @pytest.mark.asyncio()
    async def test_upload_creates_parent_dirs(
        self, storage: LocalArtifactStorage, tmp_path: Path
    ) -> None:
        source = tmp_path / "model.bin"
        source.write_bytes(b"data")
        remote = f"local://{tmp_path / 'deep' / 'nested' / 'model.bin'}"

        await storage.upload(source, remote)
        target = Path(remote.replace("local://", ""))
        assert target.exists()

    def test_init_creates_base_dir(self, storage_dir: Path) -> None:
        assert not storage_dir.exists()
        LocalArtifactStorage(storage_dir)
        assert storage_dir.exists()
