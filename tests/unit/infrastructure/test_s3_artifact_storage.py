"""Unit tests for S3ArtifactStorage."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from src.infrastructure.artifact_storage.s3_artifact_storage import S3ArtifactStorage


@pytest.fixture
def mock_boto3_client() -> MagicMock:
    with patch("src.infrastructure.artifact_storage.s3_artifact_storage.boto3") as mock:
        client = MagicMock()
        mock.client.return_value = client
        yield client


@pytest.fixture
def storage(mock_boto3_client: MagicMock) -> S3ArtifactStorage:
    return S3ArtifactStorage(
        endpoint_url="http://minio:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )


class TestS3URIParsing:
    def test_valid_uri(self, storage: S3ArtifactStorage) -> None:
        bucket, key = storage._parse_uri("s3://my-bucket/models/v1/model.onnx")
        assert bucket == "my-bucket"
        assert key == "models/v1/model.onnx"

    def test_invalid_uri_raises(self, storage: S3ArtifactStorage) -> None:
        with pytest.raises(ValueError, match="Invalid S3 URI"):
            storage._parse_uri("s3://bucket-only")


class TestS3Download:
    async def test_download_success(
        self,
        storage: S3ArtifactStorage,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "model.onnx"
        result = await storage.download("s3://bucket/key/model.onnx", local)
        assert result == local
        mock_boto3_client.download_file.assert_called_once_with(
            "bucket", "key/model.onnx", str(local)
        )

    async def test_download_failure_raises(
        self,
        storage: S3ArtifactStorage,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_boto3_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "download"
        )
        with pytest.raises(OSError, match="Failed to download"):
            await storage.download("s3://bucket/missing.onnx", tmp_path / "out")


class TestS3Upload:
    async def test_upload_success(
        self,
        storage: S3ArtifactStorage,
        mock_boto3_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        local = tmp_path / "model.onnx"
        local.write_text("fake model data")
        await storage.upload(local, "s3://bucket/models/model.onnx")
        mock_boto3_client.upload_file.assert_called_once()

    async def test_upload_missing_file_raises(
        self, storage: S3ArtifactStorage
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await storage.upload(Path("/nonexistent/model.onnx"), "s3://b/k")


class TestS3Exists:
    async def test_exists_true(
        self, storage: S3ArtifactStorage, mock_boto3_client: MagicMock
    ) -> None:
        result = await storage.exists("s3://bucket/model.onnx")
        assert result is True

    async def test_exists_false(
        self, storage: S3ArtifactStorage, mock_boto3_client: MagicMock
    ) -> None:
        mock_boto3_client.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not found"}}, "head"
        )
        result = await storage.exists("s3://bucket/missing.onnx")
        assert result is False


class TestS3Delete:
    async def test_delete_success(
        self, storage: S3ArtifactStorage, mock_boto3_client: MagicMock
    ) -> None:
        await storage.delete("s3://bucket/model.onnx")
        mock_boto3_client.delete_object.assert_called_once_with(
            Bucket="bucket", Key="model.onnx"
        )
