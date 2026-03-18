"""
S3/MinIO Artifact Storage implementation.

Uses boto3 to interact with S3-compatible object storage.
Falls back gracefully when S3 is not configured.
"""

import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.domain.model_registry.repositories.artifact_storage import ArtifactStorage

logger = logging.getLogger(__name__)


class S3ArtifactStorage(ArtifactStorage):
    """
    Stores and retrieves model artifacts from S3 or MinIO.

    URI format: s3://bucket-name/path/to/artifact
    """

    def __init__(
        self,
        endpoint_url: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str = "us-east-1",
    ) -> None:
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
        )

    def _parse_uri(self, uri: str) -> tuple[str, str]:
        """Parse s3://bucket/key into (bucket, key)."""
        path = uri.replace("s3://", "")
        parts = path.split("/", 1)
        if len(parts) != 2:  # noqa: PLR2004
            raise ValueError(f"Invalid S3 URI: {uri}. Expected s3://bucket/key")
        return parts[0], parts[1]

    async def download(self, remote_uri: str, local_path: Path) -> Path:
        """Download an artifact from S3 to a local path."""
        bucket, key = self._parse_uri(remote_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._client.download_file(bucket, key, str(local_path))
            logger.info("Downloaded %s → %s", remote_uri, local_path)
            return local_path
        except (BotoCoreError, ClientError) as e:
            raise OSError(f"Failed to download artifact from {remote_uri}: {e}") from e

    async def upload(self, local_path: Path, remote_uri: str) -> None:
        """Upload a local artifact to S3."""
        bucket, key = self._parse_uri(remote_uri)

        if not local_path.exists():
            raise FileNotFoundError(f"Local artifact {local_path} not found")

        try:
            self._client.upload_file(str(local_path), bucket, key)
            logger.info("Uploaded %s → %s", local_path, remote_uri)
        except (BotoCoreError, ClientError) as e:
            raise OSError(f"Failed to upload artifact to {remote_uri}: {e}") from e

    async def exists(self, remote_uri: str) -> bool:
        """Check if an artifact exists in S3."""
        bucket, key = self._parse_uri(remote_uri)
        try:
            self._client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False

    async def delete(self, remote_uri: str) -> None:
        """Delete an artifact from S3."""
        bucket, key = self._parse_uri(remote_uri)
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
            logger.info("Deleted %s", remote_uri)
        except (BotoCoreError, ClientError) as e:
            raise OSError(f"Failed to delete artifact {remote_uri}: {e}") from e
