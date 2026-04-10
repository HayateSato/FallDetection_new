"""
MinIO (S3-compatible object store) client for the Fall Detection datalake.

Used by ml_server to:
  - List available sensor recording CSV files
  - Upload new CSV files uploaded from the operator dashboard
  - Download CSV files for offline inference replay

Environment variables:
  MINIO_URL          — MinIO S3 endpoint  (default: http://minio:9000 in Docker)
  MINIO_ACCESS_KEY   — MinIO access key   (default: minioadmin)
  MINIO_SECRET_KEY   — MinIO secret key   (default: minioadmin)
  MINIO_BUCKET       — Bucket name        (default: sensor-recordings)
"""

import logging
import os

logger = logging.getLogger(__name__)

MINIO_URL        = os.getenv("MINIO_URL",        "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET",     "sensor-recordings")


def _get_client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=MINIO_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",   # boto3 requires a region; value is ignored by MinIO
    )


def ensure_bucket_exists() -> None:
    """Create the bucket if it does not already exist."""
    client = _get_client()
    try:
        client.head_bucket(Bucket=MINIO_BUCKET)
    except Exception:
        client.create_bucket(Bucket=MINIO_BUCKET)
        logger.info(f"Created MinIO bucket: {MINIO_BUCKET}")


def list_csv_files() -> list:
    """Return a list of CSV file metadata dicts from the bucket."""
    client = _get_client()
    ensure_bucket_exists()
    resp = client.list_objects_v2(Bucket=MINIO_BUCKET)
    files = []
    for obj in resp.get("Contents", []):
        if obj["Key"].lower().endswith(".csv"):
            files.append({
                "name": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
    return sorted(files, key=lambda f: f["last_modified"], reverse=True)


def upload_file(filename: str, file_obj) -> None:
    """Upload a file-like object to the bucket under the given filename."""
    ensure_bucket_exists()
    client = _get_client()
    client.upload_fileobj(file_obj, MINIO_BUCKET, filename)
    logger.info(f"Uploaded {filename!r} to MinIO bucket {MINIO_BUCKET!r}")


def download_file_bytes(filename: str) -> bytes:
    """Download a file from the bucket and return its raw bytes."""
    client = _get_client()
    resp = client.get_object(Bucket=MINIO_BUCKET, Key=filename)
    return resp["Body"].read()
