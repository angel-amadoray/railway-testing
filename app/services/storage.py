"""
Storage client for OCI Object Storage (S3-compatible).

Usage:
    from app.services.storage import download_fits_if_missing
    path = download_fits_if_missing(
        sector_id="southern_field_1",
        epoch="2025-04",
        detector=1,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.config import settings
from app.services.fits_loader import (
    find_fits_file,
    find_sector_files,
    parse_filename_metadata,
)

logger = logging.getLogger(__name__)


def _is_configured() -> bool:
    """Check whether the minimum configuration for accessing OCI exists.

    Check the endpoint, access key, secret, and bucket name through the
    application's global configuration.

    Returns:
        ``True`` when all values are present; otherwise, ``False``.
    """
    return bool(
        settings.oci_endpoint
        and settings.oci_access_key
        and settings.oci_secret_key
        and settings.oci_bucket
    )


def _get_client():
    """Create the boto3 client compatible with OCI Object Storage.

    Configure the endpoint, credentials, region, S3v4 signature, and checksum
    validation compatible with OCI multipart operations.

    Returns:
        Boto3 client configured for the OCI bucket.

    Raises:
        RuntimeError: If storage options or credentials are missing.
    """
    if not _is_configured():
        raise RuntimeError(
            "OCI storage not configured. Set OCI_ENDPOINT, OCI_ACCESS_KEY, "
            "OCI_SECRET_KEY, OCI_BUCKET."
        )
    return boto3.client(
        "s3",
        endpoint_url=settings.oci_endpoint,
        aws_access_key_id=settings.oci_access_key,
        aws_secret_access_key=settings.oci_secret_key,
        region_name=settings.oci_region or None,
        config=Config(
            signature_version="s3v4",
            # Required for OCI UploadPart
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _remote_key(sector_id: str, epoch: str, filename: str) -> str:
    """Build the full S3 key for a FITS file.

    The key is formed as ``<prefix>/<sector>/<epoch>/<filename>`` using the
    prefix configured in the application.
    """
    return f"{settings.oci_prefix}/{sector_id}/{epoch}/{filename}"


def _local_dir(sector_id: str, epoch: str) -> Path:
    """Build the local data directory for a sector and epoch."""
    return Path(settings.data_dir) / sector_id / epoch


def list_remote_files(sector_id: str, epoch: str) -> list[dict]:
    """
    List FITS files available remotely for a sector and epoch.

    Iterate through all ``list_objects_v2`` pages, ignore objects that do not
    end in ``.fits``, and sort results by name.

    Args:
        sector_id: Remote sector identifier.
        epoch: Epoch whose files should be queried.

    Returns:
        List of dictionaries with ``filename``, approximate size in MB, and the
        complete S3 ``key``.

    Raises:
        RuntimeError: If the client cannot be configured.
        ClientError or BotoCoreError: If OCI rejects or cannot complete bucket
            enumeration.
    """
    client = _get_client()
    prefix = f"{settings.oci_prefix}/{sector_id}/{epoch}/"
    out = []
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.oci_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".fits"):
                    continue
                out.append({
                    "filename": key.split("/")[-1],
                    "size_mb": round(obj["Size"] / 1e6, 2),
                    "key": key,
                })
    except (ClientError, BotoCoreError) as e:
        logger.exception("OCI list_objects_v2 failed for prefix=%s", prefix)
        raise
    return sorted(out, key=lambda o: o["filename"])


def download_file(remote_key: str, local_path: Path) -> Path:
    """Download a remote object and save it to a local path.

    Create parent directories before downloading. If the transfer fails and
    leaves a partial file, attempt to remove it to avoid leaving a corrupt FITS.

    Args:
        remote_key: Object key within the OCI bucket.
        local_path: Local destination path.

    Returns:
        The same local path after a successful download.

    Raises:
        RuntimeError: If OCI configuration is missing.
        ClientError or BotoCoreError: If the download fails.
    """
    client = _get_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("OCI download: s3://%s/%s → %s",
                settings.oci_bucket, remote_key, local_path)
    try:
        client.download_file(settings.oci_bucket, remote_key, str(local_path))
    except (ClientError, BotoCoreError) as e:
        logger.exception("OCI download failed for key=%s", remote_key)
        # Remove any partial file
        if local_path.exists():
            try: local_path.unlink()
            except Exception: pass
        raise
    logger.info("  ✅ %.1f MB", local_path.stat().st_size / 1e6)
    return local_path


def download_fits_if_missing(
    sector_id: str,
    epoch: str,
    detector: int,
    obs_id: str | None = None,
    visit: int | None = None,
) -> Path:
    """
    Get a local FITS file or download the only matching candidate from OCI.

    First search the disk using detector, observation, and visit. If it does not
    exist, check that downloads are enabled, list remote objects, filter their
    metadata, and download the only matching result.

    Args:
        sector_id: Sector identifier.
        epoch: Observation epoch.
        detector: Required detector number.
        obs_id: Optional observation identifier.
        visit: Optional visit number.

    Returns:
        Local path of the found or downloaded FITS file.

    Raises:
        FileNotFoundError: If no local/remote file exists, downloads are
            disabled, OCI is not configured, or enumeration fails.
        ValueError: If multiple ambiguous remote candidates exist.
    """
    # 1) Is it already local?
    try:
        return find_fits_file(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except (FileNotFoundError, ValueError):
        pass  # try downloading

    # 2) Are downloads enabled and configured?
    if not settings.oci_download_enabled:
        raise FileNotFoundError(
            f"FITS not local and OCI download disabled "
            f"(sector={sector_id} epoch={epoch} detector={detector})."
        )
    if not _is_configured():
        raise FileNotFoundError(
            f"FITS not local and OCI not configured. Set OCI_ENDPOINT, "
            f"OCI_ACCESS_KEY, OCI_SECRET_KEY, OCI_BUCKET env vars. "
            f"(sector={sector_id} epoch={epoch} detector={detector})"
        )

    # 3) List remotely and filter by detector/obs_id/visit
    try:
        remote_files = list_remote_files(sector_id, epoch)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not list OCI bucket for {sector_id}/{epoch}: {e}"
        )

    matches = []
    for f in remote_files:
        meta = parse_filename_metadata(f["filename"])
        if not meta:
            continue
        if meta.get("detector") != detector:
            continue
        if obs_id is not None and meta.get("obs_id") != obs_id:
            continue
        if visit is not None and meta.get("visit") != visit:
            continue
        matches.append(f)

    if not matches:
        raise FileNotFoundError(
            f"No FITS in OCI for sector={sector_id} epoch={epoch} "
            f"detector={detector} obs_id={obs_id} visit={visit}. "
            f"Files available: {[f['filename'] for f in remote_files[:5]]}"
        )
    if len(matches) > 1:
        names = [m["filename"] for m in matches[:5]]
        raise ValueError(
            f"Ambiguous match ({len(matches)} files). Especifica obs_id y/o visit. "
            f"Candidates: {names}"
        )

    chosen = matches[0]
    dest = _local_dir(sector_id, epoch) / chosen["filename"]
    return download_file(chosen["key"], dest)


def upload_file(local_path: Path, sector_id: str, epoch: str) -> str:
    """Upload a local file to the OCI bucket and return its remote key.

    Preserve the local filename and place it under the configured prefix, sector,
    and epoch.

    Args:
        local_path: Path of the file to upload.
        sector_id: Destination sector identifier.
        epoch: Destination observation epoch.

    Returns:
        Remote S3 key used for the object.

    Raises:
        RuntimeError: If OCI is not configured.
        ClientError or BotoCoreError: If the upload fails.
    """
    client = _get_client()
    remote_key = _remote_key(sector_id, epoch, local_path.name)
    logger.info("OCI upload: %s → s3://%s/%s",
                local_path, settings.oci_bucket, remote_key)
    client.upload_file(str(local_path), settings.oci_bucket, remote_key)
    logger.info("  ✅ subido (%.1f MB)", local_path.stat().st_size / 1e6)
    return remote_key
