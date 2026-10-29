"""
Cliente de almacenamiento para OCI Object Storage (S3-compatible).

Uso:
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
    return bool(
        settings.oci_endpoint
        and settings.oci_access_key
        and settings.oci_secret_key
        and settings.oci_bucket
    )


def _get_client():
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
            # 👇 Fix OCI: imprescindible para UploadPart
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _remote_key(sector_id: str, epoch: str, filename: str) -> str:
    return f"{settings.oci_prefix}/{sector_id}/{epoch}/{filename}"


def _local_dir(sector_id: str, epoch: str) -> Path:
    return Path(settings.data_dir) / sector_id / epoch


def list_remote_files(sector_id: str, epoch: str) -> list[dict]:
    """
    Lista los FITS disponibles en el bucket para un sector/epoch.
    Devuelve [{"filename": ..., "size_mb": ..., "key": ...}, ...].
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
    """Descarga un objeto del bucket al disco. Crea directorios si faltan."""
    client = _get_client()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("OCI download: s3://%s/%s → %s",
                settings.oci_bucket, remote_key, local_path)
    try:
        client.download_file(settings.oci_bucket, remote_key, str(local_path))
    except (ClientError, BotoCoreError) as e:
        logger.exception("OCI download failed for key=%s", remote_key)
        # Borrar archivo parcial si quedó
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
    Si el FITS ya existe localmente (cualquier combinación compatible con los
    filtros detector/obs_id/visit), lo devuelve. Si no, intenta descargarlo
    desde OCI.

    Lanza FileNotFoundError si no existe localmente y no se puede descargar.
    """
    # 1) ¿Ya está local?
    try:
        return find_fits_file(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except (FileNotFoundError, ValueError):
        pass  # intentar descarga

    # 2) ¿Descarga habilitada y configurada?
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

    # 3) Listar remoto y filtrar por detector/obs_id/visit
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
    """Sube un FITS local al bucket. Devuelve la remote_key."""
    client = _get_client()
    remote_key = _remote_key(sector_id, epoch, local_path.name)
    logger.info("OCI upload: %s → s3://%s/%s",
                local_path, settings.oci_bucket, remote_key)
    client.upload_file(str(local_path), settings.oci_bucket, remote_key)
    logger.info("  ✅ subido (%.1f MB)", local_path.stat().st_size / 1e6)
    return remote_key