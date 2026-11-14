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
    """Comprueba si existe la configuración mínima para acceder a OCI.

    Verifica endpoint, clave de acceso, secreto y nombre de bucket mediante la
    configuración global de la aplicación.

    Returns:
        ``True`` cuando todos los valores están presentes; en otro caso,
        ``False``.
    """
    return bool(
        settings.oci_endpoint
        and settings.oci_access_key
        and settings.oci_secret_key
        and settings.oci_bucket
    )


def _get_client():
    """Crea el cliente boto3 S3 compatible con OCI Object Storage.

    Configura endpoint, credenciales, región, firma S3v4 y validación de checksums
    compatible con operaciones multipart de OCI.

    Returns:
        Cliente boto3 configurado para el bucket OCI.

    Raises:
        RuntimeError: Si faltan opciones o credenciales de almacenamiento.
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
            # 👇 Fix OCI: imprescindible para UploadPart
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def _remote_key(sector_id: str, epoch: str, filename: str) -> str:
    """Construye la clave S3 completa de un archivo FITS.

    La clave se forma como ``<prefijo>/<sector>/<época>/<archivo>`` usando el
    prefijo configurado en la aplicación.
    """
    return f"{settings.oci_prefix}/{sector_id}/{epoch}/{filename}"


def _local_dir(sector_id: str, epoch: str) -> Path:
    """Construye el directorio local de datos para un sector y una época."""
    return Path(settings.data_dir) / sector_id / epoch


def list_remote_files(sector_id: str, epoch: str) -> list[dict]:
    """
    Lista los archivos FITS disponibles remotamente para un sector y época.

    Recorre todas las páginas de ``list_objects_v2``, ignora objetos que no
    terminen en ``.fits`` y ordena los resultados por nombre.

    Args:
        sector_id: Identificador del sector remoto.
        epoch: Época cuyos archivos se desean consultar.

    Returns:
        Lista de diccionarios con ``filename``, tamaño aproximado en MB y
        ``key`` S3 completa.

    Raises:
        RuntimeError: Si el cliente no puede configurarse.
        ClientError o BotoCoreError: Si OCI rechaza o no puede completar la
            enumeración del bucket.
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
    """Descarga un objeto remoto y lo guarda en una ruta local.

    Crea los directorios padres antes de descargar. Si la transferencia falla y
    dejó un archivo parcial, intenta eliminarlo para no dejar un FITS corrupto.

    Args:
        remote_key: Clave del objeto dentro del bucket OCI.
        local_path: Ruta local de destino.

    Returns:
        La misma ruta local después de una descarga exitosa.

    Raises:
        RuntimeError: Si falta configuración OCI.
        ClientError o BotoCoreError: Si falla la descarga.
    """
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
    Obtiene un FITS local o descarga el único candidato compatible desde OCI.

    Primero busca en disco usando detector, observación y visita. Si no existe,
    comprueba que las descargas estén habilitadas, lista los objetos remotos,
    filtra sus metadatos y descarga el único resultado coincidente.

    Args:
        sector_id: Identificador del sector.
        epoch: Época de observación.
        detector: Número de detector requerido.
        obs_id: Identificador de observación opcional.
        visit: Número de visita opcional.

    Returns:
        Ruta local del FITS encontrado o descargado.

    Raises:
        FileNotFoundError: Si no hay archivo local/remoto, las descargas están
            deshabilitadas, OCI no está configurado o falla la enumeración.
        ValueError: Si hay múltiples candidatos remotos ambiguos.
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
    """Sube un archivo local al bucket OCI y devuelve su clave remota.

    El nombre del archivo local se conserva y se coloca bajo el prefijo,
    sector y época configurados.

    Args:
        local_path: Ruta del archivo que se desea subir.
        sector_id: Identificador del sector de destino.
        epoch: Época de observación de destino.

    Returns:
        Clave S3 remota usada para el objeto.

    Raises:
        RuntimeError: Si OCI no está configurado.
        ClientError o BotoCoreError: Si falla la subida.
    """
    client = _get_client()
    remote_key = _remote_key(sector_id, epoch, local_path.name)
    logger.info("OCI upload: %s → s3://%s/%s",
                local_path, settings.oci_bucket, remote_key)
    client.upload_file(str(local_path), settings.oci_bucket, remote_key)
    logger.info("  ✅ subido (%.1f MB)", local_path.stat().st_size / 1e6)
    return remote_key
