"""
Sube los FITS de la timeline a OCI.

Estructura remota:
    fits/timeline/{sector}/{obs_id}/v{N}/{filename}

Uso:
    py scripts/upload_timeline_to_oci.py --sector orion_nebula --obs-id 0685
    py scripts/upload_timeline_to_oci.py --sector orion_nebula --obs-id 0685 --list
"""

import argparse
import sys
from pathlib import Path

import boto3
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def get_client():
    if not (
        settings.oci_endpoint
        and settings.oci_access_key
        and settings.oci_secret_key
        and settings.oci_bucket
    ):
        print("❌ Faltan credenciales OCI en .env")
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=settings.oci_endpoint,
        aws_access_key_id=settings.oci_access_key,
        aws_secret_access_key=settings.oci_secret_key,
        region_name=settings.oci_region or None,
        config=Config(
            signature_version="s3v4",
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def list_remote(client, prefix: str):
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=settings.oci_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(f"  {obj['Size'] / 1e6:8.1f} MB  {obj['Key']}")
            total += 1
    print(f"\nTotal: {total} archivos")


def upload(client, sector: str, obs_id: str):
    root = Path(f"data/sample_fits/timeline/{sector}/{obs_id}")
    if not root.is_dir():
        print(f"❌ No existe {root}")
        sys.exit(1)

    uploaded = 0
    skipped = 0

    # Recorrer v1/, v2/, ...
    for visit_dir in sorted(root.iterdir()):
        if not visit_dir.is_dir() or not visit_dir.name.startswith("v"):
            continue
        visit_name = visit_dir.name  # "v1"

        for fits_file in sorted(visit_dir.glob("*.fits")):
            key = f"timeline/{sector}/{obs_id}/{visit_name}/{fits_file.name}"
            size_mb = fits_file.stat().st_size / 1e6

            # Verificar si ya existe en remoto con mismo tamaño
            try:
                head = client.head_object(Bucket=settings.oci_bucket, Key=key)
                if head["ContentLength"] == fits_file.stat().st_size:
                    print(f"  ⏭️  {key} (ya existe)")
                    skipped += 1
                    continue
            except client.exceptions.NoSuchKey:
                pass
            except Exception:
                pass  # si falla head, intentar subir igual

            print(f"  ⬆️  {size_mb:6.1f} MB  {key}")
            client.upload_file(str(fits_file), settings.oci_bucket, key)
            uploaded += 1

    print(f"\n✅ Subidos: {uploaded}")
    print(f"⏭️  Saltados: {skipped}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", required=True)
    ap.add_argument("--obs-id", required=True)
    ap.add_argument(
        "--list", action="store_true", help="Solo listar lo que hay en OCI"
    )
    args = ap.parse_args()

    client = get_client()

    if args.list:
        prefix = f"timeline/{args.sector}/{args.obs_id}/"
        print(f"Archivos en s3://{settings.oci_bucket}/{prefix}\n")
        list_remote(client, prefix)
    else:
        print(
            f"Subiendo timeline de {args.sector}/{args.obs_id} a "
            f"s3://{settings.oci_bucket}/timeline/...\n"
        )
        upload(client, args.sector, args.obs_id)


if __name__ == "__main__":
    main()