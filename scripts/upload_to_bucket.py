"""
Sube FITS locales al bucket de OCI (S3-compatible).

Uso:
    # Sube TODO lo que esté en data/sample_fits/ (respeta estructura sector/epoch)
    py scripts/upload_to_oci.py

    # Sube solo un sector
    py scripts/upload_to_oci.py --sector southern_field_1

    # Sube solo una época
    py scripts/upload_to_oci.py --sector southern_field_1 --epoch 2025-04

    # Verifica qué hay en el bucket (no sube)
    py scripts/upload_to_oci.py --list
"""

import argparse
import sys
from pathlib import Path

import boto3
from botocore.config import Config

# Añadir backend/ al path para importar app.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402


def get_client():
    if not (settings.oci_endpoint and settings.oci_access_key
            and settings.oci_secret_key and settings.oci_bucket):
        print("❌ Faltan variables OCI_* en tu .env")
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


def list_remote(client):
    paginator = client.get_paginator("list_objects_v2")
    prefix = settings.oci_prefix + "/"
    total = 0
    for page in paginator.paginate(Bucket=settings.oci_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            print(f"  {obj['Size']/1e6:8.1f} MB  {obj['Key']}")
            total += 1
    print(f"\nTotal: {total} archivos")


def upload_all(client, sector_filter=None, epoch_filter=None):
    root = Path(settings.data_dir)
    if not root.is_dir():
        print(f"❌ No existe {root}")
        sys.exit(1)

    uploaded = 0
    for sector_dir in sorted(root.iterdir()):
        if not sector_dir.is_dir():
            continue
        if sector_filter and sector_dir.name != sector_filter:
            continue
        for epoch_dir in sorted(sector_dir.iterdir()):
            if not epoch_dir.is_dir():
                continue
            if epoch_filter and epoch_dir.name != epoch_filter:
                continue
            for fits_file in sorted(epoch_dir.glob("*.fits")):
                key = f"{settings.oci_prefix}/{sector_dir.name}/{epoch_dir.name}/{fits_file.name}"
                size_mb = fits_file.stat().st_size / 1e6
                print(f"⬆️  {size_mb:7.1f} MB  {key}")
                client.upload_file(str(fits_file), settings.oci_bucket, key)
                uploaded += 1
    print(f"\n✅ Subidos {uploaded} archivos")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="Lista archivos en el bucket (no sube)")
    p.add_argument("--sector", help="Solo este sector_id")
    p.add_argument("--epoch", help="Solo esta época")
    args = p.parse_args()

    client = get_client()
    if args.list:
        list_remote(client)
    else:
        upload_all(client, args.sector, args.epoch)


if __name__ == "__main__":
    main()