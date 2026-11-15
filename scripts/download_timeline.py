"""
Descarga visitas específicas del mismo obs_id en todos los planning periods.

Uso:
    # Listar visitas disponibles
    py scripts/download_timeline.py --obs-id 0685 --sector orion_nebula --list-only

    # Descargar visitas específicas (por número de visit)
    py scripts/download_timeline.py --obs-id 0685 --sector orion_nebula --visits 2 3

    # Descargar las N primeras visitas (más antiguas)
    py scripts/download_timeline.py --obs-id 0685 --sector orion_nebula --max-visits 3
"""

import argparse
import re
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "nasa-irsa-spherex"
REGION = "us-east-1"
PREFIX_ROOT = "qr2/level2/"

PATTERN = re.compile(
    r"level2_(?P<plan>.+?)_(?P<obsid>\d+)_(?P<visit>\d+)D(?P<det>\d+)"
    r"_spx_l2b-v(?P<ver>\d+)-(?P<date>\d{4}-\d{3})\.fits$"
)


def s3_client():
    return boto3.client(
        "s3", region_name=REGION, config=Config(signature_version=UNSIGNED)
    )


def list_prefixes(s3, prefix):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]


def list_all_keys(s3, prefix):
    keys, paginator = [], s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def scan_obsid(s3, obsid: str) -> dict[int, list[dict]]:
    """Devuelve {visit_number: [entries]} para el obs_id dado."""
    periods = [
        p.rstrip("/").split("/")[-1] for p in list_prefixes(s3, PREFIX_ROOT)
    ]
    print(f"Escaneando {len(periods)} planning periods...")
    all_by_visit: dict[int, list[dict]] = {}

    for period in periods:
        prefix = f"{PREFIX_ROOT}{period}/"
        try:
            keys = list_all_keys(s3, prefix)
        except Exception:
            continue
        for k in keys:
            m = PATTERN.match(k.split("/")[-1])
            if not m:
                continue
            d = m.groupdict()
            if d["obsid"] != obsid:
                continue
            d["det"] = int(d["det"])
            d["visit"] = int(d["visit"])
            d["version"] = int(d["ver"])
            d["key"] = k
            all_by_visit.setdefault(d["visit"], []).append(d)
        if any(d["obsid"] == obsid for d in (PATTERN.match(k.split("/")[-1]).groupdict() for k in keys if PATTERN.match(k.split("/")[-1]))):
            print(f"  ✅ {period}")

    return all_by_visit


def print_visits(by_visit: dict[int, list[dict]]):
    print(f"\n{'=' * 65}")
    print(f"VISITAS DISPONIBLES para obs_id")
    print(f"{'=' * 65}\n")
    for v in sorted(by_visit.keys()):
        entries = by_visit[v]
        dets = sorted({e["det"] for e in entries})
        dates = sorted({e["date"] for e in entries})
        plans = sorted({e["plan"] for e in entries})
        print(f"  Visita {v}:")
        print(f"    Detectores: {dets}")
        print(f"    Fechas: {dates}")
        print(f"    Plan: {plans[0]}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-id", required=True)
    ap.add_argument("--sector", required=True)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--visits", type=int, nargs="+", help="Visitas específicas a descargar")
    ap.add_argument("--max-visits", type=int, help="Descargar las N visitas más antiguas")
    ap.add_argument("--detectors", type=int, nargs="+", choices=range(1, 7))
    args = ap.parse_args()

    s3 = s3_client()
    by_visit = scan_obsid(s3, args.obs_id)

    if not by_visit:
        print(f"\n❌ obs_id={args.obs_id} no encontrado.")
        sys.exit(1)

    if args.detectors:
        for v in by_visit:
            by_visit[v] = [e for e in by_visit[v] if e["det"] in args.detectors]
        by_visit = {v: es for v, es in by_visit.items() if es}

    print_visits(by_visit)

    if args.list_only:
        print("💡 Usa --visits 2 3 o --max-visits 3 para descargar")
        return

    # Determinar qué visitas descargar
    if args.visits:
        to_download = args.visits
    elif args.max_visits:
        to_download = sorted(by_visit.keys())[: args.max_visits]
    else:
        print("❌ Debes indicar --visits o --max-visits")
        sys.exit(1)

    # Descargar
    dest_root = Path(f"data/sample_fits/timeline/{args.sector}/{args.obs_id}")
    dest_root.mkdir(parents=True, exist_ok=True)
    print(f"📁 Destino: {dest_root.resolve()}\n")

    downloaded = skipped = errors = 0

    for v in to_download:
        if v not in by_visit:
            print(f"⚠ Visita {v} no existe, saltando")
            continue

        # Mejor versión por detector
        best: dict[int, dict] = {}
        for e in by_visit[v]:
            if e["det"] not in best or e["version"] > best[e["det"]]["version"]:
                best[e["det"]] = e

        dest_dir = dest_root / f"v{v}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"── Visita {v} ──")
        for det in sorted(best.keys()):
            e = best[det]
            filename = e["key"].split("/")[-1]
            dest = dest_dir / filename

            if dest.exists():
                print(f"  ⏭️  D{det}: ya existe")
                skipped += 1
                continue

            print(f"  ⬇️  D{det}: {filename}")
            try:
                s3.download_file(BUCKET, e["key"], str(dest))
                print(f"      ✅ {dest.stat().st_size / 1e6:.1f} MB")
                downloaded += 1
            except Exception as ex:
                print(f"      ❌ {ex}")
                errors += 1
        print()

    print(f"{'=' * 65}")
    print(f"✅ Descargados: {downloaded}")
    print(f"⏭️  Saltados: {skipped}")
    if errors:
        print(f"❌ Errores: {errors}")


if __name__ == "__main__":
    main()