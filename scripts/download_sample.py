"""
Descarga FITS de SPHEREx (Level 2) desde s3://nasa-irsa-spherex (anónimo).

Modos:
  --list
        Lista planning periods disponibles.

  --inspect --planning-period X
        Lista todas las observaciones agrupadas por (obs_id, visita, fecha)
        con sus detectores disponibles.

  --planning-period X --obs-id Y --visit Z --detectors 1 2 3 --output-dir DIR
        Descarga los FITS seleccionados.

Ejemplo (dos épocas usando dos visitas del MISMO obs_id, misma región del cielo):
  py download_sample.py --planning-period 2025W18_2B --obs-id 0001 --visit 1 ^
      --detectors 1 2 3 --output-dir ..\\data\\sample_fits\\orion_nebula\\2025-04
  py download_sample.py --planning-period 2025W18_2B --obs-id 0001 --visit 2 ^
      --detectors 1 2 3 --output-dir ..\\data\\sample_fits\\orion_nebula\\2025-09
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "nasa-irsa-spherex"
REGION = "us-east-1"
PREFIX_ROOT = "qr2/level2/"

# level2_{plan}_{obsid}_{visit}D{det}_spx_l2b-v{ver}-{YYYY-DDD}.fits
PATTERN = re.compile(
    r"level2_(?P<plan>.+?)_(?P<obsid>\d+)_(?P<visit>\d+)D(?P<det>\d+)"
    r"_spx_l2b-v(?P<ver>\d+)-(?P<date>\d{4}-\d{3})\.fits$"
)


def s3_client():
    return boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))


def list_prefixes(s3, prefix):
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"] for p in resp.get("CommonPrefixes", [])]


def list_all_keys(s3, prefix):
    keys, paginator = [], s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def discover(s3, planning_period):
    """Devuelve lista de dicts con metadata parseada de cada FITS."""
    keys = list_all_keys(s3, f"{PREFIX_ROOT}{planning_period}/")
    entries = []
    for k in keys:
        m = PATTERN.match(k.split("/")[-1])
        if m:
            d = m.groupdict()
            d["det"] = int(d["det"])
            d["visit"] = int(d["visit"])
            d["key"] = k
            entries.append(d)
    return entries


def print_inspection(entries, planning_period):
    print(f"\nPlanning period: {planning_period}")
    if not entries:
        print("  ⚠️  No se encontraron FITS parseables. ¿Nombre del período correcto?")
        return

    dates = sorted({e["date"] for e in entries})
    print(f"  Fechas de procesado disponibles: {', '.join(dates)}")
    print(f"  Total FITS: {len(entries)}\n")

    groups = defaultdict(list)
    for e in entries:
        groups[(e["obsid"], e["visit"], e["date"])].append(e["det"])

    print(f"  {'obs_id':<8}{'visita':<8}{'fecha':<12}{'detectores'}")
    print(f"  {'-'*8}{'-'*8}{'-'*12}{'-'*20}")
    for (obsid, visit, date), dets in sorted(groups.items()):
        dets_str = " ".join(f"D{d}" for d in sorted(set(dets)))
        print(f"  {obsid:<8}{visit:<8}{date:<12}{dets_str}")
    print(f"\n  Total grupos (obs_id, visita): {len(groups)}")


def select_and_download(s3, planning_period, obsid, visit, detectors,
                        processing_date, out_dir):
    entries = discover(s3, planning_period)
    if not entries:
        print(f"❌ No hay FITS en {planning_period}", file=sys.stderr)
        return 0

    if processing_date is None:
        processing_date = max(e["date"] for e in entries)
        print(f"ℹ️  --processing-date no especificado; usando la más reciente: {processing_date}")

    wanted = {(obsid, int(visit), processing_date, det) for det in detectors}
    found = {
        (e["obsid"], e["visit"], e["date"], e["det"]): e
        for e in entries
        if e["obsid"] == obsid
        and e["visit"] == int(visit)
        and e["date"] == processing_date
        and e["det"] in detectors
    }

    missing = wanted - set(found.keys())
    if missing:
        print("⚠️  No se encontraron los siguientes FITS:")
        for o, v, d, det in sorted(missing):
            print(f"     obs={o} visit={v} date={d} D{det}")
        available_dets = sorted({
            e["det"] for e in entries
            if e["obsid"] == obsid and e["visit"] == int(visit) and e["date"] == processing_date
        })
        available_dates = sorted({e["date"] for e in entries})
        if not available_dets:
            print(f"   👉 Probablemente obs_id={obsid} o visit={visit} no existe.")
            print(f"      Ejecuta --inspect para ver combinaciones válidas.")
        else:
            print(f"   👉 Detectores disponibles para ese grupo: {available_dets}")
            print(f"      Fechas disponibles: {available_dates}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    for key in sorted(found):
        e = found[key]
        dest = out_dir / e["key"].split("/")[-1]
        print(f"\n⬇️  D{e['det']}  {e['key'].split('/')[-1]}")
        print(f"    → {dest.resolve()}")
        s3.download_file(BUCKET, e["key"], str(dest))
        print(f"    ✅ {dest.stat().st_size / 1e6:.1f} MB")
        ok += 1

    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--planning-period")
    ap.add_argument("--obs-id")
    ap.add_argument("--visit", type=int)
    ap.add_argument("--processing-date", default=None)
    ap.add_argument("--detector", type=int, choices=range(1, 7))
    ap.add_argument("--detectors", type=int, nargs="+", choices=range(1, 7))
    ap.add_argument("--output-dir", default="data/sample_fits")
    args = ap.parse_args()

    s3 = s3_client()

    if args.list:
        prefixes = list_prefixes(s3, PREFIX_ROOT)
        print("Planning periods disponibles:")
        for p in prefixes:
            print(f"  📁 {p.rstrip('/').split('/')[-1]}")
        return

    if not args.planning_period:
        ap.error("--planning-period es obligatorio (excepto con --list)")

    if args.inspect:
        print_inspection(discover(s3, args.planning_period), args.planning_period)
        return

    detectors = args.detectors or ([args.detector] if args.detector else [])
    if not detectors:
        ap.error("Debes indicar --detector o --detectors")
    if not args.obs_id or args.visit is None:
        ap.error("Debes indicar --obs-id y --visit (usa --inspect para descubrirlos)")

    ok = select_and_download(
        s3, args.planning_period, args.obs_id, args.visit,
        detectors, args.processing_date, args.output_dir,
    )
    print(f"\n{'='*50}\nDescargados {ok}/{len(detectors)}")
    sys.exit(0 if ok == len(detectors) else 2)


if __name__ == "__main__":
    main()