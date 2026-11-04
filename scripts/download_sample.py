"""
Descarga FITS de SPHEREx (Level 2) desde s3://nasa-irsa-spherex (anónimo).

Modos:
  --list                                         Lista planning periods.
  --inspect --planning-period X                  Agrupa obs por (obs_id, visit).
  --find-by-coords --planning-period X --ra Y --dec Z
                                                 Busca obs_ids cerca de (RA,Dec)
                                                 leyendo solo headers de S3.
  --planning-period X --obs-id Y --visit Z --detectors 1..6 --output-dir D
                                                 Descarga.
"""

import argparse
import io
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import boto3
from astropy.io import fits
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
        print("  ⚠️  No se encontraron FITS parseables.")
        return
    dates = sorted({e["date"] for e in entries})
    print(f"  Fechas: {', '.join(dates)}")
    print(f"  Total FITS: {len(entries)}\n")

    groups = defaultdict(list)
    for e in entries:
        groups[(e["obsid"], e["visit"], e["date"])].append(e["det"])

    print(f"  {'obs_id':<8}{'visita':<8}{'fecha':<12}{'detectores'}")
    print(f"  {'-'*8}{'-'*8}{'-'*12}{'-'*20}")
    for (obsid, visit, date), dets in sorted(groups.items()):
        dets_str = " ".join(f"D{d}" for d in sorted(set(dets)))
        print(f"  {obsid:<8}{visit:<8}{date:<12}{dets_str}")
    print(f"\n  Total grupos: {len(groups)}")


# --------------------------------------------------------------------------- #
# Lectura de headers por HTTP Range (sin descargar el FITS completo)
# --------------------------------------------------------------------------- #

def read_header_range(s3, key, max_bytes=100_000):
    """
    Lee los primeros `max_bytes` bytes de un FITS en S3 y extrae RA/Dec/DATE-OBS.

    Intenta primero astropy (parsea el MEF truncado); si falla, hace regex
    directo sobre el texto del header — robusto a truncamientos.
    """
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes=0-{max_bytes-1}")
        content = resp["Body"].read()
    except Exception as e:
        return {"error": str(e)}

    # Fast path: astropy
    try:
        with fits.open(
            io.BytesIO(content),
            memmap=False,
            ignore_missing_end=True,
            ignore_missing_simple=True,
        ) as hdul:
            for hdu in hdul:
                h = hdu.header
                if "CRVAL1" in h and "CRVAL2" in h:
                    return {
                        "ra": float(h["CRVAL1"]),
                        "dec": float(h["CRVAL2"]),
                        "detector": int(h.get("DETECTOR", 0)),
                        "date_obs": str(h.get("DATE-OBS", "")),
                    }
    except Exception:
        pass  # caer al fallback

    # Fallback: regex sobre el buffer crudo
    text = content.decode("ascii", errors="replace")

    def get_val(kw):
        m = re.search(rf"\b{kw}\s*=\s*([^/\n]+)", text)
        if not m:
            return None
        val = m.group(1).strip().strip("'").strip()
        return val or None

    ra_s = get_val("CRVAL1")
    dec_s = get_val("CRVAL2")
    if ra_s is None or dec_s is None:
        return {"error": "No CRVAL in header (astropy + regex)"}
    try:
        return {
            "ra": float(ra_s),
            "dec": float(dec_s),
            "detector": 0,
            "date_obs": get_val("DATE-OBS") or "",
        }
    except ValueError as e:
        return {"error": f"Bad CRVAL: {ra_s}, {dec_s} ({e})"}


def angular_distance(ra1, dec1, ra2, dec2):
    dra = abs(ra1 - ra2)
    if dra > 180:
        dra = 360 - dra
    dra *= math.cos(math.radians((dec1 + dec2) / 2))
    ddec = abs(dec1 - dec2)
    return math.sqrt(dra ** 2 + ddec ** 2)


def find_by_coords(s3, planning_period, target_ra, target_dec,
                   tolerance_deg=2.0, sample_every=1):
    """
    Recorre todos los grupos (obs_id, visit) del planning period, lee solo
    los headers de detector 1 en S3 y devuelve los que caen a
    <tolerance_deg del objetivo.

    sample_every: si >1, revisa solo 1 de cada N grupos (para scan rápido).
    """
    entries = discover(s3, planning_period)
    if not entries:
        print(f"❌ No hay FITS en {planning_period}")
        return []

    # Un key por grupo (obs_id, visit), solo detector 1
    by_group = {}
    for e in entries:
        if e["det"] != 1:
            continue
        key = (e["obsid"], e["visit"])
        if key not in by_group:
            by_group[key] = e["key"]

    groups = sorted(by_group.items())
    print(f"\nEscaneando {len(groups)} grupos (obs_id, visit) en {planning_period}")
    print(f"Objetivo: RA={target_ra:.2f}°, Dec={target_dec:.2f}° ± {tolerance_deg}°")
    print(f"Muestreo: 1 de cada {sample_every} grupos\n")

    matches = []
    checked = 0
    errors = 0
    for i, ((obsid, visit), key) in enumerate(groups[::sample_every], 1):
        if i % 100 == 0:
            print(f"  Revisando {i}/{len(groups[::sample_every])}…", end="\r")
        info = read_header_range(s3, key)
        if "error" in info or info.get("ra") is None:
            errors += 1
            continue
        checked += 1
        dist = angular_distance(info["ra"], info["dec"], target_ra, target_dec)
        if dist <= tolerance_deg:
            matches.append({
                "obs_id": obsid,
                "visit": visit,
                "ra": info["ra"],
                "dec": info["dec"],
                "distance_deg": dist,
                "date_obs": info.get("date_obs", ""),
            })

    print(f"\n  Escaneados OK: {checked}, errores: {errors}, coincidencias: {len(matches)}\n")
    matches.sort(key=lambda m: m["distance_deg"])
    for m in matches[:20]:
        print(f"  obs={m['obs_id']} visit={m['visit']} "
              f"RA={m['ra']:.3f} Dec={m['dec']:.3f} "
              f"dist={m['distance_deg']:.2f}° "
              f"date={m['date_obs'][:19]}")
    if len(matches) > 20:
        print(f"  … y {len(matches) - 20} más")
    return matches


# --------------------------------------------------------------------------- #
# Descarga
# --------------------------------------------------------------------------- #

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
        if not available_dets:
            print(f"   👉 obs_id={obsid} o visit={visit} no existe.")
        else:
            print(f"   👉 Detectores disponibles: {available_dets}")

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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--find-by-coords", action="store_true")
    ap.add_argument("--ra", type=float, help="RA objetivo (grados)")
    ap.add_argument("--dec", type=float, help="Dec objetivo (grados)")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Tolerancia en grados (default 2.0)")
    ap.add_argument("--sample-every", type=int, default=1,
                    help="Revisar 1 de cada N grupos (default 1)")
    ap.add_argument("--planning-period")
    ap.add_argument("--obs-id")
    ap.add_argument("--visit", type=int)
    ap.add_argument("--processing-date", default=None)
    ap.add_argument("--detector", type=int, choices=range(1, 7))
    ap.add_argument("--detectors", type=int, nargs="+", choices=range(1, 7))
    ap.add_argument("--output-dir", default="data/sample_fits")
    ap.add_argument("--scan-all", action="store_true", help="Escanea TODOS los planning periods buscando (ra,dec)")
    args = ap.parse_args()

    s3 = s3_client()

    if args.list:
        for p in list_prefixes(s3, PREFIX_ROOT):
            print(f"  📁 {p.rstrip('/').split('/')[-1]}")
        return

    #if not args.planning_period:
     #   ap.error("--planning-period es obligatorio (excepto con --list)")

    if args.inspect:
        print_inspection(discover(s3, args.planning_period), args.planning_period)
        return

    if args.find_by_coords:
        if args.ra is None or args.dec is None:
            ap.error("--find-by-coords requiere --ra y --dec")
        find_by_coords(s3, args.planning_period, args.ra, args.dec,
                       args.tolerance, args.sample_every)
        return
    
    if getattr(args, "scan_all", False):
        if args.ra is None or args.dec is None:
            ap.error("--scan-all requiere --ra y --dec")
        periods = [p.rstrip('/').split('/')[-1] for p in list_prefixes(s3, PREFIX_ROOT)]
        print(f"Escaneando {len(periods)} planning periods…\n")
        results = []
        for period in periods:
            print(f"\n=== {period} ===")
            matches = find_by_coords(
                s3, period, args.ra, args.dec,
                args.tolerance, args.sample_every,
            )
            if matches:
                results.append((period, matches))
        print(f"\n\n{'='*60}\nRESUMEN")
        print(f"{'='*60}")
        if not results:
            print("❌ No se encontró el objetivo en ningún planning period.")
        else:
            for period, matches in results:
                best = matches[0]
                print(f"\n✅ {period}")
                print(f"   Mejor match: obs={best['obs_id']} visit={best['visit']} "
                      f"dist={best['distance_deg']:.2f}°")
        return

    detectors = args.detectors or ([args.detector] if args.detector else [])
    if not detectors:
        ap.error("Debes indicar --detector o --detectors")
    if not args.obs_id or args.visit is None:
        ap.error("Debes indicar --obs-id y --visit (usa --inspect)")

    ok = select_and_download(
        s3, args.planning_period, args.obs_id, args.visit,
        detectors, args.processing_date, args.output_dir,
    )
    print(f"\n{'='*50}\nDescargados {ok}/{len(detectors)}")
    sys.exit(0 if ok == len(detectors) else 2)


if __name__ == "__main__":
    main()