import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.core import cache
from app.models.analytics import DifferenceResponse, KnownObject, KnownObjectsResponse, IdentifyResponse
from app.services import diff_engine as de
from app.services import fits_loader as fl
from app.services import jpl_api

from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])

# Helpers
def _parse_ra_deg(ra_str: str | None) -> float | None:
    """'10:16:58.22' (h:m:s) → grados decimales."""
    if not ra_str:
        return None
    try:
        coord = SkyCoord(ra_str, "0d", unit=(u.hourangle, u.deg))
        return float(coord.ra.deg)
    except Exception:
        return None


def _parse_dec_deg(dec_str: str | None) -> float | None:
    """\"+07 14'57\\\"\" → grados decimales."""
    if not dec_str:
        return None
    try:
        normalized = (
            dec_str.replace(" ", ":").replace("'", ":").replace('"', "").strip()
        )
        coord = SkyCoord("0d", normalized, unit=(u.deg, u.deg))
        return float(coord.dec.deg)
    except Exception:
        return None


@router.get("/difference", response_model=DifferenceResponse)
def difference(
    request: Request,
    sector_id: str,
    epoch_from: str,
    epoch_to: str,
    detector: int = Query(1, ge=1, le=6),
    # --- Background ---
    background_mode: str = Query("row", pattern="^(row|2d|none)$"),
    background_downsample: int = Query(8, ge=2, le=32),
    # --- Diferencia ---
    ratio: bool = Query(False),
    # --- Máscaras de estrellas ---
    mask_stars_for_threshold: bool = Query(True),
    mask_stars_for_display: bool = Query(False),
    star_percentile: float = Query(99.9, ge=90.0, le=99.999),
    # --- Normalización ---
    normalize: str = Query("absolute", pattern="^(absolute|snr)$"),
    noise_block_size: int = Query(64, ge=16, le=256),
    # --- Threshold ---
    threshold_method: str = Query("percentile", pattern="^(percentile|sigma|absolute)$"),
    threshold_value: float | None = Query(None, gt=0),
    threshold_percentile: float = Query(99.0, ge=50.0, le=99.99),
    sigma_k: float = Query(5.0, gt=0, le=20),
    # --- Resto ---
    min_coverage: float = Query(0.5, ge=0.1, le=1.0),
    style: str = Query("signed", pattern="^(signed|absolute)$"),
    quality: int = Query(85, ge=10, le=100),
    obs_id: str | None = None,
    visit_from: int | None = None,
    visit_to: int | None = None,
):
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")

    for ep in (epoch_from, epoch_to):
        if ep not in hotspot.epochs:
            raise HTTPException(
                400,
                f"Epoch '{ep}' not available for sector '{sector_id}'. "
                f"Available: {hotspot.epochs}",
            )

    if threshold_method == "absolute" and threshold_value is None:
        raise HTTPException(400, "threshold_method='absolute' requires threshold_value")

    params = {
        "sector_id": sector_id,
        "epoch_from": epoch_from,
        "epoch_to": epoch_to,
        "detector": detector,
        "background_mode": background_mode,
        "background_downsample": background_downsample,
        "ratio": ratio,
        "mask_stars_for_threshold": mask_stars_for_threshold,
        "mask_stars_for_display": mask_stars_for_display,
        "star_percentile": star_percentile,
        "normalize": normalize,
        "noise_block_size": noise_block_size,
        "threshold_method": threshold_method,
        "threshold_value": threshold_value,
        "threshold_percentile": threshold_percentile,
        "sigma_k": sigma_k,
        "min_coverage": min_coverage,
        "style": style,
        "quality": quality,
    }
    if obs_id:
        params["obs_id"] = obs_id
    if visit_from is not None:
        params["visit_from"] = visit_from
    if visit_to is not None:
        params["visit_to"] = visit_to

    hit = cache.get("difference", params)
    if hit:
        meta_path = hit.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        return DifferenceResponse(
            url=cache.url_for(hit),
            cached=True,
            parameters=params,
            threshold=meta.get("threshold", 0.0),
            n_pixels_above=meta.get("n_pixels_above", 0),
            n_pixels_total=meta.get("n_pixels_total", 0),
            metadata=meta.get("metadata", {}),
        )

    try:
        hdul_t1 = fl.load_local_fits(
            sector_id=sector_id,
            epoch=epoch_from,
            detector=detector,
            obs_id=obs_id,
            visit=visit_from,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"T1: {e}")
    except ValueError as e:
        raise HTTPException(409, f"T1: {e}")

    try:
        hdul_t2 = fl.load_local_fits(
            sector_id=sector_id,
            epoch=epoch_to,
            detector=detector,
            obs_id=obs_id,
            visit=visit_to,
        )
    except FileNotFoundError as e:
        hdul_t1.close()
        raise HTTPException(404, f"T2: {e}")
    except ValueError as e:
        hdul_t1.close()
        raise HTTPException(409, f"T2: {e}")

    try:
        result = de.compute_difference(
            hdul_t1,
            hdul_t2,
            background_mode=background_mode,
            background_downsample=background_downsample,
            ratio=ratio,
            mask_stars_for_threshold=mask_stars_for_threshold,
            mask_stars_for_display=mask_stars_for_display,
            star_percentile=star_percentile,
            normalize=normalize,
            noise_block_size=noise_block_size,
            threshold_method=threshold_method,
            threshold_value=threshold_value,
            threshold_percentile=threshold_percentile,
            sigma_k=sigma_k,
            min_coverage=min_coverage,
            style=style,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception("diff_engine failed")
        raise HTTPException(500, f"Difference computation failed: {e}")
    finally:
        hdul_t1.close()
        hdul_t2.close()

    buf = de.to_webp_bytes(result["image"], quality=quality)
    path = cache.put("difference", params, buf)

    sidecar = {
        "threshold": result["threshold"],
        "normalize": result["normalize"],
        "n_pixels_above": result["n_pixels_above"],
        "n_pixels_total": result["n_pixels_total"],
        "metadata": result["metadata"],
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar))

    return DifferenceResponse(
        url=cache.url_for(path),
        cached=False,
        parameters=params,
        threshold=result["threshold"],
        n_pixels_above=result["n_pixels_above"],
        n_pixels_total=result["n_pixels_total"],
        metadata=result["metadata"],
    )

# JBL api router

@router.get("/known-objects", response_model=KnownObjectsResponse)
def known_objects(
    request: Request,
    sector_id: str,
    epoch: str,
    detector: int = Query(1, ge=1, le=6),
    radius_deg: float = Query(0.5, gt=0, le=5.0),
    obs_time_iso: str | None = None,
    two_pass: bool = Query(False),
    mag_required: bool = Query(False),
    max_vmag: float | None = Query(20.0),
    max_objects: int | None = Query(200, ge=1, le=5000),
    sort_by: str = Query("vmag", pattern="^(vmag|offset|name)$"),
    offset_margin: float = Query(1.4142, gt=1.0, le=3.0),
    include_pixels: bool = Query(True),
    ra_override_deg: float | None = Query(None, ge=0, le=360),
    dec_override_deg: float | None = Query(None, ge=-90, le=90),
    obs_id: str | None = None,
    visit: int | None = None,
):
    """
    Consulta JPL SB_Ident para identificar asteroides conocidos en el FOV.

    Por defecto usa two_pass=false y mag_required=false → respuesta rápida
    y generosa (decenas o miles de objetos en la eclíptica).

    `ra_override_deg` y `dec_override_deg` permiten probar otros centros sin
    modificar hotspots.json (útil para diagnóstico).
    """
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(
            400,
            f"Epoch '{epoch}' not available for sector '{sector_id}'. "
            f"Available: {hotspot.epochs}",
        )

    # 1) Abrir FITS una sola vez para DATE-OBS + vector de estado
    try:
        hdul = fl.load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"No FITS found: {e}")
    except ValueError as e:
        raise HTTPException(409, str(e))

    try:
        header = hdul["IMAGE"].header

        if obs_time_iso is None:
            date_obs = header.get("DATE-OBS")
            if not date_obs:
                raise HTTPException(400, "No DATE-OBS in FITS. Pasa obs_time_iso.")
            obs_time_iso = date_obs[:19]

        x_sc = header.get("X_SC"); y_sc = header.get("Y_SC"); z_sc = header.get("Z_SC")
        vx_sc = header.get("VX_SC"); vy_sc = header.get("VY_SC"); vz_sc = header.get("VZ_SC")
        sgt_lat = header.get("SGT_LAT_MIDPT", 0.0)
        sgt_lon = header.get("SGT_LON_MIDPT", 0.0)
    finally:
        hdul.close()

    # 2) Observador
    if all(v is not None for v in (x_sc, y_sc, z_sc)):
        parts = [str(x_sc), str(y_sc), str(z_sc)]
        if all(v is not None for v in (vx_sc, vy_sc, vz_sc)):
            parts.extend([str(vx_sc), str(vy_sc), str(vz_sc)])
        observer = {"xobs": ",".join(parts)}
    else:
        logger.warning("FITS sin X_SC/Y_SC/Z_SC; usando geodésico aproximado")
        observer = {"lat": sgt_lat, "lon": sgt_lon, "alt": 0.0}

    # 3) Centro del FOV
    ra_center = ra_override_deg if ra_override_deg is not None else hotspot.ra
    dec_center = dec_override_deg if dec_override_deg is not None else hotspot.dec

     # 4) Parámetros de cache (incluye TODOS los filtros)
    params = {
        "sector_id": sector_id,
        "epoch": epoch,
        "detector": detector,
        "radius_deg": radius_deg,
        "obs_time_iso": obs_time_iso,
        "two_pass": two_pass,
        "mag_required": mag_required,
        "max_vmag": max_vmag,
        "max_objects": max_objects,
        "sort_by": sort_by,
        "include_pixels": include_pixels,
        "offset_margin": offset_margin,
        "ra_center_deg": ra_center,
        "dec_center_deg": dec_center,
    }
    if obs_id: params["obs_id"] = obs_id
    if visit is not None: params["visit"] = visit

    hit = cache.get("known_objects", params)
    if hit:
        try:
            payload = json.loads(hit.read_text(encoding="utf-8"))
            return KnownObjectsResponse(**payload, cached=True)
        except Exception:
            logger.warning("Corrupt cache entry %s; recomputing", hit)

    # 5) Llamar a JPL con TODOS los filtros
    try:
        result = jpl_api.find_known_objects(
            ra_center_deg=ra_center,
            dec_center_deg=dec_center,
            radius_deg=radius_deg,
            obs_time_iso=obs_time_iso,
            observer=observer,
            two_pass=two_pass,
            mag_required=mag_required,
            filter_by_offset=True,
            offset_margin=offset_margin,
            max_vmag=max_vmag,
            max_objects=max_objects,
            sort_by=sort_by,
        )
    except jpl_api.JPLAPIError as e:
        logger.exception("JPL API failed")
        raise HTTPException(503, f"JPL API unavailable: {e}")

    # 6) Parsear (y opcionalmente calcular píxeles)
    objects_parsed = []
    if include_pixels and result["objects"]:
        # Batch: parsear todas las coordenadas
        ra_deg_list = [_parse_ra_deg(o.get("Astrometric RA (hh:mm:ss)")) for o in result["objects"]]
        dec_deg_list = [_parse_dec_deg(o.get('Astrometric Dec (dd mm\'ss")')) for o in result["objects"]]

        valid_ra = [r if r is not None else np.nan for r in ra_deg_list]
        valid_dec = [d if d is not None else np.nan for d in dec_deg_list]
        ra_arr = np.array(valid_ra, dtype=np.float64)
        dec_arr = np.array(valid_dec, dtype=np.float64)

        # Abrir FITS de nuevo para tener el WCS (una sola vez)
        try:
            hdul_for_wcs = fl.load_local_fits(
                sector_id=sector_id, epoch=epoch, detector=detector,
                obs_id=obs_id, visit=visit,
            )
            px, py = fl.world_to_pixel_batch(hdul_for_wcs, ra_arr, dec_arr)
            ny, nx = hdul_for_wcs["IMAGE"].data.shape
            hdul_for_wcs.close()
        except Exception as e:
            logger.warning("Could not compute pixels for known objects: %s", e)
            px = np.full_like(ra_arr, np.nan)
            py = np.full_like(dec_arr, np.nan)
            ny, nx = 0, 0

        for i, obj in enumerate(result["objects"]):
            xi = px[i] if np.isfinite(px[i]) else None
            yi = py[i] if np.isfinite(py[i]) else None
            in_frame = (
                xi is not None and yi is not None
                and 0 <= xi < nx and 0 <= yi < ny
            )
            objects_parsed.append(
                KnownObject(
                    name=obj.get("Object name", "Unknown"),
                    ra=obj.get("Astrometric RA (hh:mm:ss)"),
                    dec=obj.get('Astrometric Dec (dd mm\'ss")'),
                    vmag=obj.get("Visual magnitude (V)"),
                    ra_offset_arcsec=obj.get('Dist. from center RA (")'),
                    dec_offset_arcsec=obj.get('Dist. from center Dec (")'),
                    total_offset_arcsec=obj.get('Dist. from center Norm (")'),
                    rate_arcsec_min=obj.get("Rate of motion (arcsec/min)"),
                    ra_deg=float(ra_arr[i]) if np.isfinite(ra_arr[i]) else None,
                    dec_deg=float(dec_arr[i]) if np.isfinite(dec_arr[i]) else None,
                    pixel_x=float(xi) if xi is not None else None,
                    pixel_y=float(yi) if yi is not None else None,
                    in_frame=bool(in_frame),
                )
            )
    else:
        # Sin cálculo de píxeles (modo rápido)
        objects_parsed = [
            KnownObject(
                name=obj.get("Object name", "Unknown"),
                ra=obj.get("Astrometric RA (hh:mm:ss)"),
                dec=obj.get('Astrometric Dec (dd mm\'ss")'),
                vmag=obj.get("Visual magnitude (V)"),
                ra_offset_arcsec=obj.get('Dist. from center RA (")'),
                dec_offset_arcsec=obj.get('Dist. from center Dec (")'),
                total_offset_arcsec=obj.get('Dist. from center Norm (")'),
                rate_arcsec_min=obj.get("Rate of motion (arcsec/min)"),
            )
            for obj in result["objects"]
        ]

    response = KnownObjectsResponse(
        sector_id=sector_id,
        epoch=epoch,
        ra_center_deg=ra_center,
        dec_center_deg=dec_center,
        obs_time_iso=obs_time_iso,
        two_pass=two_pass,
        mag_required=mag_required,
        mode=result["mode"],
        n_objects_raw=result["n_objects_raw"],
        n_objects_after_offset=result["n_objects_after_offset"],
        n_objects_after_vmag=result["n_objects_after_vmag"],
        n_objects=result["n_objects"],
        filtered_out_offset=result["filtered_out_offset"],
        filtered_out_vmag=result["filtered_out_vmag"],
        truncated=result["truncated"],
        max_offset_arcsec_allowed=result["max_offset_arcsec_allowed"],
        max_vmag=max_vmag,
        max_objects=max_objects,
        n_first_pass=result["n_first_pass"],
        n_second_pass=result["n_second_pass"],
        objects=objects_parsed,
        cached=False,
    )

    data = response.model_dump()
    data.pop("cached", None)
    cache.put("known_objects", params, json.dumps(data, indent=2).encode("utf-8"))

    return response

@router.get("/identify", response_model=IdentifyResponse)
def identify(
    request: Request,
    sector_id: str,
    epoch: str,
    detector: int = Query(1, ge=1, le=6),
    pixel_x: int = Query(..., ge=0),
    pixel_y: int = Query(..., ge=0),
    tolerance_arcsec: float = Query(15.0, gt=0, le=120.0),
    max_vmag: float | None = Query(22.0),
    obs_id: str | None = None,
    visit: int | None = None,
):
    """
    El usuario hace click en (pixel_x, pixel_y) → devuelve RA/Dec y objetos
    conocidos dentro de `tolerance_arcsec`.

    - is_known=true     → hay ≥1 objeto catalogado a menos de tolerance_arcsec.
    - is_candidate=true → no hay objetos conocidos → candidato a descubrimiento.
    """
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(400, f"Epoch '{epoch}' not available")

    # 1) Convertir pixel → RA/Dec
    try:
        hdul = fl.load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector,
            obs_id=obs_id, visit=visit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"No FITS found: {e}")
    except ValueError as e:
        raise HTTPException(409, str(e))

    try:
        ny, nx = hdul["IMAGE"].data.shape
        if not (0 <= pixel_x < nx and 0 <= pixel_y < ny):
            raise HTTPException(
                400, f"pixel ({pixel_x},{pixel_y}) fuera de la imagen {nx}x{ny}",
            )
        ra_deg, dec_deg = fl.pixel_to_world(hdul, pixel_x, pixel_y)
        # También extraemos DATE-OBS + observer para la consulta JPL
        header = hdul["IMAGE"].header
        date_obs = header.get("DATE-OBS")
        x_sc, y_sc, z_sc = header.get("X_SC"), header.get("Y_SC"), header.get("Z_SC")
        vx_sc, vy_sc, vz_sc = header.get("VX_SC"), header.get("VY_SC"), header.get("VZ_SC")
    finally:
        hdul.close()

    obs_time_iso = date_obs[:19] if date_obs else None
    if obs_time_iso is None:
        raise HTTPException(400, "No DATE-OBS in FITS")

    # 2) Observer
    if all(v is not None for v in (x_sc, y_sc, z_sc)):
        parts = [str(x_sc), str(y_sc), str(z_sc)]
        if all(v is not None for v in (vx_sc, vy_sc, vz_sc)):
            parts.extend([str(vx_sc), str(vy_sc), str(vz_sc)])
        observer = {"xobs": ",".join(parts)}
    else:
        observer = {
            "lat": header.get("SGT_LAT_MIDPT", 0.0),
            "lon": header.get("SGT_LON_MIDPT", 0.0),
            "alt": 0.0,
        }

    # 3) Consultar JPL con FOV centrado en el pixel
    radius_deg = tolerance_arcsec / 3600.0
    params = {
        "sector_id": sector_id, "epoch": epoch, "detector": detector,
        "pixel_x": pixel_x, "pixel_y": pixel_y,
        "tolerance_arcsec": tolerance_arcsec,
        "max_vmag": max_vmag, "obs_time_iso": obs_time_iso,
    }

    hit = cache.get("identify", params)
    if hit:
        try:
            payload = json.loads(hit.read_text(encoding="utf-8"))
            return IdentifyResponse(**payload, cached=True)
        except Exception:
            logger.warning("Corrupt identify cache, recomputing")

    try:
        result = jpl_api.find_known_objects(
            ra_center_deg=ra_deg,
            dec_center_deg=dec_deg,
            radius_deg=radius_deg,
            obs_time_iso=obs_time_iso,
            observer=observer,
            two_pass=False,
            mag_required=False,
            filter_by_offset=True,
            offset_margin=1.0,
            max_vmag=max_vmag,
            max_objects=50,
            sort_by="offset",
        )
    except jpl_api.JPLAPIError as e:
        logger.exception("JPL identify failed")
        raise HTTPException(503, f"JPL API unavailable: {e}")

    # 4) Parsear objetos cercanos
    known = []
    for obj in result["objects"]:
        offset_arcsec = None
        try:
            offset_arcsec = float(
                str(obj.get('Dist. from center Norm (")', "")).strip() or "nan"
            )
        except ValueError:
            offset_arcsec = None

        if offset_arcsec is not None and offset_arcsec > tolerance_arcsec:
            continue

        known.append(
            KnownObject(
                name=obj.get("Object name", "Unknown"),
                ra=obj.get("Astrometric RA (hh:mm:ss)"),
                dec=obj.get('Astrometric Dec (dd mm\'ss")'),
                vmag=obj.get("Visual magnitude (V)"),
                ra_offset_arcsec=obj.get('Dist. from center RA (")'),
                dec_offset_arcsec=obj.get('Dist. from center Dec (")'),
                total_offset_arcsec=obj.get('Dist. from center Norm (")'),
                rate_arcsec_min=obj.get("Rate of motion (arcsec/min)"),
                ra_deg=_parse_ra_deg(obj.get("Astrometric RA (hh:mm:ss)")),
                dec_deg=_parse_dec_deg(obj.get('Astrometric Dec (dd mm\'ss")')),
            )
        )

    response = IdentifyResponse(
        sector_id=sector_id,
        epoch=epoch,
        detector=detector,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        tolerance_arcsec=tolerance_arcsec,
        n_known=len(known),
        known_objects=known,
        is_known=len(known) > 0,
        is_candidate=len(known) == 0,
        cached=False,
    )

    data = response.model_dump()
    data.pop("cached", None)
    cache.put("identify", params, json.dumps(data, indent=2).encode("utf-8"))

    return response