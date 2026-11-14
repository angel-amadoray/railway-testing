import json
import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.core import cache
from app.models.advanced import (
    PairSpectrumResponse,
    BrightPixel,
    BrightestPixelsResponse,
    SuggestedTarget,
    SuggestTargetResponse
)
from app.services import advanced_analyzer as aa
import numpy as np
from app.services import fits_loader as fl

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/advanced-analytics", tags=["advanced-analytics"])


@router.get("/pair-spectrum", response_model=PairSpectrumResponse)
def pair_spectrum(
    request: Request,
    sector_id: str,
    epoch: str,
    ra_deg: float = Query(..., ge=0, le=360),
    dec_deg: float = Query(..., ge=-90, le=90),
    obs_id: str | None = None,
    visit: int | None = None,
):
    """
    Análisis de 2 bandas (par de detectores del mismo telescopio).

    Para un (RA, Dec) determina qué par cubre el punto
    (D1+D4, D2+D5 o D3+D6), mide flux en ambos y clasifica según
    el ratio F(long) / F(short).
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

    params = {
        "sector_id": sector_id,
        "epoch": epoch,
        "ra_deg": round(ra_deg, 5),
        "dec_deg": round(dec_deg, 5),
    }
    if obs_id:
        params["obs_id"] = obs_id
    if visit is not None:
        params["visit"] = visit

    hit = cache.get("advanced_pair", params)
    if hit:
        try:
            payload = json.loads(hit.read_text(encoding="utf-8"))
            return PairSpectrumResponse(**payload, cached=True)
        except Exception:
            logger.warning("Corrupt advanced_pair cache; recomputing")

    try:
        result = aa.analyze_pair(
            sector_id=sector_id,
            epoch=epoch,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            obs_id=obs_id,
            visit=visit,
        )
    except Exception as e:
        logger.exception("advanced_analyzer failed")
        raise HTTPException(500, f"Analysis failed: {e}")

    response = PairSpectrumResponse(**result, cached=False)

    data = response.model_dump()
    data.pop("cached", None)
    cache.put(
        "advanced_pair",
        params,
        json.dumps(data, indent=2).encode("utf-8"),
    )

    return response

@router.get("/brightest-pixels", response_model=BrightestPixelsResponse)
def brightest_pixels(
    request: Request,
    sector_id: str,
    epoch: str,
    detector: int = Query(1, ge=1, le=6),
    top_n: int = Query(10, ge=1, le=50),
    obs_id: str | None = None,
    visit: int | None = None,
):
    """
    Devuelve los N píxeles con mayor flujo del detector, con sus RA/Dec
    correspondientes. Útil para encontrar un target real para el análisis.
    """
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(400, f"Epoch '{epoch}' not available")

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
        clean = fl.get_clean_image(hdul, subtract_zodi=True)
        valid = (clean > 0) & np.isfinite(clean)
        if valid.sum() == 0:
            return BrightestPixelsResponse(
                sector_id=sector_id, epoch=epoch, detector=detector,
                n_pixels=0, pixels=[],
            )

        masked = np.where(valid, clean, 0.0)
        flat = masked.ravel()
        top_idx = np.argpartition(flat, -top_n)[-top_n:]

        ys, xs = np.unravel_index(top_idx, clean.shape)

        wcs = fl.get_spatial_wcs(hdul)
        pixels = []
        for x, y in zip(xs, ys):
            try:
                ra, dec = wcs.all_pix2world(float(x), float(y), 0)
                pixels.append(BrightPixel(
                    pixel_x=int(x),
                    pixel_y=int(y),
                    flux=float(clean[y, x]),
                    ra_deg=float(ra),
                    dec_deg=float(dec),
                ))
            except Exception:
                continue
    finally:
        hdul.close()

    pixels.sort(key=lambda p: -p.flux)
    return BrightestPixelsResponse(
        sector_id=sector_id, epoch=epoch, detector=detector,
        n_pixels=len(pixels), pixels=pixels[:top_n],
    )

@router.get("/suggest-target", response_model=SuggestTargetResponse)
def suggest_target(
    request: Request,
    sector_id: str,
    epoch: str,
    detector_short: int = Query(1, ge=1, le=6),
    detector_long: int = Query(4, ge=1, le=6),
    obs_id: str | None = None,
    visit: int | None = None,
    min_snr: float = Query(10.0, gt=0),
    max_candidates: int = Query(30, ge=5, le=100),
):
    """
    Encuentra un pixel que tenga flujo válido (>0) en AMBOS detectores.

    Estrategia:
      1. Carga D_short y D_long.
      2. Toma los top `max_candidates` píxeles brillantes de D_short.
      3. Para cada uno: convierte a RA/Dec, verifica que D_long tenga flujo
         positivo en la misma posición.
      4. Devuelve el primero que cumpla snr > min_snr en ambos.
    """
    hotspot = request.app.state.hotspots.get(sector_id)
    if hotspot is None:
        raise HTTPException(404, f"Sector '{sector_id}' not found")
    if epoch not in hotspot.epochs:
        raise HTTPException(400, f"Epoch '{epoch}' not available")

    try:
        hdul_s = fl.load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector_short,
            obs_id=obs_id, visit=visit,
        )
        hdul_l = fl.load_local_fits(
            sector_id=sector_id, epoch=epoch, detector=detector_long,
            obs_id=obs_id, visit=visit,
        )
    except FileNotFoundError as e:
        raise HTTPException(404, f"Missing FITS: {e}")

    try:
        clean_s = fl.get_clean_image(hdul_s, subtract_zodi=True)
        clean_l = fl.get_clean_image(hdul_l, subtract_zodi=True)

        valid_s = (clean_s > 0) & np.isfinite(clean_s)
        if valid_s.sum() == 0:
            return SuggestTargetResponse(
                sector_id=sector_id, epoch=epoch,
                pair=f"D{detector_short}+D{detector_long}",
                reason="No valid pixels in short detector",
            )

        masked_s = np.where(valid_s, clean_s, 0.0)
        flat = masked_s.ravel()
        top_idx = np.argpartition(flat, -max_candidates)[-max_candidates:]
        ys, xs = np.unravel_index(top_idx, clean_s.shape)

        wcs_s = fl.get_spatial_wcs(hdul_s)
        wcs_l = fl.get_spatial_wcs(hdul_l)

        # Importar helpers del advanced_analyzer
        from app.services.advanced_analyzer import (
            _photometry_detector, _compute_ratio,
        )
        from app.services.fits_loader import get_spatial_wcs as _gwcs

        for x_s, y_s in zip(xs, ys):
            try:
                ra, dec = wcs_s.all_pix2world(float(x_s), float(y_s), 0)
                ra = float(np.asarray(ra).ravel()[0])
                dec = float(np.asarray(dec).ravel()[0])
            except Exception:
                continue

            # Verificar que el punto cae en D_long
            try:
                raw = wcs_l.all_world2pix(ra, dec, 0)
                arr = np.asarray(raw).ravel()
                if arr.size < 2:
                    continue
                x_l = int(round(float(arr[0])))
                y_l = int(round(float(arr[1])))
                ny_l, nx_l = clean_l.shape
                if not (0 <= x_l < nx_l and 0 <= y_l < ny_l):
                    continue
            except Exception:
                continue

            # Medir flujo en ambos usando la fotometría estándar
            short_r = _photometry_detector(
                sector_id, epoch, detector_short, ra, dec, obs_id, visit
            )
            long_r = _photometry_detector(
                sector_id, epoch, detector_long, ra, dec, obs_id, visit
            )

            if not (short_r["available"] and long_r["available"]):
                continue
            if short_r["flux"] is None or long_r["flux"] is None:
                continue
            if short_r["flux"] <= 0 or long_r["flux"] <= 0:
                continue
            if (short_r["snr"] or 0) < min_snr or (long_r["snr"] or 0) < min_snr:
                continue

            return SuggestTargetResponse(
                sector_id=sector_id, epoch=epoch,
                pair=f"D{detector_short}+D{detector_long}",
                target=SuggestedTarget(
                    ra_deg=ra,
                    dec_deg=dec,
                    flux_short=short_r["flux"],
                    flux_long=long_r["flux"],
                    snr_short=short_r["snr"],
                    snr_long=long_r["snr"],
                    pixel_short=(short_r["pixel_x"], short_r["pixel_y"]),
                    pixel_long=(long_r["pixel_x"], long_r["pixel_y"]),
                ),
                candidates_examined=max_candidates,
                reason="Found pixel valid in both detectors",
            )
    finally:
        hdul_s.close()
        hdul_l.close()

    return SuggestTargetResponse(
        sector_id=sector_id, epoch=epoch,
        pair=f"D{detector_short}+D{detector_long}",
        candidates_examined=max_candidates,
        reason=(
            "No candidate passed the filter. All top pixels in the short "
            "detector are either saturated or invalid in the long detector. "
            "Try reducing min_snr or increasing max_candidates."
        ),
    )